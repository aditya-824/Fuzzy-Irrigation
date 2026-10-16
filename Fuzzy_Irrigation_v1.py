"""
Fuzzy Logic Controller (Mamdani) for irrigation duration - India
=================================================================
Inputs : T  = daily max air temperature (deg C)       [NASA POWER: T2M_MAX]
         SM = root-zone soil wetness (% of saturation) [NASA POWER: GWETROOT x 100]
Output : irrigation duration (minutes per irrigation event)

Pipeline
  1. Download daily data for any lat/lon in India from NASA POWER (no API key).
  2. Build a "teacher" duration from FAO-56 water balance
     (Hargreaves ET0 x Kc, effective rain, root-zone depletion).
  3. Mamdani FLC: triangular/trapezoidal MFs, 9 rules, min-AND, max-aggregation,
     centroid defuzzification.
  4. (Optional) tune MF breakpoints with Differential Evolution against the teacher.
  5. Report MAE / R2 on a held-out test period.

Usage:
  pip install numpy pandas requests scipy
  python Fuzzy_Irrigation_v1.py --lat 30.90 --lon 75.85 --crop wheat
  python Fuzzy_Irrigation_v1.py --offline          # synthetic data, no internet
"""
import argparse
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
KC_MID = {"wheat": 1.15, "rice": 1.20, "maize": 1.20, "cotton": 1.15,
          "sugarcane": 1.25, "tomato": 1.15, "potato": 1.15}   # FAO-56 Table 12 (mid-season)
TAW_MM = 100.0        # total available water in root zone (mm) - set from soil/root depth
APP_RATE_MM_H = 15.0  # sprinkler ~10-20 mm/h; drip is lower - change to your system
MAX_MIN = 90.0        # maximum controller output (minutes)
Y = np.linspace(0, MAX_MIN, 181)  # output universe of discourse


# ----------------------------------------------------------------------------
# Data acquisition
# ----------------------------------------------------------------------------
def fetch_power(lat, lon, start="20190101", end="20241231"):
    import requests
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = dict(parameters="T2M_MAX,T2M_MIN,GWETROOT,PRECTOTCORR", community="AG",
                  latitude=lat, longitude=lon, start=start, end=end, format="JSON")
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    p = r.json()["properties"]["parameter"]
    df = pd.DataFrame(p)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.replace(-999, np.nan).dropna()
    df.columns = ["tmax", "tmin", "wetness", "rain"]
    df["sm"] = df["wetness"] * 100.0
    return df


def synthetic(lat, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n)
    doy = idx.dayofyear.values
    tmax = 32 + 9 * np.sin(2 * np.pi * (doy - 90) / 365) + rng.normal(0, 2, n)
    tmin = tmax - rng.uniform(8, 14, n)
    monsoon = ((doy > 170) & (doy < 270))
    rain = np.where(rng.random(n) < np.where(monsoon, 0.5, 0.05), rng.gamma(2, 8, n), 0.0)
    sm = np.clip(45 + 25 * monsoon + rng.normal(0, 12, n), 5, 100)
    return pd.DataFrame(dict(tmax=tmax, tmin=tmin, rain=rain, sm=sm), index=idx)


# ----------------------------------------------------------------------------
# FAO-56 teacher signal (proxy for "ideal" irrigation duration)
# ----------------------------------------------------------------------------
def extraterrestrial_radiation_mm(lat_deg, doy):
    phi = np.deg2rad(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))
    return 0.408 * ra  # mm/day


def hargreaves_et0(df, lat):
    tmean = (df.tmax + df.tmin) / 2
    ra = extraterrestrial_radiation_mm(lat, df.index.dayofyear.values)
    return 0.0023 * (tmean + 17.8) * np.sqrt(np.clip(df.tmax - df.tmin, 0, None)) * ra


def teacher_duration(df, lat, crop):
    etc = hargreaves_et0(df, lat) * KC_MID[crop]
    pe = 0.8 * df.rain                                   # effective rain (simple)
    depletion = (1 - df.sm / 100.0) * TAW_MM + etc - pe  # mm below field capacity
    depth = np.clip(depletion - 0.45 * TAW_MM, 0, None)   # irrigate only beyond ~45 % depletion (MAD)
    return np.clip(60 * depth / APP_RATE_MM_H, 0, MAX_MIN)


# ----------------------------------------------------------------------------
# Fuzzy machinery (vectorised Mamdani)
# ----------------------------------------------------------------------------
def trapmf(x, a, b, c, d):
    x = np.asarray(x, float)
    up = np.where(x >= b, 1.0, np.where(x <= a, 0.0, (x - a) / max(b - a, 1e-9)))
    dn = np.where(x <= c, 1.0, np.where(x >= d, 0.0, (d - x) / max(d - c, 1e-9)))
    return np.minimum(up, dn)


def trimf(x, a, b, c):
    return trapmf(x, a, b, b, c)


DEFAULT_P = np.array([20, 40, 60, 80,        # soil-moisture breakpoints m1<m2<m3<m4
                      10, 25, 45, 70])       # output centres: Short, Medium, Long, VeryLong

# (temp set, moisture set, output set)
# temp: 0 Cool 1 Warm 2 Hot | moisture: 0 Dry 1 Moist 2 Wet | out: 0 None 1 Short 2 Med 3 Long 4 VLong
RULES = [(0, 0, 2), (1, 0, 3), (2, 0, 4),
         (0, 1, 1), (1, 1, 2), (2, 1, 3),
         (0, 2, 0), (1, 2, 0), (2, 2, 1)]


def temp_mfs(T):
    return np.stack([trapmf(T, 0, 0, 22, 28),
                     trimf(T, 24, 31, 38),
                     trapmf(T, 34, 40, 55, 55)], axis=1)


def moist_mfs(M, p):
    m1, m2, m3, m4 = np.sort(p[:4])
    return np.stack([trapmf(M, 0, 0, m1, m2),
                     trimf(M, m1, (m2 + m3) / 2, m4),
                     trapmf(M, m3, m4, 100, 100)], axis=1)


def out_mfs(p):
    c1, c2, c3, c4 = np.sort(p[4:])
    return np.stack([trimf(Y, 0, 0, c1), trimf(Y, 0, c1, c2), trimf(Y, c1, c2, c3),
                     trimf(Y, c2, c3, c4), trimf(Y, c3, c4, MAX_MIN)])


def flc(T, M, p=DEFAULT_P):
    T, M = np.atleast_1d(T).astype(float), np.atleast_1d(M).astype(float)
    tm, mm, om = temp_mfs(T), moist_mfs(M, p), out_mfs(p)
    w = np.stack([np.minimum(tm[:, t], mm[:, m]) for t, m, _ in RULES], axis=1)  # (N,R)
    consequents = om[[o for _, _, o in RULES]]                                   # (R,Y)
    agg = np.minimum(w[:, :, None], consequents[None]).max(axis=1)               # (N,Y)
    den = agg.sum(axis=1)
    return np.where(den > 1e-9, (agg * Y).sum(axis=1) / np.maximum(den, 1e-9), 0.0)


# ----------------------------------------------------------------------------
# Tuning + evaluation
# ----------------------------------------------------------------------------
def tune(T, M, target, maxiter=25, seed=1):
    from scipy.optimize import differential_evolution
    bounds = [(5, 30), (25, 50), (45, 70), (65, 95),
              (3, 20), (15, 40), (30, 60), (50, 85)]
    res = differential_evolution(lambda p: np.mean(np.abs(flc(T, M, p) - target)),
                                 bounds, maxiter=maxiter, popsize=10, seed=seed, polish=False)
    return res.x


def metrics(y, yhat):
    mae = np.mean(np.abs(y - yhat))
    ss_res, ss_tot = np.sum((y - yhat) ** 2), np.sum((y - y.mean()) ** 2)
    return mae, 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=30.90)   # Ludhiana, Punjab
    ap.add_argument("--lon", type=float, default=75.85)
    ap.add_argument("--crop", default="wheat", choices=KC_MID)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-tune", action="store_true")
    a = ap.parse_args()

    df = synthetic(a.lat) if a.offline else fetch_power(a.lat, a.lon)
    df["target"] = teacher_duration(df, a.lat, a.crop)

    split = int(len(df) * 0.7)                       # chronological split
    tr, te = df.iloc[:split], df.iloc[split:]

    p = DEFAULT_P
    if not a.no_tune:
        p = tune(tr.tmax.values, tr.sm.values, tr.target.values)
        print("Tuned parameters:", np.round(np.sort(p[:4]), 1), np.round(np.sort(p[4:]), 1))

    for name, d in (("train", tr), ("test", te)):
        mae, r2 = metrics(d.target.values, flc(d.tmax.values, d.sm.values, p))
        print(f"{name:5s}  MAE = {mae:5.1f} min   R2 = {r2:5.2f}")

    print("\nSample decisions (Tmax degC, soil wetness %) -> minutes")
    for T, M in [(42, 15), (35, 35), (28, 50), (38, 70), (25, 85)]:
        print(f"  T={T:>2}  SM={M:>2}  ->  {flc(T, M, p)[0]:5.1f} min")


if __name__ == "__main__":
    main()