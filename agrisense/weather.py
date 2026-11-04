"""Weather inputs: synthetic Nile-Delta demo series, user CSV, NASA POWER CSV."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from . import et0

COLUMNS = ["date", "tmax", "tmin", "rhmean", "wind2", "rs", "rain"]
BENHA = dict(name="Benha, Qalyubia", lat=30.46, lon=31.18, elev=13.0)
_RAIN_P = np.array([.06, .045, .035, .015, .003, 0, 0, 0, 0, .015, .03, .05])  # P(rain day) per month


def synthetic_weather(start="2022-01-01", end="2025-12-31", lat=BENHA["lat"], elev=BENHA["elev"], seed=7):
    """Synthetic daily weather with a Nile-Delta-like seasonal cycle.

    FOR DEMOS AND UNIT TESTS ONLY - this is not measured data and must never be used for
    scientific conclusions. Replace it with NASA POWER or weather-station data.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n, doy = len(dates), dates.dayofyear.to_numpy()

    def seasonal(peak_doy):
        return np.cos(2 * np.pi * (doy - peak_doy) / 365)

    def ar1(phi, sd):
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + rng.normal(0, sd)
        return x

    rain = np.where(rng.random(n) < _RAIN_P[dates.month.to_numpy() - 1], rng.exponential(4.0, n) + 0.5, 0.0)
    clear = 1 / (1 + np.exp(-(1.6 + 0.9 * seasonal(200) + ar1(0.6, 0.8))))   # 0 overcast ... 1 clear
    clear = np.where(rain > 0, clear * 0.5, clear)
    ra = et0.extraterrestrial_radiation(lat, doy)
    rs = (0.55 + 0.45 * clear) * (0.75 + 2e-5 * elev) * ra
    tmean = 21.5 + 7.6 * seasonal(205) + ar1(0.7, 1.6)
    dtr = (11.0 + 1.2 * seasonal(200)) * (0.6 + 0.5 * clear)                 # diurnal temperature range
    rh = 56 + 6 * seasonal(10) - 10 * (clear - 0.8) + rng.normal(0, 6, n) + np.where(rain > 0, 12, 0)
    wind = np.exp(rng.normal(np.log(2.0 * (1 + 0.15 * seasonal(100))), 0.35, n))
    df = pd.DataFrame({"date": dates, "tmax": tmean + dtr / 2, "tmin": tmean - dtr / 2,
                       "rhmean": np.clip(rh, 20, 97), "wind2": np.clip(wind, 0.5, 7.0),
                       "rs": rs, "rain": rain})
    df[COLUMNS[1:]] = df[COLUMNS[1:]].round(2)
    return df


def _clean(df):
    num = [c for c in COLUMNS if c != "date"]
    df = df[COLUMNS].sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df[num] = df[num].astype(float).interpolate(limit=5, limit_direction="both")
    return df.dropna().reset_index(drop=True)


def load_csv(file, lat=None):
    """Read a CSV with columns date,tmax,tmin,rhmean,wind2[,rs][,rain] (case-insensitive).

    rhmean in %, wind2 = wind at 2 m (m/s), rs in MJ m-2 day-1, rain in mm.
    A missing rs is estimated from the temperature range (needs lat); missing rain becomes 0.
    """
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = {"date", "tmax", "tmin", "rhmean", "wind2"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}. Expected: {COLUMNS}")
    df["date"] = pd.to_datetime(df["date"])
    if "rain" not in df:
        df["rain"] = 0.0
    if "rs" not in df:
        if lat is None:
            raise ValueError("Column 'rs' is missing and no latitude was given to estimate it")
        df["rs"] = et0.rs_from_temperature(df["tmax"], df["tmin"], lat, df["date"].dt.dayofyear.to_numpy())
    return _clean(df)


_NASA = {"T2M_MAX": "tmax", "T2M_MIN": "tmin", "RH2M": "rhmean", "WS2M": "wind2",
         "ALLSKY_SFC_SW_DWN": "rs", "PRECTOTCORR": "rain"}


def _solar_to_mj(header):
    """Factor that turns the unit named in the file header for ALLSKY_SFC_SW_DWN into MJ m-2 day-1."""
    for line in header.splitlines():
        if line.strip().startswith("ALLSKY_SFC_SW_DWN") and "(" in line:
            unit = line[line.rfind("(") + 1:].split(")")[0].replace(" ", "").lower()
            if unit.startswith("mj"):
                return 1.0
            if unit.startswith(("kw-hr", "kwh")):
                return 3.6
            raise ValueError(f"Unexpected unit '{unit}' for ALLSKY_SFC_SW_DWN (expected MJ/m^2/day)")
    return 1.0   # no unit line in the file: the AG community delivers MJ m-2 day-1


def _parse_nasa(text):
    """NASA POWER daily CSV text -> table with this package's columns; -999 gaps stay NaN."""
    header, _, body = text.rpartition("-END HEADER-")
    df = pd.read_csv(io.StringIO(body.strip())).replace(-999, np.nan)
    if "DOY" in df.columns:
        df["date"] = pd.to_datetime(df["YEAR"].astype(int).astype(str) + "-01-01") \
            + pd.to_timedelta(df["DOY"].astype(int) - 1, unit="D")
    else:
        df["date"] = pd.to_datetime(dict(year=df["YEAR"], month=df["MO"], day=df["DY"]))
    missing = set(_NASA) - set(df.columns)
    if missing:
        raise ValueError(f"NASA POWER file lacks columns: {sorted(missing)}")
    df = df.rename(columns=_NASA)
    df["rs"] = df["rs"].astype(float) * _solar_to_mj(header)
    return df


def load_nasa_power(file):
    """Read a NASA POWER daily CSV (community AG) fetched with scripts/fetch_nasa_power.py."""
    text = file.read() if hasattr(file, "read") else open(file, encoding="utf-8").read()
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return _clean(_parse_nasa(text))


# ---------- automatic download (NASA POWER, no API key) ----------
NASA_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_LAG_DAYS = 8   # the newest days are not published yet: meteorology ~2-3 days, solar radiation ~5-7 days behind


class WeatherDownloadError(ValueError):
    """The online weather could not be downloaded or read (a ValueError, so the app shows it as a message)."""


def auto_window(plant_date, today=None, years=4, lag_days=NASA_LAG_DAYS):
    """(start, end) to download for the automatic source: the planting date and at least `years` of history."""
    end = pd.Timestamp(today if today is not None else pd.Timestamp.today()).normalize() - pd.Timedelta(days=lag_days)
    start = min(pd.Timestamp(plant_date).normalize(), end - pd.DateOffset(years=years))
    return max(start, pd.Timestamp("1984-01-01")), end   # NASA POWER solar data starts in 1984


def fetch_nasa_power(lat, lon, start, end, timeout=60, retries=2, pause=2.0):
    """Download daily NASA POWER weather (community AG, no API key) for one point; returns the clean table.

    Needs internet. Days NASA has not published yet (still -999 at the end of the period) are cut off, not
    filled. Raises WeatherDownloadError with a readable message when the download fails.
    """
    import time

    import requests

    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Latitude must be within -90..90 and longitude within -180..180")
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if end < start:
        raise ValueError("The end date is before the start date")
    params = {"parameters": ",".join(_NASA), "community": "ag", "longitude": round(float(lon), 2),
              "latitude": round(float(lat), 2), "start": start.strftime("%Y%m%d"),
              "end": end.strftime("%Y%m%d"), "format": "CSV"}
    problem = "no answer"
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(pause * attempt)
        try:
            r = requests.get(NASA_URL, params=params, timeout=timeout)
        except requests.RequestException as err:           # no internet, DNS failure, timeout ...
            problem = type(err).__name__
            continue
        if r.status_code == 429 or r.status_code >= 500:   # server busy or down: try again
            problem = f"HTTP {r.status_code}"
            continue
        if r.status_code >= 400:                            # our request is wrong: retrying will not help
            raise WeatherDownloadError(f"NASA POWER rejected the request (HTTP {r.status_code}): "
                                       f"{' '.join(r.text.split())[:200]}")
        text = r.text
        break
    else:
        raise WeatherDownloadError(f"Could not download weather from NASA POWER ({problem}). Check the internet "
                                   "connection and try again, or use the demo / CSV source.")
    if "-END HEADER-" not in text:
        raise WeatherDownloadError("NASA POWER sent an unexpected reply: " + " ".join(text.split())[:150])
    try:
        raw = _parse_nasa(text)
    except ValueError as err:
        raise WeatherDownloadError(f"Could not read the NASA POWER reply: {err}") from err
    complete = raw[COLUMNS[1:]].notna().all(axis=1).to_numpy()
    if not complete.any():
        raise WeatherDownloadError("NASA POWER has no data for this period and location")
    return _clean(raw.iloc[: np.flatnonzero(complete)[-1] + 1])


# ---------- finishing a season that is not over yet ----------
def _doy365(dates):
    """Day of year 1..365 with 29 February folded into 28 February, so every year lines up."""
    d = pd.DatetimeIndex(dates)
    month, day = d.month.to_numpy(), d.day.to_numpy()
    doy = np.where(d.is_leap_year & (month > 2), d.dayofyear.to_numpy() - 1, d.dayofyear.to_numpy())
    return np.where((month == 2) & (day == 29), 59, doy)


def extend_with_climatology(df, until, min_years=3, smooth=7):
    """Return the table with an `observed` column; days after the last observed one, up to `until`, are appended.

    Appended days are an ESTIMATE, not weather: the mean of the same calendar day in all earlier years,
    smoothed over `smooth` days. Their rain is the long-term daily mean, which is below the 2 mm threshold of
    irrigation.simulate in most climates, so it is effectively ignored. Needs every calendar day in at least
    `min_years` different years.
    """
    num = COLUMNS[1:]
    out = df[COLUMNS].copy()
    out["observed"] = True
    until, last = pd.Timestamp(until).normalize(), out["date"].max()
    if until <= last:
        return out
    groups = out[num].groupby(_doy365(out["date"]))
    if len(groups) < 365 or groups.size().min() < min_years:
        raise ValueError(f"Completing the season needs every calendar day in at least {min_years} different "
                         "years of observed weather")
    clim = groups.mean()                                            # index 1..365
    k = smooth // 2
    arr = clim.to_numpy()
    padded = np.concatenate([arr[-k:], arr, arr[:k]]) if k else arr   # wrap around the year end
    kernel = np.ones(2 * k + 1) / (2 * k + 1)
    smoothed = np.column_stack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(arr.shape[1])])
    clim = pd.DataFrame(smoothed, index=clim.index, columns=num)
    days = pd.date_range(last + pd.Timedelta(days=1), until, freq="D")
    fill = clim.loc[_doy365(days)].reset_index(drop=True).round(2)
    fill.insert(0, "date", days)
    fill["observed"] = False
    return pd.concat([out, fill], ignore_index=True)


def add_eto(df, lat, elev):
    """Return a copy of df with FAO-56 Penman-Monteith ET0 (mm/day) in column 'eto'."""
    out = df.copy()
    out["eto"] = et0.eto_penman_monteith(out["tmax"], out["tmin"], out["rs"], out["wind2"], lat, elev,
                                         out["date"].dt.dayofyear.to_numpy(), rhmean=out["rhmean"])
    return out
