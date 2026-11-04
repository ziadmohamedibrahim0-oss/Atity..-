"""Reference evapotranspiration (ET0): FAO-56 Penman-Monteith and Hargreaves-Samani.

Equation numbers refer to FAO Irrigation and Drainage Paper 56 (Allen et al., 1998).
All functions accept scalars, numpy arrays or pandas Series.
"""
from __future__ import annotations

import numpy as np

GSC = 0.0820       # solar constant, MJ m-2 min-1
SIGMA = 4.903e-9   # Stefan-Boltzmann constant, MJ K-4 m-2 day-1


def sat_vp(t):
    """Saturation vapour pressure (kPa) at air temperature t (deg C) - Eq. 11."""
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def vp_slope(t):
    """Slope of the saturation vapour-pressure curve (kPa/deg C) - Eq. 13."""
    return 4098.0 * sat_vp(t) / (t + 237.3) ** 2


def pressure(z):
    """Atmospheric pressure (kPa) at elevation z (m) - Eq. 7."""
    return 101.3 * ((293.0 - 0.0065 * z) / 293.0) ** 5.26


def wind_at_2m(uz, z=10.0):
    """Convert wind speed measured at height z (m) to 2 m - Eq. 47."""
    return np.asarray(uz, float) * 4.87 / np.log(67.8 * z - 5.42)


def _geometry(lat_deg, doy):
    phi = np.radians(lat_deg)
    doy = np.asarray(doy, float)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)                    # Eq. 23
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)              # Eq. 24
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))      # Eq. 25
    return phi, dr, delta, ws


def extraterrestrial_radiation(lat_deg, doy):
    """Extraterrestrial radiation Ra (MJ m-2 day-1) - Eq. 21."""
    phi, dr, delta, ws = _geometry(lat_deg, doy)
    return (24 * 60 / np.pi) * GSC * dr * (
        ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))


def daylight_hours(lat_deg, doy):
    """Maximum possible sunshine duration N (h) - Eq. 34."""
    return 24 / np.pi * _geometry(lat_deg, doy)[3]


def rs_from_sunshine(n, big_n, ra, a_s=0.25, b_s=0.50):
    """Solar radiation from measured sunshine hours - Eq. 35 (Angstrom)."""
    return (a_s + b_s * np.asarray(n, float) / big_n) * ra


def rs_from_temperature(tmax, tmin, lat_deg, doy, krs=0.16):
    """Solar radiation from the temperature range - Eq. 50 (krs 0.16 inland, 0.19 coastal)."""
    ra = extraterrestrial_radiation(lat_deg, doy)
    return krs * np.sqrt(np.maximum(np.asarray(tmax, float) - np.asarray(tmin, float), 0)) * ra


def eto_penman_monteith(tmax, tmin, rs, u2, lat_deg, z, doy,
                        rhmean=None, rhmax=None, rhmin=None):
    """Daily FAO-56 Penman-Monteith reference evapotranspiration (mm/day) - Eq. 6.

    rs: solar radiation (MJ m-2 day-1); u2: wind speed at 2 m (m/s); z: elevation (m).
    Humidity: give (rhmax and rhmin) or rhmean (%); with neither, Tdew ~ Tmin (Eq. 48).
    Soil heat flux G is taken as 0 for daily time steps.
    """
    tmax, tmin = np.asarray(tmax, float), np.asarray(tmin, float)
    rs, u2 = np.asarray(rs, float), np.asarray(u2, float)
    tmean = (tmax + tmin) / 2
    es = (sat_vp(tmax) + sat_vp(tmin)) / 2                              # Eq. 12
    if rhmax is not None and rhmin is not None:                         # Eq. 17
        ea = (sat_vp(tmin) * np.asarray(rhmax, float) / 100
              + sat_vp(tmax) * np.asarray(rhmin, float) / 100) / 2
    elif rhmean is not None:                                            # Eq. 19
        ea = np.asarray(rhmean, float) / 100 * es
    else:                                                               # Eq. 48
        ea = sat_vp(tmin)
    ra = extraterrestrial_radiation(lat_deg, doy)
    rso = (0.75 + 2e-5 * z) * ra                                        # Eq. 37
    rns = 0.77 * rs                                                     # Eq. 38 (albedo 0.23)
    rnl = (SIGMA * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2
           * (0.34 - 0.14 * np.sqrt(ea))
           * (1.35 * np.clip(rs / rso, 0.3, 1.0) - 0.35))               # Eq. 39
    rn = rns - rnl                                                      # Eq. 40
    delta = vp_slope(tmean)
    gamma = 0.000665 * pressure(z)                                      # Eq. 8
    num = 0.408 * delta * rn + gamma * 900 / (tmean + 273) * u2 * (es - ea)
    return np.maximum(num / (delta + gamma * (1 + 0.34 * u2)), 0.0)


def eto_hargreaves(tmax, tmin, lat_deg, doy, c=0.0023):
    """Hargreaves-Samani (1985) ET0 (mm/day) from temperature only - Eq. 52."""
    tmax, tmin = np.asarray(tmax, float), np.asarray(tmin, float)
    ra_mm = 0.408 * extraterrestrial_radiation(lat_deg, doy)
    return np.maximum(c * ((tmax + tmin) / 2 + 17.8) * np.sqrt(np.maximum(tmax - tmin, 0)) * ra_mm, 0.0)
