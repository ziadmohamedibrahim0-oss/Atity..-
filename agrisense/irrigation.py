"""Tomato crop parameters, Kc curve and an FAO-56 root-zone water balance with irrigation strategies."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

M3_PER_FEDDAN_PER_MM = 4.2     # 1 feddan = 4200 m2, so 1 mm of water = 4.2 m3


@dataclass(frozen=True)
class Crop:
    """Tomato defaults from FAO-56 (Tables 11, 12, 22, 24). Adjust to variety and planting date."""
    name: str = "Tomato"
    stages: tuple = (30, 40, 45, 30)   # initial, development, mid, late season (days)
    kc_ini: float = 0.60
    kc_mid: float = 1.15
    kc_end: float = 0.80
    zr_min: float = 0.25               # root depth at planting (m)
    zr_max: float = 0.70               # maximum root depth (m)
    p: float = 0.40                    # fraction of TAW that can be depleted without stress
    ky: float = 1.05                   # seasonal yield response factor (FAO-33)

    @property
    def length(self) -> int:
        return int(sum(self.stages))


@dataclass(frozen=True)
class Soil:
    name: str
    fc: float   # water content at field capacity (m3/m3)
    wp: float   # water content at wilting point (m3/m3)


# Typical mid-range values inside the ranges of FAO-56 Table 19 - replace with measured values.
SOILS = {s.name: s for s in (Soil("Sandy loam", 0.23, 0.10), Soil("Loam", 0.25, 0.12),
                             Soil("Clay loam", 0.32, 0.18), Soil("Clay", 0.36, 0.22))}

SYSTEMS = {   # application efficiency and share of the foliage that gets wet
    "drip": dict(eff=0.90, wetting=0.0, ar="تنقيط", en="Drip"),
    "sprinkler": dict(eff=0.75, wetting=1.0, ar="رش", en="Sprinkler"),
    "surface": dict(eff=0.60, wetting=0.3, ar="سطحي (غمر/خطوط)", en="Surface (furrow)"),
}


@dataclass(frozen=True)
class Strategy:
    name: str
    ar: str
    kind: str = "threshold"   # "none" | "threshold" (soil-water based) | "fixed" (calendar based)
    trigger: float = 1.0      # threshold: irrigate when expected depletion exceeds trigger x RAW
    refill: float = 1.0       # threshold: share of the current depletion that is refilled
    interval: int = 3         # fixed: days between irrigations
    depth: float = 20.0       # fixed: gross water applied per irrigation (mm)


DEFAULT_STRATEGIES = [
    Strategy("Rainfed (no irrigation)", "بدون ري", kind="none"),
    Strategy("Smart (refill at RAW)", "ري ذكي (عند RAW)", trigger=1.0),
    Strategy("Regulated deficit (1.6 x RAW)", "ري بالعجز المنظم", trigger=1.6),
    Strategy("Fixed: 20 mm / 3 days", "جدول ثابت 20 مم كل 3 أيام", kind="fixed", interval=3, depth=20),
    Strategy("Heavy: 20 mm / 2 days", "إفراط 20 مم كل يومين", kind="fixed", interval=2, depth=20),
]
_AR = {s.name: s.ar for s in DEFAULT_STRATEGIES}


def kc_curve(crop: Crop, dap):
    """Single crop coefficient for days after planting `dap` (1-based) - FAO-56 Fig. 34."""
    li, ld, lm, ll = crop.stages
    d = np.asarray(dap, float)
    return np.where(d <= li, crop.kc_ini,
           np.where(d <= li + ld, crop.kc_ini + (d - li) / ld * (crop.kc_mid - crop.kc_ini),
           np.where(d <= li + ld + lm, crop.kc_mid,
                    crop.kc_mid + np.minimum((d - li - ld - lm) / ll, 1.0) * (crop.kc_end - crop.kc_mid))))


def stage_names(crop: Crop, dap):
    li, ld, lm, _ = crop.stages
    d = np.asarray(dap)
    return np.select([d <= li, d <= li + ld, d <= li + ld + lm], ["initial", "development", "mid"], "late")


def simulate(wx, crop, soil, plant_date, strategy, efficiency=0.9, initial_depletion=0.0, rain_min=2.0):
    """Daily root-zone water balance for one season (FAO-56 chapter 8, single Kc).

    wx: DataFrame with columns date, eto (mm/day) and rain (mm).
    Depletion Dr (mm) is the water missing relative to field capacity. Irrigation is decided at the
    start of the day from the depletion expected at the end of that day; Ks follows FAO-56 Eq. 84.
    Simplifications: no runoff, no capillary rise, rain < rain_min mm ignored, Dr kept when TAW grows.
    """
    w = wx[wx["date"] >= pd.Timestamp(plant_date)].head(crop.length).reset_index(drop=True)
    if len(w) < crop.length:
        raise ValueError(f"Weather data covers only {len(w)} of the {crop.length} season days; "
                         "choose an earlier planting date or a longer weather record")
    n = len(w)
    dap = np.arange(1, n + 1)
    kc = kc_curve(crop, dap)
    zr = crop.zr_min + (crop.zr_max - crop.zr_min) * np.minimum(1.0, dap / (crop.stages[0] + crop.stages[1]))
    taw = 1000.0 * (soil.fc - soil.wp) * zr
    raw = crop.p * taw
    eto, rain = w["eto"].to_numpy(float), w["rain"].to_numpy(float)
    peff = np.where(rain >= rain_min, rain, 0.0)

    dr, ks, eta, dp, net, gross = (np.zeros(n) for _ in range(6))
    prev = min(initial_depletion, taw[0])
    for i in range(n):
        etc_i = kc[i] * eto[i]
        if strategy.kind == "threshold" and prev + etc_i > strategy.trigger * raw[i]:
            net[i] = strategy.refill * prev
            gross[i] = net[i] / efficiency
        elif strategy.kind == "fixed" and i % strategy.interval == 0:
            gross[i] = strategy.depth
            net[i] = strategy.depth * efficiency
        d0 = prev - net[i] - peff[i]
        dp[i] = max(0.0, -d0)                    # water above field capacity drains below the roots
        d0 = max(0.0, d0)
        ks[i] = 1.0 if d0 <= raw[i] else max(0.0, (taw[i] - d0) / ((1 - crop.p) * taw[i]))
        eta[i] = min(ks[i] * etc_i, taw[i] - d0)
        dr[i] = d0 + eta[i]
        prev = dr[i]
    return pd.DataFrame({"date": w["date"], "dap": dap, "stage": stage_names(crop, dap), "eto": eto,
                         "kc": kc, "etc": kc * eto, "ks": ks, "eta": eta, "rain": rain, "rain_eff": peff,
                         "irr_net": net, "irr_gross": gross, "dp": dp, "dr": dr, "raw": raw, "taw": taw})


def summarize(df, crop):
    """Season totals and the relative yield Ya/Ym = 1 - Ky (1 - ETa/ETc) (Doorenbos & Kassam, 1979)."""
    etc, eta, gross = df["etc"].sum(), df["eta"].sum(), df["irr_gross"].sum()
    return {"etc_mm": etc, "eta_mm": eta, "rain_eff_mm": df["rain_eff"].sum(),
            "irrigation_mm": gross, "irrigation_m3_feddan": gross * M3_PER_FEDDAN_PER_MM,
            "events": int((df["irr_gross"] > 0).sum()), "deep_percolation_mm": df["dp"].sum(),
            "stress_days": int((df["ks"] < 0.999).sum()),
            "relative_yield_pct": 100 * max(0.0, 1 - crop.ky * (1 - eta / etc))}


def run_all(wx, crop, soil, plant_date, efficiency=0.9, strategies=None):
    """Simulate every strategy; returns {strategy name: daily DataFrame}."""
    return {s.name: simulate(wx, crop, soil, plant_date, s, efficiency)
            for s in (strategies or DEFAULT_STRATEGIES)}


SUMMARY_COLS = {   # key: (English, Arabic)
    "etc_mm": ("ETc (mm)", "الاحتياج ETc (مم)"),
    "eta_mm": ("ETa (mm)", "الاستهلاك الفعلي ETa (مم)"),
    "rain_eff_mm": ("Effective rain (mm)", "المطر الفعال (مم)"),
    "irrigation_mm": ("Irrigation (mm)", "مياه الري (مم)"),
    "irrigation_m3_feddan": ("Irrigation (m3/feddan)", "مياه الري (م³/فدان)"),
    "events": ("Irrigations (#)", "عدد الريات"),
    "deep_percolation_mm": ("Deep percolation (mm)", "الرشح العميق (مم)"),
    "stress_days": ("Stress days", "أيام الإجهاد"),
    "relative_yield_pct": ("Relative yield (%)", "الإنتاج النسبي (%)"),
}


def compare_table(results, crop, lang="en"):
    """One row per strategy with the season summary; lang = 'en' or 'ar'."""
    i = 0 if lang == "en" else 1
    rows = {(_AR.get(name, name) if i else name): {SUMMARY_COLS[k][i]: v for k, v in summarize(df, crop).items()}
            for name, df in results.items()}
    return pd.DataFrame(rows).T
