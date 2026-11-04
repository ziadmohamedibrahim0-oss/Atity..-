"""Integrated decision engine: foliar-disease risk (heuristic), irrigation advice and Markdown reports."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .irrigation import M3_PER_FEDDAN_PER_MM, SYSTEMS

KB_PATH = Path(__file__).resolve().parents[1] / "data" / "disease_knowledge.json"
WET_DRIVEN = {"bacterial_spot", "early_blight", "late_blight", "leaf_mold", "septoria_leaf_spot", "target_spot"}


def load_kb(path=KB_PATH):
    """Disease knowledge base: name, type and general management tips in Arabic and English."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def foliar_risk(wx, irrigation_gross=None, system="drip"):
    """Heuristic daily index (0-100) of weather favourability for leaf-wetness diseases.

    NOT a validated epidemiological model: the weights are assumptions that the team must
    calibrate against field observations. Components: humidity (45%), temperature suitability
    around 22 deg C (30%) and leaf wetting by rain or irrigation (25%), smoothed over 3 days.
    wx needs columns tmax, tmin, rhmean, rain.
    """
    tmean = (wx["tmax"].to_numpy(float) + wx["tmin"].to_numpy(float)) / 2
    humid = np.clip((wx["rhmean"].to_numpy(float) - 65) / 30, 0, 1)
    temp_ok = np.clip(1 - np.abs(tmean - 22) / 12, 0, 1)
    wet = (wx["rain"].to_numpy(float) >= 1).astype(float)
    if irrigation_gross is not None:
        wet = np.clip(wet + (np.asarray(irrigation_gross) > 0) * SYSTEMS[system]["wetting"], 0, 1)
    score = 100 * (0.45 * humid + 0.30 * temp_ok + 0.25 * wet)
    return pd.Series(score, index=wx.index).rolling(3, min_periods=1).mean()


def risk_level(score):
    return "high" if score >= 55 else "moderate" if score >= 30 else "low"


def advise(day, system, risk_score, stress_days, disease=None, kb=None, lang="ar"):
    """Integrated recommendations for one simulated day (a row of the daily irrigation table)."""
    t = (lambda ar, en: ar) if lang == "ar" else (lambda ar, en: en)
    etc = max(float(day["etc"]), 0.5)
    days_left = int(max(0.0, float(day["raw"]) - float(day["dr"])) // etc)
    net = float(day["dr"]) + days_left * etc
    vol = net / SYSTEMS[system]["eff"] * M3_PER_FEDDAN_PER_MM
    if days_left == 0:
        recs = [t(f"اروِ اليوم: عمق صافٍ ≈ {net:.0f} مم (≈ {vol:.0f} م³/فدان بعد كفاءة النظام).",
                  f"Irrigate today: net depth ≈ {net:.0f} mm (≈ {vol:.0f} m³/feddan after system efficiency).")]
    else:
        days_ar = {1: "يوم", 2: "يومين"}.get(days_left) or (f"{days_left} أيام" if days_left <= 10 else f"{days_left} يومًا")
        recs = [t(f"الري القادم بعد ≈ {days_ar} بعمق صافٍ ≈ {net:.0f} مم (≈ {vol:.0f} م³/فدان).",
                  f"Next irrigation in ≈ {days_left} day(s), net ≈ {net:.0f} mm (≈ {vol:.0f} m³/feddan).")]
    level = risk_level(risk_score)
    recs.append({
        "high": t("خطر الأمراض الورقية مرتفع: اروِ صباحًا وبالتنقيط قدر الإمكان، وحسّن التهوية، وافحص الأوراق السفلية كل 2–3 أيام.",
                  "Foliar disease risk is HIGH: irrigate in the morning (drip if possible), improve ventilation and scout lower leaves every 2-3 days."),
        "moderate": t("خطر الأمراض الورقية متوسط: تجنّب بلل المجموع الخضري وتابع الأوراق السفلية.",
                      "Foliar disease risk is MODERATE: avoid wetting the foliage and keep scouting lower leaves."),
        "low": t("خطر الأمراض الورقية منخفض حاليًا.", "Foliar disease risk is currently LOW."),
    }[level])
    if level != "low" and SYSTEMS[system]["wetting"] > 0:
        recs.append(t("نظام الري الحالي يبلّل الأوراق؛ التحويل للتنقيط يقلل خطر الأمراض ويرفع كفاءة المياه.",
                      "The current irrigation system wets the leaves; drip irrigation lowers disease risk and saves water."))
    if disease and kb and disease in kb and disease != "healthy":
        info = kb[disease]
        tips = info["tips_ar" if lang == "ar" else "tips_en"][:2]
        recs.append(t(f"آخر تشخيص: {info['ar']} — ", f"Latest diagnosis: {info['en']} — ") + " ".join(tips))
        if disease == "spider_mites" and stress_days > 0:
            recs.append(t(f"الإجهاد المائي ({stress_days} يوم حتى الآن) يزيد نشاط العنكبوت الأحمر؛ أبقِ رطوبة التربة ضمن RAW.",
                          f"Water stress ({stress_days} days so far) favours spider mites; keep soil moisture within RAW."))
    return recs


def _fmt(v):
    return f"{v:,.0f}" if abs(v - round(v)) < 1e-9 else f"{v:,.1f}"


def build_report(field, area, info, table, advice, diagnoses, lang="ar"):
    """Markdown monitoring report. info: dict of label -> value; table: irrigation comparison table."""
    t = (lambda ar, en: ar) if lang == "ar" else (lambda ar, en: en)
    lines = [f"# AgriSense AI - {t('تقرير متابعة', 'Monitoring report')}", "",
             f"**{t('الحقل', 'Field')}:** {field} | **{t('المساحة', 'Area')}:** {area:g} {t('فدان', 'feddan')}", ""]
    lines += [f"- **{k}:** {v}" for k, v in info.items()]
    lines += ["", f"## {t('مقارنة استراتيجيات الري', 'Irrigation strategies')}", ""]
    cols = list(table.columns)
    lines += ["| " + " | ".join([t("الاستراتيجية", "Strategy")] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for name, row in table.iterrows():
        lines.append("| " + " | ".join([str(name)] + [_fmt(float(v)) for v in row]) + " |")
    lines += ["", f"## {t('التوصيات', 'Recommendations')}", ""] + [f"- {r}" for r in advice]
    if diagnoses:
        lines += ["", f"## {t('آخر التشخيصات', 'Latest diagnoses')}", ""]
        lines += [f"- {d['created_at']} | {d['label']} | {d['confidence']:.0%} | "
                  f"{t('شدة الإصابة', 'lesion area')} {d['severity_pct']:.1f}%" for d in diagnoses]
    lines += ["", "---", t("التوصيات إرشادية ولا تُغني عن رأي المرشد الزراعي أو مركز البحوث الزراعية.",
                           "Recommendations are advisory and do not replace an agricultural extension officer.")]
    return "\n".join(lines)
