"""AgriSense AI - Streamlit decision-support dashboard.      Run:  streamlit run app.py"""
import io
from datetime import date

import pandas as pd
import streamlit as st
from PIL import Image

from agrisense import db, decision, disease, gps, irrigation as irr, location, ml_et0, weather

st.set_page_config(page_title="AgriSense AI", page_icon="🌱", layout="wide")

# ---------- language (Arabic / English) ----------
AR = st.sidebar.radio("اللغة / Language", ["العربية", "English"], horizontal=True) == "العربية"
LANG = "ar" if AR else "en"


def t(ar, en):
    return ar if AR else en


if AR:   # right-to-left layout
    st.markdown("<style>.stApp,[data-testid='stSidebar']{direction:rtl;text-align:right}</style>",
                unsafe_allow_html=True)

CROP, KB = irr.Crop(), decision.load_kb()
LABEL = {s.name: (s.ar if AR else s.name) for s in irr.DEFAULT_STRATEGIES}

@st.cache_data(show_spinner=False, ttl=24 * 3600)
def find_places(query, language):
    return location.search_places(query, language=language)


def location_picker():
    """Sidebar widgets that set latitude, longitude and elevation: browser GPS, place search, or typing them."""
    for key, value in (("loc_lat", weather.BENHA["lat"]), ("loc_lon", weather.BENHA["lon"]),
                       ("loc_elev", weather.BENHA["elev"])):
        st.session_state.setdefault(key, value)

    def move_to(lat_, lon_, elev_=None):   # call it only BEFORE the three number inputs below are created
        st.session_state["loc_lat"], st.session_state["loc_lon"] = round(float(lat_), 2), round(float(lon_), 2)
        if elev_ is not None:
            st.session_state["loc_elev"] = location.clamp_elevation(elev_)

    # 1) browser GPS: asks for permission once when the app opens, and again whenever the button is pressed
    try:
        fix = gps.gps_fix({"button": t("📍 حدد موقعي تلقائيًا", "📍 Detect my location"),
                           "wait": t("جارٍ تحديد الموقع…", "Locating…")}, rtl=AR)
    except Exception:   # very old Streamlit or a blocked component: place search and typing still work
        fix = None
    if isinstance(fix, dict) and fix.get("ts") != st.session_state.get("gps_ts"):
        st.session_state["gps_ts"] = fix.get("ts")
        if "error" in fix:
            st.session_state["gps_note"] = (fix["error"], None, True)
        elif not location.in_range(fix["lat"], fix["lon"]):
            st.session_state["gps_note"] = ("range", None, True)
        else:
            found = location.lookup_elevation(fix["lat"], fix["lon"])
            move_to(fix["lat"], fix["lon"], found)
            st.session_state["gps_note"] = ("ok", fix.get("accuracy"), found is not None)
    note = st.session_state.get("gps_note")
    if note:
        kind, accuracy, had_elev = note
        if kind == "ok":
            st.caption(t("📍 تم تحديد موقعك تلقائيًا", "📍 Your location was detected")
                       + (t(f" (دقة ≈ {accuracy:.0f} م)", f" (accuracy ≈ {accuracy:.0f} m)") if accuracy else "")
                       + ("" if had_elev else t("؛ تعذّر جلب الارتفاع فعدّله يدويًا.",
                                                "; elevation not found - edit it by hand.")))
        elif kind == "range":
            st.warning(t("موقعك خارج النطاق المدعوم (خط العرض بين −60 و60).",
                         "Your location is outside the supported range (latitude -60 to 60)."))
        else:
            st.caption(t("تعذّر تحديد الموقع تلقائيًا (لم يُسمح به أو غير متاح): اضغط الزر واسمح به، أو ابحث بالاسم، "
                         "أو اكتب الإحداثيات.",
                         "Could not detect the location (not allowed or unavailable): press the button and allow "
                         "it, search by name, or type the coordinates."))

    # 2) place search (Open-Meteo geocoder; also gives the elevation)
    query = st.text_input(t("🔎 ابحث باسم مدينة أو قرية", "🔎 Search a city or village"), key="place_q").strip()
    if query:
        try:
            places = find_places(query, LANG)
        except location.LocationError as err:
            places = []
            st.caption(t("المكان خارج النطاق المدعوم (خط العرض بين −60 و60).",
                         "That place is outside the supported range (latitude -60 to 60).") if err.kind == "range"
                       else t("تعذّر البحث (لا إنترنت أو الخدمة مشغولة)؛ اكتب الإحداثيات يدويًا.",
                              "Search failed (no internet or the service is busy); type the coordinates instead."))
        else:
            if places:
                pick = st.selectbox(t("النتائج", "Results"), range(len(places)),
                                    format_func=lambda i: places[i]["label"])
                if st.button(t("استخدم هذا الموقع", "Use this location")):
                    chosen = places[pick]
                    move_to(chosen["lat"], chosen["lon"], chosen["elev"] if chosen["elev"] is not None
                            else location.lookup_elevation(chosen["lat"], chosen["lon"]))
            else:
                st.caption(t("لا توجد نتائج؛ جرّب كتابة الاسم بشكل آخر.", "No results; try another spelling."))

    # 3) the values themselves (also editable by hand)
    lat = st.number_input(t("خط العرض", "Latitude"), -60.0, 60.0, step=0.01, format="%.2f", key="loc_lat")
    lon = st.number_input(t("خط الطول", "Longitude"), -180.0, 180.0, step=0.01, format="%.2f", key="loc_lon")
    elev = st.number_input(t("الارتفاع (م)", "Elevation (m)"), -50.0, 3000.0, step=1.0, key="loc_elev")
    with st.expander(t("🗺️ الموقع على الخريطة", "🗺️ Location on the map")):
        point = pd.DataFrame({"lat": [lat], "lon": [lon]})
        try:
            st.map(point, zoom=8)
        except TypeError:   # older Streamlit without the zoom argument
            st.map(point)
        st.caption(t("المصادر: أسماء الأماكن GeoNames، والارتفاع Copernicus DEM، عبر Open-Meteo.",
                     "Sources: place names GeoNames, elevation Copernicus DEM, via Open-Meteo."))
    return lat, lon, elev


# ---------- sidebar: field, soil, irrigation system, location, weather ----------
with st.sidebar:
    st.header(t("بيانات الحقل", "Field setup"))
    field = st.text_input(t("اسم الحقل", "Field name"), "Field-1")
    area = st.number_input(t("المساحة (فدان)", "Area (feddan)"), 0.25, 500.0, 1.0, 0.25)
    soil_name = st.selectbox(t("نوع التربة", "Soil type"), list(irr.SOILS), index=2)
    system = st.selectbox(t("نظام الري", "Irrigation system"), list(irr.SYSTEMS),
                          format_func=lambda k: irr.SYSTEMS[k][LANG])
    plant_date = st.date_input(t("تاريخ الزراعة", "Planting date"), date(2025, 2, 15))
    st.header(t("الموقع", "Location"))
    lat, lon, elev = location_picker()
    st.header(t("بيانات الطقس", "Weather data"))
    sources = {"auto": t("تلقائي من الإنترنت (NASA POWER)", "Automatic online (NASA POWER)"),
               "demo": t("تجريبية (محاكاة)", "Demo (synthetic)"), "csv": "CSV", "nasa": "NASA POWER CSV"}
    source = st.radio(t("المصدر", "Source"), list(sources), format_func=sources.get)
    upload = st.file_uploader(t("ملف الطقس", "Weather file"), type="csv") if source in ("csv", "nasa") else None
    complete = (st.checkbox(t("أكمل الموسم غير المنتهي بمتوسط السنين الماضية",
                              "Complete an unfinished season with the multi-year average"), value=True)
                if source == "auto" else False)


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def load_weather(source, raw, lat, lon, elev, start, end):
    if source == "auto":
        df = weather.fetch_nasa_power(lat, lon, start, end)
    elif source == "demo":
        df = weather.synthetic_weather(lat=lat, elev=elev)
    elif source == "nasa":
        df = weather.load_nasa_power(io.BytesIO(raw))
    else:
        df = weather.load_csv(io.BytesIO(raw), lat)
    return weather.add_eto(df, lat, elev)


@st.cache_data(show_spinner=False)
def complete_season(wx_obs, until, lat, elev):
    return weather.add_eto(weather.extend_with_climatology(wx_obs, until), lat, elev)


@st.cache_data(show_spinner=False)
def run_scenarios(wx, plant_date, soil_name, efficiency):
    return irr.run_all(wx, CROP, irr.SOILS[soil_name], plant_date, efficiency)


@st.cache_data(show_spinner=False)
def run_ml(wx, lat, feature_set, model):
    return ml_et0.compare(wx, lat, feature_set, model)


@st.cache_resource
def get_classifier():
    return disease.load_classifier()


st.title("🌱 AgriSense AI")
st.caption(t("دعم القرار الزراعي بالذكاء الاصطناعي: الري وأمراض أوراق الطماطم",
             "AI decision support for tomato: irrigation and leaf diseases"))

if source in ("csv", "nasa") and upload is None:
    st.info(t("ارفع ملف الطقس من الشريط الجانبي للمتابعة.", "Upload a weather file in the sidebar to continue."))
    st.stop()
projected = None   # (first, last) estimated day when the season is completed with the multi-year average
try:
    if source == "auto":
        start, end = weather.auto_window(plant_date)
        with st.spinner(t("جارٍ تنزيل الطقس من NASA POWER…", "Downloading weather from NASA POWER…")):
            wx_obs = load_weather(source, b"", lat, lon, elev, str(start.date()), str(end.date()))
        wx, newest = wx_obs, wx_obs["date"].max()
        season_end = pd.Timestamp(plant_date) + pd.Timedelta(days=CROP.length - 1)
        if season_end > newest:   # NASA has no observed weather for the end of this season yet
            latest_plant = newest - pd.Timedelta(days=CROP.length - 1)
            if not complete:
                raise ValueError(t(
                    f"آخر يوم متاح من NASA POWER هو {newest:%Y-%m-%d}، والموسم يحتاج {CROP.length} يومًا من الطقس الفعلي. "
                    f"اختر تاريخ زراعة لا يتجاوز {latest_plant:%Y-%m-%d}، أو فعّل «أكمل الموسم» من الشريط الجانبي.",
                    f"The newest NASA POWER day is {newest:%Y-%m-%d} and the season needs {CROP.length} days of "
                    f"observed weather. Choose a planting date no later than {latest_plant:%Y-%m-%d}, or enable "
                    "'Complete an unfinished season' in the sidebar."))
            wx = complete_season(wx_obs, season_end, lat, elev)
            projected = (max(newest + pd.Timedelta(days=1), pd.Timestamp(plant_date)), season_end)
    else:
        wx = wx_obs = load_weather(source, upload.getvalue() if upload else b"", lat, lon, elev, None, None)
    results = run_scenarios(wx, pd.Timestamp(plant_date), soil_name, irr.SYSTEMS[system]["eff"])
except (ValueError, KeyError) as err:   # bad weather file, failed download, or the season does not fit
    st.error(str(err))
    if isinstance(err, weather.WeatherDownloadError):
        st.info(t("لو الإنترنت أو الخدمة غير متاحة، اختر مصدر «تجريبية» أو ارفع ملف CSV من الشريط الجانبي.",
                  "If the internet or the service is unavailable, pick the demo source or upload a CSV in the sidebar."))
    st.stop()
if source == "demo":
    st.warning(t("⚠️ بيانات الطقس الحالية محاكاة للعرض فقط ولا تصلح لأي استنتاج علمي.",
                 "⚠️ The weather is synthetic demo data - not valid for scientific conclusions."))
elif source == "auto":
    st.caption(t(f"🛰️ طقس NASA POWER (تلقائي) من {wx_obs['date'].min():%Y-%m-%d} إلى {wx_obs['date'].max():%Y-%m-%d} "
                 f"· النقطة {lat:.2f}, {lon:.2f} · بيانات شبكية تقريبية (≈55 كم) وليست قياس محطة.",
                 f"🛰️ NASA POWER weather (automatic) {wx_obs['date'].min():%Y-%m-%d} to {wx_obs['date'].max():%Y-%m-%d} "
                 f"· point {lat:.2f}, {lon:.2f} · gridded estimate (~55 km), not a station measurement."))
    if projected:
        st.warning(t(f"⚠️ الأيام من {projected[0]:%Y-%m-%d} إلى {projected[1]:%Y-%m-%d} مُقدَّرة بمتوسط السنين الماضية "
                     "وليست طقسًا فعليًا (والمطر المتوقع يُهمل غالبًا لأنه أقل من 2 مم). تتحدّث النتائج تلقائيًا كلما "
                     "نزلت بيانات NASA الجديدة.",
                     f"⚠️ {projected[0]:%Y-%m-%d} to {projected[1]:%Y-%m-%d} is estimated from the multi-year average, "
                     "not observed weather (expected rain is mostly ignored: it is below 2 mm). Results update as "
                     "new NASA data arrives."))
table = irr.compare_table(results, CROP, LANG)

tab_dash, tab_irr, tab_ml, tab_dis, tab_rep = st.tabs([
    t("لوحة القرار", "Dashboard"), t("الري", "Irrigation"), t("نموذج ET₀ الذكي", "AI ET₀ model"),
    t("أمراض الأوراق", "Leaf diseases"), t("التقارير", "Reports")])

# ---------- dashboard: one integrated recommendation ----------
with tab_dash:
    name = st.selectbox(t("استراتيجية الري المتّبعة", "Irrigation strategy in use"), list(LABEL), index=1,
                        format_func=LABEL.get, key="dash_strategy")
    sim = results[name]
    day = st.slider(t("يوم الموسم (بعد الزراعة)", "Day of season (after planting)"), 1, CROP.length, 60)
    if projected and sim["date"].iloc[day - 1] >= projected[0]:
        st.caption(t("⚠️ هذا اليوم بعد آخر طقس فعلي: الأرقام مبنية على المتوسط المناخي.",
                     "⚠️ This day is after the last observed weather: numbers use the multi-year average."))
    season = wx.set_index("date").loc[sim["date"]].reset_index()
    risk = decision.foliar_risk(season, sim["irr_gross"], system)
    summary = irr.summarize(sim, CROP)
    last = db.recent("diagnoses", field, 1)
    latest = disease.canonical(last[0]["label"]) if last else None
    recs = decision.advise(sim.iloc[day - 1], system, float(risk.iloc[day - 1]),
                           int((sim["ks"].iloc[:day] < 0.999).sum()), latest, KB, LANG)

    k = st.columns(4)
    k[0].metric(t("الاحتياج الموسمي ETc", "Seasonal ETc"), f"{summary['etc_mm']:.0f} mm")
    k[1].metric(t("مياه الري للحقل", "Irrigation water (field)"), f"{summary['irrigation_m3_feddan'] * area:,.0f} m³")
    k[2].metric(t("الإنتاج النسبي", "Relative yield"), f"{summary['relative_yield_pct']:.0f}%")
    k[3].metric(t("أيام الإجهاد المائي", "Water-stress days"), summary["stress_days"])

    st.subheader(t("توصيات اليوم", "Recommendations for this day"))
    for rec in recs:
        st.markdown(f"- {rec}")
    st.subheader(t("مؤشر خطر الأمراض الورقية (استرشادي)", "Foliar-disease risk index (heuristic)"))
    st.line_chart(pd.Series(risk.to_numpy(), index=sim["date"], name="risk"))
    st.caption(t("المؤشر يعتمد على افتراضات ويحتاج معايرة ببيانات حقلية قبل الاعتماد عليه.",
                 "The index rests on assumptions and needs calibration with field data before it is relied on."))

# ---------- irrigation simulation ----------
with tab_irr:
    st.subheader(t("مقارنة استراتيجيات الري", "Irrigation strategy comparison"))
    st.dataframe(table.round(1))
    detail = st.selectbox(t("تفاصيل استراتيجية", "Strategy details"), list(LABEL), index=1,
                          format_func=LABEL.get, key="irr_detail")
    daily = results[detail].set_index("date")
    st.caption(t("العجز المائي Dr مقابل RAW وTAW (مم)", "Root-zone depletion Dr vs RAW and TAW (mm)"))
    st.line_chart(daily[["dr", "raw", "taw"]])
    st.caption(t("الاحتياج ETc والري المضاف (مم/يوم)", "Crop demand ETc and applied irrigation (mm/day)"))
    st.bar_chart(daily[["etc", "irr_gross"]])
    st.caption(t("معامل الإجهاد Ks (1 = لا إجهاد)", "Stress coefficient Ks (1 = no stress)"))
    st.line_chart(daily["ks"])
    if st.button(t("حفظ هذه المحاكاة في قاعدة البيانات", "Save this simulation to the database")):
        db.save_simulation(field, detail, plant_date.isoformat(), soil_name, system, irr.summarize(results[detail], CROP))
        st.success(t("تم الحفظ.", "Saved."))

# ---------- ML estimation of ET0 ----------
with tab_ml:
    st.markdown(t("يُدرَّب النموذج على أول سنوات السجل المناخي ويُختبر على السنة الأخيرة، ثم يُقارن بمعادلة "
                  "Hargreaves-Samani مقابل مرجع FAO-56.",
                  "The model is trained on the early years, tested on the last year, and compared with "
                  "Hargreaves-Samani against the FAO-56 reference."))
    if source == "demo":
        st.warning(t("النتائج هنا على بيانات محاكاة: تثبت أن الكود يعمل فقط.",
                     "Results on synthetic data only prove that the pipeline runs."))
    c1, c2 = st.columns(2)
    feature_set = c1.selectbox(t("مدخلات النموذج", "Model inputs"), list(ml_et0.FEATURES))
    model_name = c2.selectbox(t("النموذج", "Model"), list(ml_et0.MODELS))
    if st.button(t("تدريب ومقارنة", "Train and compare")):
        try:
            metrics, series = run_ml(wx_obs, lat, feature_set, model_name)
            st.dataframe(metrics)
            st.line_chart(series.set_index("date").rename(columns={"eto": "FAO-56 PM", "hs": "Hargreaves", "pred": "ML"}))
        except ValueError as err:
            st.error(str(err))

# ---------- leaf disease ----------
with tab_dis:
    clf = get_classifier()
    if clf is None:
        st.info(t("مصنّف الأمراض لم يُدرَّب بعد (models/tomato_resnet18.pt)؛ المتاح الآن هو تقدير شدّة الإصابة "
                  "اللوني فقط. شغّل scripts/train_disease.py للتدريب.",
                  "The disease classifier is not trained yet (models/tomato_resnet18.pt); only the colour-based "
                  "lesion estimate is active. Run scripts/train_disease.py to train it."))
    photo = st.file_uploader(t("صورة ورقة طماطم", "Tomato leaf photo"), type=["jpg", "jpeg", "png"], key="leaf")
    if photo is not None:
        img = Image.open(photo).convert("RGB")
        sev = disease.estimate_severity(img)
        left, right = st.columns(2)
        left.image(img, caption=t("الصورة", "Photo"))
        right.image(sev["overlay"], caption=t("المناطق المصابة المقدّرة (أحمر)", "Estimated lesions (red)"))
        st.metric(t("نسبة المساحة المصابة (تقدير لوني)", "Lesion area (colour rule)"), f"{sev['severity_pct']:.1f}%")
        if sev["leaf_pct"] < 10:
            st.warning(t("الصورة لا تبدو ورقة قريبة وواضحة؛ قد تكون النتائج غير دقيقة.",
                         "This does not look like a close-up of a single leaf; results may be unreliable."))
        label, conf = "n/a", 0.0
        if clf is not None:
            preds = clf.predict(img)
            label, conf = preds[0]
            entry = KB.get(disease.canonical(label), {})
            st.subheader(f"{entry.get(LANG, label)} ({conf:.0%})")
            if conf < 0.6:
                st.warning(t("الثقة منخفضة: صوّر الورقة بوضوح أو استشر مختصًا.",
                             "Low confidence: retake the photo or consult a specialist."))
            st.bar_chart(pd.Series({KB.get(disease.canonical(lbl), {}).get(LANG, lbl): p for lbl, p in preds}))
            for tip in entry.get("tips_ar" if AR else "tips_en", []):
                st.markdown(f"- {tip}")
        if st.button(t("حفظ التشخيص", "Save diagnosis")):
            db.save_diagnosis(field, photo.name, label, conf, sev["severity_pct"], "resnet18" if clf else "rule-based")
            st.success(t("تم الحفظ.", "Saved."))
    st.caption(t("إرشادات عامة ولا تُغني عن المرشد الزراعي.",
                 "General guidance only - it does not replace an agricultural extension officer."))
    history = db.recent("diagnoses", field, 10)
    if history:
        st.dataframe(pd.DataFrame(history).drop(columns=["id"]))

# ---------- report ----------
with tab_rep:
    info = {t("تاريخ الزراعة", "Planting date"): plant_date.isoformat(), t("التربة", "Soil"): soil_name,
            t("نظام الري", "Irrigation system"): irr.SYSTEMS[system][LANG],
            t("بيانات الطقس", "Weather source"): sources[source], t("الاستراتيجية", "Strategy"): LABEL[name],
            t("يوم الموسم", "Day of season"): day}
    if projected:
        info[t("ملاحظة الطقس", "Weather note")] = t(
            f"تقدير بمتوسط السنين الماضية للفترة {projected[0]:%Y-%m-%d} إلى {projected[1]:%Y-%m-%d}",
            f"multi-year average estimate for {projected[0]:%Y-%m-%d} to {projected[1]:%Y-%m-%d}")
    report = decision.build_report(field, area, info, table, recs, db.recent("diagnoses", field, 10), LANG)
    st.markdown(report)
    st.download_button(t("تنزيل التقرير (Markdown)", "Download report (Markdown)"), report,
                       file_name=f"{field}_report.md", mime="text/markdown")
    st.download_button(t("تنزيل نتائج المحاكاة (CSV)", "Download simulation (CSV)"),
                       results[name].to_csv(index=False), file_name=f"{field}_simulation.csv", mime="text/csv")
