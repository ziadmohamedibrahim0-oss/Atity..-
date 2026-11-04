"""Runs app.py end to end with a FAKE streamlit module and FAKE web services (no Streamlit, no internet needed).

    python tests/smoke_app.py

It checks the app logic: location picker (GPS / place search / typing), automatic NASA weather, completing an
unfinished season, error messages. It also enforces Streamlit's rule that a widget's session_state key may only
be set BEFORE the widget is created in a run. It cannot show how the real browser UI looks.
"""
import contextlib
import datetime
import os
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agrisense import location, ml_et0, weather  # noqa: E402

TODAY = "2026-11-04"
NEWEST_OBSERVED = pd.Timestamp("2026-10-24")      # fake NASA stops 3 days before the requested end (-999)
_orig_window, _orig_fetch = weather.auto_window, weather.fetch_nasa_power
weather.auto_window = lambda plant_date, **k: _orig_window(plant_date, today=TODAY, **k)
weather.fetch_nasa_power = lambda *a, **k: _orig_fetch(*a, pause=0, **k)


def fake_nasa(params):
    s, e = params["start"], params["end"]
    df = weather.synthetic_weather(f"{s[:4]}-{s[4:6]}-{s[6:]}", f"{e[:4]}-{e[4:6]}-{e[6:]}")
    rows = []
    for i, r in enumerate(df.itertuples()):
        vals = [r.tmax, r.tmin, r.rhmean, r.wind2, r.rs, r.rain]
        if i >= len(df) - 3:
            vals = [-999] * 6
        rows.append(f"{r.date.year},{r.date.dayofyear}," + ",".join(str(v) for v in vals))
    head = ("-BEGIN HEADER-\nNASA/POWER Source Native Resolution Daily Data\nParameter(s):\n"
            "ALLSKY_SFC_SW_DWN   CERES SYN1deg All Sky Surface Shortwave Downward Irradiance (MJ/m^2/day)\n-END HEADER-\n")
    body = "YEAR,DOY,T2M_MAX,T2M_MIN,RH2M,WS2M,ALLSKY_SFC_SW_DWN,PRECTOTCORR\n" + "\n".join(rows) + "\n"
    return SimpleNamespace(status_code=200, text=head + body)


MANSOURA = {"results": [
    {"name": "Mansoura", "latitude": 31.04, "longitude": 31.38, "elevation": 9.0, "admin1": "Dakahlia", "country": "Egypt"},
    {"name": "El Mansoura", "latitude": 36.3, "longitude": 2.5, "country": "Algeria"}]}


class Stop(Exception):
    pass


class SessionState(dict):
    """dict that refuses writes to keys of widgets already created in this run (like Streamlit does)."""
    created = set()

    def __setitem__(self, key, value):
        if key in self.created:
            raise RuntimeError(f"session_state['{key}'] was set after its widget was created")
        super().__setitem__(key, value)

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]


def run_app(script=None, session=None):
    """One script run. `script` = what the 'user' does; `session` is reused to emulate a rerun."""
    script = {"source": "auto", "plant": datetime.date(2025, 2, 15), "ar": True, **(script or {})}
    ss = session if session is not None else SessionState()
    ss.created = set()
    out, log = [], []

    def keyed(key, default):
        if key is not None:
            ss.created.add(key)
            if key not in ss:
                dict.__setitem__(ss, key, default)
            return ss[key]
        return default

    class UI(contextlib.AbstractContextManager):
        def __exit__(self, *a):
            return False

        def radio(self, label, options, **k):
            return ("العربية" if script["ar"] else "English") if "العربية" in list(options) else script["source"]

        def selectbox(self, label, options, index=0, key=None, **k):
            return script.get("pick", 0) if isinstance(options, range) else list(options)[index]

        def number_input(self, label, mn=None, mx=None, value=None, step=None, key=None, **k):
            return keyed(key, value if value is not None else mn)

        def text_input(self, label, value="", key=None, **k):
            keyed(key, value)
            return script.get("query", "") if key == "place_q" else value

        def checkbox(self, label, value=False, key=None, **k):
            return script.get("complete", value)

        def date_input(self, label, value=None, **k):
            return script["plant"] if ("الزراعة" in label or "Planting" in label) else value

        def slider(self, label, mn, mx, value=None, **k):
            return value

        def file_uploader(self, *a, **k):
            return None

        def button(self, label, **k):
            if label in ("استخدم هذا الموقع", "Use this location"):
                return script.get("click_use", False)
            if label in ("تدريب ومقارنة", "Train and compare"):
                return script.get("click_ml", False)
            return False

        def columns(self, n):
            return [UI() for _ in range(n)]

        def tabs(self, labels):
            return [UI() for _ in labels]

        def __getattr__(self, name):          # markdown, caption, info, warning, error, metric, map, spinner ...
            def record(*a, **k):
                out.append((name, " ".join(str(x) for x in a[:2])))
                return UI()
            return record

    ui = UI()
    st = types.ModuleType("streamlit")
    st.sidebar, st.session_state = ui, ss
    for name in ("radio selectbox number_input text_input checkbox date_input slider file_uploader button columns tabs "
                 "spinner expander map markdown caption info warning error success title header subheader metric "
                 "line_chart bar_chart dataframe image download_button set_page_config").split():
        setattr(st, name, getattr(ui, name))

    def stop():
        raise Stop()
    st.stop = stop
    st.cache_data = st.cache_resource = lambda *a, **k: a[0] if a and callable(a[0]) else (lambda f: f)

    components_v1 = types.ModuleType("streamlit.components.v1")

    def declare_component(name, path=None, url=None):
        assert (Path(path) / "index.html").is_file(), "GPS component page is missing"
        def component(**kwargs):
            out.append(("gps_component", kwargs["key"]))
            return script.get("gps")
        return component
    components_v1.declare_component = declare_component
    st.components = types.ModuleType("streamlit.components")
    st.components.v1 = components_v1
    sys.modules.update({"streamlit": st, "streamlit.components": st.components, "streamlit.components.v1": components_v1})
    import agrisense.gps as gps_pkg
    gps_pkg._component = None

    def fake_get(url, params=None, timeout=None):
        log.append((url, params))
        if url == weather.NASA_URL:
            if script.get("nasa_offline"):
                raise requests.ConnectionError("offline")
            return fake_nasa(params)
        if url == location.GEOCODING_URL:
            if script.get("geo_offline"):
                raise requests.ConnectionError("offline")
            return SimpleNamespace(status_code=200, json=lambda: MANSOURA)
        if url == location.ELEVATION_URL:
            return SimpleNamespace(status_code=200, json=lambda: {"elevation": [7.0]})
        raise AssertionError(f"unexpected URL {url}")

    seen_ml = []
    dummy = (pd.DataFrame({"model": ["x"], "rmse": [0.1]}),
             pd.DataFrame({"date": pd.date_range("2026-01-01", periods=3), "eto": [1, 2, 3], "hs": [1, 2, 3], "pred": [1, 2, 3]}))
    stopped = False
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            with mock.patch("requests.get", side_effect=fake_get), \
                    mock.patch.object(ml_et0, "compare", lambda wx, *a, **k: (seen_ml.append(wx), dummy)[1]):
                try:
                    exec(compile((ROOT / "app.py").read_text(encoding="utf-8"), "app.py", "exec"), {"__name__": "__main__"})
                except Stop:
                    stopped = True
        finally:
            os.chdir(cwd)
    return SimpleNamespace(out=out, stopped=stopped, ss=ss, log=log, ml=seen_ml,
                           nasa=[p for u, p in log if u == weather.NASA_URL])


def texts(run, kind):
    return [t for k, t in run.out if k == kind]


def main():
    d = datetime.date
    # 1) first open, no GPS answer yet: default location, real-looking NASA request, app renders fully
    r = run_app()
    assert not r.stopped and not texts(r, "error"), texts(r, "error")
    assert (r.ss["loc_lat"], r.ss["loc_lon"], r.ss["loc_elev"]) == (weather.BENHA["lat"], weather.BENHA["lon"], weather.BENHA["elev"])
    assert len(r.nasa) == 1 and r.nasa[0]["community"] == "ag" and r.nasa[0]["latitude"] == weather.BENHA["lat"]
    assert ("gps_component", "gps") in r.out and any(k == "map" for k, _ in r.out)
    assert any("NASA POWER" in c for c in texts(r, "caption")) and any(k == "download_button" for k, _ in r.out)
    print("1 OK  first open (Arabic): default location, NASA request", r.nasa[0]["start"], "->", r.nasa[0]["end"])
    ss = r.ss

    # 2) GPS answer arrives -> coordinates + elevation set, weather requested for the NEW point
    fix = {"lat": 31.0412, "lon": 31.3785, "accuracy": 30.0, "ts": "1-1"}
    r = run_app({"gps": fix}, ss)
    assert not r.stopped and (ss["loc_lat"], ss["loc_lon"], ss["loc_elev"]) == (31.04, 31.38, 7.0)
    assert r.nasa[-1]["latitude"] == 31.04 and r.nasa[-1]["longitude"] == 31.38
    assert any("تم تحديد موقعك" in c for c in texts(r, "caption"))
    print("2 OK  GPS fix applied:", ss["loc_lat"], ss["loc_lon"], "elev", ss["loc_elev"], "-> NASA asked for the new point")
    dict.__setitem__(ss, "loc_lat", 29.0)                     # user edits latitude by hand ...
    run_app({"gps": fix}, ss)                                 # ... the same GPS answer on a rerun must not undo it
    assert ss["loc_lat"] == 29.0
    run_app({"gps": {**fix, "ts": "1-2", "lat": 30.0}}, ss)   # a NEW answer (button pressed again) does
    assert ss["loc_lat"] == 30.0
    print("  OK  manual edit survives reruns; a new GPS answer replaces it")

    # 3) GPS denied / unsupported / outside the supported latitudes
    r = run_app({"gps": {"error": "denied", "ts": "2-1"}})
    assert not r.stopped and r.ss["loc_lat"] == weather.BENHA["lat"] and any("تعذّر تحديد الموقع" in c for c in texts(r, "caption"))
    r = run_app({"gps": {"lat": 70.0, "lon": 20.0, "accuracy": 10, "ts": "2-2"}, "ar": False})
    assert r.ss["loc_lat"] == weather.BENHA["lat"] and any("outside the supported range" in w for w in texts(r, "warning"))
    print("3 OK  GPS denied / out of range: app keeps working with the previous location")

    # 4) place search
    r = run_app({"query": "Mansoura", "pick": 0})
    assert r.ss["loc_lat"] == weather.BENHA["lat"], "search alone must not move the location"
    r = run_app({"query": "Mansoura", "pick": 0, "click_use": True})
    assert (r.ss["loc_lat"], r.ss["loc_lon"], r.ss["loc_elev"]) == (31.04, 31.38, 9.0)          # elevation from the geocoder
    r = run_app({"query": "Mansoura", "pick": 1, "click_use": True})
    assert (r.ss["loc_lat"], r.ss["loc_lon"], r.ss["loc_elev"]) == (36.3, 2.5, 7.0)             # elevation looked up
    r = run_app({"query": "Mansoura", "geo_offline": True})
    assert not r.stopped and any("تعذّر البحث" in c for c in texts(r, "caption"))
    print("4 OK  place search: pick + 'use' moves the location, offline search degrades gracefully")

    # 5) season not finished: completed with the multi-year average (flagged), ML sees observed days only
    r = run_app({"plant": d(2026, 9, 1), "ar": False, "click_ml": True})
    assert not r.stopped and not texts(r, "error"), texts(r, "error")
    assert any("estimated from the multi-year average" in w for w in texts(r, "warning"))
    assert any("multi-year average estimate for" in m for m in texts(r, "markdown")), "report note missing"
    assert r.ml and r.ml[0]["date"].max() == NEWEST_OBSERVED, "ML tab must not see estimated days"
    print("5 OK  unfinished season completed + flagged; ML uses observed days only (last:", r.ml[0]["date"].max().date(), ")")
    r = run_app({"plant": d(2026, 9, 1), "ar": False, "complete": False})
    assert r.stopped and any("no later than" in e and "Complete an unfinished season" in e for e in texts(r, "error"))
    print("  OK  with the option off: clear error with the latest possible planting date")
    r = run_app({"plant": d(2025, 2, 15), "ar": False})
    assert not r.stopped and not any("estimated" in w for w in texts(r, "warning"))
    print("  OK  a finished season gets no estimate")

    # 6) no internet for NASA: readable error + hint, no crash
    r = run_app({"nasa_offline": True})
    assert r.stopped and texts(r, "error") and texts(r, "info")
    print("6 OK  NASA offline:", texts(r, "error")[0][:80])

    # 7) other sources unchanged
    r = run_app({"source": "demo", "ar": False})
    assert not r.stopped and not texts(r, "error") and texts(r, "warning")
    r = run_app({"source": "csv", "ar": False})
    assert r.stopped and any("Upload" in i for i in texts(r, "info"))
    print("7 OK  demo + csv sources unchanged")
    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
