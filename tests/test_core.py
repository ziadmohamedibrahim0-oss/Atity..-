"""Unit tests.  Run from the project root:  python -m unittest discover -s tests -v   (pytest also works)."""
import io
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests
from PIL import Image, ImageDraw

from agrisense import db, decision, disease, et0, irrigation as irr, location, ml_et0, weather

LAT, ELEV = weather.BENHA["lat"], weather.BENHA["elev"]
WX = weather.add_eto(weather.synthetic_weather(), LAT, ELEV)
CROP, SOIL = irr.Crop(), irr.SOILS["Clay loam"]
RES = irr.run_all(WX, CROP, SOIL, "2025-02-15", 0.9)
SUM = {k: irr.summarize(v, CROP) for k, v in RES.items()}
NONE, SMART, _, FIXED, HEAVY = (s.name for s in irr.DEFAULT_STRATEGIES)


class TestET0(unittest.TestCase):
    def test_extraterrestrial_radiation_fao56_example8(self):
        # FAO-56 Example 8: 3 September (day 246) at 20 deg S -> 32.2 MJ m-2 day-1
        self.assertAlmostEqual(float(et0.extraterrestrial_radiation(-20, 246)), 32.2, delta=0.1)

    def test_penman_monteith_fao56_example18(self):
        # FAO-56 Example 18: Uccle (Brussels), 6 July, 50.8 N, 100 m -> ET0 = 3.9 mm/day
        doy, lat = 187, 50.80
        ra = et0.extraterrestrial_radiation(lat, doy)
        rs = et0.rs_from_sunshine(9.25, et0.daylight_hours(lat, doy), ra)
        eto = et0.eto_penman_monteith(21.5, 12.3, rs, et0.wind_at_2m(10 / 3.6, 10), lat, 100, doy,
                                      rhmax=84, rhmin=63)
        self.assertAlmostEqual(float(eto), 3.88, delta=0.05)


class TestWeather(unittest.TestCase):
    def test_synthetic_climate_is_plausible(self):
        monthly = WX.groupby(WX["date"].dt.month)["eto"].mean()
        self.assertTrue(1.5 < monthly[1] < 3.0 and 6.0 < monthly[7] < 8.5)

    def test_nasa_power_parser(self):
        text = ("-BEGIN HEADER-\nNASA/POWER\n-END HEADER-\n"
                "YEAR,DOY,T2M_MAX,T2M_MIN,RH2M,WS2M,ALLSKY_SFC_SW_DWN,PRECTOTCORR\n"
                "2024,1,20.0,9.5,60.2,2.3,12.5,0.0\n2024,2,-999,9.1,61.0,2.1,12.8,0.1\n"
                "2024,3,19.0,8.7,62.0,2.0,13.0,0.0\n")
        df = weather.load_nasa_power(io.BytesIO(text.encode()))
        self.assertEqual(len(df), 3)
        self.assertEqual(df["date"].iloc[0], pd.Timestamp("2024-01-01"))
        self.assertAlmostEqual(df["tmax"].iloc[1], 19.5)            # -999 gap interpolated

    def test_csv_loader_estimates_missing_radiation(self):
        csv = "date,Tmax,Tmin,RHmean,Wind2\n2024-06-01,34,21,45,2.0\n2024-06-02,35,22,44,2.2\n"
        df = weather.load_csv(io.StringIO(csv), lat=LAT)
        self.assertTrue((df["rs"] > 15).all() and (df["rain"] == 0).all())


def _nasa_reply(rows, unit="MJ/m^2/day", status=200):
    """Fake requests.Response with a NASA POWER style CSV (-999 = not published yet)."""
    head = ("-BEGIN HEADER-\nNASA/POWER Source Native Resolution Daily Data\nParameter(s):\n"
            f"ALLSKY_SFC_SW_DWN   CERES SYN1deg All Sky Surface Shortwave Downward Irradiance ({unit})\n"
            "-END HEADER-\n")
    body = "YEAR,DOY,T2M_MAX,T2M_MIN,RH2M,WS2M,ALLSKY_SFC_SW_DWN,PRECTOTCORR\n" + "\n".join(rows) + "\n"
    return SimpleNamespace(status_code=status, text=head + body)


GOOD_ROWS = ["2025,1,20.0,9.5,60.2,2.3,12.5,0.0", "2025,2,21.0,9.1,61.0,2.1,12.8,0.1",
             "2025,3,19.0,8.7,62.0,2.0,13.0,0.0", "2025,4,-999,-999,-999,-999,-999,-999",
             "2025,5,-999,-999,-999,-999,-999,-999"]


class TestAutoWeather(unittest.TestCase):
    """Online source (NASA POWER) - tested with fake replies; the live service was not reachable when written."""

    def fetch(self, reply, **kw):
        with mock.patch("requests.get", return_value=reply) as get:
            df = weather.fetch_nasa_power(30.456, 31.184, "2025-01-01", "2025-01-05", pause=0, **kw)
        return df, get

    def test_request_parameters(self):
        _, get = self.fetch(_nasa_reply(GOOD_ROWS))
        params = get.call_args.kwargs["params"]
        self.assertEqual(params["community"], "ag")
        self.assertEqual((params["start"], params["end"], params["format"]), ("20250101", "20250105", "CSV"))
        self.assertEqual((params["latitude"], params["longitude"]), (30.46, 31.18))
        self.assertEqual(params["parameters"], "T2M_MAX,T2M_MIN,RH2M,WS2M,ALLSKY_SFC_SW_DWN,PRECTOTCORR")

    def test_unpublished_days_are_cut_not_filled(self):
        df, _ = self.fetch(_nasa_reply(GOOD_ROWS))
        self.assertEqual(len(df), 3)
        self.assertEqual(df["date"].iloc[-1], pd.Timestamp("2025-01-03"))
        self.assertAlmostEqual(df["rs"].iloc[0], 12.5)

    def test_kwh_header_is_converted_to_mj(self):
        df, _ = self.fetch(_nasa_reply(GOOD_ROWS, unit="kW-hr/m^2/day"))
        self.assertAlmostEqual(df["rs"].iloc[0], 12.5 * 3.6)

    def test_unknown_radiation_unit_is_refused(self):
        with self.assertRaises(weather.WeatherDownloadError):
            self.fetch(_nasa_reply(GOOD_ROWS, unit="W/m^2"))

    def test_rejected_request_is_not_retried(self):
        with mock.patch("requests.get", return_value=_nasa_reply([], status=422)) as get:
            with self.assertRaises(weather.WeatherDownloadError):
                weather.fetch_nasa_power(30.46, 31.18, "2025-01-01", "2025-01-05", pause=0)
        self.assertEqual(get.call_count, 1)

    def test_no_internet_retries_then_explains(self):
        with mock.patch("requests.get", side_effect=requests.ConnectionError("offline")) as get:
            with self.assertRaises(weather.WeatherDownloadError) as ctx:
                weather.fetch_nasa_power(30.46, 31.18, "2025-01-01", "2025-01-05", retries=2, pause=0)
        self.assertEqual(get.call_count, 3)
        self.assertIn("internet", str(ctx.exception))

    def test_busy_server_then_success(self):
        replies = [SimpleNamespace(status_code=503, text="busy"), _nasa_reply(GOOD_ROWS)]
        with mock.patch("requests.get", side_effect=replies) as get:
            df = weather.fetch_nasa_power(30.46, 31.18, "2025-01-01", "2025-01-05", pause=0)
        self.assertEqual((get.call_count, len(df)), (2, 3))

    def test_unexpected_reply_and_empty_period(self):
        with self.assertRaises(weather.WeatherDownloadError):
            self.fetch(SimpleNamespace(status_code=200, text='{"messages": ["oops"]}'))
        with self.assertRaises(weather.WeatherDownloadError):
            self.fetch(_nasa_reply(GOOD_ROWS[3:]))

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            weather.fetch_nasa_power(95, 31, "2025-01-01", "2025-01-05")
        with self.assertRaises(ValueError):
            weather.fetch_nasa_power(30, 31, "2025-02-01", "2025-01-05")

    def test_auto_window_covers_season_and_history(self):
        start, end = weather.auto_window("2025-02-15", today="2026-11-04")
        self.assertEqual(end, pd.Timestamp("2026-10-27"))
        self.assertLessEqual(start, pd.Timestamp("2025-02-15"))
        self.assertGreaterEqual((end - start).days, 4 * 365)           # enough years for the ML tab
        old_start, _ = weather.auto_window("2010-06-01", today="2026-11-04")
        self.assertEqual(old_start, pd.Timestamp("2010-06-01"))        # an old planting date extends the window
        self.assertEqual(weather.auto_window("1970-01-01", today="2026-11-04")[0], pd.Timestamp("1984-01-01"))


class TestLocation(unittest.TestCase):
    """Place search + elevation (Open-Meteo) with fake replies; the live service was not reachable when written."""

    @staticmethod
    def reply(body, status=200):
        return SimpleNamespace(status_code=status, json=lambda: body)

    def search(self, body, **kw):
        with mock.patch("requests.get", return_value=self.reply(body)) as get:
            return location.search_places("Benha", **kw), get

    def test_search_parses_results(self):
        body = {"results": [
            {"name": "Benha", "latitude": 30.4667, "longitude": 31.1833, "elevation": 13.0,
             "admin1": "Qalyubia", "country": "Egypt"},
            {"name": "Banha", "latitude": 30.0, "longitude": 31.0, "country": "Egypt"}]}
        places, get = self.search(body, language="en")
        self.assertEqual(places[0], {"label": "Benha, Qalyubia, Egypt", "lat": 30.4667, "lon": 31.1833, "elev": 13.0})
        self.assertIsNone(places[1]["elev"])
        params = get.call_args.kwargs["params"]
        self.assertEqual((params["name"], params["language"], params["format"]), ("Benha", "en", "json"))

    def test_no_results_key_means_empty_list(self):
        self.assertEqual(self.search({"generationtime_ms": 0.4})[0], [])

    def test_short_query_does_not_call_the_service(self):
        with mock.patch("requests.get") as get:
            self.assertEqual(location.search_places(" a "), [])
        get.assert_not_called()

    def test_unsupported_latitudes_are_dropped_or_explained(self):
        far = {"name": "Longyearbyen", "latitude": 78.2, "longitude": 15.6, "country": "Norway"}
        near = {"name": "Oslo", "latitude": 59.9, "longitude": 10.7}
        places, _ = self.search({"results": [far, near]})
        self.assertEqual([p["label"] for p in places], ["Oslo"])
        with self.assertRaises(location.LocationError) as ctx:
            self.search({"results": [far]})
        self.assertEqual(ctx.exception.kind, "range")

    def test_search_failures_are_location_errors(self):
        with mock.patch("requests.get", side_effect=requests.ConnectionError("offline")):
            with self.assertRaises(location.LocationError) as ctx:
                location.search_places("Cairo")
        self.assertEqual(ctx.exception.kind, "network")
        with mock.patch("requests.get", return_value=self.reply({}, status=500)):
            with self.assertRaises(location.LocationError):
                location.search_places("Cairo")

    def test_elevation_lookup(self):
        with mock.patch("requests.get", return_value=self.reply({"elevation": [13.0]})) as get:
            self.assertEqual(location.lookup_elevation(30.46, 31.18), 13.0)
        self.assertEqual(get.call_args.kwargs["params"], {"latitude": 30.46, "longitude": 31.18})
        with mock.patch("requests.get", return_value=self.reply({"elevation": [9999.0]})):
            self.assertEqual(location.lookup_elevation(0, 0), 3000.0)            # clamped to the app limit
        for bad in (self.reply({}, status=500), self.reply({"elevation": []}), self.reply({"oops": 1})):
            with mock.patch("requests.get", return_value=bad):
                self.assertIsNone(location.lookup_elevation(0, 0))
        with mock.patch("requests.get", side_effect=requests.Timeout()):
            self.assertIsNone(location.lookup_elevation(0, 0))

    def test_range_helpers(self):
        self.assertTrue(location.in_range(30.46, 31.18))
        self.assertFalse(location.in_range(61, 0))
        self.assertFalse(location.in_range(0, 181))
        self.assertEqual(location.clamp_elevation(-500), -50.0)


class TestClimatology(unittest.TestCase):
    """Finishing a season that is not over yet with the multi-year average (clearly flagged as estimated)."""

    @classmethod
    def setUpClass(cls):
        cls.obs = weather.synthetic_weather("2022-10-27", "2026-10-24")

    def test_appends_days_and_flags_them(self):
        out = weather.extend_with_climatology(self.obs, "2027-03-20")
        new = out[~out["observed"]]
        self.assertEqual(new["date"].iloc[0], pd.Timestamp("2026-10-25"))
        self.assertEqual(new["date"].iloc[-1], pd.Timestamp("2027-03-20"))
        self.assertEqual(int(out["observed"].sum()), len(self.obs))
        self.assertTrue((out["date"].diff().dropna() == pd.Timedelta(days=1)).all())     # no gap, no repeat
        self.assertTrue(out[weather.COLUMNS[1:]].notna().all().all())

    def test_estimate_is_the_smoothed_mean_of_earlier_years(self):
        out = weather.extend_with_climatology(self.obs, "2026-11-30")
        got = out[out["date"] == "2026-11-10"].iloc[0]["tmax"]
        same_days = self.obs[(self.obs["date"].dt.month == 11) & (self.obs["date"].dt.day.between(7, 13))]
        self.assertAlmostEqual(got, same_days["tmax"].mean(), delta=0.02)

    def test_nothing_is_added_when_the_record_covers_the_season(self):
        out = weather.extend_with_climatology(self.obs, "2026-10-01")
        self.assertEqual(len(out), len(self.obs))
        self.assertTrue(out["observed"].all())

    def test_needs_several_years(self):
        with self.assertRaises(ValueError):
            weather.extend_with_climatology(self.obs[self.obs["date"] < "2024-06-01"], "2025-01-01")

    def test_leap_day(self):
        out = weather.extend_with_climatology(self.obs, "2028-03-05")                  # 2028 is a leap year
        feb29 = out[out["date"] == "2028-02-29"]
        self.assertEqual(len(feb29), 1)
        self.assertTrue(feb29[weather.COLUMNS[1:]].notna().all().all())

    def test_completed_season_runs_through_the_simulation(self):
        full = weather.add_eto(weather.extend_with_climatology(self.obs, "2027-03-20"), LAT, ELEV)
        res = irr.run_all(full, CROP, SOIL, "2026-10-27", 0.9)        # starts on the third estimated day
        self.assertEqual(len(res[SMART]), CROP.length)
        self.assertGreater(irr.summarize(res[SMART], CROP)["etc_mm"], 100)


@unittest.skipUnless(shutil.which("node"), "Node.js is not installed")
class TestGpsComponent(unittest.TestCase):
    def test_component_behaviour(self):
        done = subprocess.run(["node", str(Path(__file__).with_name("gps_component_check.js"))],
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)


class TestIrrigation(unittest.TestCase):
    def test_kc_curve(self):
        kc = irr.kc_curve(CROP, [1, 30, 70, 115, 145])
        self.assertAlmostEqual(kc[0], 0.60)
        self.assertAlmostEqual(kc[1], 0.60)
        self.assertAlmostEqual(kc[2], 1.15)
        self.assertAlmostEqual(kc[3], 1.15)
        self.assertAlmostEqual(kc[4], 0.80)

    def test_water_balance_closes(self):
        for name, d in RES.items():          # inputs = ETa + drainage + storage gain; initial depletion is 0
            inputs = (d["rain_eff"] + d["irr_net"]).sum()
            outputs = (d["eta"] + d["dp"]).sum() - d["dr"].iloc[-1]
            self.assertAlmostEqual(inputs, outputs, places=6, msg=name)

    def test_strategy_ranking(self):
        self.assertEqual(SUM[SMART]["stress_days"], 0)
        self.assertGreater(SUM[SMART]["relative_yield_pct"], 99)
        self.assertLess(SUM[NONE]["relative_yield_pct"], 20)
        self.assertLess(SUM[SMART]["irrigation_mm"], SUM[HEAVY]["irrigation_mm"])
        self.assertLess(SUM[SMART]["deep_percolation_mm"], SUM[HEAVY]["deep_percolation_mm"])
        self.assertLess(SUM[SMART]["irrigation_mm"], SUM[FIXED]["irrigation_mm"])

    def test_season_beyond_weather_record_raises(self):
        with self.assertRaises(ValueError):
            irr.simulate(WX, CROP, SOIL, "2025-11-01", irr.DEFAULT_STRATEGIES[1])

    def test_comparison_table_languages(self):
        self.assertEqual(len(irr.compare_table(RES, CROP, "en")), 5)
        self.assertIn("ري ذكي (عند RAW)", irr.compare_table(RES, CROP, "ar").index)


class TestDecision(unittest.TestCase):
    def test_risk_index(self):
        dry = pd.DataFrame({"tmax": [33] * 5, "tmin": [20] * 5, "rhmean": [40] * 5, "rain": [0] * 5})
        wet = pd.DataFrame({"tmax": [24] * 5, "tmin": [16] * 5, "rhmean": [92] * 5, "rain": [3] * 5})
        r_dry, r_wet = decision.foliar_risk(dry), decision.foliar_risk(wet)
        self.assertTrue((r_dry >= 0).all() and (r_wet <= 100).all())
        self.assertEqual(decision.risk_level(r_dry.iloc[-1]), "low")
        self.assertEqual(decision.risk_level(r_wet.iloc[-1]), "high")

    def test_sprinkler_raises_risk_on_irrigation_days(self):
        wx = pd.DataFrame({"tmax": [28] * 4, "tmin": [17] * 4, "rhmean": [75] * 4, "rain": [0] * 4})
        irrigated = [0, 20, 0, 0]
        self.assertGreater(decision.foliar_risk(wx, irrigated, "sprinkler").max(),
                           decision.foliar_risk(wx, irrigated, "drip").max())

    def test_advice_and_report(self):
        kb, row = decision.load_kb(), RES[SMART].iloc[60]
        for lang in ("ar", "en"):
            self.assertGreaterEqual(len(decision.advise(row, "drip", 70.0, 3, "spider_mites", kb, lang)), 4)
        en = decision.advise(row, "sprinkler", 70.0, 3, "spider_mites", kb, "en")
        self.assertTrue(any("Water stress" in r for r in en) and any("drip" in r for r in en))
        report = decision.build_report("F1", 2, {"Soil": "Clay loam"}, irr.compare_table(RES, CROP, "en"), en, [], "en")
        self.assertIn("| Smart (refill at RAW) |", report)


class TestML(unittest.TestCase):
    def test_compare_runs_and_learns(self):
        table, series = ml_et0.compare(WX, LAT, "T only", "Random Forest")
        self.assertEqual(len(table), 3)
        self.assertEqual(set(series.columns), {"date", "eto", "hs", "pred"})
        self.assertGreater(table.iloc[2]["R2"], 0.5)

    def test_needs_enough_data(self):
        with self.assertRaises(ValueError):
            ml_et0.compare(WX.head(200), LAT)


def synthetic_leaf():
    img = Image.new("RGB", (400, 400), (205, 205, 205))            # plain grey background
    draw = ImageDraw.Draw(img)
    draw.ellipse([60, 40, 340, 360], fill=(60, 140, 60))           # leaf: area = pi*140*160
    draw.ellipse([150, 150, 230, 230], fill=(120, 70, 30))         # large interior lesion
    draw.ellipse([250, 220, 280, 250], fill=(110, 60, 25))         # small lesion
    return img


class TestDisease(unittest.TestCase):
    def test_canonical_plantvillage_names(self):
        names = {"Tomato___Bacterial_spot": "bacterial_spot", "Tomato___Early_blight": "early_blight",
                 "Tomato___Late_blight": "late_blight", "Tomato___Leaf_Mold": "leaf_mold",
                 "Tomato___Septoria_leaf_spot": "septoria_leaf_spot",
                 "Tomato___Spider_mites Two-spotted_spider_mite": "spider_mites",
                 "Tomato___Target_Spot": "target_spot",
                 "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "yellow_leaf_curl_virus",
                 "Tomato___Tomato_mosaic_virus": "mosaic_virus", "Tomato___healthy": "healthy"}
        kb = decision.load_kb()
        for folder, key in names.items():
            self.assertEqual(disease.canonical(folder), key)
            self.assertIn(key, kb)

    def test_severity_on_synthetic_leaf(self):
        expected = 100 * (3.14159 * 40 ** 2 + 3.14159 * 15 ** 2) / (3.14159 * 140 * 160)      # about 8.1 %
        out = disease.estimate_severity(synthetic_leaf())
        self.assertAlmostEqual(out["severity_pct"], expected, delta=2.0)
        self.assertAlmostEqual(out["leaf_pct"], 100 * 3.14159 * 140 * 160 / 400 ** 2, delta=2.0)
        self.assertEqual(out["overlay"].size, (400, 400))

    def test_no_leaf_means_no_lesions(self):
        out = disease.estimate_severity(Image.new("RGB", (300, 300), (205, 205, 205)))
        self.assertEqual((out["leaf_pct"], out["severity_pct"]), (0.0, 0.0))


class TestDB(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "test.db")
            db.save_diagnosis("F1", "a.jpg", "Tomato___Late_blight", 0.93, 12.5, "resnet18", path=path)
            db.save_simulation("F1", SMART, "2025-02-15", "Clay loam", "drip", SUM[SMART], path=path)
            diag = db.recent("diagnoses", "F1", path=path)
            self.assertEqual((len(diag), diag[0]["label"]), (1, "Tomato___Late_blight"))
            self.assertEqual(len(db.recent("simulations", path=path)), 1)
            self.assertEqual(db.recent("diagnoses", "other-field", path=path), [])
            with self.assertRaises(ValueError):
                db.recent("users", path=path)


if __name__ == "__main__":
    unittest.main()
