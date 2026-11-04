"""Machine-learning estimation of ET0 from limited weather data, compared with Hargreaves-Samani.

Target = FAO-56 Penman-Monteith ET0 computed from the full weather record (column 'eto').
Validation = temporal hold-out: the last `test_years` are never seen during training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import et0

BASE = ["tmax", "tmin", "ra", "doy_sin", "doy_cos"]
FEATURES = {"T only": BASE, "T + RH": BASE + ["rhmean"], "T + RH + wind": BASE + ["rhmean", "wind2"]}
MODELS = {
    "Random Forest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1),
    "Gradient Boosting": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0),
}


def make_features(df, lat):
    out = df.copy()
    doy = out["date"].dt.dayofyear.to_numpy()
    out["ra"] = et0.extraterrestrial_radiation(lat, doy)
    out["doy_sin"], out["doy_cos"] = np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)
    out["hs"] = et0.eto_hargreaves(out["tmax"], out["tmin"], lat, doy)
    return out


def metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return {"RMSE": float(np.sqrt(mean_squared_error(y, yhat))), "MAE": float(mean_absolute_error(y, yhat)),
            "R2": float(r2_score(y, yhat)), "Bias": float(np.mean(yhat - y))}


def compare(df, lat, feature_set="T only", model="Random Forest", test_years=1):
    """Return (metrics table, test-period series with columns date, eto, hs, pred).

    df needs the weather columns plus 'eto' (see weather.add_eto). Compared on the held-out years:
    Hargreaves-Samani (original), Hargreaves-Samani calibrated on the training years, and the ML model.
    """
    d = make_features(df, lat)
    cut = d["date"].max() - pd.DateOffset(years=test_years)
    tr, te = d[d["date"] <= cut], d[d["date"] > cut]
    if len(tr) < 365 or len(te) < 30:
        raise ValueError("Need at least ~2 years of daily data (training years + a held-out test period)")
    feats = FEATURES[feature_set]
    cal = LinearRegression().fit(tr[["hs"]], tr["eto"])
    ml = MODELS[model]().fit(tr[feats], tr["eto"])
    pred = ml.predict(te[feats])
    table = pd.DataFrame({"Hargreaves-Samani (original)": metrics(te["eto"], te["hs"]),
                          "Hargreaves-Samani (calibrated)": metrics(te["eto"], cal.predict(te[["hs"]])),
                          f"{model} [{feature_set}]": metrics(te["eto"], pred)}).T.round(3)
    return table, te.assign(pred=pred)[["date", "eto", "hs", "pred"]].reset_index(drop=True)
