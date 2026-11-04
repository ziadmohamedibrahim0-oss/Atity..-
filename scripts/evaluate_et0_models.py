"""Compare ET0 estimators (Hargreaves-Samani vs machine learning) against FAO-56 Penman-Monteith.

Examples:
  python scripts/evaluate_et0_models.py                                   # synthetic DEMO data
  python scripts/evaluate_et0_models.py --nasa data/nasa_benha.csv --lat 30.46 --elev 13
Writes outputs/et0_model_comparison.csv and outputs/et0_scatter.png.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from agrisense import ml_et0, weather


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="CSV with date,tmax,tmin,rhmean,wind2[,rs][,rain]")
    ap.add_argument("--nasa", help="NASA POWER CSV from scripts/fetch_nasa_power.py")
    ap.add_argument("--lat", type=float, default=weather.BENHA["lat"])
    ap.add_argument("--elev", type=float, default=weather.BENHA["elev"])
    ap.add_argument("--test-years", type=int, default=1)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    if a.nasa:
        df = weather.load_nasa_power(a.nasa)
    elif a.csv:
        df = weather.load_csv(a.csv, a.lat)
    else:
        print("!! No weather file given: using SYNTHETIC demo data. The numbers below are NOT scientific evidence.")
        df = weather.synthetic_weather(lat=a.lat, elev=a.elev)
    df = weather.add_eto(df, a.lat, a.elev)

    tables, series = [], {}
    for feature_set in ml_et0.FEATURES:
        for model in ml_et0.MODELS:
            table, ts = ml_et0.compare(df, a.lat, feature_set, model, a.test_years)
            tables.append(table)
            series[table.index[-1]] = ts
    result = pd.concat(tables)
    result = result[~result.index.duplicated()]            # Hargreaves rows repeat in every comparison
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "et0_model_comparison.csv")
    print(result.to_string())

    best = result.loc[list(series)].sort_values("RMSE").index[0]
    ts = series[best]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    for ax, col, title in ((axes[0], "hs", "Hargreaves-Samani (original)"), (axes[1], "pred", best)):
        ax.scatter(ts["eto"], ts[col], s=8, alpha=0.6)
        ax.plot([0, ts["eto"].max()], [0, ts["eto"].max()], "k--", lw=1)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("FAO-56 Penman-Monteith ET0 (mm/day)")
    axes[0].set_ylabel("Estimated ET0 (mm/day)")
    fig.tight_layout()
    fig.savefig(out / "et0_scatter.png", dpi=150)
    print(f"\nsaved {out / 'et0_model_comparison.csv'} and {out / 'et0_scatter.png'}")


if __name__ == "__main__":
    main()
