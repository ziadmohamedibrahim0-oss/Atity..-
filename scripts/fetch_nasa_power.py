"""Download daily NASA POWER weather for one point and save it as CSV (needs internet).

Example:
  python scripts/fetch_nasa_power.py --lat 30.46 --lon 31.18 --start 20200101 --end 20241231 --out data/nasa_benha.csv
Then choose "NASA POWER CSV" in the app, or call agrisense.weather.load_nasa_power(path).
"""
import argparse
from pathlib import Path

import requests

URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = "T2M_MAX,T2M_MIN,RH2M,WS2M,ALLSKY_SFC_SW_DWN,PRECTOTCORR"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, default=30.46)
    ap.add_argument("--lon", type=float, default=31.18)
    ap.add_argument("--start", default="20200101", help="YYYYMMDD")
    ap.add_argument("--end", default="20241231", help="YYYYMMDD")
    ap.add_argument("--out", default="data/nasa_benha.csv")
    a = ap.parse_args()
    params = {"parameters": PARAMETERS, "community": "AG", "longitude": a.lon, "latitude": a.lat,
              "start": a.start, "end": a.end, "format": "CSV"}
    r = requests.get(URL, params=params, timeout=120)
    r.raise_for_status()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(r.text, encoding="utf-8")
    print(f"saved {out} ({len(r.text.splitlines())} lines)")


if __name__ == "__main__":
    main()
