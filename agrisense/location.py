"""Place search and terrain elevation for the location picker (Open-Meteo, free, no API key).

Credits: elevation = Copernicus DEM (90 m) served by Open-Meteo; place names = GeoNames via Open-Meteo.
"""
from __future__ import annotations

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
LAT_RANGE, LON_RANGE, ELEV_RANGE = (-60.0, 60.0), (-180.0, 180.0), (-50.0, 3000.0)   # limits of the app inputs


class LocationError(ValueError):
    """The place search failed. kind: 'network' (service unreachable) or 'range' (only unsupported latitudes)."""

    def __init__(self, message, kind="network"):
        super().__init__(message)
        self.kind = kind


def in_range(lat, lon):
    return LAT_RANGE[0] <= lat <= LAT_RANGE[1] and LON_RANGE[0] <= lon <= LON_RANGE[1]


def clamp_elevation(elev):
    return min(max(float(elev), ELEV_RANGE[0]), ELEV_RANGE[1])


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def search_places(query, count=5, language="ar", timeout=15):
    """Find places by name. Returns [{'label','lat','lon','elev'}] (elev None if unknown); [] if nothing matches."""
    import requests

    query = " ".join(str(query).split())
    if len(query) < 2:                      # the service returns nothing for 0-1 characters
        return []
    try:
        r = requests.get(GEOCODING_URL, timeout=timeout,
                         params={"name": query, "count": count, "language": language, "format": "json"})
        status, body = r.status_code, (r.json() if r.status_code == 200 else None)
    except (requests.RequestException, ValueError) as err:
        raise LocationError("Could not reach the place search service") from err
    if status != 200 or not isinstance(body, dict):
        raise LocationError(f"The place search answered HTTP {status}")
    sep = "، " if language == "ar" else ", "
    places, skipped = [], 0
    for x in body.get("results") or []:
        lat, lon = _num(x.get("latitude")), _num(x.get("longitude"))
        if lat is None or lon is None:
            continue
        if not in_range(lat, lon):
            skipped += 1
            continue
        parts = []
        for part in (x.get("name"), x.get("admin1"), x.get("country")):
            if part and str(part) not in parts:
                parts.append(str(part))
        elev = _num(x.get("elevation"))
        places.append({"label": sep.join(parts) or "?", "lat": lat, "lon": lon,
                       "elev": None if elev is None else clamp_elevation(elev)})
    if skipped and not places:
        raise LocationError("Only places outside the supported latitude range matched", kind="range")
    return places


def lookup_elevation(lat, lon, timeout=15):
    """Terrain elevation (m) at a point; None when it cannot be found. Never raises."""
    import requests

    try:
        r = requests.get(ELEVATION_URL, timeout=timeout,
                         params={"latitude": round(float(lat), 4), "longitude": round(float(lon), 4)})
        if r.status_code != 200:
            return None
        value = _num(r.json()["elevation"][0])
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        return None
    return None if value is None else clamp_elevation(value)
