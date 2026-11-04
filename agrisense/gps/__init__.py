"""Browser GPS button for Streamlit (no extra package needed).

frontend/index.html calls navigator.geolocation: it asks for permission once when the app is opened and again
whenever the button is pressed. Browsers only allow this on https:// pages (Streamlit Cloud) or on localhost.
"""
from __future__ import annotations

from pathlib import Path

_FRONTEND = Path(__file__).resolve().parent / "frontend"
_component = None


def gps_fix(texts, auto=True, rtl=False, key="gps"):
    """Show the GPS button. Returns None until the browser answered, then {'lat','lon','accuracy','ts'}
    or {'error': 'denied'|'failed'|'unsupported', 'ts'}; 'ts' is unique per answer."""
    global _component
    import streamlit.components.v1 as components

    if _component is None:
        _component = components.declare_component("agrisense_gps", path=str(_FRONTEND))
    return _component(texts=texts, auto=auto, rtl=rtl, key=key, default=None)
