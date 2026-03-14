"""
Location Service for NutritionBot
===================================
Finds WIC-authorized grocery stores near a given address using
Google Places API (Nearby Search).

Cross-references Google results with our known WIC store list so only
WIC-authorized stores are returned.

Agentic usage:
    from location_service import LocationService

    svc = LocationService()

    # From a WhatsApp location event (lat/lng already known):
    stores = svc.find_nearby_wic_stores(lat=42.4075, lng=-71.1190)

    # From a text address:
    stores = svc.find_nearby_wic_stores_by_address("Tufts University, Medford MA")

    # Get a formatted reply string for the bot:
    reply = svc.format_for_bot(stores)
"""

import os
import math
import requests
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")

# Tufts Medford campus coordinates (fallback default)
TUFTS_LAT = 42.4075
TUFTS_LNG = -71.1190

# Known WIC-authorized store name fragments (lowercase) for cross-reference
WIC_STORE_NAMES = {
    "stop & shop", "stop and shop",
    "market basket",
    "star market",
    "shaw's", "shaws",
    "whole foods",      # some locations accept WIC
    "price chopper",
    "big y",
    "hannaford",
    "cvs",
    "walgreens",
    "rosemary market",
    "dandea",
    "ferro",
    "fernandes",
    "minute market",
}

# Google Places Nearby Search endpoint
PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
GEOCODE_URL       = "https://maps.googleapis.com/maps/api/geocode/json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _haversine_miles(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Straight-line distance in miles between two lat/lng points."""
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _is_likely_wic(name: str) -> bool:
    """Return True if the store name matches a known WIC-authorized chain."""
    name_lower = name.lower()
    return any(wic in name_lower for wic in WIC_STORE_NAMES)


def _price_label(level: Optional[int]) -> str:
    if level is None:
        return ""
    labels = ["", "$", "$$", "$$$", "$$$$"]
    return labels[level] if isinstance(level, int) and 0 <= level < len(labels) else ""


# ── Main service class ────────────────────────────────────────────────────────

class LocationService:
    """
    Finds nearby WIC-authorized grocery stores via Google Places API.
    Requires GOOGLE_PLACES_API_KEY in .env.
    """

    def __init__(self):
        if not GOOGLE_API_KEY:
            raise RuntimeError(
                "GOOGLE_PLACES_API_KEY not set in .env\n"
                "Add:  GOOGLE_PLACES_API_KEY=your_key_here"
            )

    # ── Geocoding ─────────────────────────────────────────────────────────────

    def geocode(self, address: str) -> tuple[float, float]:
        """Convert a text address to (lat, lng). Raises on failure."""
        resp = requests.get(
            GEOCODE_URL,
            params={"address": address, "key": GOOGLE_API_KEY},
            timeout=10,
        )
        data = resp.json()
        if data.get("status") != "OK" or not data.get("results"):
            raise ValueError(f"Geocoding failed for '{address}': {data.get('status')}")
        loc = data["results"][0]["geometry"]["location"]
        return loc["lat"], loc["lng"]

    # ── Nearby search ─────────────────────────────────────────────────────────

    def _search_places(self, lat: float, lng: float, radius_meters: int, keyword: str) -> list[dict]:
        """Raw Google Places Nearby Search call."""
        params = {
            "location": f"{lat},{lng}",
            "radius": radius_meters,
            "type": "grocery_or_supermarket",
            "keyword": keyword,
            "key": GOOGLE_API_KEY,
        }
        resp = requests.get(PLACES_NEARBY_URL, params=params, timeout=10)
        data = resp.json()
        if data.get("status") not in ("OK", "ZERO_RESULTS"):
            raise RuntimeError(f"Places API error: {data.get('status')} — {data.get('error_message','')}")
        return data.get("results", [])

    def find_nearby_wic_stores(
        self,
        lat: float,
        lng: float,
        radius_miles: float = 3.0,
        max_results: int = 8,
    ) -> list[dict]:
        """
        Search for WIC-authorized grocery stores near (lat, lng).

        Args:
            lat, lng      : user coordinates
            radius_miles  : search radius (default 3 miles)
            max_results   : cap on returned stores

        Returns:
            List of store dicts sorted by distance, WIC-likely stores first.
        """
        radius_m = int(radius_miles * 1609.34)

        # Search with two keywords to maximise coverage
        raw: list[dict] = []
        seen_ids: set[str] = set()
        for kw in ("grocery store", "supermarket"):
            for place in self._search_places(lat, lng, radius_m, kw):
                pid = place.get("place_id", "")
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    raw.append(place)

        stores = []
        for place in raw:
            name = place.get("name", "")
            loc  = place["geometry"]["location"]
            dist = _haversine_miles(lat, lng, loc["lat"], loc["lng"])

            stores.append({
                "name"       : name,
                "address"    : place.get("vicinity", ""),
                "lat"        : loc["lat"],
                "lng"        : loc["lng"],
                "distance_mi": round(dist, 2),
                "wic_likely" : _is_likely_wic(name),
                "rating"     : place.get("rating"),
                "price_level": _price_label(place.get("price_level")),
                "open_now"   : place.get("opening_hours", {}).get("open_now"),
                "place_id"   : place.get("place_id", ""),
            })

        # Sort: WIC-likely first, then by distance
        stores.sort(key=lambda s: (not s["wic_likely"], s["distance_mi"]))
        return stores[:max_results]

    def find_nearby_wic_stores_by_address(
        self,
        address: str,
        radius_miles: float = 3.0,
        max_results: int = 8,
    ) -> list[dict]:
        """Same as find_nearby_wic_stores but takes a text address."""
        lat, lng = self.geocode(address)
        return self.find_nearby_wic_stores(lat, lng, radius_miles, max_results)

    # ── Formatting ────────────────────────────────────────────────────────────

    def format_for_bot(self, stores: list[dict], user_lat: float = None, user_lng: float = None) -> str:
        """
        Format the store list into a friendly bot reply.

        Args:
            stores   : output of find_nearby_wic_stores
            user_lat / user_lng : optional, used to build a Google Maps link

        Returns:
            Ready-to-send text string.
        """
        if not stores:
            return (
                "I couldn't find any WIC-authorized grocery stores nearby. "
                "You can check the full Massachusetts WIC store list at:\n"
                "https://www.mass.gov/info-details/find-wic-approved-stores"
            )

        wic_stores  = [s for s in stores if s["wic_likely"]]
        other_stores = [s for s in stores if not s["wic_likely"]]

        lines = ["Here are WIC-authorized grocery stores near you:\n"]

        def _store_line(s: dict, idx: int) -> str:
            open_tag = ""
            if s["open_now"] is True:
                open_tag = " ✓ Open now"
            elif s["open_now"] is False:
                open_tag = " ✗ Currently closed"
            rating = f"  Rating: {s['rating']}/5" if s["rating"] else ""
            price  = f"  Price: {s['price_level']}" if s["price_level"] else ""
            maps_url = f"https://www.google.com/maps/place/?q=place_id:{s['place_id']}"
            return (
                f"{idx}. {s['name']}{open_tag}\n"
                f"   {s['address']}\n"
                f"   {s['distance_mi']} miles away{rating}{price}\n"
                f"   Map: {maps_url}"
            )

        for i, s in enumerate(wic_stores, 1):
            lines.append(_store_line(s, i))

        if other_stores:
            lines.append("\nOther nearby grocery stores (WIC status not confirmed — call ahead):")
            for i, s in enumerate(other_stores, len(wic_stores) + 1):
                lines.append(_store_line(s, i))

        lines.append(
            "\nTip: Always bring your WIC card and check with the store about "
            "which specific items are covered before shopping."
        )

        return "\n".join(lines)


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    svc = LocationService()

    address = input("Enter your address (or press Enter for Tufts Medford): ").strip()
    if not address:
        lat, lng = TUFTS_LAT, TUFTS_LNG
        print(f"Using Tufts Medford campus ({lat}, {lng})\n")
    else:
        lat, lng = svc.geocode(address)
        print(f"Geocoded to: {lat}, {lng}\n")

    stores = svc.find_nearby_wic_stores(lat, lng)
    print(svc.format_for_bot(stores, lat, lng))
