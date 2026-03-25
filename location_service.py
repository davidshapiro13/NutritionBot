"""
Location Service for NutritionBot
===================================
Finds the nearest WIC-authorized stores from the official MA WIC store list
(rag_data/ma_wic_approved_stores.csv), using pre-geocoded lat/lng coordinates.

Usage:
    svc = LocationService()
    stores = svc.find_nearby_wic_stores(lat=42.4075, lng=-71.1190)
    reply  = svc.format_for_bot(stores)
"""

import csv
import math
from pathlib import Path

CSV_PATH = Path(__file__).parent / "rag_data" / "ma_wic_approved_stores.csv"


def _haversine_miles(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _load_wic_stores() -> list[dict]:
    """Load all WIC stores from CSV. Skips rows without coordinates."""
    stores = []
    with CSV_PATH.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                lat = float(row["lat"])
                lng = float(row["lng"])
            except (ValueError, KeyError):
                continue
            stores.append({
                "name":          row.get("store", "").strip(),
                "address":       row.get("address", "").strip(),
                "city":          row.get("city", "").strip(),
                "zip":           row.get("zip", "").strip(),
                "phone":         row.get("phone", "").strip(),
                "self_checkout": row.get("self_checkout", "").strip(),
                "lat":           lat,
                "lng":           lng,
            })
    return stores


# Load once at import time
_WIC_STORES = _load_wic_stores()


class LocationService:

    def find_nearby_wic_stores(
        self,
        lat: float,
        lng: float,
        max_results: int = 5,
        max_miles: float = 10.0,
    ) -> list[dict]:
        """
        Return the nearest WIC-authorized stores to (lat, lng),
        sorted by distance, up to max_results within max_miles.
        """
        results = []
        for store in _WIC_STORES:
            dist = _haversine_miles(lat, lng, store["lat"], store["lng"])
            if dist <= max_miles:
                results.append({**store, "distance_mi": round(dist, 2)})

        results.sort(key=lambda s: s["distance_mi"])
        return results[:max_results]

    def format_for_bot(self, stores: list[dict]) -> str:
        if not stores:
            return (
                "I couldn't find any WIC-authorized stores nearby. "
                "You can search the full list at:\n"
                "https://www.mass.gov/info-details/find-wic-approved-stores"
            )

        lines = ["Here are the nearest WIC-authorized stores:\n"]
        for i, s in enumerate(stores, 1):
            checkout = " | Self-checkout ✓" if s["self_checkout"].lower() == "yes" else ""
            phone = f" | 📞 {s['phone']}" if s["phone"] else ""
            lines.append(
                f"{i}. {s['name']}{checkout}\n"
                f"   {s['address']}, {s['city']} {s['zip']}\n"
                f"   {s['distance_mi']} miles away{phone}"
            )

        lines.append(
            "\nTip: Bring your WIC card and confirm covered items with the store before shopping."
        )
        return "\n".join(lines)


if __name__ == "__main__":
    svc = LocationService()
    stores = svc.find_nearby_wic_stores(lat=42.4075, lng=-71.1190)
    print(svc.format_for_bot(stores))
