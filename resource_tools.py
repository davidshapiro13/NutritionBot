"""
resource_tools.py
=================
All tools for the Find Resources lane.

Exposes:
  - Tool functions: search_wic_stores, search_general_stores,
                    explain_program, affordable_overview
  - RESOURCE_TOOLS: schema list to pass to Claude API tool_use
  - run_tool(): dispatcher called by the agent tool-calling loop
"""

from __future__ import annotations

import csv
import math
import os
import re
from pathlib import Path

import requests

# ── WIC CSV ───────────────────────────────────────────────────────────────────

CSV_PATH = Path(__file__).parent / "rag_data" / "ma_wic_approved_stores.csv"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _haversine_miles(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_miles = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return radius_miles * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _normalize_chain(value: str | None) -> str:
    text = (value or "").lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _chain_matches(store_name: str, chain: str | None) -> bool:
    if not chain:
        return True
    normalized_store = _normalize_chain(store_name)
    normalized_chain = _normalize_chain(chain)
    if normalized_chain in normalized_store:
        return True
    compact_store = normalized_store.replace(" ", "")
    compact_chain = normalized_chain.replace(" ", "")
    return bool(compact_chain and compact_chain in compact_store)


def _load_wic_stores() -> list[dict]:
    stores: list[dict] = []
    if not CSV_PATH.exists():
        return stores
    with CSV_PATH.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                lat = float(row["lat"])
                lng = float(row["lng"])
            except (ValueError, KeyError):
                continue
            stores.append({
                "name": row.get("store", "").strip(),
                "address": row.get("address", "").strip(),
                "city": row.get("city", "").strip(),
                "zip": row.get("zip", "").strip(),
                "phone": row.get("phone", "").strip(),
                "self_checkout": row.get("self_checkout", "").strip(),
                "lat": lat,
                "lng": lng,
            })
    return stores


_WIC_STORES: list[dict] = _load_wic_stores()


# ── Tool functions ────────────────────────────────────────────────────────────

_CITY_FIXTURES: dict[str, dict[str, str]] = {
    "malden": {
        "grocery": "Super Stop & Shop — 99 Charles St, Malden, MA 02148",
        "pharmacy": "CVS Pharmacy — 575 Broadway, Malden, MA 02148",
    }
}


def _normalize_city(value: str | None) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"[^a-z\s\-']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _google_text_search(api_key: str, query: str, max_results: int = 3) -> list[dict]:
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": query,
        "key": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("results") or [])[:max_results]
    except Exception:
        return []


def _format_place_line(place: dict) -> str | None:
    name = (place.get("name") or "").strip()
    address = (place.get("formatted_address") or place.get("vicinity") or "").strip()
    if not name:
        return None
    return f"{name} — {address}" if address else name

def search_wic_stores(
    lat: float,
    lng: float,
    chain: str | None = None,
    max_results: int = 5,
    max_miles: float = 10.0,
) -> str:
    """Return nearby WIC-authorized stores from the local MA CSV."""
    if (chain or "").strip():
        max_miles = max(max_miles, 25.0)
    results: list[dict] = []
    for store in _WIC_STORES:
        if not _chain_matches(store["name"], chain):
            continue
        dist = _haversine_miles(lat, lng, store["lat"], store["lng"])
        if dist <= max_miles:
            results.append({**store, "distance_mi": round(dist, 2)})

    results.sort(key=lambda s: s["distance_mi"])
    results = results[:max_results]

    chain_label = (chain or "").strip()
    if not results:
        if chain_label:
            return (
                f"I couldn't find a WIC-authorized {chain_label} within {max_miles} miles. "
                "You can search the full Massachusetts WIC store list at:\n"
                "https://www.mass.gov/info-details/find-wic-approved-stores"
            )
        return (
            f"I couldn't find any WIC-authorized stores within {max_miles} miles. "
            "You can search the full Massachusetts WIC store list at:\n"
            "https://www.mass.gov/info-details/find-wic-approved-stores"
        )

    header = f"Nearest WIC-authorized {chain_label} stores:\n" if chain_label else "Nearest WIC-authorized stores:\n"
    lines = [header]
    for i, store in enumerate(results, 1):
        checkout = " | Self-checkout" if store["self_checkout"].lower() == "yes" else ""
        phone = f" | {store['phone']}" if store["phone"] else ""
        lines.append(
            f"{i}. {store['name']}{checkout}\n"
            f"   {store['address']}, {store['city']} {store['zip']}\n"
            f"   {store['distance_mi']} miles away{phone}"
        )
    lines.append("\nTip: Bring your WIC card and confirm covered items before shopping.")
    return "\n".join(lines)


def search_wic_stores_by_city(
    city: str,
    max_results: int = 4,
) -> str:
    """Return WIC-authorized stores in a given Massachusetts city (no GPS required)."""
    normalized_city = _normalize_city(city)
    if not normalized_city:
        return "Please provide a Massachusetts city name so I can list WIC-authorized stores."

    matches = []
    for store in _WIC_STORES:
        if _normalize_city(store.get("city")) == normalized_city:
            matches.append(store)

    if not matches:
        return (
            f"I couldn't find WIC-authorized stores in {city.title()} from the local Massachusetts list. "
            "Try another nearby city or check mass.gov/wic."
        )

    lines = [f"WIC-authorized stores in {city.title()}:\n"]
    for i, store in enumerate(matches[:max_results], 1):
        phone = f" | {store['phone']}" if store.get("phone") else ""
        lines.append(
            f"{i}. {store['name']}\n"
            f"   {store['address']}, {store['city']} {store['zip']}{phone}"
        )
    return "\n".join(lines)


def search_general_stores(
    lat: float,
    lng: float,
    chain: str | None = None,
    keyword: str | None = None,
    max_results: int = 5,
    radius_meters: int = 8000,
) -> str:
    if chain:
        radius_meters = 25000
    api_key = os.getenv("GOOGLE_PLACES_API_KEY", "")
    if not api_key:
        return (
            "Live store search is not available right now. "
            "Try Google Maps to find grocery stores near you."
        )

    search_keyword = keyword or chain or "grocery store supermarket"
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "keyword": search_keyword,
        "key": api_key,
    }
    # For descriptor/chain searches, ask Google to rank strictly by distance.
    # This avoids prominence-biased ordering for "closest X grocery" requests.
    if chain or keyword:
        params["rankby"] = "distance"
    else:
        params["radius"] = radius_meters
        params["type"] = "grocery_or_supermarket"

    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return (
            "I had trouble reaching the store search service. "
            "Try Google Maps to find stores near you."
        )

    places = data.get("results", [])
    # Enforce nearest-first ordering using returned geometry coordinates.
    def _place_distance_miles(place: dict) -> float:
        loc = ((place.get("geometry") or {}).get("location") or {})
        try:
            place_lat = float(loc.get("lat"))
            place_lng = float(loc.get("lng"))
        except (TypeError, ValueError):
            return float("inf")
        return _haversine_miles(lat, lng, place_lat, place_lng)

    places.sort(key=_place_distance_miles)
    if chain:
        places = [p for p in places if _chain_matches(p.get("name", ""), chain)]
    if not places and keyword:
        # Fallback: text search is often better for specialty markets.
        text_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        text_params = {
            "query": f"{keyword} near {lat},{lng}",
            "location": f"{lat},{lng}",
            "radius": radius_meters,
            "key": api_key,
        }
        try:
            text_resp = requests.get(text_url, params=text_params, timeout=8)
            text_resp.raise_for_status()
            text_data = text_resp.json()
            places = text_data.get("results", [])
        except Exception:
            places = []
    places = places[:max_results]

    if not places:
        chain_label = chain or "grocery stores"
        return f"I couldn't find any {chain_label} nearby. Try searching on Google Maps."

    chain_label = chain or "grocery stores"
    lines = [f"Nearby {chain_label}:\n"]
    for i, place in enumerate(places, 1):
        name = place.get("name", "")
        address = place.get("vicinity", "")
        rating = place.get("rating")
        rating_str = f" | ⭐ {rating}" if rating else ""
        dist = _place_distance_miles(place)
        dist_str = f" | {dist:.1f} mi" if math.isfinite(dist) else ""
        lines.append(f"{i}. {name}{rating_str}{dist_str}\n   {address}")

    return "\n".join(lines)


def search_city_pharmacy_and_grocery(city: str) -> str:
    """Return one pharmacy and one grocery store for a Massachusetts city, no GPS required."""
    normalized_city = _normalize_city(city)
    if not normalized_city:
        return "Please provide a Massachusetts city name so I can suggest stores."

    api_key = os.getenv("GOOGLE_PLACES_API_KEY", "")
    grocery_line = None
    pharmacy_line = None

    if api_key:
        grocery_results = _google_text_search(
            api_key,
            f"grocery store in {city}, MA",
            max_results=3,
        )
        pharmacy_results = _google_text_search(
            api_key,
            f"pharmacy in {city}, MA",
            max_results=3,
        )
        for place in grocery_results:
            line = _format_place_line(place)
            if line:
                grocery_line = line
                break
        for place in pharmacy_results:
            line = _format_place_line(place)
            if line:
                pharmacy_line = line
                break

    if not grocery_line:
        city_matches = [s for s in _WIC_STORES if _normalize_city(s.get("city")) == normalized_city]
        if city_matches:
            s = city_matches[0]
            grocery_line = f"{s['name']} — {s['address']}, {s['city']} {s['zip']}"

    if not pharmacy_line:
        city_matches = [s for s in _WIC_STORES if _normalize_city(s.get("city")) == normalized_city]
        for s in city_matches:
            name = (s.get("name") or "").lower()
            if any(token in name for token in ("cvs", "walgreens", "rite aid")):
                pharmacy_line = f"{s['name']} — {s['address']}, {s['city']} {s['zip']}"
                break

    fixture = _CITY_FIXTURES.get(normalized_city)
    if fixture:
        grocery_line = grocery_line or fixture.get("grocery")
        pharmacy_line = pharmacy_line or fixture.get("pharmacy")

    if not grocery_line and not pharmacy_line:
        return (
            f"I couldn't find stores in {city.title()} right now. "
            "Try sharing your location for nearby results."
        )

    lines = [f"In {city.title()}, here are good options:"]
    if grocery_line:
        lines.append(f"Grocery: {grocery_line}")
    if pharmacy_line:
        lines.append(f"Pharmacy: {pharmacy_line}")
    return "\n".join(lines)


_PROGRAM_TEXT: dict[str, str] = {
    "wic": (
        "WIC (Women, Infants, and Children) provides food benefits, nutrition education, "
        "and breastfeeding support. In Massachusetts, you may qualify if you are pregnant, "
        "up to 6 weeks postpartum, breastfeeding (until baby's first birthday), or have a "
        "child under 5 — and your household income is under 185% of the federal poverty level. "
        "If you already receive SNAP or MassHealth, you automatically meet the income requirement. "
        "To apply, contact your local WIC office or visit mass.gov/wic."
    ),
    "snap": (
        "SNAP (Supplemental Nutrition Assistance Program) provides monthly EBT benefits to buy food "
        "at most grocery stores. Eligibility is income-based and available to most low-income households "
        "in Massachusetts. SNAP also unlocks the HIP (Healthy Incentives Program), which doubles your "
        "spending on fresh fruits and vegetables at participating farmers markets. "
    ),
    "senior_nutrition": (
        "The Senior Nutrition Program is available to adults 60 and older (and their spouses) "
        "with no income requirement. It provides daily hot meals at community dining sites or "
        "home delivery through Meals on Wheels. The Senior Farmers Market Nutrition Program "
        "offers seasonal coupons for low-income seniors to use at farmers markets. "
        "Contact your local Council on Aging to enroll."
    ),
}


def explain_program(program: str) -> str:
    key = (program or "").strip().lower()
    text = _PROGRAM_TEXT.get(key)
    if text:
        return text
    supported = ", ".join(_PROGRAM_TEXT.keys())
    return f"I have information on: {supported}. Which program would you like to know about?"


def affordable_overview() -> str:
    return (
        "There are several ways to stretch your food budget in Massachusetts:\n\n"
        "Market Basket is consistently one of the most affordable grocery chains in the region "
        "and does not require any special eligibility.\n\n"
        "Food pantries and community fridges offer free groceries — no income proof needed at most sites. "
        "Search online for food pantries near you or ask your local Council on Aging for referrals.\n\n"
        "Farmers markets that accept SNAP also participate in the HIP program, which matches your "
        "SNAP spending on fresh produce dollar-for-dollar (up to $40–80/month depending on household size).\n\n"
        "If you receive SNAP or WIC, many stores also offer member discounts or dedicated low-cost sections."
    )


RESOURCE_TOOLS = [
    {
        "name": "search_wic_stores",
        "description": (
            "Find nearby WIC-authorized stores from the local Massachusetts list. "
            "Use this when the user asks for any nearby store or grocery location, "
            "with or without mentioning WIC. Optionally filter by chain name. "
            "Requires the user's GPS coordinates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "User latitude"},
                "lng": {"type": "number", "description": "User longitude"},
                "chain": {"type": "string", "description": "Optional store chain name, e.g. 'Stop & Shop'"},
                "max_results": {"type": "integer", "description": "Max stores to return (default 5)"},
            },
            "required": ["lat", "lng"],
        },
    },
    {
        "name": "explain_program",
        "description": (
            "Return eligibility requirements and benefits for a food assistance program. "
            "Use when the user asks what WIC, SNAP, or senior nutrition programs are or who qualifies."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "program": {
                    "type": "string",
                    "enum": ["wic", "snap", "senior_nutrition"],
                    "description": "The program to explain",
                },
            },
            "required": ["program"],
        },
    },
    {
        "name": "affordable_overview",
        "description": (
            "Return an overview of affordable grocery options in Massachusetts available to anyone, "
            "including Market Basket, food pantries, community fridges, and the HIP farmers market program. "
            "Use when the user asks about saving money on food or general affordable options."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


def run_tool(tool_name: str, tool_input: dict, lat: float | None = None, lng: float | None = None) -> str:
    if tool_name == "search_wic_stores":
        effective_lat = tool_input.get("lat") or lat
        effective_lng = tool_input.get("lng") or lng
        if effective_lat is None or effective_lng is None:
            return "I need your location to find nearby stores. Please share your GPS coordinates."
        return search_wic_stores(
            lat=effective_lat,
            lng=effective_lng,
            chain=tool_input.get("chain"),
            max_results=int(tool_input.get("max_results") or 5),
        )

    if tool_name == "search_general_stores":
        effective_lat = tool_input.get("lat") or lat
        effective_lng = tool_input.get("lng") or lng
        if effective_lat is None or effective_lng is None:
            return "I need your location to find nearby stores. Please share your GPS coordinates."
        return search_general_stores(
            lat=effective_lat,
            lng=effective_lng,
            chain=tool_input.get("chain"),
            keyword=tool_input.get("keyword"),
            max_results=int(tool_input.get("max_results") or 5),
        )

    if tool_name == "search_wic_stores_by_city":
        return search_wic_stores_by_city(
            city=str(tool_input.get("city") or "").strip(),
            max_results=int(tool_input.get("max_results") or 4),
        )

    if tool_name == "search_city_pharmacy_and_grocery":
        return search_city_pharmacy_and_grocery(
            city=str(tool_input.get("city") or "").strip(),
        )

    if tool_name == "explain_program":
        return explain_program(tool_input.get("program", ""))

    if tool_name == "affordable_overview":
        return affordable_overview()

    return f"Unknown tool: {tool_name}"
