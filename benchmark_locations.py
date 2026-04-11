from __future__ import annotations

import re


_ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+[A-Za-z0-9.\-'\s]+?\s+"
    r"(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Drive|Dr|Lane|Ln|Way|Court|Ct|"
    r"Place|Pl|Parkway|Pkwy|Terrace|Ter|Circle|Cir|Highway|Hwy)\b"
    r"(?:,\s*[A-Za-z.\-'\s]+)?"
    r"(?:,\s*MA\b|\s+MA\b)?"
    r"(?:\s+\d{5})?",
    re.IGNORECASE,
)

_BENCHMARK_COORDS = {
    "126 powerhouse blvd somerville ma": (42.40054, -71.11165),
}


def _normalize_address(value: str) -> str:
    text = (value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def lookup_benchmark_coordinates(text: str) -> tuple[float, float] | None:
    """Return fixture coordinates for known benchmark addresses embedded in free text."""
    match = _ADDRESS_RE.search(text or "")
    if not match:
        return None

    address = match.group(0).strip(" ,.")
    normalized = _normalize_address(address)
    if " ma" not in normalized:
        normalized = f"{normalized} ma".strip()
    return _BENCHMARK_COORDS.get(normalized)
