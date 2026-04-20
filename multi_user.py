"""Household profile routing for shared-device use.

The WhatsApp/device user_id still identifies the physical conversation, while
an active household profile chooses which memory file and LLM session to use.
The default profile intentionally keeps using the original user_id so existing
memory files keep working.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


PROFILE_DIR = Path(__file__).parent / "user_profiles"
DEFAULT_PROFILE_ID = "default"
DEFAULT_PROFILE_LABEL = "Me"


def _profile_file(device_user_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.@+-]+", "_", device_user_id).strip("_") or "unknown"
    return PROFILE_DIR / f"{safe}.json"


def _slug(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    return slug or "person"


def _load(device_user_id: str) -> dict:
    path = _profile_file(device_user_id)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {
        "active_profile_id": DEFAULT_PROFILE_ID,
        "profiles": {
            DEFAULT_PROFILE_ID: {
                "label": DEFAULT_PROFILE_LABEL,
                "user_key": device_user_id,
            }
        },
    }


def _save(device_user_id: str, data: dict) -> None:
    PROFILE_DIR.mkdir(exist_ok=True)
    path = _profile_file(device_user_id)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _ensure_default(device_user_id: str, data: dict) -> dict:
    profiles = data.setdefault("profiles", {})
    profiles.setdefault(
        DEFAULT_PROFILE_ID,
        {"label": DEFAULT_PROFILE_LABEL, "user_key": device_user_id},
    )
    data.setdefault("active_profile_id", DEFAULT_PROFILE_ID)
    if data["active_profile_id"] not in profiles:
        data["active_profile_id"] = DEFAULT_PROFILE_ID
    return data


def effective_user_id(device_user_id: str) -> str:
    """Return the active household profile key used for memory/session state."""
    data = _ensure_default(device_user_id, _load(device_user_id))
    active = data["profiles"][data["active_profile_id"]]
    return active.get("user_key") or device_user_id


def active_profile_label(device_user_id: str) -> str:
    data = _ensure_default(device_user_id, _load(device_user_id))
    active = data["profiles"][data["active_profile_id"]]
    return active.get("label") or DEFAULT_PROFILE_LABEL


def list_profiles(device_user_id: str) -> list[dict[str, str | bool]]:
    data = _ensure_default(device_user_id, _load(device_user_id))
    active_id = data["active_profile_id"]
    return [
        {
            "id": profile_id,
            "label": profile.get("label") or DEFAULT_PROFILE_LABEL,
            "active": profile_id == active_id,
        }
        for profile_id, profile in data["profiles"].items()
    ]


def switch_profile_by_id(device_user_id: str, profile_id: str) -> str | None:
    """Switch active profile by stored profile id and return its label."""
    data = _ensure_default(device_user_id, _load(device_user_id))
    profiles = data["profiles"]
    if profile_id not in profiles:
        return None
    data["active_profile_id"] = profile_id
    _save(device_user_id, data)
    return profiles[profile_id].get("label") or DEFAULT_PROFILE_LABEL


def add_or_switch_profile(device_user_id: str, label: str) -> tuple[str, bool]:
    """Create profile if needed, make it active, and return (label, created)."""
    cleaned = re.sub(r"\s+", " ", label or "").strip()
    if not cleaned:
        cleaned = DEFAULT_PROFILE_LABEL

    data = _ensure_default(device_user_id, _load(device_user_id))
    profiles = data["profiles"]

    if cleaned.lower() in {"me", "myself", "main", "default"}:
        data["active_profile_id"] = DEFAULT_PROFILE_ID
        _save(device_user_id, data)
        return profiles[DEFAULT_PROFILE_ID].get("label", DEFAULT_PROFILE_LABEL), False

    for profile_id, profile in profiles.items():
        if (profile.get("label") or "").strip().lower() == cleaned.lower():
            data["active_profile_id"] = profile_id
            _save(device_user_id, data)
            return profile.get("label") or cleaned, False

    base_id = _slug(cleaned)
    profile_id = base_id
    suffix = 2
    while profile_id in profiles:
        profile_id = f"{base_id}_{suffix}"
        suffix += 1

    profiles[profile_id] = {
        "label": cleaned,
        "user_key": f"{device_user_id}__{profile_id}",
    }
    data["active_profile_id"] = profile_id
    _save(device_user_id, data)
    return cleaned, True


def parse_profile_command(text: str) -> tuple[str, str | None] | None:
    """Parse explicit text commands for shared-device profile management."""
    normalized = re.sub(r"\s+", " ", text or "").strip()
    lower = normalized.lower()
    if not lower:
        return None

    if lower in {"who am i", "whoami", "current user", "current profile"}:
        return "current", None
    if lower in {"switch user", "switch users", "change user", "change users"}:
        return "help", None
    if lower in {"list users", "users", "list profiles", "profiles"}:
        return "list", None

    match = re.match(r"^(?:add|new|create)\s+(?:user|profile)\s+(.+)$", normalized, flags=re.I)
    if match:
        return "switch", match.group(1).strip()

    match = re.match(r"^switch\s+(?:to\s+)?(.+)$", normalized, flags=re.I)
    if match:
        label = match.group(1).strip()
        if label.lower() not in {"user", "users", "profile", "profiles"}:
            return "switch", label

    return None


_NON_NAME_WORDS = {
    "adult",
    "child",
    "kid",
    "senior",
    "pregnant",
    "vegetarian",
    "vegan",
    "diabetic",
    "hungry",
    "back",
    "using",
    "my",
    "mom",
    "mother",
    "dad",
    "father",
    "husband",
    "wife",
    "spouse",
    "partner",
    "son",
    "daughter",
}


def _clean_name(candidate: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", candidate or "").strip(" .,!?:;")
    if not cleaned or len(cleaned) > 32:
        return None
    parts = cleaned.split()
    if len(parts) > 3:
        return None
    if any(part.lower() in _NON_NAME_WORDS for part in parts):
        return None
    if not all(re.match(r"^[A-Za-z][A-Za-z'-]*$", part) for part in parts):
        return None
    return " ".join(part[:1].upper() + part[1:] for part in parts)


def parse_profile_intent(text: str) -> tuple[str, str | None] | None:
    """Parse natural shared-device identity cues.

    This intentionally handles only high-confidence phrasing. Relationship
    mentions like "my daughter..." are not profile switches by themselves,
    because they usually describe the care target rather than the speaker.
    """
    command = parse_profile_command(text)
    if command is not None:
        return command

    normalized = re.sub(r"\s+", " ", text or "").strip()
    lower = normalized.lower()
    if not normalized:
        return None

    if re.search(
        r"\b(?:someone else|another person|my (?:mom|mother|dad|father|husband|wife|spouse|partner|son|daughter|child|kid)) "
        r"(?:is )?(?:using|on|here|talking)\b",
        lower,
    ):
        return "ask_name", None

    patterns = [
        r"^(?:this is|it's|it is)\s+(.+?)(?:\s+(?:now|here))?[.!]?$",
        r"^(?:i am|i'm)\s+(.+?)\s+(?:now|here)[.!]?$",
        r"^(.+?)\s+here[.!]?$",
    ]
    for pattern in patterns:
        match = re.match(pattern, normalized, flags=re.I)
        if match:
            name = _clean_name(match.group(1))
            if name:
                return "switch", name

    return None


def relationship_reference(text: str) -> str | None:
    """Return a relationship term when a message is likely about another person."""
    lower = re.sub(r"\s+", " ", text or "").strip().lower()
    if not lower:
        return None
    if re.search(r"\b(?:using|on|here|talking)\b", lower):
        return None
    match = re.search(
        r"\bmy\s+(mom|mother|dad|father|husband|wife|spouse|partner|son|daughter|child|kid)\b",
        lower,
    )
    if not match:
        return None
    relation = match.group(1)
    aliases = {
        "mother": "mom",
        "father": "dad",
        "kid": "child",
    }
    return aliases.get(relation, relation)
