"""Structured shared-device profile intent routing."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re

from multi_user import parse_profile_command, parse_profile_intent, relationship_reference


PROFILE_ROUTER_SYSTEM_PROMPT = """
You classify whether a user message is about shared-device identity/profile routing.

Return JSON only with this schema:
{
  "action": "none | current | help | list | ask_name | switch | relationship_clarify",
  "profile_name": string or null,
  "relationship": string or null,
  "confidence": number between 0 and 1
}

Meanings:
- none: ordinary nutrition, food safety, resources, health, preference, allergy, or goal message.
- current: user asks which profile is active.
- help: user asks how to switch/change users.
- list: user asks to list profiles/users.
- ask_name: user says a different/new/another person is using the device, but gives no name.
- switch: user clearly identifies the active device user by name.
- relationship_clarify: user mentions their mom/daughter/son/spouse/etc. as a possible care target, and it is ambiguous whether that person is using the device or the current profile is asking on their behalf.

Rules:
- Prefer "none" unless the message is clearly about who is using the device, whose profile should be active, or whether advice is for another person.
- Do not classify diet, allergies, medical conditions, pregnancy, goals, dislikes, or ordinary questions as profile switching.
- "I'm vegetarian", "I'm pregnant", and "I have diabetes" are not profile switches by themselves.
- "my daughter is 4", "can you help my mom", or "my son has allergies" should be relationship_clarify.
- "my daughter is using this", "someone else is using this", or "I'm a different user" should be ask_name unless a name is provided.
- If a known saved profile name is clearly referenced as the active user, action should be switch with that name.
"""

PROFILE_CONTRADICTION_SYSTEM_PROMPT = """
You decide whether a new user message conflicts with the active saved profile enough to ask a brief profile-safety clarification.

Return JSON only with this schema:
{
  "needs_clarification": true or false,
  "saved_fact": string or null,
  "reason": string or null,
  "confidence": number between 0 and 1
}

Ask for clarification only when all are true:
- The saved profile has a strong fact such as diabetes, pregnancy, severe allergy, child/older adult age context, vegetarian/vegan, high blood pressure, or medication context.
- The new message asks for advice that would be materially different or potentially unsafe if that saved fact applies.
- The user did not already make clear they are asking for someone else or switching users.

Do not ask for clarification for ordinary questions that can be answered safely using the profile.
Examples that should be false:
- Saved diabetes, message: "Can I eat rice?"
- Saved diabetes, message: "I'm diabetic, what breakfast is good?"
- Saved peanut allergy, message: "Are peanuts healthy?"
- Saved vegetarian, message: "My family wants chicken ideas."

Examples that should be true:
- Saved diabetes, message asks for daily soda/donuts/sugary drinks.
- Saved peanut allergy, message asks what peanut snacks to pack for myself.
- Saved child profile, message says "I'm pregnant."
- Saved vegetarian, message asks for chicken meal prep for myself.
"""


@dataclass(frozen=True)
class ProfileRoute:
    action: str
    profile_name: str | None = None
    relationship: str | None = None
    confidence: float = 0.0


@dataclass(frozen=True)
class ProfileContradiction:
    needs_clarification: bool
    saved_fact: str | None = None
    reason: str | None = None
    confidence: float = 0.0


_VALID_ACTIONS = {
    "none",
    "current",
    "help",
    "list",
    "ask_name",
    "switch",
    "relationship_clarify",
}


def _parse_json_object(raw: str) -> dict | None:
    text = (raw or "").strip().strip("`").strip()
    if text.lower().startswith("json"):
        text = text[4:].lstrip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _clean_string(value) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", str(value)).strip(" .,!?:;")
    return cleaned or None


def _route_from_command(text: str) -> ProfileRoute | None:
    command = parse_profile_command(text)
    if command is None:
        return None
    action, label = command
    if action == "switch":
        return ProfileRoute("switch", profile_name=label, confidence=1.0)
    return ProfileRoute(action, confidence=1.0)


def _normalize_model_route(data: dict | None) -> ProfileRoute:
    if not data:
        return ProfileRoute("none")

    action = _clean_string(data.get("action")) or "none"
    action = action.lower()
    if action not in _VALID_ACTIONS:
        action = "none"

    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    profile_name = _clean_string(data.get("profile_name"))
    relationship = _clean_string(data.get("relationship"))

    if action == "switch" and not profile_name:
        return ProfileRoute("none")
    if action == "relationship_clarify" and not relationship:
        return ProfileRoute("none")

    if action in {"switch", "ask_name"} and confidence < 0.82:
        return ProfileRoute("none", confidence=confidence)
    if action == "relationship_clarify" and confidence < 0.74:
        return ProfileRoute("none", confidence=confidence)
    if action in {"current", "help", "list"} and confidence < 0.8:
        return ProfileRoute("none", confidence=confidence)

    return ProfileRoute(
        action=action,
        profile_name=profile_name,
        relationship=relationship,
        confidence=confidence,
    )


def _normalize_contradiction(data: dict | None) -> ProfileContradiction:
    if not data:
        return ProfileContradiction(False)
    needs = bool(data.get("needs_clarification"))
    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    if not needs or confidence < 0.78:
        return ProfileContradiction(False, confidence=confidence)
    return ProfileContradiction(
        True,
        saved_fact=_clean_string(data.get("saved_fact")),
        reason=_clean_string(data.get("reason")),
        confidence=confidence,
    )


def _profile_text(profile: dict) -> str:
    return "\n".join(
        f"{key}: {value}"
        for key, value in profile.items()
        if _clean_string(value)
    )


def _should_check_contradiction(text: str, profile: dict) -> bool:
    message = re.sub(r"\s+", " ", text or "").strip().lower()
    facts = _profile_text(profile).lower()
    if not message or not facts:
        return False

    if re.search(r"\b(for someone else|not for me|for my friend|for a friend|my family|for my family)\b", message):
        return False

    if "diabet" in facts and re.search(
        r"\b(sugary|sugar|candy|soda|donut|doughnut|dessert|juice|milkshake|cake|cookie|sweet drink)\b",
        message,
    ):
        return True

    allergies = str(profile.get("allergies", "")).lower()
    if allergies:
        allergy_terms = [
            term.strip()
            for term in re.split(r"[,;/]|\band\b", allergies)
            if len(term.strip()) >= 4
        ]
        expanded_terms = set(allergy_terms)
        expanded_terms.update(term[:-1] for term in allergy_terms if term.endswith("s"))
        if any(re.search(rf"\b{re.escape(term)}s?\b", message) for term in expanded_terms):
            return True

    restrictions = " ".join(
        str(profile.get(field, "")).lower()
        for field in ("dietary_restriction", "preferences", "durable_extras")
    )
    if re.search(r"\b(vegetarian|vegan)\b", restrictions) and re.search(
        r"\b(chicken|beef|pork|bacon|turkey|fish|meat|burger|steak)\b",
        message,
    ):
        return True

    age_group = str(profile.get("age_group", "")).lower()
    if age_group == "child" and re.search(r"\b(pregnan|breastfeed|beer|wine|alcohol)\b", message):
        return True

    if re.search(r"\b(pregnan|breastfeed)\b", facts) and re.search(
        r"\b(alcohol|beer|wine|raw sushi|raw fish|deli meat)\b",
        message,
    ):
        return True

    if re.search(r"\b(hypertension|blood pressure|low sodium)\b", facts) and re.search(
        r"\b(high sodium|salty|salt-heavy|chips every day|processed meat)\b",
        message,
    ):
        return True

    return False


def _fallback_route(text: str) -> ProfileRoute:
    intent = parse_profile_intent(text)
    if intent is not None:
        action, label = intent
        if action == "switch":
            return ProfileRoute("switch", profile_name=label, confidence=0.9)
        return ProfileRoute(action, confidence=0.9)

    relation = relationship_reference(text)
    if relation:
        return ProfileRoute("relationship_clarify", relationship=relation, confidence=0.8)
    return ProfileRoute("none")


def route_profile_intent(
    *,
    text: str,
    active_profile: str,
    profiles: list[dict],
    ai,
    session_id: str,
) -> ProfileRoute:
    """Return a conservative profile routing decision for one user message."""
    command_route = _route_from_command(text)
    if command_route is not None:
        return command_route

    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return ProfileRoute("none")

    profile_labels = [
        str(profile.get("label", "")).strip()
        for profile in profiles
        if str(profile.get("label", "")).strip()
    ]
    query = json.dumps(
        {
            "message": normalized,
            "active_profile": active_profile,
            "saved_profiles": profile_labels,
        },
        ensure_ascii=True,
    )

    try:
        raw = ai.ask(
            PROFILE_ROUTER_SYSTEM_PROMPT,
            query,
            session_id,
            lastk_override=0,
        )
        route = _normalize_model_route(_parse_json_object(raw))
        if route.action != "none":
            return route
    except Exception:
        pass

    return _fallback_route(normalized)


def route_profile_contradiction(
    *,
    text: str,
    profile: dict,
    active_profile: str,
    ai,
    session_id: str,
) -> ProfileContradiction:
    """Return whether a rare profile-safety clarification is warranted."""
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not _should_check_contradiction(normalized, profile):
        return ProfileContradiction(False)

    query = json.dumps(
        {
            "message": normalized,
            "active_profile": active_profile,
            "saved_profile": profile,
        },
        ensure_ascii=True,
    )
    try:
        raw = ai.ask(
            PROFILE_CONTRADICTION_SYSTEM_PROMPT,
            query,
            session_id,
            lastk_override=0,
        )
        return _normalize_contradiction(_parse_json_object(raw))
    except Exception:
        return ProfileContradiction(False)
