from dataclasses import dataclass
import json
import re

from wa_service_sdk import Button


@dataclass
class TurnContext:
    user_id: str
    user_message: str
    session: str
    profile: dict
    profile_context: str


def make_buttons(buttons_data: list[dict]) -> list[Button]:
    """Convert list of {id, title} dicts to SDK Button objects."""
    return [Button(id=b["id"], title=b["title"]) for b in buttons_data]


def user_session(user_id: str) -> str:
    return f"NutritionBot_User_{user_id}"


def parse_resources_json(raw: str) -> dict | None:
    """Extract a single JSON object from model output; return dict or None."""
    text = (raw or "").strip().strip("`").strip()
    if text.lower().startswith("json"):
        text = text[4:].lstrip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start : i + 1]
                try:
                    out = json.loads(chunk)
                    return out if isinstance(out, dict) else None
                except Exception:
                    return None
    return None


def resource_suggested_buttons(items: list | None) -> list[Button]:
    out: list[Button] = []
    if not items:
        return out
    for i, item in enumerate(items[:3]):
        if isinstance(item, str):
            title = item[:20]
        elif isinstance(item, dict):
            title = str(item.get("title", ""))[:20]
        else:
            title = ""
        if title:
            out.append(Button(id=f"resources_dyn_{i}", title=title))
    return out


def resources_action_type(action: dict) -> str:
    return (action.get("type") or "").strip().upper()


def wants_wic_store_by_location(text: str) -> bool:
    """True if the user is asking for WIC-accepting stores near them."""
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    if "wic" not in t:
        return False
    return bool(
        re.search(
            r"\b(nearest|closest|nearby|near me|around me)\b|"
            r"\bwhere\b.{0,60}\b(store|shop|retailer)\b|"
            r"\b(find|which)\b.{0,40}\bstore|"
            r"\bstores?\b.{0,30}\b(near|close|around)",
            t,
        )
    )


def is_synthetic_resources_hub_opener(text: str) -> bool:
    """True when this turn is the scripted open from the Find Resources button."""
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    return "opened find resources from the main menu" in t


def is_greeting(text: str) -> bool:
    """Return True for simple greetings that should show the welcome menu."""
    normalized = text.strip().lower()
    greetings = {
        "",
        "hi",
        "hello",
        "hey",
        "hiya",
        "good morning",
        "good afternoon",
        "good evening",
        "start",
        "menu",
        "help",
    }
    if normalized in greetings:
        return True
    if normalized.startswith("@") or "switch to" in normalized or "nutritionbot" in normalized:
        return True
    return False


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def has_profile_value(profile: dict, *fields: str) -> bool:
    return any(normalize_text(profile.get(field, "")) for field in fields)


def choose_profile_target(profile: dict) -> str | None:
    """Pick the next profile area to ask about, based on questionnaire-style priorities."""
    if not has_profile_value(profile, "asking_for"):
        return "asking_for"
    if not has_profile_value(profile, "age_group"):
        return "age_group"
    if not has_profile_value(profile, "main_goal"):
        return "main_goal"
    if not has_profile_value(
        profile,
        "health_conditions",
        "medications",
        "allergies",
        "dietary_restriction",
    ):
        return "health_context"
    if not has_profile_value(profile, "preferences", "disliked_foods", "recurring_needs"):
        return "routine"
    return None


def should_continue_profile_flow(user_message: str) -> bool:
    """Treat short, direct replies as answers to the most recent profile question."""
    text = normalize_text(user_message)
    if not text:
        return True
    if "?" in text:
        return False
    lower = text.lower()
    interrupting_phrases = {
        "food safety",
        "nutrition",
        "find stores",
        "find resources",
        "wic",
        "snap",
        "help",
        "menu",
        "start",
        "hi",
        "hello",
    }
    if lower in interrupting_phrases:
        return False
    return len(text.split()) <= 12


def looks_like_profile_answer(target: str, user_message: str) -> bool:
    """Use target-specific heuristics so new requests do not get swallowed as profile answers."""
    text = normalize_text(user_message)
    if not text:
        return True
    if not should_continue_profile_flow(text):
        return False

    lower = text.lower()
    request_starters = (
        "can you",
        "could you",
        "would you",
        "what ",
        "how ",
        "should ",
        "do i ",
        "is it ",
        "are there",
        "give me",
        "tell me",
        "help me",
    )
    if lower.startswith(request_starters):
        return False

    if target == "asking_for":
        return bool(
            re.search(
                r"\b(for me|myself|self|me|my child|my kid|my son|my daughter|my baby|my mom|my mother|my dad|my father|my parent|my husband|my wife|my spouse|someone else)\b",
                lower,
            )
        )

    if target == "age_group":
        return bool(re.search(r"\b(under|adult|child|kid|teen|young|middle|senior|elder|\d{1,3})\b", lower))

    if target == "health_context":
        return bool(
            re.search(
                r"\b(allerg|diabet|pregnan|gluten|vegan|vegetarian|halal|kosher|medication|metformin|warfarin|hypertension|blood pressure|none|no allergies)\b",
                lower,
            )
        )

    if target == "main_goal":
        return len(text.split()) <= 10 and not lower.startswith(("food ", "meal ", "store "))

    if target == "routine":
        return len(text.split()) <= 12

    return False


def format_profile_for_prompt(profile: dict, blank_profile: str) -> str:
    if not profile:
        return blank_profile
    return "\n".join(f"{k}: {v}" for k, v in profile.items() if normalize_text(v))
