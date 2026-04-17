"""
NutritionBot Agent
===================
Central router that classifies user intent and calls the right tool.

Intents:
    food_safety      → hub + optional RAG router on follow-up; typed questions use query_rag
    nutrition_advice → LLM router: optional query_rag else main_system _ai.ask
    find resources   → LLM-led resources_mode + JSON; optional KB snippets via router + get_context
    find_stores      → enters resources_mode (same)
    find_wic_stores  → request_location → resource_tools (WIC / general stores)
    out_of_scope     → LLM-generated refusal

Usage (from Nutrition_Bot.py):
    agent = NutritionAgent()

    # User sends a text message
    text, buttons = agent.run(user_message, user_id)

    # User clicks a button
    text, buttons = agent.run_tool(interaction_id, user_id)

    # User sends a location
    text, buttons = agent.run_location(lat, lng, user_id)
"""

import os
import mimetypes
from AI import AI
from rag_pipeline import RAGPipeline
import json
from benchmark_locations import lookup_benchmark_coordinates as _lookup_benchmark_coordinates

from prompts import (
    main_system_prompt,
    button_creator_prompt,
    button_title_repair_prompt,
    eligibility_check_prompt,
    intent_classifier_prompt,
    profile_nudge_prompt,
    WELCOME_BUTTONS,
    WELCOME_FALLBACK_MESSAGE,
    welcome_generator_prompt,
    food_safety_hub_prompt,
    FOOD_SAFETY_HUB_FALLBACK_MESSAGE,
    FOOD_SAFETY_HUB_BUTTON_FALLBACK,
    rag_router_prompt,
    kb_retrieval_router_prompt,
    thanks_tailor_prompt,
    image_analysis_prompt,
    RESOURCES_FALLBACK_BUTTONS,
    WIC_POST_SCREENING_BUTTONS,
    NUTRITION_FALLBACK_BUTTONS,
    resources_tool_selector_prompt,
    resources_synthesizer_prompt,
)
from resource_tools import run_tool as _run_resource_tool

from wa_service_sdk import Button

try:
    from wa_service_sdk.responses import MAX_BUTTON_TITLE_CHARS, MAX_INTERNAL_ID_CHARS
except ImportError:
    MAX_BUTTON_TITLE_CHARS = 20
    MAX_INTERNAL_ID_CHARS = 120
from agent_state import STATE as _state
from resource_tools import run_tool as _run_resource_tool, start_eligibility as _wic_eligibility_opening
from agent_profile import (
    answer_saved_profile_task,
    build_profile_question,
    continue_profile_conversation,
    maybe_append_profile_nudge,
    maybe_start_profile_from_welcome,
    profile_buttons_for_target,
    profile_button_value,
    save_and_reload_profile,
)
from agent_helpers import (
    TurnContext,
    choose_profile_target as _choose_profile_target,
    format_profile_for_prompt as _format_profile_for_prompt,
    is_greeting as _is_greeting,
    is_synthetic_resources_hub_opener as _is_synthetic_resources_hub_opener,
    looks_like_profile_answer as _looks_like_profile_answer,
    make_buttons as _make_buttons,
    normalize_text as _normalize_text,
    parse_resources_json as _parse_resources_json,
    resource_suggested_buttons as _resource_suggested_buttons,
    resources_action_type as _resources_action_type,
    should_continue_profile_flow as _should_continue_profile_flow,
    user_session as _user_session,
    wants_wic_store_by_location as _wants_wic_store_by_location,
)

import ast
import re


# ── Shared instances (loaded once at startup) ─────────────────────────────────
_ai  = AI()
_rag = RAGPipeline()
_rag.build_public_index()
# Same UserMemory as RAGPipeline (shared SentenceTransformer + per-user FAISS cache).
_mem = _rag.memory
BLANK_PROFILE = "(no profile info)"
DISCLAIMER_BUTTONS = [
    {"id": "disclaimer_agree", "title": "Agree"},
    {"id": "disclaimer_decline", "title": "Decline"},
]
FIRST_USE_DISCLAIMER = (
    "Hi, I'm Nura 😊. I provide general nutrition, food safety, and Massachusetts resource information, "
    "but I am not a doctor and I do not diagnose medical conditions or provide emergency care. "
    "Like anyone, I sometimes get things wrong but I hope to help you as best as I can! "
    "If you have severe symptoms or think it may be an emergency, seek urgent medical care. "
    "Please tap Agree to continue."
)

_DISALLOWED_BUTTON_PATTERNS = (
    r"\bcall\b",
    r"\blocation\b",
    r"\bmap\b",
    r"\bdirections?\b",
    # Allow "apply" in titles (WIC apply / eligibility); block only bot-action phrasing.
    r"\bapply\s+now\b",
    r"\border\b",
    r"\bbuy\b",
    r"\bvisit\b",
    r"\bopen\b",
    r"\bcontact\b",
    r"\btext\b",
    r"\bemail\b",
)


def _parse_llm_button_items(raw: str) -> list[dict] | None:
    """Parse first LLM reply into up to 3 dicts with id and title; None on failure."""
    import json as _json

    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = text.strip().strip("`").strip()
    if text.startswith("json"):
        text = text[4:].strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        items = ast.literal_eval(text)
    except Exception:
        try:
            items = _json.loads(text)
        except Exception:
            return None
    if not isinstance(items, list) or not items:
        return None
    out: list[dict] = []
    for item in items[:3]:
        if isinstance(item, str):
            try:
                info = ast.literal_eval(item)
            except Exception:
                try:
                    info = _json.loads(item)
                except Exception:
                    continue
        elif isinstance(item, dict):
            info = item
        else:
            continue
        title = str(info.get("title", "")).strip()
        bid = str(info.get("id", "btn")).strip() or "btn"
        if not title:
            continue
        out.append({"id": bid, "title": title})
    return out if out else None


def _any_button_title_over_limit(items: list[dict], limit: int = MAX_BUTTON_TITLE_CHARS) -> bool:
    return any(len(str(d.get("title", "")).strip()) > limit for d in items)


def _repair_button_items_with_llm(items: list[dict], session_id: str) -> list[dict] | None:
    """One rewrite pass so every title is a full short phrase ≤ limit (no mechanical truncation)."""
    try:
        payload = json.dumps(items[:3], ensure_ascii=False)
        raw = _ai.ask(button_title_repair_prompt, payload, session_id + "_btnfix")
        fixed = _parse_llm_button_items(raw)
        if not fixed:
            return None
        if _any_button_title_over_limit(fixed):
            return None
        return fixed[:3]
    except Exception:
        return None


def _generate_buttons(
    response: str,
    session_id: str,
    fallback_buttons: list[dict] | None = None,
) -> list[Button]:
    """Ask LLM to generate contextual follow-up buttons based on a response."""
    fb = fallback_buttons if fallback_buttons is not None else WELCOME_BUTTONS

    def _is_allowed_button_title(title: str) -> bool:
        normalized = (title or "").strip().lower()
        if not normalized or len(normalized) > MAX_BUTTON_TITLE_CHARS:
            return False
        return not any(re.search(pattern, normalized) for pattern in _DISALLOWED_BUTTON_PATTERNS)

    try:
        raw = _ai.ask(button_creator_prompt, response, session_id + "_btn")
        items = _parse_llm_button_items(raw)
        if not items:
            retry_hint = (
                "\n\n[SYSTEM] Your previous reply could not be parsed as a JSON array. "
                'Output ONLY: [{"id":"snake_case","title":"20chars max"},...] — 2 or 3 items, no markdown, no prose.'
            )
            raw = _ai.ask(button_creator_prompt, (response or "")[:6000] + retry_hint, session_id + "_btn_retry")
            items = _parse_llm_button_items(raw)
        if not items:
            return _make_buttons(fb)
        if _any_button_title_over_limit(items):
            repaired = _repair_button_items_with_llm(items, session_id)
            if repaired:
                items = repaired
        buttons = []
        for info in items:
            title = str(info.get("title", ""))
            bid = str(info.get("id", "btn")).strip() or "btn"
            if len(bid) > MAX_INTERNAL_ID_CHARS:
                bid = bid[:MAX_INTERNAL_ID_CHARS]
            if not _is_allowed_button_title(title):
                continue
            buttons.append(Button(id=bid, title=title.strip()))
        return buttons[:3] if buttons else _make_buttons(fb)
    except Exception:
        print("Error")
        return _make_buttons(fb)


def _debug_log(message: str) -> None:
    print(f"[DEBUG] {message}")


def _should_use_rag_food_safety(user_text: str, session_id: str) -> bool:
    """LLM routes whether this food-safety turn should use RAG (default yes if unclear)."""
    text = (user_text or "").strip().lower()
    if not text:
        return False
    if len(text.split()) <= 3 and text in {"hi", "hello", "thanks", "ok", "okay"}:
        return False
    if re.search(
        r"\b(leftover|leftovers|safe|safety|fridge|freezer|storage|expire|expired|expiration|spoil|spoiled|bad|reheat|thaw|raw|cooked|temperature|temp|canned|milk|chicken|meat|rice|egg|eggs)\b",
        text,
    ):
        return True
    try:
        raw = _ai.ask(rag_router_prompt, user_text, session_id + "_ragroute").strip().lower()
        if raw.startswith("no"):
            return False
        if raw.startswith("yes"):
            return True
    except Exception:
        pass
    return True


def _should_retrieve_public_kb(user_message: str, session_id: str, lane: str) -> bool:
    """LLM routes nutrition/resources KB retrieval (default yes if unclear)."""
    text = (user_message or "").strip().lower()
    if not text or text in {"thanks", "thank you", "ok", "okay", "yes", "no"}:
        return False
    if lane == "resources":
        if _wants_wic_store_by_location(text):
            return True
        if re.search(
            r"\b(wic|wic-authorized|market basket|farmers market|eligib|qualif|benefit|apply|application|store|stores|retailer|resource)\b",
            text,
        ):
            return True
        if len(text.split()) <= 4:
            return False
    if lane == "nutrition":
        # Simple meal-idea questions should go straight to answer generation.
        if "?" in text and len(text.split()) <= 8 and re.search(r"\b(breakfast|lunch|dinner|snack|meal)\b", text):
            return False
        if re.search(
            r"\b(calorie|protein|fiber|sodium|cholesterol|vitamin|mineral|serving|portion|diet|meal plan|food safety|pregnan|diabetes|allerg|gluten|vegan|vegetarian)\b",
            text,
        ):
            return True
        if len(text.split()) <= 4:
            return False
    payload = f"[LANE]\n{lane}\n[USER MESSAGE]\n{(user_message or '').strip() or '(empty)'}"
    try:
        raw = _ai.ask(kb_retrieval_router_prompt, payload, session_id + "_kbroute").strip().lower()
        if raw.startswith("no"):
            return False
        if raw.startswith("yes"):
            return True
    except Exception:
        pass
    return True


def _is_exact_store_fact_question(user_message: str) -> bool:
    """Return True for store-fact lookups that should use RAG, not location search."""
    text = (user_message or "").strip().lower()
    if not text:
        return False

    asks_store_fact = bool(
        re.search(
            r"\b(phone number|phone|call|hours|open|close|closing|opening|when does|what time|"
            r"wic|accept wic|take wic|cover wic|covered by wic)\b",
            text,
        )
    )
    mentions_specific_store = bool(
        re.search(
            r"\b(stop\s*&?\s*shop|market basket|whole foods|h[\-\s]?mart|pemberton farms|cvs|walgreens|shaw'?s|hannaford)\b",
            text,
        )
    ) or bool(re.search(r"\b\d{1,6}\s+.+\b(street|st|avenue|ave|boulevard|blvd|road|rd|drive|dr|lane|ln|parkway|pkwy)\b", text))

    asks_proximity = bool(
        re.search(
            r"\b(closest|nearest|nearby|near me|around me|closer|distance|how far|from my house|from where i live)\b",
            text,
        )
    )

    return asks_store_fact and mentions_specific_store and not asks_proximity


def _is_proximity_store_search(user_message: str) -> bool:
    """Return True when user asks for nearby/closest stores."""
    text = (user_message or "").strip().lower()
    if not text:
        return False
    asks_store = bool(
        re.search(
            r"\b(store|stores|grocery|supermarket|market)\b",
            text,
        )
    )
    asks_proximity = bool(
        re.search(
            r"\b(closest|nearest|nearby|near me|around me|closer|distance|how far)\b",
            text,
        )
    )
    return asks_store and asks_proximity


def _derive_store_keyword(user_message: str) -> str | None:
    """Infer a focused keyword for general store searches from cuisine/specialty descriptors."""
    text = (user_message or "").strip().lower()
    if not text:
        return None
    cuisine_keywords = {
        "korean": "korean grocery store",
        "chinese": "chinese grocery store",
        "japanese": "japanese grocery store",
        "indian": "indian grocery store",
        "mexican": "mexican grocery store",
        "middle eastern": "middle eastern grocery store",
        "halal": "halal grocery store",
        "asian": "asian grocery store",
    }
    for token, keyword in cuisine_keywords.items():
        if token in text:
            return keyword

    # Generic patterns for open-ended descriptors, e.g.:
    # "closest brazilian grocery store", "I want to make ethiopian food"
    explicit_match = re.search(
        r"\b([a-z][a-z\s\-]{2,30})\s+(?:grocery\s+store|supermarket|market)\b",
        text,
    )
    if explicit_match:
        descriptor = re.sub(r"\s+", " ", explicit_match.group(1)).strip()
        if descriptor and descriptor not in {"closest", "nearest", "nearby", "local"}:
            return f"{descriptor} grocery store"

    cuisine_match = re.search(
        r"\b(?:make|cook|cooking)\s+([a-z][a-z\s\-]{2,30})\s+food\b",
        text,
    )
    if cuisine_match:
        descriptor = re.sub(r"\s+", " ", cuisine_match.group(1)).strip()
        if descriptor:
            return f"{descriptor} grocery store"
    return None


def _is_wic_item_coverage_question(user_message: str) -> bool:
    """Return True for WIC eligibility questions about a food item/product."""
    text = (user_message or "").strip().lower()
    if not text:
        return False

    mentions_wic = bool(re.search(r"\bwic\b", text))
    if not mentions_wic:
        return False

    asks_coverage = bool(
        re.search(
            r"\b(cover|covered|eligible|eligibility|approved|allowed|qualif|"
            r"can i buy|can i get|does wic (cover|pay|include)|wic (cover|pay|include))\b",
            text,
        )
    )
    if not asks_coverage:
        return False

    # Keep store/location lookups on the store tool path.
    asks_store_lookup = bool(
        re.search(
            r"\b(store|stores|nearest|nearby|near me|closest|address|hours|phone|"
            r"open|close|location|map|directions)\b",
            text,
        )
    )
    return not asks_store_lookup


def _classify_intent(user_message: str, session_id: str) -> str:
    """Classify user message into one of the four intents."""
    if _wants_wic_store_by_location(user_message):
        return "find resources"
    text = (user_message or "").strip().lower()
    if re.search(
        r"\b(wic|snap|ebt|food pantry|pantry|community fridge|market basket|farmers market|hip program|eligib|qualif|benefit|apply|resource|store|stores|retailer)\b",
        text,
    ):
        return "find resources"
    if re.search(
        r"\b(safe|safety|leftover|leftovers|fridge|freezer|storage|expire|expired|expiration|spoil|spoiled|reheat|thaw|raw|undercooked|food poisoning)\b",
        text,
    ):
        return "food_safety"
    if re.search(
        r"\b(healthy|healthier|nutrition|eat better|diet|meal|snack|recipe|recipes|breakfast|lunch|dinner|protein|fiber|vegetable|vegetables|fruit|pregnan|diabetes|allerg|gluten|vegan|vegetarian)\b",
        text,
    ):
        return "nutrition_advice"
    result = _ai.ask(intent_classifier_prompt, user_message, session_id)
    intent = re.sub(r"\s+", " ", (result or "").strip().lower())
    if intent.startswith("find resources") or intent == "find_stores" or (
        "find" in intent and "resource" in intent
    ):
        intent = "find resources"
    elif intent.startswith("food_safety"):
        intent = "food_safety"
    elif intent.startswith("nutrition"):
        intent = "nutrition_advice"
    elif intent.startswith("out_of_scope") or intent.startswith("out of scope"):
        intent = "out_of_scope"
    valid = {"food_safety", "nutrition_advice", "find resources", "out_of_scope"}
    return intent if intent in valid else "nutrition_advice"


def _should_extract_profile_from_message(user_message: str) -> bool:
    """Only run memory extraction when a message likely contains durable profile facts."""
    text = _normalize_text(user_message)
    if not text:
        return False
    lower = text.lower()
    profile_signals = (
        r"\b(i am|i'm|for me|myself|my child|my kid|my son|my daughter|"
        r"my mom|my mother|my dad|my father|my parent|my spouse|"
        r"allerg|diabet|pregnan|gluten|vegan|vegetarian|halal|kosher|"
        r"medication|blood pressure|hypertension|goal|prefer|preferences|"
        r"dislike|don't like|budget|wic|snap)\b"
    )
    if re.search(profile_signals, lower):
        return True
    # Plain questions are rarely profile facts; skip the extra LLM extraction call.
    if "?" in text:
        return False
    return len(text.split()) <= 12


def _append_button_intro(response: str, buttons: list[Button], session_id: str) -> str:
    """Append a fixed sentence introducing follow-up buttons without another model call."""
    if not buttons:
        return response
    main, sources_block = response, ""
    if "\n\nSources:" in response:
        main, _, rest = response.rpartition("\n\nSources:")
        sources_block = "\n\nSources:" + rest
    intro = "You can tap the button or ask another question:"
    return f"{main.rstrip()}\n\n{intro}{sources_block}"


def _profile_acknowledgement(profile: dict) -> str:
    if _normalize_text(profile.get("asking_for")) in {"child", "parent", "spouse", "other"}:
        return _ai.ask(thanks_tailor_prompt, "user is writing about somone else", "thank-you")
    return _ai.ask(thanks_tailor_prompt, "user is writing about themselves", "thank-you")


def _parse_resources_json(raw: str) -> dict | None:
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


def _normalize_resources_decision_wic_only(
    tool: str, params: dict | None, reply: str
) -> tuple[str, dict, str]:
    """Find Resources is WIC-only: coerce tools/params and drop non-WIC tool names."""
    t = (tool or "none").strip().lower()
    p = dict(params or {})
    r = (reply or "").strip()

    if t == "affordable_overview":
        return "explain_program", {"program": "wic"}, r

    allowed = {
        "search_wic_stores",
        "search_general_stores",
        "explain_program",
        "start_eligibility",
        "none",
    }
    if t not in allowed:
        return "none", {}, r

    if t == "explain_program":
        p["program"] = "wic"

    return t, p, r


def _resource_suggested_buttons(items: list | None) -> list[Button]:
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


def _resources_action_type(action: dict) -> str:
    return (action.get("type") or "").strip().upper()


# ── WIC eligibility screening (after start_eligibility tool: buttons + typed HH size) ──
WIC_STEP_CATEGORY = "category"
WIC_STEP_RESIDENCY = "residency"
WIC_STEP_ADJUNCTIVE = "adjunctive"
WIC_STEP_HOUSEHOLD_INPUT = "household_input"
WIC_STEP_INCOME = "income"

# Massachusetts WIC gross income standards (185% FPL–style table; yearly + monthly).
_WIC_YEARLY: dict[int, int] = {
    1: 28_953,
    2: 39_128,
    3: 49_303,
    4: 59_478,
    5: 69_653,
    6: 79_828,
    7: 90_003,
    8: 100_178,
}
_WIC_MONTHLY: dict[int, int] = {
    1: 2_413,
    2: 3_261,
    3: 4_109,
    4: 4_957,
    5: 5_805,
    6: 6_653,
    7: 7_501,
    8: 8_349,
}
_WIC_YEARLY_PER_EXTRA = 10_175
_WIC_MONTHLY_PER_EXTRA = 848  # 10175/12 rounded; matches published add-on column


def _wic_clear_tracking(user_id: str) -> None:
    _state.wic_eligibility_steps.pop(user_id, None)
    _state.wic_eligibility_answers.pop(user_id, None)


def _wic_category_buttons() -> list[Button]:
    return _make_buttons(
        [
            {"id": "wic_elig_cat_yes", "title": "Yes, applies to me"},
            {"id": "wic_elig_cat_no", "title": "No, none of those"},
            {"id": "wic_elig_cat_unsure", "title": "Not sure"},
        ]
    )


def _wic_residency_buttons() -> list[Button]:
    return _make_buttons(
        [
            {"id": "wic_elig_ma_yes", "title": "Yes, I live in MA"},
            {"id": "wic_elig_ma_no", "title": "No, not in MA"},
            {"id": "wic_elig_ma_unsure", "title": "Not sure"},
        ]
    )


def _wic_adjunctive_buttons() -> list[Button]:
    return _make_buttons(
        [
            {"id": "wic_elig_adj_yes", "title": "Yes, one of those"},
            {"id": "wic_elig_adj_no", "title": "No"},
            {"id": "wic_elig_adj_unsure", "title": "Not sure"},
        ]
    )


def _wic_parse_household_size_text(text: str) -> int | None:
    """Parse a household count from free text (1–60)."""
    t = (text or "").strip()
    if not t:
        return None
    low = t.lower()
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
    }
    if low in words:
        return words[low]
    m = re.match(r"^(\d{1,2})\s*$", t)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 60:
            return n
    m = re.search(r"\b(\d{1,2})\b", t)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 60:
            return n
    return None


def _wic_monthly_income_limit(hh_size: int) -> int:
    """Gross monthly income guideline for household size (MA WIC table; 9+ extrapolated)."""
    if hh_size <= 0:
        return _WIC_MONTHLY[1]
    if hh_size <= 8:
        return _WIC_MONTHLY[hh_size]
    return _WIC_MONTHLY[8] + (hh_size - 8) * _WIC_MONTHLY_PER_EXTRA


def _wic_yearly_income_limit(hh_size: int) -> int:
    if hh_size <= 0:
        return _WIC_YEARLY[1]
    if hh_size <= 8:
        return _WIC_YEARLY[hh_size]
    return _WIC_YEARLY[8] + (hh_size - 8) * _WIC_YEARLY_PER_EXTRA


def _wic_income_buttons_monthly(monthly: int) -> list[Button]:
    """Under/Over use the same dollar cutoff shown in the question (≤20 char titles)."""
    m_str = f"${monthly:,}/mo"
    under = f"Under {m_str}"
    over = f"Over {m_str}"
    if len(under) > MAX_BUTTON_TITLE_CHARS:
        under = f"Under ${monthly}/mo"[:MAX_BUTTON_TITLE_CHARS]
    if len(over) > MAX_BUTTON_TITLE_CHARS:
        over = f"Over ${monthly}/mo"[:MAX_BUTTON_TITLE_CHARS]
    return _make_buttons(
        [
            {"id": "wic_elig_inc_under", "title": under},
            {"id": "wic_elig_inc_over", "title": over},
            {"id": "wic_elig_inc_unsure", "title": "Not sure"},
        ]
    )


def _wic_income_question_text(hh_size: int) -> str:
    monthly = _wic_monthly_income_limit(hh_size)
    yearly = _wic_yearly_income_limit(hh_size)
    tail = ""
    if hh_size >= 9:
        tail = (
            " For households larger than 9, limits increase by about $848 per month (about $10,175 per year) "
            "for each extra person—confirm the exact figure with WIC or mass.gov/wic."
        )
    return (
        f"For {hh_size} people in your household, WIC often compares your gross monthly income before taxes "
        f"to about ${monthly:,}/month (roughly ${yearly:,}/year). Is your usual gross monthly household income "
        f"under that amount, over it, or are you not sure? General information only—not a legal determination.{tail}"
    )


def _wic_step_expected_prefix(step: str) -> str | None:
    return {
        WIC_STEP_CATEGORY: "wic_elig_cat_",
        WIC_STEP_RESIDENCY: "wic_elig_ma_",
        WIC_STEP_ADJUNCTIVE: "wic_elig_adj_",
        WIC_STEP_HOUSEHOLD_INPUT: None,
        WIC_STEP_INCOME: "wic_elig_inc_",
    }.get(step)


def _wic_eligibility_map_text_to_id(step: str, text: str) -> str | None:
    t = (text or "").strip().lower()
    if not t:
        return None
    if step == WIC_STEP_CATEGORY:
        if t in {"yes", "y", "yeah", "yep"}:
            return "wic_elig_cat_yes"
        if t in {"no", "n", "nope", "nah"}:
            return "wic_elig_cat_no"
        if "not sure" in t or t == "unsure" or "maybe" in t or t == "?" or "不知道" in t:
            return "wic_elig_cat_unsure"
    if step == WIC_STEP_RESIDENCY:
        if t in {"yes", "y", "yeah", "yep"} or "massachusetts" in t or t == "ma":
            return "wic_elig_ma_yes"
        if t in {"no", "n", "nope"} or "not in ma" in t or "not massachusetts" in t:
            return "wic_elig_ma_no"
        if "not sure" in t or t == "unsure" or "maybe" in t or "不知道" in t:
            return "wic_elig_ma_unsure"
    if step == WIC_STEP_ADJUNCTIVE:
        if t in {"yes", "y", "yeah", "yep"}:
            return "wic_elig_adj_yes"
        if t in {"no", "n", "nope"}:
            return "wic_elig_adj_no"
        if "not sure" in t or t == "unsure" or "maybe" in t or "不知道" in t:
            return "wic_elig_adj_unsure"
    if step == WIC_STEP_INCOME:
        if "under" in t or "below" in t or "less" in t or t in {"yes", "y"}:
            return "wic_elig_inc_under"
        if "over" in t or "above" in t or "more than" in t:
            return "wic_elig_inc_over"
        if "not sure" in t or t == "unsure" or "maybe" in t or "不知道" in t:
            return "wic_elig_inc_unsure"
    return None


# ══════════════════════════════════════════════════════════════════════════════
# AGENT CLASS
# ══════════════════════════════════════════════════════════════════════════════


class NutritionAgent:
    _RESOURCE_LEGACY_HINTS = {
        "affordable_shopping": "User wants budget-friendly food shopping; answer using WIC foods and WIC-authorized stores only.",
        "check_eligibility": "User wants to know if they might qualify for WIC in Massachusetts and how WIC works.",
        "elig_wic": "User asked specifically about WIC eligibility.",
        "elig_snap": "User tapped a SNAP-related option; this lane is WIC-only—give a one-line boundary then WIC eligibility and WIC stores only (no SNAP rules or benefits).",
        "elig_not_sure": "User is unsure about programs; explain WIC only and who typically qualifies for WIC in Massachusetts.",
        "elig_i_qualify": "User thinks they may qualify; give WIC next steps (how to apply, WIC stores) only.",
    }

    _STATIC_PROMPT_BUTTONS = {
        "for_myself": "Give me practical healthy eating advice for an adult.",
        "for_child": "Give me practical healthy eating advice for a young child under 5.",
        "special_nutrition": (
            "Ask what special nutrition situation the user wants help with, "
            "such as pregnancy, allergies, diabetes, or dietary restrictions."
        ),
        "wic_apply": "How do I apply for WIC benefits in Massachusetts?",
    }

    def _format_profile_context(self, profile: dict) -> str:
        return _format_profile_for_prompt(profile, BLANK_PROFILE)

    def _get_profile(self, user_id: str) -> dict:
        """Parse user profile from memory file."""
        raw = _mem.load_all(user_id)
        profile = {}
        for line in raw.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                profile[k.strip()] = v.strip()
        return profile

    def _food_safety_answer_turn(
        self,
        user_text: str,
        user_id: str,
        profile_context: str,
    ) -> tuple[str, list[Button]]:
        """One food-safety turn: router picks RAG vs main prompt; then dynamic buttons."""
        session = _user_session(user_id)
        full_query = f"[USER PROFILE]\n{profile_context}\n[QUESTION]\n{user_text}"
        if _should_use_rag_food_safety(user_text, session):
            response = _rag.query_rag(
                full_query,
                session_id=session,
                user_id=user_id,
                memory_source_message=user_text,
            )
        else:
            response = _ai.ask(main_system_prompt, full_query, session)
        clean = re.sub(r"\[Source:[^\]]+\]", "", response).strip()
        buttons = _generate_buttons(
            clean,
            session + "_fsans_btn",
            fallback_buttons=FOOD_SAFETY_HUB_BUTTON_FALLBACK,
        )
        return clean, buttons

    def _resources_turn(
        self,
        user_text: str,
        user_id: str,
        lat: float | None = None,
        lng: float | None = None,
    ) -> tuple[str, list[Button] | str]:
        """Find Resources: tool selector, resource_tools.run_tool, then synthesize."""
        session = _user_session(user_id)
        profile_context = self._format_profile_context(self._get_profile(user_id))

        if _is_exact_store_fact_question(user_text) or _is_wic_item_coverage_question(user_text):
            full_query = f"[USER PROFILE]\n{profile_context}\n[QUESTION]\n{user_text}"
            response = _rag.query_rag(
                full_query,
                session_id=session,
                user_id=user_id,
                memory_source_message=user_text,
            )
            clean = re.sub(r"\[Source:[^\]]+\]", "", response).strip()
            buttons = _generate_buttons(
                clean,
                session,
                fallback_buttons=RESOURCES_FALLBACK_BUTTONS,
            )
            return clean, buttons

        if lat is None or lng is None:
            cached = _state.user_last_location.get(user_id)
            if cached:
                lat, lng = cached
        if lat is None or lng is None:
            coords = _lookup_benchmark_coordinates(user_text or "")
            if coords:
                lat, lng = coords
                _state.user_last_location[user_id] = coords

        kb_block = ""
        try:
            retrieval_q = (user_text or "").strip() or "Massachusetts WIC authorized vendors and WIC program"
            if _should_retrieve_public_kb(retrieval_q, session, "resources"):
                ctx, _, _ = _rag.get_context(retrieval_q, user_id=user_id)
                if ctx and str(ctx).strip():
                    kb_block = "\n\n[KNOWLEDGE BASE]\n" + str(ctx).strip()
        except Exception:
            pass

        um = (user_text or "").strip() or "(empty)"
        selector_query = f"[USER PROFILE]\n{profile_context}\n[USER MESSAGE]\n{um}{kb_block}"
        raw = _ai.ask(resources_tool_selector_prompt, selector_query, session + "_tool_sel")
        decision = _parse_resources_json(raw) or {"tool": "none", "params": {}, "reply": ""}

        tool = (decision.get("tool") or "none").strip().lower()
        params = decision.get("params") or {}
        reply = (decision.get("reply") or "").strip()
        tool, params, reply = _normalize_resources_decision_wic_only(tool, params, reply)

        # Deterministic override for nearby ethnic/specialty grocery requests.
        derived_keyword = _derive_store_keyword(user_text)
        if _is_proximity_store_search(user_text) and derived_keyword:
            tool = "search_general_stores"
            params = dict(params)
            params.setdefault("keyword", derived_keyword)
            if "max_results" not in params and re.search(r"\b(closest|nearest)\b", (user_text or "").lower()):
                params["max_results"] = 1

        if tool == "start_eligibility":
            tool_result = _run_resource_tool(tool, params)
            _state.resources_mode_users.discard(user_id)
            _state.eligibility_state.add(user_id)
            _state.wic_eligibility_users.add(user_id)
            _state.wic_eligibility_steps[user_id] = WIC_STEP_CATEGORY
            _state.wic_eligibility_answers[user_id] = {}
            final = ((reply + "\n\n" + tool_result) if reply else tool_result).strip()
            return final, _wic_category_buttons()

        if tool in ("search_wic_stores", "search_general_stores"):
            if lat is None or lng is None:
                _state.pending_store_search[user_id] = {
                    "tool": tool,
                    "params": params,
                    "user_text": user_text or "",
                }
                msg = reply or (
                    "Tap the button below to share your location — I'll find WIC-authorized stores near you."
                )
                return msg, "request_location"
            tool_result = _run_resource_tool(tool, params, lat=lat, lng=lng)

        elif tool == "explain_program":
            tool_result = _run_resource_tool(tool, params)

        else:
            if not reply:
                reply = "How can I help with WIC in Massachusetts today?"
            buttons = _generate_buttons(reply, session, fallback_buttons=RESOURCES_FALLBACK_BUTTONS)
            return reply, buttons

        ut = (user_text or "").strip()
        synth_query = f"[USER MESSAGE]\n{ut}\n\n[TOOL RESULT]\n{tool_result}"
        final = _ai.ask(resources_synthesizer_prompt, synth_query, session + "_synth")
        if not (final or "").strip():
            final = tool_result

        buttons = _generate_buttons(final.strip(), session, fallback_buttons=RESOURCES_FALLBACK_BUTTONS)
        return final.strip(), buttons

    def _remember_user_message(self, user_id: str, user_message: str) -> dict:
        """Save structured facts from raw user text and return the refreshed profile."""
        if not _should_extract_profile_from_message(user_message):
            return self._get_profile(user_id)
        return save_and_reload_profile(
            user_id,
            user_message,
            mem=_mem,
            normalize_text=_normalize_text,
        )

    def _build_profile_question(self, profile: dict, user_message: str, target: str, session: str) -> str:
        return build_profile_question(
            profile,
            user_message,
            target,
            session,
            ai=_ai,
            profile_nudge_prompt=profile_nudge_prompt,
            format_profile_for_prompt=_format_profile_for_prompt,
            normalize_text=_normalize_text,
            blank_profile=BLANK_PROFILE,
        )

    def _start_profile_conversation(self, user_id: str, profile: dict, session: str, seed_message: str) -> tuple[str, list[Button]]:
        target = _choose_profile_target(profile) or "asking_for"
        _state.nutrition_ob_state[user_id] = {"target": target}
        question = self._build_profile_question(profile, seed_message, target, session)
        return question, profile_buttons_for_target(target)

    def _maybe_start_profile_from_welcome(
        self,
        user_id: str,
        profile: dict,
        session: str,
        welcome_response: str,
        user_message: str,
    ) -> tuple[str, list[Button]] | None:
        return maybe_start_profile_from_welcome(
            user_id=user_id,
            profile=profile,
            session=session,
            welcome_response=welcome_response,
            user_message=user_message,
            choose_profile_target=_choose_profile_target,
            build_profile_question_fn=self._build_profile_question,
            nutrition_ob_state=_state.nutrition_ob_state,
        )

    def _answer_saved_profile_task(self, user_id: str, profile: dict, session: str, state: dict) -> tuple[str, list[Button]]:
        return answer_saved_profile_task(
            user_id=user_id,
            profile=profile,
            session=session,
            state=state,
            ai=_ai,
            main_system_prompt=main_system_prompt,
            format_profile_context=self._format_profile_context,
            generate_buttons=_generate_buttons,
            make_buttons=_make_buttons,
            welcome_buttons=WELCOME_BUTTONS,
        )

    def _continue_profile_conversation(self, user_id: str, user_message: str, session: str) -> tuple[str, list[Button]] | None:
        return continue_profile_conversation(
            user_id=user_id,
            user_message=user_message,
            session=session,
            remember_user_message=self._remember_user_message,
            save_profile_value=lambda uid, field, value: _mem.save(uid, f"{field}: {value}"),
            choose_profile_target=_choose_profile_target,
            looks_like_profile_answer=_looks_like_profile_answer,
            build_profile_question_fn=self._build_profile_question,
            answer_saved_profile_task_fn=self._answer_saved_profile_task,
            nutrition_ob_state=_state.nutrition_ob_state,
        )

    def _maybe_append_profile_nudge(
        self,
        user_id: str,
        response: str,
        profile: dict,
        user_message: str,
        session: str,
        intent: str,
    ) -> tuple[str, list[Button]]:
        return maybe_append_profile_nudge(
            user_id=user_id,
            response=response,
            profile=profile,
            user_message=user_message,
            session=session,
            intent=intent,
            choose_profile_target=_choose_profile_target,
            normalize_text=_normalize_text,
            build_profile_question_fn=self._build_profile_question,
            nutrition_ob_state=_state.nutrition_ob_state,
        )

    def _build_turn_context(self, user_message: str, user_id: str) -> TurnContext:
        profile = self._get_profile(user_id)
        return TurnContext(
            user_id=user_id,
            user_message=user_message,
            session=_user_session(user_id),
            profile=profile,
            profile_context=self._format_profile_context(profile),
        )

    def _reset_navigation_state(self, user_id: str) -> None:
        self._leave_food_safety_mode(user_id)
        self._clear_resources_mode(user_id)
        _state.eligibility_state.discard(user_id)
        _state.wic_eligibility_users.discard(user_id)
        _wic_clear_tracking(user_id)
        _state.nutrition_ob_state.pop(user_id, None)
        _state.pending_store_type.pop(user_id, None)
        _state.pending_store_search.pop(user_id, None)

    def _menu_response(
        self,
        text: str,
        user_id: str,
        *,
        menu_buttons: list[dict] | None = None,
    ) -> tuple[str, list[Button]]:
        """Reset lane state, then attach buttons.

        ``menu_buttons``:
          * ``None`` (default) — ``WELCOME_BUTTONS`` from ``prompts.py`` (Eating Better / Food Safety / Find Resources).
          * ``[]`` — no buttons (text-only).
          * Non-empty list — custom ``{id, title}`` rows (same shape as ``WELCOME_BUTTONS``; titles ≤20 chars).
        """
        self._reset_navigation_state(user_id)
        if menu_buttons is None:
            return text, _make_buttons(WELCOME_BUTTONS)
        if len(menu_buttons) == 0:
            return text, []
        return text, _make_buttons(menu_buttons)

    def _question_response(
        self,
        text: str,
        buttons: list[Button] | None = None,
    ) -> tuple[str, list[Button]]:
        return text, buttons or []

    def _task_response(
        self,
        text: str,
        buttons: list[Button] | str,
        session: str,
    ) -> tuple[str, list[Button] | str]:
        if buttons != "request_location":
            text = _append_button_intro(
                text,
                buttons if isinstance(buttons, list) else [],
                session,
            )
        return text, buttons

    def _profile_flow_response(
        self,
        text: str,
        buttons: list[Button],
        user_id: str,
        session: str,
    ) -> tuple[str, list[Button] | str]:
        if not buttons:
            return self._question_response(text)
        button_ids = [getattr(button, "id", "") for button in buttons]
        if button_ids == [item["id"] for item in WELCOME_BUTTONS]:
            return self._menu_response(text, user_id)
        return self._task_response(text, buttons, session)

    def _disclaimer_prompt(self) -> tuple[str, list[Button]]:
        return self._question_response(
            FIRST_USE_DISCLAIMER,
            _make_buttons(DISCLAIMER_BUTTONS),
        )

    def _has_disclaimer_consent(self, user_id: str) -> bool:
        return user_id in _state.accepted_disclaimer_users

    def _handle_disclaimer_gate_for_text(
        self,
        user_id: str,
    ) -> tuple[str, list[Button]] | None:
        _debug_log(
            f"text gate check user_id={user_id} accepted={self._has_disclaimer_consent(user_id)}"
        )
        if self._has_disclaimer_consent(user_id):
            return None
        _debug_log(f"text gate returning disclaimer for user_id={user_id}")
        return self._disclaimer_prompt()

    def _handle_disclaimer_gate_for_tool(
        self,
        interaction_id: str,
        user_id: str,
    ) -> tuple[str, list[Button] | str] | None:
        _debug_log(
            f"tool gate check user_id={user_id} interaction_id={interaction_id} accepted={self._has_disclaimer_consent(user_id)}"
        )
        if interaction_id == "disclaimer_agree":
            _debug_log(f"user_id={user_id} accepted disclaimer via button")
            _state.accepted_disclaimer_users.add(user_id)
            return self.run("", user_id)
        if interaction_id == "disclaimer_decline":
            _debug_log(f"user_id={user_id} declined disclaimer")
            return self._question_response(
                "You need to agree to the disclaimer before using Nura.",
                _make_buttons(DISCLAIMER_BUTTONS),
            )
        if self._has_disclaimer_consent(user_id):
            return None
        _debug_log(f"tool gate returning disclaimer for user_id={user_id}")
        return self._disclaimer_prompt()

    def _handle_disclaimer_gate_for_location(
        self,
        user_id: str,
    ) -> tuple[str, list[Button]] | None:
        _debug_log(
            f"location gate check user_id={user_id} accepted={self._has_disclaimer_consent(user_id)}"
        )
        if self._has_disclaimer_consent(user_id):
            return None
        _debug_log(f"location gate returning disclaimer for user_id={user_id}")
        return self._disclaimer_prompt()

    def _clear_resources_mode(self, user_id: str) -> None:
        _state.resources_mode_users.discard(user_id)
        _state.resources_conversation_summary.pop(user_id, None)

    def _enter_resources_mode(self, user_id: str) -> None:
        _state.resources_mode_users.add(user_id)

    def _leave_food_safety_mode(self, user_id: str) -> None:
        _state.food_safety_flow_users.discard(user_id)

    def _enter_food_safety_mode(self, user_id: str) -> None:
        _state.food_safety_flow_users.add(user_id)

    def _start_eligibility_flow(self, user_id: str) -> None:
        self._clear_resources_mode(user_id)
        _state.eligibility_state.add(user_id)

    def _request_location_response(self, user_id: str, store_type: str) -> tuple[str, str]:
        _state.pending_store_type[user_id] = store_type
        if store_type == "find_wic_stores":
            return (
                "Tap the button below to share your location and I'll find the nearest WIC stores for you. 📍",
                "request_location",
            )
        return (
            "Tap the button below to share your location and I'll find nearby stores for you. 📍",
            "request_location",
        )

    def _wic_eligibility_render_step(self, user_id: str) -> tuple[str, list[Button]]:
        step = _state.wic_eligibility_steps.get(user_id) or WIC_STEP_CATEGORY
        answers = _state.wic_eligibility_answers.get(user_id) or {}
        if step == WIC_STEP_CATEGORY:
            return _wic_eligibility_opening(), _wic_category_buttons()
        if step == WIC_STEP_RESIDENCY:
            return "Next: are you a Massachusetts resident?", _wic_residency_buttons()
        if step == WIC_STEP_ADJUNCTIVE:
            return (
                "Does anyone in your household who needs WIC receive SNAP, MassHealth (Medicaid), TAFDC, "
                "or certain other cash assistance programs that Massachusetts can count for WIC?",
                _wic_adjunctive_buttons(),
            )
        if step == WIC_STEP_HOUSEHOLD_INPUT:
            return (
                "Rough income limits matter when nobody is on those programs. "
                "How many people are in your household—everyone who lives together and shares income? "
                "Reply with one whole number only (for example 1, 3, or 6).",
                [],
            )
        if step == WIC_STEP_INCOME:
            hh_size = int(answers.get("hh_size") or 1)
            monthly = _wic_monthly_income_limit(hh_size)
            return _wic_income_question_text(hh_size), _wic_income_buttons_monthly(monthly)
        return _wic_eligibility_opening(), _wic_category_buttons()

    def _wic_eligibility_goto_income(self, user_id: str, hh_size: int) -> tuple[str, list[Button]]:
        answers = _state.wic_eligibility_answers.setdefault(user_id, {})
        n = min(max(int(hh_size), 1), 60)
        answers["hh_size"] = n
        _state.wic_eligibility_steps[user_id] = WIC_STEP_INCOME
        m = _wic_monthly_income_limit(n)
        return self._question_response(_wic_income_question_text(n), _wic_income_buttons_monthly(m))

    def _wic_eligibility_apply_interaction(
        self, user_id: str, interaction_id: str
    ) -> tuple[str, list[Button] | str]:
        step = _state.wic_eligibility_steps.get(user_id) or WIC_STEP_CATEGORY
        if step == WIC_STEP_HOUSEHOLD_INPUT:
            return self._question_response(
                "Please type one whole number for how many people live in your home (example: 3 or 6).",
                [],
            )

        prefix = _wic_step_expected_prefix(step)
        if prefix is not None and not interaction_id.startswith(prefix):
            hint = "Please use one of the buttons below to continue (or reply yes / no / not sure)."
            base, btns = self._wic_eligibility_render_step(user_id)
            if hint not in base:
                base = f"{hint}\n\n{base}"
            return self._question_response(base, btns)

        answers = _state.wic_eligibility_answers.setdefault(user_id, {})

        if step == WIC_STEP_CATEGORY:
            if interaction_id == "wic_elig_cat_no":
                _state.eligibility_state.discard(user_id)
                _state.wic_eligibility_users.discard(user_id)
                _wic_clear_tracking(user_id)
                return self._menu_response(
                    "WIC in Massachusetts mainly supports pregnancy, recent pregnancy or postpartum (about six weeks), "
                    "breastfeeding through baby's first birthday, and children under 5. From what you shared, you may "
                    "not qualify under the usual WIC categories—WIC staff can say for sure. I recommend contacting a "
                    "local WIC clinic if you have any doubt. You can still use this chat for WIC stores and general WIC "
                    "information.",
                    user_id,
                    menu_buttons=[],
                )
            answers["category"] = "yes" if interaction_id == "wic_elig_cat_yes" else "unsure"
            _state.wic_eligibility_steps[user_id] = WIC_STEP_RESIDENCY
            return self._question_response(
                "Next: are you a Massachusetts resident?",
                _wic_residency_buttons(),
            )

        if step == WIC_STEP_RESIDENCY:
            if interaction_id == "wic_elig_ma_no":
                _state.eligibility_state.discard(user_id)
                _state.wic_eligibility_users.discard(user_id)
                _wic_clear_tracking(user_id)
                return self._menu_response(
                    "Massachusetts WIC is for people who live in Massachusetts. If you're elsewhere, I recommend "
                    "searching for WIC in your state for how to apply. This chat can't determine eligibility outside MA; "
                    "your local WIC office is the right next step.",
                    user_id,
                    menu_buttons=[],
                )
            answers["ma"] = "yes" if interaction_id == "wic_elig_ma_yes" else "unsure"
            _state.wic_eligibility_steps[user_id] = WIC_STEP_ADJUNCTIVE
            return self._question_response(
                "Does anyone in your household who needs WIC receive SNAP, MassHealth (Medicaid), TAFDC, "
                "or certain other cash assistance programs that Massachusetts can count for WIC?",
                _wic_adjunctive_buttons(),
            )

        if step == WIC_STEP_ADJUNCTIVE:
            if interaction_id == "wic_elig_adj_yes":
                _state.eligibility_state.discard(user_id)
                _state.wic_eligibility_users.discard(user_id)
                _wic_clear_tracking(user_id)
                return self._menu_response(
                    "Many people qualify for WIC when a household member is enrolled in programs like SNAP, "
                    "MassHealth (Medicaid), or TAFDC that Massachusetts counts for WIC (adjunctive eligibility). "
                    "From what you shared, you may qualify—this is not a final eligibility decision. I strongly "
                    "recommend you apply through a local WIC clinic or mass.gov/wic for the next steps.",
                    user_id,
                    menu_buttons=WIC_POST_SCREENING_BUTTONS,
                )
            answers["adjunctive"] = "no" if interaction_id == "wic_elig_adj_no" else "unsure"
            _state.wic_eligibility_steps[user_id] = WIC_STEP_HOUSEHOLD_INPUT
            return self._question_response(
                "Rough income limits matter when nobody is on those programs. "
                "How many people are in your household—everyone who lives together and shares income? "
                "Reply with one whole number only (for example 1, 3, or 6).",
                [],
            )

        if step == WIC_STEP_INCOME:
            _state.eligibility_state.discard(user_id)
            _state.wic_eligibility_users.discard(user_id)
            _wic_clear_tracking(user_id)
            if interaction_id == "wic_elig_inc_under":
                return self._menu_response(
                    "Based on the household size and income you indicated—and WIC's general rules—you may meet typical "
                    "WIC income guidelines in Massachusetts. This is not a final eligibility decision. I recommend you "
                    "apply and talk with WIC staff about your situation; they can help with paperwork and what's next.",
                    user_id,
                    menu_buttons=WIC_POST_SCREENING_BUTTONS,
                )
            if interaction_id == "wic_elig_inc_over":
                return self._menu_response(
                    "From what you shared, your income may be over the rough WIC cap we used for your household size. "
                    "Families can still qualify in some situations, and limits change. I recommend you ask a WIC "
                    "clinic or mass.gov/wic; they do the official eligibility determination.",
                    user_id,
                    menu_buttons=WIC_POST_SCREENING_BUTTONS,
                )
            return self._menu_response(
                "When income is hard to pin down, WIC staff are the right people to help you apply and compare your "
                "situation to current rules. I recommend contacting a local WIC clinic or mass.gov/wic for the next "
                "steps—you may still qualify depending on details.",
                user_id,
                menu_buttons=WIC_POST_SCREENING_BUTTONS,
            )

        return self._question_response(*self._wic_eligibility_render_step(user_id))

    def _handle_eligibility_turn(
        self,
        ctx: TurnContext,
    ) -> tuple[str, list[Button] | str] | None:
        if ctx.user_id not in _state.eligibility_state:
            return None

        if ctx.user_id in _state.wic_eligibility_users:
            step = _state.wic_eligibility_steps.get(ctx.user_id) or WIC_STEP_CATEGORY

            if step == WIC_STEP_HOUSEHOLD_INPUT:
                parsed = _wic_parse_household_size_text(ctx.user_message)
                if parsed is not None and parsed >= 1:
                    return self._wic_eligibility_goto_income(ctx.user_id, parsed)
                if ctx.user_message.strip():
                    return self._question_response(
                        "That doesn't look like one whole number (1–60). Please send a single number only—for "
                        "example 4 or 6.",
                        [],
                    )
                return self._question_response(*self._wic_eligibility_render_step(ctx.user_id))

            mapped = _wic_eligibility_map_text_to_id(step, ctx.user_message)
            if mapped:
                return self._wic_eligibility_apply_interaction(ctx.user_id, mapped)
            base, btns = self._wic_eligibility_render_step(ctx.user_id)
            hint = (
                "Tap a button below to answer, or type a number if this step asked for one."
                if not btns
                else "Tap a button below to answer, or reply with yes, no, or not sure."
            )
            if ctx.user_message.strip():
                return self._question_response(f"{hint}\n\n{base}", btns)
            return self._question_response(base, btns)

        prompt = eligibility_check_prompt
        response = _ai.ask(prompt, ctx.user_message, ctx.session)
        recommendation_keywords = [
            "qualify",
            "eligible",
            "recommend",
            "apply",
            "snap",
            "wic",
            "senior nutrition",
        ]
        if any(kw in response.lower() for kw in recommendation_keywords):
            _state.eligibility_state.discard(ctx.user_id)
            _state.wic_eligibility_users.discard(ctx.user_id)
            _wic_clear_tracking(ctx.user_id)
            return self._menu_response(response, ctx.user_id)
        buttons: list[Button] = []
        return self._question_response(response, buttons)

    def _handle_greeting_turn(
        self,
        ctx: TurnContext,
    ) -> tuple[str, list[Button] | str] | None:
        if not _is_greeting(ctx.user_message):
            return None

        self._reset_navigation_state(ctx.user_id)
        user_line = ctx.user_message.strip() or "The user just opened the chat."
        response = WELCOME_FALLBACK_MESSAGE

        welcome_with_profile = self._maybe_start_profile_from_welcome(
            user_id=ctx.user_id,
            profile=ctx.profile,
            session=ctx.session,
            welcome_response=response,
            user_message=user_line,
        )
        if welcome_with_profile is not None:
            return self._question_response(*welcome_with_profile)
        return self._menu_response(response, ctx.user_id)

    def _handle_resources_mode_turn(
        self,
        ctx: TurnContext,
    ) -> tuple[str, list[Button] | str] | None:
        if ctx.user_id not in _state.resources_mode_users:
            return None
        self._remember_user_message(ctx.user_id, ctx.user_message)
        text, btns = self._resources_turn(ctx.user_message, ctx.user_id)
        return self._task_response(text, btns, ctx.session)

    def _handle_food_safety_mode_turn(
        self,
        ctx: TurnContext,
    ) -> tuple[str, list[Button] | str] | None:
        if ctx.user_id not in _state.food_safety_flow_users:
            return None
        response, buttons = self._food_safety_answer_turn(
            ctx.user_message,
            ctx.user_id,
            ctx.profile_context,
        )
        return self._task_response(response, buttons, ctx.session)

    def _handle_new_text_intent(
        self,
        ctx: TurnContext,
    ) -> tuple[str, list[Button] | str]:
        intent = _classify_intent(ctx.user_message, ctx.session)
        _debug_log(
            f"new_text_intent user_id={ctx.user_id} intent={intent} message={ctx.user_message!r}"
        )

        if intent == "find resources":
            _state.resources_mode_users.add(ctx.user_id)
            _state.food_safety_flow_users.discard(ctx.user_id)
            self._remember_user_message(ctx.user_id, ctx.user_message)
            text, btns = self._resources_turn(ctx.user_message, ctx.user_id)
            return self._task_response(text, btns, ctx.session)

        if intent == "food_safety":
            full_query = (
                f"[USER PROFILE]\n{ctx.profile_context}\n[QUESTION]\n{ctx.user_message}"
            )
            if _should_use_rag_food_safety(ctx.user_message, ctx.session):
                response = _rag.query_rag(
                    full_query,
                    session_id=ctx.session,
                    user_id=ctx.user_id,
                    memory_source_message=ctx.user_message,
                )
            else:
                response = _ai.ask(main_system_prompt, full_query, ctx.session)
            remembered_profile = self._get_profile(ctx.user_id)
        else:
            remembered_profile = self._remember_user_message(
                ctx.user_id,
                ctx.user_message,
            )
            profile_context = self._format_profile_context(remembered_profile)
            full_query = (
                f"[USER PROFILE]\n{profile_context}\n[QUESTION]\n{ctx.user_message}"
            )
            if intent == "nutrition_advice":
                if _should_retrieve_public_kb(ctx.user_message, ctx.session, "nutrition"):
                    response = _rag.query_rag(
                        full_query,
                        session_id=ctx.session,
                        user_id=ctx.user_id,
                        memory_source_message=ctx.user_message,
                    )
                else:
                    response = _ai.ask(main_system_prompt, full_query, ctx.session)
            else:
                response = _ai.ask(main_system_prompt, full_query, ctx.session)

        response, nudge_buttons = self._maybe_append_profile_nudge(
            user_id=ctx.user_id,
            response=response,
            profile=remembered_profile,
            user_message=ctx.user_message,
            session=ctx.session,
            intent=intent,
        )
        buttons = nudge_buttons or _generate_buttons(response, ctx.session)
        if nudge_buttons:
            return self._question_response(response, nudge_buttons)
        return self._task_response(response, buttons, ctx.session)

    def _handle_food_safety_button(
        self,
        user_id: str,
        session: str,
        profile_context: str,
    ) -> tuple[str, list[Button] | str]:
        self._enter_food_safety_mode(user_id)
        self._clear_resources_mode(user_id)
        hub_query = (
            f"[USER PROFILE]\n{profile_context}\n[CONTEXT]\n"
            "The user opened Food Safety from the main menu."
        )
        try:
            response = _ai.ask(
                food_safety_hub_prompt, hub_query, session + "_fshub"
            ).strip()
            if len(response) < 30:
                response = FOOD_SAFETY_HUB_FALLBACK_MESSAGE
        except Exception:
            response = FOOD_SAFETY_HUB_FALLBACK_MESSAGE
        buttons = _generate_buttons(
            response,
            session + "_fshub_btn",
            fallback_buttons=FOOD_SAFETY_HUB_BUTTON_FALLBACK,
        )
        return self._task_response(response, buttons, session)

    def _handle_nutrition_button(
        self,
        user_id: str,
        session: str,
        profile: dict,
        profile_context: str,
    ) -> tuple[str, list[Button] | str]:
        target = _choose_profile_target(profile)
        if target:
            _state.nutrition_ob_state[user_id] = {
                "target": target,
                "pending_question": "Give me practical, personalized healthy eating advice based on this user's profile.",
            }
            question = self._build_profile_question(
                profile,
                "The user tapped the nutrition option and is ready to talk about eating habits and health goals.",
                target,
                session,
            )
            return self._question_response(question, profile_buttons_for_target(target))
        nutrition_open = (
            f"[USER PROFILE]\n{profile_context}\n[QUESTION]\n"
            "Give me practical, personalized healthy eating advice based on this user's profile."
        )
        nut_router_msg = (
            "User tapped Eating Better; give practical, personalized healthy eating advice based on profile."
        )
        try:
            if _should_retrieve_public_kb(nut_router_msg, session, "nutrition"):
                response = _rag.query_rag(
                    nutrition_open,
                    session_id=session,
                    user_id=user_id,
                    memory_source_message=(
                        "The user tapped Eating Better from the main menu; "
                        "give practical, personalized healthy eating advice based on their profile."
                    ),
                )
            else:
                response = _ai.ask(main_system_prompt, nutrition_open, session)
        except Exception:
            response = (
                "I’m here to help with eating better. What would you like to work on first—meals, snacks, or something else?"
            )
        buttons = _generate_buttons(
            response, session, fallback_buttons=NUTRITION_FALLBACK_BUTTONS
        )
        return self._task_response(response, buttons, session)

    def _handle_resources_prompt(
        self,
        prompt_text: str,
        user_id: str,
        session: str,
        activate_mode: bool = False,
    ) -> tuple[str, list[Button] | str]:
        if activate_mode:
            self._enter_resources_mode(user_id)
        text, btns = self._resources_turn(prompt_text, user_id)
        return self._task_response(text, btns, session)

    def _handle_unknown_tool_follow_up(
        self,
        follow_up: str,
        user_id: str,
        session: str,
        profile_context: str,
    ) -> tuple[str, list[Button] | str]:
        if user_id in _state.resources_mode_users:
            text, btns = self._resources_turn(follow_up, user_id)
            return self._task_response(text, btns, session)
        if user_id in _state.food_safety_flow_users:
            response, buttons = self._food_safety_answer_turn(
                follow_up, user_id, profile_context
            )
            return self._task_response(response, buttons, session)
        return self.run(follow_up, user_id)

    def run(self, user_message: str, user_id: str) -> tuple[str, list[Button] | str]:
        """Handle a free-text message from the user. Injects user profile into context."""
        _debug_log(f"run start user_id={user_id} message={user_message!r}")
        disclaimer_gate = self._handle_disclaimer_gate_for_text(user_id)
        if disclaimer_gate is not None:
            _debug_log(f"run returning disclaimer gate for user_id={user_id}")
            return disclaimer_gate
        ctx = self._build_turn_context(user_message, user_id)

        if ctx.user_message.strip().lower() in {"menu", "start", "help"}:
            return self._menu_response(WELCOME_FALLBACK_MESSAGE, user_id)

        eligibility_reply = self._handle_eligibility_turn(ctx)
        if eligibility_reply is not None:
            return eligibility_reply

        profile_reply = self._continue_profile_conversation(
            user_id,
            user_message,
            ctx.session,
        )
        if profile_reply is not None:
            return self._profile_flow_response(
                profile_reply[0],
                profile_reply[1],
                user_id,
                ctx.session,
            )

        greeting_reply = self._handle_greeting_turn(ctx)
        if greeting_reply is not None:
            return greeting_reply

        resources_reply = self._handle_resources_mode_turn(ctx)
        if resources_reply is not None:
            return resources_reply

        food_safety_reply = self._handle_food_safety_mode_turn(ctx)
        if food_safety_reply is not None:
            return food_safety_reply

        return self._handle_new_text_intent(ctx)

    def run_image(
        self,
        image_path: str,
        user_id: str,
        caption: str | None = None,
        mime_type: str | None = None,
    ) -> tuple[str, list[Button] | str]:
        """Handle an image by uploading it into the user's session and prompting with context."""
        _debug_log(f"run_image start user_id={user_id} image_path={image_path!r}")
        disclaimer_gate = self._handle_disclaimer_gate_for_text(user_id)
        if disclaimer_gate is not None:
            return disclaimer_gate

        session = _user_session(user_id)
        profile = self._get_profile(user_id)
        profile_context = self._format_profile_context(profile)
        user_caption = (caption or "").strip() or "(no caption provided)"
        try:
            resolved_mime = mime_type or mimetypes.guess_type(image_path)[0]
            if not resolved_mime:
                return (
                    "I couldn't read that image type. Please send a JPG, PNG, or HEIC image.",
                    _make_buttons(WELCOME_BUTTONS),
                )
            if not resolved_mime.startswith("image/"):
                return (
                    "I can only analyze images right now. Please send a JPG, PNG, or HEIC image.",
                    _make_buttons(WELCOME_BUTTONS),
                )

            upload_result = _ai.client.upload_media(
                file_path=image_path,
                session_id=session,
                content_type=resolved_mime,
            )
            _debug_log(f"run_image upload_result user_id={user_id} result={upload_result!r}")
            upload_error = upload_result.get("error")
            if upload_error or not upload_result.get("ok"):
                upload_payload = upload_result.get("upload") if isinstance(upload_result, dict) else None
                upload_status = None
                upload_init = None
                if isinstance(upload_payload, dict):
                    upload_status = upload_payload.get("status_code")
                    upload_init = upload_payload.get("upload_init")
                _debug_log(
                    "run_image upload failure summary "
                    f"user_id={user_id} "
                    f"session={session} "
                    f"image_path={image_path!r} "
                    f"mime={resolved_mime!r} "
                    f"error={upload_error!r} "
                    f"status_code={upload_result.get('status_code')!r} "
                    f"upload_status_code={upload_status!r} "
                    f"upload_init={upload_init!r}"
                )
                _debug_log(
                    f"run_image upload error user_id={user_id} "
                    f"error={upload_error or upload_result}"
                )
                rate_limited = False
                if upload_result.get("status_code") == 429 or upload_status == 429:
                    rate_limited = True
                if isinstance(upload_init, dict):
                    init_status = upload_init.get("status_code")
                    init_error = str(upload_init.get("error", "")).lower()
                    if init_status == 429 or "429" in init_error or "limit exceeded" in init_error:
                        rate_limited = True
                if rate_limited:
                    return (
                        "Image service is busy right now. Please wait a few seconds and send the image again.",
                        _make_buttons(WELCOME_BUTTONS),
                    )
                return (
                    "I couldn't read that image right now. Please try sending it again.",
                    _make_buttons(WELCOME_BUTTONS),
                )

            query = (
                f"[USER PROFILE]\n{profile_context}\n"
                f"[USER CAPTION]\n{user_caption}"
            )
            media_refs = [{"id": upload_result["id"], "type": upload_result["type"]}]
            image_model = os.getenv("IMAGE_MODEL", "").strip() or None
            image_lastk_raw = os.getenv("IMAGE_LASTK", "").strip()
            image_lastk = int(image_lastk_raw) if image_lastk_raw.isdigit() else None
            response = _ai.ask(
                image_analysis_prompt,
                query,
                session,
                media=media_refs,
                model_override=image_model,
                lastk_override=image_lastk,
            )
            _debug_log(f"run_image llm_response user_id={user_id} response={response!r}")
            clean = re.sub(r"\[Source:[^\]]+\]", "", response).strip()
            buttons = _generate_buttons(clean, session, fallback_buttons=WELCOME_BUTTONS)
            return self._task_response(clean, buttons, session)
        except Exception as exc:
            _debug_log(f"run_image exception user_id={user_id} error={exc!r}")
            return (
                "I couldn't analyze that image right now. Please try again in a moment.",
                _make_buttons(WELCOME_BUTTONS),
            )

    def run_tool(
        self,
        interaction_id: str,
        user_id: str,
        interaction_title: str | None = None,
    ) -> tuple[str, list[Button] | str]:
        """Handle a button click (InteractiveEvent)."""
        _debug_log(
            f"run_tool start user_id={user_id} interaction_id={interaction_id} title={interaction_title!r}"
        )
        disclaimer_gate = self._handle_disclaimer_gate_for_tool(interaction_id, user_id)
        if disclaimer_gate is not None:
            _debug_log(f"run_tool returning disclaimer gate for user_id={user_id}")
            return disclaimer_gate
        session = _user_session(user_id)
        profile = self._get_profile(user_id)
        profile_context = self._format_profile_context(profile)
        profile_button_text = profile_button_value(interaction_id, interaction_title)
        if profile_button_text and user_id in _state.nutrition_ob_state:
            profile_reply = self._continue_profile_conversation(
                user_id,
                profile_button_text,
                session,
            )
            if profile_reply is not None:
                return self._profile_flow_response(
                    profile_reply[0],
                    profile_reply[1],
                    user_id,
                    session,
                )

        if interaction_id.startswith("wic_elig_"):
            if user_id not in _state.wic_eligibility_users:
                return self._menu_response(
                    "That WIC screening button is no longer active. You can keep exploring WIC below.",
                    user_id,
                    menu_buttons=WIC_POST_SCREENING_BUTTONS,
                )
            return self._wic_eligibility_apply_interaction(user_id, interaction_id)

        if interaction_id in ("nutrition", "find_stores"):
            self._leave_food_safety_mode(user_id)
        if interaction_id == "nutrition":
            self._clear_resources_mode(user_id)

        if interaction_id == "food_safety":
            return self._handle_food_safety_button(user_id, session, profile_context)

        elif interaction_id == "nutrition":
            return self._handle_nutrition_button(
                user_id,
                session,
                profile,
                profile_context,
            )

        elif interaction_id == "find_stores":
            return self._handle_resources_prompt(
                "The user opened Find Resources from the main menu.",
                user_id,
                session,
                activate_mode=True,
            )

        elif interaction_id.startswith("resources_dyn_"):
            label = (interaction_title or "").strip() or interaction_id
            return self._handle_resources_prompt(label, user_id, session)

        elif interaction_id == "wic_info":
            return self._handle_resources_prompt(
                "The user tapped WIC Help and wants to know about WIC in Massachusetts.",
                user_id,
                session,
                activate_mode=True,
            )

        elif interaction_id in ("elig_still_unsure", "elig_answers"):
            self._start_eligibility_flow(user_id)
            response = _ai.ask(
                eligibility_check_prompt,
                "Start the eligibility check now. Ask one question at a time.",
                session,
            )
            return self._question_response(response)

        elif interaction_id in (
            "affordable_shopping",
            "check_eligibility",
            "elig_wic",
            "elig_snap",
            "elig_not_sure",
            "elig_i_qualify",
        ):
            legacy_hint = self._RESOURCE_LEGACY_HINTS.get(
                interaction_id,
                interaction_title or interaction_id,
            )
            return self._handle_resources_prompt(
                legacy_hint,
                user_id,
                session,
                activate_mode=True,
            )

        elif interaction_id == "find_wic_stores":
            _state.pending_store_search[user_id] = {
                "tool": "search_wic_stores",
                "params": {},
                "user_text": interaction_title or "find WIC stores near me",
            }
            return (
                "Tap the button below to share your location — I'll find WIC-authorized stores near you.",
                "request_location",
            )

        elif interaction_id == "find_all_stores":
            _state.pending_store_search[user_id] = {
                "tool": "search_general_stores",
                "params": {},
                "user_text": interaction_title or "find WIC-authorized stores near me",
            }
            return (
                "Tap the button below to share your location — I'll find WIC-authorized stores near you.",
                "request_location",
            )

        elif interaction_id in self._STATIC_PROMPT_BUTTONS:
            query = self._STATIC_PROMPT_BUTTONS[interaction_id]
            response = _ai.ask(main_system_prompt, query, session)
            clean = re.sub(r"\[Source:[^\]]+\]", "", response).strip()
            buttons = _generate_buttons(clean, session)
            response = clean

        else:
            follow_up = interaction_title or interaction_id
            return self._handle_unknown_tool_follow_up(
                follow_up,
                user_id,
                session,
                profile_context,
            )

        return self._task_response(response, buttons, session)

    def run_location(
        self,
        lat: float,
        lng: float,
        user_id: str,
    ) -> tuple[str, list[Button] | str]:
        """Handle location: pending tool context, legacy store-type buttons, or default WIC search."""
        _debug_log(f"run_location start user_id={user_id} lat={lat} lng={lng}")
        disclaimer_gate = self._handle_disclaimer_gate_for_location(user_id)
        if disclaimer_gate is not None:
            _debug_log(f"run_location returning disclaimer gate for user_id={user_id}")
            return disclaimer_gate
        session = _user_session(user_id)
        _state.user_last_location[user_id] = (lat, lng)

        ctx = _state.pending_store_search.pop(user_id, None)
        if ctx and isinstance(ctx, dict) and "tool" in ctx:
            # User already chose a store search; do not re-run the LLM tool selector (it can
            # return none/eligibility and skip the store list).
            _state.resources_mode_users.add(user_id)
            tool = (ctx.get("tool") or "search_wic_stores").strip().lower()
            params = dict(ctx.get("params") or {})
            user_text = (ctx.get("user_text") or "").strip() or "find WIC stores near me"
            if tool not in ("search_wic_stores", "search_general_stores"):
                tool, params = "search_wic_stores", {}
            try:
                tool_result = _run_resource_tool(tool, params, lat=lat, lng=lng)
            except Exception as e:
                err = f"Sorry, I couldn't find stores right now. Please try again. ({e})"
                return self._menu_response(err, user_id)
            synth_query = f"[USER MESSAGE]\n{user_text}\n\n[TOOL RESULT]\n{tool_result}"
            final = _ai.ask(
                resources_synthesizer_prompt, synth_query, session + "_synth_pending"
            ).strip()
            if not final:
                final = tool_result
            buttons = _generate_buttons(
                final.strip(), session, fallback_buttons=RESOURCES_FALLBACK_BUTTONS
            )
            return self._task_response(final, buttons, session)

        store_type = _state.pending_store_type.pop(user_id, None)
        _state.resources_conversation_summary.pop(user_id, None)

        if store_type in ("find_wic_stores", "find_all_stores"):
            tool = "search_general_stores" if store_type == "find_all_stores" else "search_wic_stores"
            user_text = (
                "find WIC-authorized stores near me"
                if store_type == "find_all_stores"
                else "find WIC stores near me"
            )
            try:
                tool_result = _run_resource_tool(tool, {}, lat=lat, lng=lng)
            except Exception as e:
                err = f"Sorry, I couldn't find stores right now. Please try again. ({e})"
                return self._menu_response(err, user_id)
            synth_query = (
                f"[USER MESSAGE]\n{user_text}\n\n[TOOL RESULT]\n{tool_result}"
            )
            final = _ai.ask(
                resources_synthesizer_prompt, synth_query, session + "_synth_loc"
            ).strip()
            if not final:
                final = tool_result
            buttons = _generate_buttons(
                final, session, fallback_buttons=RESOURCES_FALLBACK_BUTTONS
            )
            return self._task_response(final, buttons, session)

        try:
            result = _run_resource_tool("search_wic_stores", {}, lat=lat, lng=lng)
        except Exception as e:
            result = f"Sorry, I couldn't find stores right now. Please try again. ({e})"
        return self._menu_response(result, user_id)
