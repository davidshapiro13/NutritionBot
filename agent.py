"""
NutritionBot Agent
===================
Central router that classifies user intent and calls the right tool.

Intents:
    food_safety      → hub + optional RAG router on follow-up; typed questions use query_rag
    nutrition_advice → LLM router: optional query_rag else main_system _ai.ask
    find resources   → LLM-led resources_mode + JSON; optional KB snippets via router + get_context
    find_stores      → enters resources_mode (same)
    find_wic_stores  → request_location → location_service (WIC CSV)
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
from AI import AI
from rag_pipeline import RAGPipeline
from prompts import (
    main_system_prompt,
    button_creator_prompt,
    button_intro_prompt,
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
    resources_lead_system_prompt,
    resources_lead_json_repair_prompt,
)
from web_search import WebSearch
from wa_service_sdk import Button
from user_memory import UserMemory

import ast
import json
import re


# ── Shared instances (loaded once at startup) ─────────────────────────────────
_ai  = AI()
_rag = RAGPipeline()
_rag.build_public_index()
_mem = UserMemory(embed_model=None)
BLANK_PROFILE = "(no profile info)"


# Tracks whether a user last requested WIC-only or all stores before sharing location
_pending_store_type: dict[str, str] = {}

# Nutrition inline onboarding state: {user_id: {"step": ..., "data": {...}}}
# Used for conversational, one-question-at-a-time profile building.
_nutrition_ob_state: dict[str, dict] = {}

# Users currently in eligibility check conversation
_eligibility_state: set[str] = set()

# After Food Safety hub: follow-ups use RAG router until user hits main nav.
_food_safety_flow_users: set[str] = set()

# Find Resources: LLM-led turns until greeting / main nav / nutrition / food_safety.
_resources_mode_users: set[str] = set()
_resources_conversation_summary: dict[str, str] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_buttons(buttons_data: list[dict]) -> list[Button]:
    """Convert list of {id, title} dicts to SDK Button objects."""
    return [Button(id=b["id"], title=b["title"]) for b in buttons_data]


def _generate_buttons(
    response: str,
    session_id: str,
    fallback_buttons: list[dict] | None = None,
) -> list[Button]:
    """Ask LLM to generate contextual follow-up buttons based on a response."""
    import json as _json
    fb = fallback_buttons if fallback_buttons is not None else WELCOME_BUTTONS
    try:
        raw = _ai.ask(button_creator_prompt, response, session_id + "_btn")
        # Strip markdown code fences
        raw = raw.strip().strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()
        # Extract the list portion in case LLM added prose
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            raw = m.group(0)
        # Try ast first, fall back to json.loads
        try:
            items = ast.literal_eval(raw)
        except Exception:
            items = _json.loads(raw)
        buttons = []
        for item in items:
            if isinstance(item, str):
                try:
                    info = ast.literal_eval(item)
                except Exception:
                    info = _json.loads(item)
            else:
                info = item
            title = str(info.get("title", ""))
            bid   = str(info.get("id", "btn"))
            if not title or len(title) > 20:
                continue
            buttons.append(Button(id=bid, title=title))
        return buttons[:3] if buttons else _make_buttons(fb)
    except Exception:
        print("Error")
        return _make_buttons(fb)


def _should_use_rag_food_safety(user_text: str, session_id: str) -> bool:
    """LLM routes whether this food-safety turn should use RAG (default yes if unclear)."""
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


def _wants_wic_store_by_location(text: str) -> bool:
    """True if the user is asking for WIC-accepting stores near them (needs share-location flow)."""
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


def _is_synthetic_resources_hub_opener(text: str) -> bool:
    """True when this turn is the scripted open from the Find Resources button (not user-typed)."""
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    return "opened find resources from the main menu" in t


def _classify_intent(user_message: str, session_id: str) -> str:
    """Classify user message into one of the four intents."""
    if _wants_wic_store_by_location(user_message):
        return "find resources"
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


def _user_session(user_id: str) -> str:
    return f"NutritionBot_User_{user_id}"


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


def _is_greeting(text: str) -> bool:
    """Return True for simple greetings that should show the welcome menu."""
    normalized = text.strip().lower()
    greetings = {
        "", "hi", "hello", "hey", "hiya",
        "good morning", "good afternoon", "good evening",
        "start", "menu", "help",
    }
    if normalized in greetings:
        return True
    # Mode-switch messages from WhatsApp platform
    if normalized.startswith("@") or "switch to" in normalized or "nutritionbot" in normalized:
        return True
    return False


def _append_button_intro(response: str, buttons: list[Button], session_id: str) -> str:
    """Append a short LLM-generated sentence introducing the follow-up buttons."""
    if not buttons:
        return response
    main, sources_block = response, ""
    if "\n\nSources:" in response:
        main, _, rest = response.rpartition("\n\nSources:")
        sources_block = "\n\nSources:" + rest
    try:
        button_titles = ", ".join(b.title for b in buttons)
        prompt = button_intro_prompt.format(
            response=main[:400],
            button_titles=button_titles,
        )
        intro = _ai.ask(prompt, "Write the sentence now.", session_id + "_intro").strip()
        if intro:
            return f"{main.rstrip()}\n\n{intro}{sources_block}"
    except Exception:
        pass
    return response


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _has_profile_value(profile: dict, *fields: str) -> bool:
    return any(_normalize_text(profile.get(field, "")) for field in fields)


def _choose_profile_target(profile: dict) -> str | None:
    """Pick the next profile area to ask about, based on questionnaire-style priorities."""
    if not _has_profile_value(profile, "asking_for"):
        return "asking_for"
    if not _has_profile_value(profile, "age_group"):
        return "age_group"
    if not _has_profile_value(profile, "main_goal"):
        return "main_goal"
    if not _has_profile_value(profile, "health_conditions", "medications", "allergies", "dietary_restriction"):
        return "health_context"
    if not _has_profile_value(profile, "preferences", "disliked_foods", "recurring_needs"):
        return "routine"
    return None


def _should_continue_profile_flow(user_message: str) -> bool:
    """Treat short, direct replies as answers to the most recent profile question."""
    text = _normalize_text(user_message)
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


def _looks_like_profile_answer(target: str, user_message: str) -> bool:
    """Use target-specific heuristics so new requests do not get swallowed as profile answers."""
    text = _normalize_text(user_message)
    if not text:
        return True
    if not _should_continue_profile_flow(text):
        return False

    lower = text.lower()
    request_starters = (
        "can you", "could you", "would you", "what ", "how ", "should ", "do i ",
        "is it ", "are there", "give me", "tell me", "help me",
    )
    if lower.startswith(request_starters):
        return False

    if target == "asking_for":
        return bool(re.search(r"\b(for me|myself|self|me|my child|my kid|my son|my daughter|my baby|my mom|my mother|my dad|my father|my parent|my husband|my wife|my spouse|someone else)\b", lower))

    if target == "age_group":
        return bool(re.search(r"\b(under|adult|child|kid|teen|young|middle|senior|elder|\d{1,3})\b", lower))

    if target == "health_context":
        return bool(re.search(r"\b(allerg|diabet|pregnan|gluten|vegan|vegetarian|halal|kosher|medication|metformin|warfarin|hypertension|blood pressure|none|no allergies)\b", lower))

    if target == "main_goal":
        return len(text.split()) <= 10 and not lower.startswith(("food ", "meal ", "store "))

    if target == "routine":
        return len(text.split()) <= 12

    return False


def _build_profile_question(profile: dict, user_message: str, target: str, session_id: str) -> str:
    """Generate one natural follow-up question to keep profile-building conversational."""
    prompt = (
        f"[USER PROFILE]\n{_format_profile_for_prompt(profile)}\n\n"
        f"[LATEST MESSAGE]\n{_normalize_text(user_message) or 'The user just tapped the nutrition option.'}\n\n"
        f"[TARGET]\n{target}"
    )
    fallback_map = {
        "asking_for": "Are you asking for yourself, or for someone else?",
        "age_group": "What age range should I keep in mind?",
        "main_goal": "What would you most like help with right now?",
        "health_context": "Any health conditions, medications, allergies, or food restrictions I should keep in mind?",
        "routine": "Any foods you avoid, budget concerns, or cooking limits that would help me tailor ideas?",
    }
    try:
        question = _ai.ask(profile_nudge_prompt, prompt, session_id + "_profile_nudge").strip()
        if question:
            return question
    except Exception:
        pass
    return fallback_map[target]


def _profile_acknowledgement(profile: dict) -> str:
    if _normalize_text(profile.get("asking_for")) in {"child", "parent", "spouse", "other"}:
        return "Thanks, that helps me tailor this for them."
    return "Thanks, that helps me tailor this for you."


def _format_profile_for_prompt(profile: dict) -> str:
    if not profile:
        return BLANK_PROFILE
    return "\n".join(f"{k}: {v}" for k, v in profile.items() if _normalize_text(v))


def _save_and_reload_profile(user_id: str, user_message: str):
    if _normalize_text(user_message):
        try:
            _mem.auto_extract_and_save(user_id, user_message)
        except Exception:
            pass
    raw = _mem.load_all(user_id)
    profile = {}
    for line in raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            profile[k.strip()] = v.strip()
    return profile


_web_search = WebSearch()

def _add_web_search(response: str, query: str) -> str:
    try:
        results = _web_search.search(query, max_results=3)
        if results:
            search_text = "\n\nSources:\n" + "\n".join([f"- {r['title']}: {r['link']}" for r in results])
            response = f"{response}\n{search_text}"
    except Exception:
        pass
    return response

# ══════════════════════════════════════════════════════════════════════════════
# AGENT CLASS
# ══════════════════════════════════════════════════════════════════════════════


class NutritionAgent:
    def _format_profile_context(self, profile: dict) -> str:
        return _format_profile_for_prompt(profile)

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
    ) -> tuple[str, list[Button] | str]:
        """One LLM-led Find Resources turn: JSON reply + actions + optional dynamic buttons."""
        session = _user_session(user_id)
        profile_context = self._format_profile_context(self._get_profile(user_id))
        summary = _resources_conversation_summary.get(user_id, "(none)")
        retrieval_q = (user_text or "").strip() or "Massachusetts WIC SNAP food assistance resources"
        kb_block = ""
        if _should_retrieve_public_kb(retrieval_q, session, "resources"):
            try:
                ctx, _has_rel, _src = _rag.get_context(retrieval_q, user_id=user_id)
                if ctx and str(ctx).strip():
                    kb_block = f"[KNOWLEDGE BASE SNIPPETS]\n{str(ctx).strip()}\n\n"
            except Exception:
                pass
        query = (
            f"[USER PROFILE]\n{profile_context}\n"
            f"[CONVERSATION SUMMARY]\n{summary}\n"
            f"{kb_block}"
            f"[USER MESSAGE]\n{(user_text or '').strip() or '(empty)'}"
        )
        raw = _ai.ask(resources_lead_system_prompt, query, session + "_rlead")
        data = _parse_resources_json(raw)
        if not data:
            repair = _ai.ask(
                resources_lead_json_repair_prompt,
                f"Invalid or missing JSON. Fix it.\n\nOriginal:\n{raw[:2000]}",
                session + "_rlead_fix",
            )
            data = _parse_resources_json(repair)
        if not data:
            return (
                "I'm having trouble with that request. Could you say what you need in your own words "
                "(for example WIC, SNAP, affordable groceries, or nearby stores)?",
                [],
            )

        reply = str(data.get("reply") or "").strip()
        cs = str(data.get("conversation_summary") or "").strip()[:200]
        if cs:
            _resources_conversation_summary[user_id] = cs
        actions_raw = data.get("actions") or []
        if not isinstance(actions_raw, list):
            actions_raw = []

        actions: list[dict] = []
        for a in actions_raw:
            if isinstance(a, str):
                actions.append({"type": a})
            elif isinstance(a, dict):
                actions.append(a)

        act_types = {_resources_action_type(a) for a in actions}
        if _wants_wic_store_by_location(user_text) and not act_types & {
            "REQUEST_WIC_LOCATION",
            "REQUEST_ALL_STORES",
        }:
            actions.append({"type": "REQUEST_WIC_LOCATION"})

        if _is_synthetic_resources_hub_opener(user_text):
            actions = [
                a
                for a in actions
                if _resources_action_type(a)
                not in ("REQUEST_WIC_LOCATION", "REQUEST_ALL_STORES")
            ]

        if any(_resources_action_type(a) == "START_ELIGIBILITY" for a in actions):
            _resources_mode_users.discard(user_id)
            _resources_conversation_summary.pop(user_id, None)
            _eligibility_state.add(user_id)
            elig_msg = _ai.ask(
                eligibility_check_prompt,
                "Start the eligibility check now. Ask one question at a time.",
                session,
            )
            if reply:
                elig_msg = f"{reply}\n\n{elig_msg}"
            return elig_msg, []

        extras: list[str] = []
        wants_wic_loc = False
        wants_all_loc = False
        for a in actions:
            t = _resources_action_type(a)
            if t == "AFFORDABLE_OVERVIEW":
                aff_q = (
                    "Tell me about affordable grocery options available to everyone in Massachusetts "
                    "regardless of income or eligibility. Include Market Basket, food pantries, "
                    "community fridges, and farmers markets with the HIP program. Keep it concise."
                )
                block = _ai.ask(main_system_prompt, aff_q, session + "_r_aff").strip()
                if block:
                    extras.append(block)
            elif t == "EXPLAIN_PROGRAM":
                prog = str(a.get("program") or "").lower()
                if prog == "wic":
                    q = (
                        "In 3-4 sentences, explain who qualifies for WIC in Massachusetts: "
                        "pregnant, postpartum, breastfeeding women, or children under 5, with income under "
                        "185% of federal poverty level. End by asking if they think they qualify."
                    )
                elif prog == "snap":
                    q = (
                        "In 3-4 sentences, explain who qualifies for SNAP in Massachusetts: "
                        "income-based, available to most low-income households, also unlocks the HIP program "
                        "for fresh produce. End by asking if they think they qualify."
                    )
                else:
                    continue
                block = _ai.ask(main_system_prompt, q, session + "_r_exp").strip()
                if block:
                    extras.append(block)
            elif t == "REQUEST_WIC_LOCATION":
                wants_wic_loc = True
            elif t == "REQUEST_ALL_STORES":
                wants_all_loc = True

        parts = [p for p in extras if p]
        if reply:
            parts.append(reply)
        combined = "\n\n".join(parts) if parts else "How can I help with local food resources today?"

        if wants_wic_loc:
            _pending_store_type[user_id] = "find_wic_stores"
            loc_note = (
                "Tap the button below to share your location — I'll list nearby WIC-authorized stores."
            )
            combined = f"{combined}\n\n{loc_note}"
            return combined.strip(), "request_location"
        if wants_all_loc:
            _pending_store_type[user_id] = "find_all_stores"
            loc_note = "Tap the button below to share your location for nearby store ideas."
            combined = f"{combined}\n\n{loc_note}"
            return combined.strip(), "request_location"

        buttons = _resource_suggested_buttons(data.get("suggested_buttons"))
        return combined.strip(), buttons

    def _remember_user_message(self, user_id: str, user_message: str) -> dict:
        """Save structured facts from raw user text and return the refreshed profile."""
        return _save_and_reload_profile(user_id, user_message)

    def _start_profile_conversation(self, user_id: str, profile: dict, session: str, seed_message: str) -> tuple[str, list[Button]]:
        target = _choose_profile_target(profile) or "asking_for"
        _nutrition_ob_state[user_id] = {"target": target}
        question = _build_profile_question(profile, seed_message, target, session)
        return question, []

    def _maybe_start_profile_from_welcome(
        self,
        user_id: str,
        profile: dict,
        session: str,
        welcome_response: str,
        user_message: str,
    ) -> tuple[str, list[Button]] | None:
        """For new users, kick off profile-building directly from the welcome flow."""
        target = _choose_profile_target(profile)
        if not target:
            return None
        if target != "asking_for":
            return None
        _nutrition_ob_state[user_id] = {"target": target}
        question = _build_profile_question(profile, user_message, target, session)
        return f"{welcome_response}\n\n{question}", []

    def _answer_saved_profile_task(self, user_id: str, profile: dict, session: str, state: dict) -> tuple[str, list[Button]]:
        pending_question = (state or {}).get("pending_question")
        if pending_question:
            full_query = f"[USER PROFILE]\n{self._format_profile_context(profile)}\n[QUESTION]\n{pending_question}"
            response = _ai.ask(main_system_prompt, full_query, session)
            buttons = _generate_buttons(response, session)
            return response, buttons

        response = _ai.ask(
            main_system_prompt,
            f"[USER PROFILE]\n{self._format_profile_context(profile)}\n[QUESTION]\nGive one short, encouraging sentence that says you can now tailor nutrition help better and invites the user to ask anything.",
            session,
        ).strip()
        if not response:
            response = "Thanks, that gives me a good sense of what to tailor for you. What would you like help with first?"
        return response, _make_buttons(WELCOME_BUTTONS)

    def _continue_profile_conversation(self, user_id: str, user_message: str, session: str) -> tuple[str, list[Button]] | None:
        state = _nutrition_ob_state.get(user_id)
        target = (state or {}).get("target")
        if not state or not target or not _looks_like_profile_answer(target, user_message):
            return None

        profile = self._remember_user_message(user_id, user_message)
        next_target = _choose_profile_target(profile)
        if next_target:
            next_state = dict(state)
            next_state["target"] = next_target
            _nutrition_ob_state[user_id] = next_state
            question = _build_profile_question(profile, user_message, next_target, session)
            return f"{_profile_acknowledgement(profile)} {question}", []

        _nutrition_ob_state.pop(user_id, None)
        return self._answer_saved_profile_task(user_id, profile, session, state)

    def _maybe_append_profile_nudge(
        self,
        user_id: str,
        response: str,
        profile: dict,
        user_message: str,
        session: str,
        intent: str,
    ) -> tuple[str, list[Button]]:
        if intent != "nutrition_advice":
            return response, []
        target = _choose_profile_target(profile)
        if not target:
            _nutrition_ob_state.pop(user_id, None)
            return response, []
        if len(_normalize_text(user_message).split()) < 3:
            return response, []
        question = _build_profile_question(profile, user_message, target, session)
        _nutrition_ob_state[user_id] = {"target": target}
        return f"{response}\n\n{question}", []

    def run(self, user_message: str, user_id: str) -> tuple[str, list[Button] | str]:
        """Handle a free-text message from the user. Injects user profile into context."""
        profile = self._get_profile(user_id)
        profile_context = self._format_profile_context(profile)
        # Eligibility check conversation
        if user_id in _eligibility_state:
            session = _user_session(user_id)
            response = _ai.ask(eligibility_check_prompt, user_message, session)
            recommendation_keywords = ["qualify", "eligible", "recommend", "apply", "snap", "wic", "senior nutrition"]
            if any(kw in response.lower() for kw in recommendation_keywords):
                _eligibility_state.discard(user_id)
                buttons = _generate_buttons(
                    response,
                    session + "_elig_end",
                    fallback_buttons=WELCOME_BUTTONS,
                )
            else:
                buttons = []
            return response, buttons

        session = _user_session(user_id)
        profile_reply = self._continue_profile_conversation(user_id, user_message, session)
        if profile_reply is not None:
            return profile_reply

        # Normal flow
        if _is_greeting(user_message):
            _food_safety_flow_users.discard(user_id)
            _resources_mode_users.discard(user_id)
            _resources_conversation_summary.pop(user_id, None)
            user_line = user_message.strip() or "The user just opened the chat."
            welcome_query = (
                f"[USER PROFILE]\n{profile_context}\n[USER SAID]\n{user_line}"
            )
            try:
                response = _ai.ask(
                    welcome_generator_prompt,
                    welcome_query,
                    session + "_welcome",
                ).strip()
                if len(response) < 30:
                    response = WELCOME_FALLBACK_MESSAGE
            except Exception:
                response = WELCOME_FALLBACK_MESSAGE
            welcome_with_profile = self._maybe_start_profile_from_welcome(
                user_id=user_id,
                profile=profile,
                session=session,
                welcome_response=response,
                user_message=user_line,
            )
            if welcome_with_profile is not None:
                return welcome_with_profile
            return response, _make_buttons(WELCOME_BUTTONS)

        if user_id in _resources_mode_users:
            self._remember_user_message(user_id, user_message)
            text, btns = self._resources_turn(user_message, user_id)
            if btns != "request_location":
                text = _append_button_intro(text, btns if isinstance(btns, list) else [], session)
            return text, btns

        if user_id in _food_safety_flow_users:
            response, buttons = self._food_safety_answer_turn(
                user_message, user_id, profile_context
            )
            response = _append_button_intro(response, buttons, session)
            return response, buttons

        intent = _classify_intent(user_message, session)

        if intent == "find resources":
            _resources_mode_users.add(user_id)
            _food_safety_flow_users.discard(user_id)
            self._remember_user_message(user_id, user_message)
            text, btns = self._resources_turn(user_message, user_id)
            if btns != "request_location":
                text = _append_button_intro(text, btns if isinstance(btns, list) else [], session)
            return text, btns

        if intent == "food_safety":
            full_query = f"[USER PROFILE]\n{profile_context}\n[QUESTION]\n{user_message}"
            response = _rag.query_rag(full_query, session_id=session, user_id=user_id, memory_source_message=user_message)
            remembered_profile = self._get_profile(user_id)
        else:
            remembered_profile = self._remember_user_message(user_id, user_message)
            profile_context = self._format_profile_context(remembered_profile)
            full_query = f"[USER PROFILE]\n{profile_context}\n[QUESTION]\n{user_message}"
            if intent == "nutrition_advice":
                if _should_retrieve_public_kb(user_message, session, "nutrition"):
                    response = _rag.query_rag(
                        full_query,
                        session_id=session,
                        user_id=user_id,
                        memory_source_message=user_message,
                    )
                else:
                    response = _ai.ask(main_system_prompt, full_query, session)
            else:
                response = _ai.ask(main_system_prompt, full_query, session)

        nudge_buttons: list[Button] = []
        response, nudge_buttons = self._maybe_append_profile_nudge(
            user_id=user_id,
            response=response,
            profile=remembered_profile,
            user_message=user_message,
            session=session,
            intent=intent,
        )
        buttons = nudge_buttons or _generate_buttons(response, session)

        response = _append_button_intro(response, buttons, session)
        return response, buttons

    def run_tool(
        self,
        interaction_id: str,
        user_id: str,
        interaction_title: str | None = None,
    ) -> tuple[str, list[Button] | str]:
        """Handle a button click (InteractiveEvent)."""
        session = _user_session(user_id)
        profile = self._get_profile(user_id)
        profile_context = self._format_profile_context(profile)

        if interaction_id in ("nutrition", "find_stores"):
            _food_safety_flow_users.discard(user_id)
        if interaction_id == "nutrition":
            _resources_mode_users.discard(user_id)
            _resources_conversation_summary.pop(user_id, None)

        if interaction_id == "food_safety":
            _food_safety_flow_users.add(user_id)
            _resources_mode_users.discard(user_id)
            _resources_conversation_summary.pop(user_id, None)
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

        elif interaction_id == "nutrition":
            target = _choose_profile_target(profile)
            if target:
                _nutrition_ob_state[user_id] = {
                    "target": target,
                    "pending_question": "Give me practical, personalized healthy eating advice based on this user's profile.",
                }
                question = _build_profile_question(
                    profile,
                    "The user tapped the nutrition option and is ready to talk about eating habits and health goals.",
                    target,
                    session,
                )
                return question, []
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
            buttons = _generate_buttons(response, session)
            return response, buttons

        elif interaction_id == "find_stores":
            _resources_mode_users.add(user_id)
            text, btns = self._resources_turn(
                "The user opened Find Resources from the main menu.", user_id
            )
            if btns != "request_location":
                text = _append_button_intro(text, btns if isinstance(btns, list) else [], session)
            return text, btns

        elif interaction_id.startswith("resources_dyn_"):
            label = (interaction_title or "").strip() or interaction_id
            text, btns = self._resources_turn(label, user_id)
            if btns != "request_location":
                text = _append_button_intro(text, btns if isinstance(btns, list) else [], session)
            return text, btns

        elif interaction_id == "wic_info":
            _resources_mode_users.add(user_id)
            text, btns = self._resources_turn(
                "The user tapped WIC Help and wants to know about WIC in Massachusetts.", user_id
            )
            if btns != "request_location":
                text = _append_button_intro(text, btns if isinstance(btns, list) else [], session)
            return text, btns

        elif interaction_id in ("elig_still_unsure", "elig_answers"):
            _resources_mode_users.discard(user_id)
            _resources_conversation_summary.pop(user_id, None)
            _eligibility_state.add(user_id)
            response = _ai.ask(
                eligibility_check_prompt,
                "Start the eligibility check now. Ask one question at a time.",
                session,
            )
            return response, []

        elif interaction_id in (
            "affordable_shopping",
            "check_eligibility",
            "elig_wic",
            "elig_snap",
            "elig_not_sure",
            "elig_i_qualify",
        ):
            _resources_mode_users.add(user_id)
            legacy_hint = {
                "affordable_shopping": "User wants affordable groceries, pantries, and HIP.",
                "check_eligibility": "User wants to explore WIC, SNAP, or program eligibility.",
                "elig_wic": "User asked specifically about WIC eligibility.",
                "elig_snap": "User asked specifically about SNAP eligibility.",
                "elig_not_sure": "User is not sure which program fits; guide them gently.",
                "elig_i_qualify": "User thinks they may qualify and wants concrete next steps.",
            }.get(interaction_id, interaction_title or interaction_id)
            text, btns = self._resources_turn(legacy_hint, user_id)
            if btns != "request_location":
                text = _append_button_intro(text, btns if isinstance(btns, list) else [], session)
            return text, btns

        elif interaction_id == "find_wic_stores":
            _pending_store_type[user_id] = "find_wic_stores"
            return "Tap the button below to share your location and I'll find the nearest WIC stores for you. 📍", "request_location"

        elif interaction_id == "find_all_stores":
            _pending_store_type[user_id] = "find_all_stores"
            return "Tap the button below to share your location and I'll find nearby stores for you. 📍", "request_location"

        elif interaction_id in (
            "for_myself",
            "for_child",
            "special_nutrition",
            "wic_apply",
        ):
            label_map = {
                "for_myself":        "Give me practical healthy eating advice for an adult.",
                "for_child":         "Give me practical healthy eating advice for a young child under 5.",
                "special_nutrition": "Ask what special nutrition situation the user wants help with, such as pregnancy, allergies, diabetes, or dietary restrictions.",
                "wic_apply":         "How do I apply for WIC benefits in Massachusetts?",
            }
            query = label_map[interaction_id]
            response = _ai.ask(main_system_prompt, query, session)
            clean = re.sub(r"\[Source:[^\]]+\]", "", response).strip()
            buttons = _generate_buttons(clean, session)
            response = clean

        else:
            follow_up = interaction_title or interaction_id
            if user_id in _resources_mode_users:
                text, btns = self._resources_turn(follow_up, user_id)
                if btns != "request_location":
                    text = _append_button_intro(text, btns if isinstance(btns, list) else [], session)
                return text, btns
            if user_id in _food_safety_flow_users:
                response, buttons = self._food_safety_answer_turn(
                    follow_up, user_id, profile_context
                )
                response = _append_button_intro(response, buttons, session)
                return response, buttons
            return self.run(follow_up, user_id)

        response = _append_button_intro(response, buttons, session)
        return response, buttons

    def run_location(
        self,
        lat: float,
        lng: float,
        user_id: str,
    ) -> tuple[str, list[Button]]:
        """Handle a location message — finds nearest WIC stores from CSV."""
        _pending_store_type.pop(user_id, None)
        _resources_mode_users.discard(user_id)
        _resources_conversation_summary.pop(user_id, None)
        try:
            from location_service import LocationService
            svc    = LocationService()
            stores = svc.find_nearby_wic_stores(lat, lng)
            response = svc.format_for_bot(stores)
        except Exception as e:
            response = f"Sorry, I couldn't find stores right now. Please try again later. ({e})"

        buttons = _make_buttons(WELCOME_BUTTONS)
        return response, buttons
