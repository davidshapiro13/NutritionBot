"""
NutritionBot Agent
===================
Central router that classifies user intent and calls the right tool.

Intents:
    food_safety      → rag_pipeline.query_rag()
    nutrition_advice → AI.ask()
    find_stores      → location_service (requires lat/lng)
    find_wic_stores  → location_service (WIC-only filter)
    find_all_stores  → location_service (all grocery stores)
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
    guided_transition_prompt,
    button_intro_prompt,
    eligibility_check_prompt,
    intent_classifier_prompt,
    WELCOME_BUTTONS,
    WELCOME_MESSAGE,
    FOOD_SAFETY_BUTTONS,
    NUTRITION_BUTTONS,
    ASKING_FOR_BUTTONS,
    AGE_GROUP_BUTTONS,
    ADULT_AGE_BUTTONS,
    ALLERGY_BUTTONS,
    STORE_TYPE_BUTTONS,
    ELIGIBILITY_PROGRAM_BUTTONS,
    ELIGIBILITY_QUALIFY_BUTTONS,
    ELIGIBILITY_NOTSURE_BUTTONS,
    ELIGIBILITY_ACTION_BUTTONS,
    WIC_INFO_BUTTONS,
    LOCATION_PROMPT,
)
from web_search import WebSearch
from wa_service_sdk import Button
from user_memory import UserMemory

import ast
import re


# ── Shared instances (loaded once at startup) ─────────────────────────────────
_ai  = AI()
_rag = RAGPipeline()
_rag.build_public_index()
_mem = UserMemory(embed_model=None)


# Tracks whether a user last requested WIC-only or all stores before sharing location
_pending_store_type: dict[str, str] = {}

# Nutrition inline onboarding state: {user_id: {"step": ..., "data": {...}}}
# Steps: "asking_for" → "age" → "allergy" → "allergy_text"
_nutrition_ob_state: dict[str, dict] = {}

# Users currently in eligibility check conversation
_eligibility_state: set[str] = set()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_buttons(buttons_data: list[dict]) -> list[Button]:
    """Convert list of {id, title} dicts to SDK Button objects."""
    return [Button(id=b["id"], title=b["title"]) for b in buttons_data]


def _generate_buttons(response: str, session_id: str) -> list[Button]:
    """Ask LLM to generate contextual follow-up buttons based on a response."""
    import json as _json
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
        return buttons[:3] if buttons else _make_buttons(WELCOME_BUTTONS)
    except Exception:
        return _make_buttons(WELCOME_BUTTONS)


def _generate_guided_transition(
    selected_button: str,
    target_goal: str,
    next_buttons: list[dict],
    session_id: str,
    fallback: str,
) -> str:
    """Generate a short bridge message while keeping button flow fixed."""
    prompt = guided_transition_prompt.format(
        selected_button=selected_button,
        target_goal=target_goal,
        next_buttons=", ".join(button["title"] for button in next_buttons),
    )
    try:
        response = _ai.ask(prompt, "Write the transition message now.", session_id + "_transition").strip()
        if response:
            return response
    except Exception:
        pass
    return fallback


def _classify_intent(user_message: str, session_id: str) -> str:
    """Classify user message into one of the four intents."""
    result = _ai.ask(intent_classifier_prompt, user_message, session_id)
    intent = result.strip().lower()
    valid = {"food_safety", "nutrition_advice", "find resources", "out_of_scope"}
    return intent if intent in valid else "nutrition_advice"


def _user_session(user_id: str) -> str:
    return f"NutritionBot_User_{user_id}"


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


def _merge_buttons(primary: list[Button], extra: list[Button], limit: int = 3) -> list[Button]:
    """Keep button order stable while adding a small number of fixed extras."""
    merged: list[Button] = []
    seen: set[str] = set()
    for button in primary + extra:
        if button.id not in seen:
            merged.append(button)
            seen.add(button.id)
        if len(merged) >= limit:
            break
    return merged


def _append_button_intro(response: str, buttons: list[Button], session_id: str) -> str:
    """Append a short LLM-generated sentence introducing the follow-up buttons."""
    if not buttons:
        return response
    try:
        button_titles = ", ".join(b.title for b in buttons)
        prompt = button_intro_prompt.format(
            response=response[:400],
            button_titles=button_titles,
        )
        intro = _ai.ask(prompt, "Write the sentence now.", session_id + "_intro").strip()
        if intro:
            return f"{response}\n\n{intro}"
    except Exception:
        pass
    return response


def _should_offer_wic(text: str) -> bool:
    """Detect cases where a short WIC nudge is likely helpful."""
    text = text.lower()
    patterns = [
        r"\bpregnant", r"\bbreastfeed", r"\bpostpartum\b",
        r"\bchild\b", r"\bchildren\b", r"\bkid\b",
        r"\bbaby\b", r"\binfant\b", r"\btoddler\b", r"\bnewborn\b",
        r"\bformula\b", r"\bfamily\b", r"\bbudget\b",
        r"\bafford\b", r"\blow income\b", r"\bunder 5\b", r"\bwic\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _maybe_add_wic_nudge(response: str, buttons: list[Button], context: str, session_id: str) -> tuple[str, list[Button]]:
    """Add a LLM-generated WIC nudge only when the user context suggests it."""
    if not _should_offer_wic(context):
        return response, buttons
    if "WIC" not in response:
        nudge = _ai.ask(
            main_system_prompt,
            f"In one sentence, gently suggest that the user may benefit from WIC support based on this context: {context}",
            session_id,
        ).strip()
        response = f"{response}\n\n{nudge}"
    wic_button = [Button(id="wic_info", title="💡 WIC Help")]
    return response, _merge_buttons(buttons, wic_button)

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
        if not profile:
            return "(no profile info)"
        return "\n".join(f"{k}: {v}" for k, v in profile.items() if v)

    def run(self, user_message: str, user_id: str) -> tuple[str, list[Button]]:
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
                buttons = _make_buttons([
                    {"id": "find_wic_stores", "title": "📍 Find WIC Stores"},
                    {"id": "wic_apply",       "title": "✅ How to Apply"},
                    {"id": "find_stores",     "title": "🔙 Other Resources"},
                ])
            else:
                buttons = []
            return response, buttons
        
        # Nutrition inline onboarding: waiting for typed allergy info
        
        nob = _nutrition_ob_state.get(user_id)
        if nob and nob.get("step") == "allergy_text":
            nob["data"]["allergy"] = user_message.strip()
            return self._finish_nutrition_onboarding(user_id, nob["data"])
        
        # Normal flow
        if _is_greeting(user_message):
            return WELCOME_MESSAGE, _make_buttons(WELCOME_BUTTONS)

        session = _user_session(user_id)
        intent  = _classify_intent(user_message, session)

        if intent in ("food_safety", "wic_food"):
            # Inject profile context into RAG
            full_query = f"[USER PROFILE]\n{profile_context}\n[QUESTION]\n{user_message}"
            response = _rag.query_rag(full_query, session_id=session, user_id=user_id)
            response = _add_web_search(response, user_message)
            buttons  = _generate_buttons(response, session)

        elif intent == "nutrition_advice":
            # Inject profile context into AI
            full_prompt = f"[USER PROFILE]\n{profile_context}\n[QUESTION]\n{user_message}"
            response = _ai.ask(main_system_prompt, full_prompt, session)
            response = _add_web_search(response, user_message)
            buttons  = _generate_buttons(response, session)
            response, buttons = _maybe_add_wic_nudge(response, buttons, user_message, session)

        elif intent == "find_stores":
            response = _generate_guided_transition(
                selected_button="Find Resources",
                target_goal="Guide the user toward nearby stores or WIC-related help.",
                next_buttons=STORE_TYPE_BUTTONS,
                session_id=session,
                fallback="I can help you look for nearby stores or point you to WIC support. What would you like to find?"
            )
            buttons = _make_buttons(STORE_TYPE_BUTTONS)

        else:  # out_of_scope — let the LLM decline and redirect naturally
            response = _ai.ask(main_system_prompt, user_message, session)
            buttons  = _make_buttons(WELCOME_BUTTONS)

        response = _append_button_intro(response, buttons, session)
        return response, buttons

    def _get_profile(self, user_id: str) -> dict:
        """Parse user profile from memory file."""
        raw = _mem.load_all(user_id)
        profile = {}
        for line in raw.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                profile[k.strip()] = v.strip()
        return profile

    def run_tool(
        self,
        interaction_id: str,
        user_id: str,
        interaction_title: str | None = None,
    ) -> tuple[str, list[Button]]:
        """Handle a button click (InteractiveEvent)."""
        session = _user_session(user_id)

        if interaction_id == "food_safety":
            response = _generate_guided_transition(
                selected_button="Food Safety",
                target_goal="Help the user choose the kind of food safety question they want to start with.",
                next_buttons=FOOD_SAFETY_BUTTONS,
                session_id=session,
                fallback="I can help with storage questions or a specific food safety concern. What fits best?"
            )
            buttons = _make_buttons(FOOD_SAFETY_BUTTONS)

        elif interaction_id == "nutrition":
            _nutrition_ob_state[user_id] = {"step": "asking_for", "data": {}}
            return "Great! Let's personalize this for you. Who are you asking for?", _make_buttons(ASKING_FOR_BUTTONS)

        elif interaction_id in ("nq_for_self", "nq_for_other"):
            nob = _nutrition_ob_state.setdefault(user_id, {"step": "asking_for", "data": {}})
            nob["data"]["asking_for"] = "self" if interaction_id == "nq_for_self" else "other"
            nob["step"] = "age"
            return "Got it! What's the age group?", _make_buttons(AGE_GROUP_BUTTONS)

        elif interaction_id == "nq_age_adult":
            nob = _nutrition_ob_state.setdefault(user_id, {"step": "age", "data": {}})
            nob["step"] = "age_adult"
            return "Which age range fits best?", _make_buttons(ADULT_AGE_BUTTONS)

        elif interaction_id in ("nq_age_under18", "nq_age_young", "nq_age_middle", "nq_age_senior"):
            age_map = {
                "nq_age_under18": "under 18",
                "nq_age_young":   "young adult (18–35)",
                "nq_age_middle":  "middle-aged (36–64)",
                "nq_age_senior":  "senior (65+)",
            }
            nob = _nutrition_ob_state.setdefault(user_id, {"step": "age", "data": {}})
            nob["data"]["age_group"] = age_map[interaction_id]
            nob["step"] = "allergy"
            return "Thanks! Do you have any food allergies?", _make_buttons(ALLERGY_BUTTONS)

        elif interaction_id == "nq_no_allergy":
            nob = _nutrition_ob_state.get(user_id, {"data": {}})
            nob["data"]["allergy"] = None
            return self._finish_nutrition_onboarding(user_id, nob["data"])

        elif interaction_id == "nq_has_allergy":
            nob = _nutrition_ob_state.setdefault(user_id, {"step": "allergy", "data": {}})
            nob["step"] = "allergy_text"
            return "Please type your allergies (e.g. peanuts, dairy, gluten).", []

        elif interaction_id == "find_stores":
            return (
                "Let's find the best options for you!\n\n"
                "🛒 Affordable Options — places anyone can use, no eligibility needed.\n"
                "📋 Check Eligibility — see if you qualify for WIC, SNAP, or other MA programs."
            ), _make_buttons(STORE_TYPE_BUTTONS)

        elif interaction_id == "affordable_shopping":
            query = (
                "Tell me about affordable grocery options available to everyone in Massachusetts "
                "regardless of income or eligibility. Include Market Basket, food pantries, "
                "community fridges, and farmers markets with the HIP program. Keep it concise."
            )
            response = _ai.ask(main_system_prompt, query, session)
            buttons = _generate_buttons(response, session)

        elif interaction_id == "check_eligibility":
            return (
                "Let's see what you may qualify for. Which program are you looking into?"
            ), _make_buttons(ELIGIBILITY_PROGRAM_BUTTONS)

        elif interaction_id == "elig_wic":
            response = _ai.ask(
                main_system_prompt,
                "In 3-4 sentences, explain who qualifies for WIC in Massachusetts: "
                "pregnant, postpartum, breastfeeding women, or children under 5, with income under 185% of federal poverty level. "
                "End by asking if they think they qualify.",
                session,
            )
            return response, _make_buttons(ELIGIBILITY_QUALIFY_BUTTONS)

        elif interaction_id == "elig_snap":
            response = _ai.ask(
                main_system_prompt,
                "In 3-4 sentences, explain who qualifies for SNAP in Massachusetts: "
                "income-based, available to most low-income households, also unlocks the HIP program for fresh produce. "
                "End by asking if they think they qualify.",
                session,
            )
            return response, _make_buttons(ELIGIBILITY_QUALIFY_BUTTONS)

        elif interaction_id == "elig_not_sure":
            return (
                "No problem! You can answer a few quick questions to find out what you qualify for, "
                "or explore affordable food options available to everyone."
            ), _make_buttons(ELIGIBILITY_NOTSURE_BUTTONS)

        elif interaction_id == "elig_i_qualify":
            return (
                "Great! Here's what you can do next:"
            ), _make_buttons(ELIGIBILITY_ACTION_BUTTONS)

        elif interaction_id in ("elig_still_unsure", "elig_answers"):
            _eligibility_state.add(user_id)
            response = _ai.ask(
                eligibility_check_prompt,
                "Start the eligibility check now. Ask one question at a time.",
                session,
            )
            return response, []

        elif interaction_id == "wic_info":
            response = _ai.ask(
                main_system_prompt,
                "Explain what WIC is, who qualifies, and what benefits it provides in Massachusetts.",
                session,
            )
            buttons = _make_buttons(WIC_INFO_BUTTONS)

        elif interaction_id == "find_wic_stores":
            _pending_store_type[user_id] = "find_wic_stores"
            return "Tap the button below to share your location and I'll find the nearest WIC stores for you. 📍", "request_location"

        elif interaction_id == "find_all_stores":
            _pending_store_type[user_id] = "find_all_stores"
            return "Tap the button below to share your location and I'll find nearby stores for you. 📍", "request_location"

        elif interaction_id in ("for_myself", "for_child", "special_nutrition",
                                "meat_storage", "dairy_storage", "ask_freely",
                                "wic_apply"):
            label_map = {
                "for_myself":        "Give me practical healthy eating advice for an adult.",
                "for_child":         "Give me practical healthy eating advice for a young child under 5.",
                "special_nutrition": "Ask what special nutrition situation the user wants help with, such as pregnancy, allergies, diabetes, or dietary restrictions.",
                "meat_storage":      "How long can I safely keep meat , and how should I store it?",
                "dairy_storage":     "How long can I safely keep dairy products , and how should I store them?",
                "ask_freely":        "Help me ask a food safety question in my own words.",
                "wic_apply":         "How do I apply for WIC benefits in Massachusetts?",
            }
            query = label_map[interaction_id]
            if interaction_id in ("meat_storage", "dairy_storage"):
                response = _rag.query_rag(query, session_id=session, user_id=user_id)
            else:
                response = _ai.ask(main_system_prompt, query, session)
            clean = re.sub(r"\[Source:[^\]]+\]", "", response).strip()
            buttons = _generate_buttons(clean, session)
            response, buttons = _maybe_add_wic_nudge(response, buttons, query, session)

        else:
            follow_up = interaction_title or interaction_id
            return self.run(follow_up, user_id)

        response = _append_button_intro(response, buttons, session)
        return response, buttons

    def _finish_nutrition_onboarding(self, user_id: str, data: dict) -> tuple[str, list[Button]]:
        """Save collected nutrition profile and give first personalized advice."""
        _nutrition_ob_state.pop(user_id, None)
        profile = self._get_profile(user_id)
        if data.get("asking_for"):
            profile["asking_for"] = data["asking_for"]
        if data.get("age_group"):
            profile["age_group"] = data["age_group"]
        if data.get("allergy"):
            profile["allergy"] = data["allergy"]
        _mem.save_profile(user_id, profile)

        session = _user_session(user_id)
        context_parts = [f"The user is {data.get('age_group', 'an adult')}"]
        if data.get("asking_for") == "other":
            context_parts.append("asking on behalf of someone else")
        if data.get("allergy"):
            context_parts.append(f"with allergies to {data['allergy']}")
        context = ", ".join(context_parts) + ". Give them a short, personalized healthy eating tip."
        response = _ai.ask(main_system_prompt, context, session)
        buttons = _generate_buttons(response, session)
        response, buttons = _maybe_add_wic_nudge(response, buttons, context, session)
        return response, buttons

    def run_location(
        self,
        lat: float,
        lng: float,
        user_id: str,
    ) -> tuple[str, list[Button]]:
        """Handle a location message — finds nearest WIC stores from CSV."""
        _pending_store_type.pop(user_id, None)
        try:
            from location_service import LocationService
            svc    = LocationService()
            stores = svc.find_nearby_wic_stores(lat, lng)
            response = svc.format_for_bot(stores)
        except Exception as e:
            response = f"Sorry, I couldn't find stores right now. Please try again later. ({e})"

        buttons = _make_buttons(WELCOME_BUTTONS)
        return response, buttons
