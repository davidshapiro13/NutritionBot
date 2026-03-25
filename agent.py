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
    intent_classifier_prompt,
    WELCOME_BUTTONS,
    WELCOME_MESSAGE,
    FOOD_SAFETY_BUTTONS,
    NUTRITION_BUTTONS,
    STORE_TYPE_BUTTONS,
    WIC_INFO_BUTTONS,
    LOCATION_PROMPT,
)
from web_search import WebSearch
from sdk.wa_service_sdk import Button
from user_memory import UserMemory

import ast
import re


# ── Shared instances (loaded once at startup) ─────────────────────────────────
_ai  = AI()
_rag = RAGPipeline()
_rag.build_public_index()
_mem = UserMemory(embed_model=None)

# Onboarding state: {user_id: {"field": ..., "profile": {...}}}
_onboarding_state = {}

# Tracks whether a user last requested WIC-only or all stores before sharing location
_pending_store_type: dict[str, str] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_buttons(buttons_data: list[dict]) -> list[Button]:
    """Convert list of {id, title} dicts to SDK Button objects."""
    return [Button(id=b["id"], title=b["title"]) for b in buttons_data]


def _generate_buttons(response: str, session_id: str) -> list[Button]:
    """Ask LLM to generate contextual follow-up buttons based on a response."""
    try:
        raw = _ai.ask(button_creator_prompt, response, session_id)
        items = ast.literal_eval(raw)
        buttons = []
        for item in items:
            info = ast.literal_eval(item) if isinstance(item, str) else item
            title = info["title"][:20]  # SDK enforces max 20 chars
            buttons.append(Button(id=info["id"], title=title))
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
        response = _ai.ask(prompt, "Write the transition message now.", session_id).strip()
        if response:
            return response
    except Exception:
        pass
    return fallback


def _classify_intent(user_message: str, session_id: str) -> str:
    """Classify user message into one of the four intents."""
    result = _ai.ask(intent_classifier_prompt, user_message, session_id)
    intent = result.strip().lower()
    valid = {"food_safety", "nutrition_advice", "wic_food", "find_stores", "out_of_scope"}
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


REQUIRED_PROFILE_FIELDS = ["name", "age_group", "main_goal"]

class NutritionAgent:

    def run(self, user_message: str, user_id: str) -> tuple[str, list[Button]]:
        """Handle a free-text message from the user, with onboarding if needed."""
        # Onboarding state machine
        state = _onboarding_state.get(user_id)
        if state:
            # Continue onboarding: save last answer, ask next
            last_field = state["field"]
            state["profile"][last_field] = user_message.strip()
            missing = [f for f in REQUIRED_PROFILE_FIELDS if not state["profile"].get(f)]
            if missing:
                next_field = missing[0]
                _onboarding_state[user_id] = {"field": next_field, "profile": state["profile"]}
                prompt = self._onboarding_prompt(next_field)
                return prompt, _make_buttons(WELCOME_BUTTONS)
            # All required fields collected
            _mem.save_profile(user_id, state["profile"])
            del _onboarding_state[user_id]
            return "Thank you! Your profile is saved. How can I help you today?", _make_buttons(WELCOME_BUTTONS)

        # Check if onboarding is needed
        profile = self._get_profile(user_id)
        missing = [f for f in REQUIRED_PROFILE_FIELDS if not profile.get(f)]
        if missing:
            next_field = missing[0]
            _onboarding_state[user_id] = {"field": next_field, "profile": profile}
            prompt = self._onboarding_prompt(next_field)
            return prompt, _make_buttons(WELCOME_BUTTONS)

        # Normal flow
        if _is_greeting(user_message):
            return WELCOME_MESSAGE, _make_buttons(WELCOME_BUTTONS)

        session = _user_session(user_id)
        intent  = _classify_intent(user_message, session)

        if intent in ("food_safety", "wic_food"):
            response = _rag.query_rag(user_message, session_id=session, user_id=user_id)
            response = _add_web_search(response, user_message)
            buttons  = _generate_buttons(response, session)

        elif intent == "nutrition_advice":
            response = _ai.ask(main_system_prompt, user_message, session)
            response = _add_web_search(response, user_message)
            buttons  = _generate_buttons(response, session)
            response, buttons = _maybe_add_wic_nudge(response, buttons, user_message, session)

        elif intent == "find_stores":
            response = _generate_guided_transition(
                selected_button="Find Resources",
                target_goal="Guide the user toward nearby stores or WIC-related help.",
                next_buttons=STORE_TYPE_BUTTONS,
                session_id=session,
                fallback="I can help you look for nearby stores or point you to WIC support. What would you like to find?",
            )
            buttons = _make_buttons(STORE_TYPE_BUTTONS)

        else:  # out_of_scope — let the LLM decline and redirect naturally
            response = _ai.ask(main_system_prompt, user_message, session)
            buttons  = _make_buttons(WELCOME_BUTTONS)

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

    def _onboarding_prompt(self, field: str) -> str:
        prompts = {
            "name": "Hi! To personalize your experience, what's your name?",
            "age_group": "What is your age group? (child, adult, or elder)",
            "main_goal": "What is your main health or nutrition goal?",
        }
        return prompts.get(field, f"Please provide your {field}.")

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
                fallback="I can help with storage questions or a specific food safety concern. What fits best?",
            )
            buttons = _make_buttons(FOOD_SAFETY_BUTTONS)

        elif interaction_id == "nutrition":
            response = _generate_guided_transition(
                selected_button="Eating Better",
                target_goal="Guide the user toward nutrition help for themselves, their child, or a special situation like pregnancy or allergies.",
                next_buttons=NUTRITION_BUTTONS,
                session_id=session,
                fallback="I can help with everyday eating, nutrition for a child, or a special nutrition need. Which fits?",
            )
            buttons = _make_buttons(NUTRITION_BUTTONS)

        elif interaction_id == "find_stores":
            response = _generate_guided_transition(
                selected_button="Find Resources",
                target_goal="Guide the user toward nearby stores or WIC-related help.",
                next_buttons=STORE_TYPE_BUTTONS,
                session_id=session,
                fallback="I can help you find nearby stores or WIC support. What would you like?",
            )
            buttons = _make_buttons(STORE_TYPE_BUTTONS)

        elif interaction_id == "wic_info":
            response = _ai.ask(
                main_system_prompt,
                "Explain what WIC is, who qualifies, and what benefits it provides in Massachusetts.",
                session,
            )
            buttons = _make_buttons(WIC_INFO_BUTTONS)

        elif interaction_id in ("find_wic_stores", "find_all_stores"):
            _pending_store_type[user_id] = interaction_id
            response = LOCATION_PROMPT
            buttons  = _make_buttons(WELCOME_BUTTONS)

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
            buttons = _generate_buttons(response, session)
            response, buttons = _maybe_add_wic_nudge(response, buttons, query, session)

        else:
            follow_up = interaction_title or interaction_id
            return self.run(follow_up, user_id)

        return response, buttons

    def run_location(
        self,
        lat: float,
        lng: float,
        user_id: str,
    ) -> tuple[str, list[Button]]:
        """Handle a location message — finds nearby stores based on user's prior selection."""
        wic_only = _pending_store_type.pop(user_id, None) == "find_wic_stores"
        try:
            from location_service import LocationService
            svc    = LocationService()
            stores = svc.find_nearby_wic_stores(lat, lng)
            if wic_only:
                stores = [s for s in stores if s["wic_likely"]]
            response = svc.format_for_bot(stores, lat, lng)
        except Exception as e:
            response = f"Sorry, I couldn't find stores right now. Please try again later. ({e})"

        buttons = _make_buttons(WELCOME_BUTTONS)
        return response, buttons
