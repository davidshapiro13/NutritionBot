from wa_service_sdk import Button
import re


def build_profile_question(
    profile: dict,
    user_message: str,
    target: str,
    session_id: str,
    *,
    ai,
    profile_nudge_prompt: str,
    format_profile_for_prompt,
    normalize_text,
    blank_profile: str,
) -> str:
    """Generate one deterministic follow-up question to keep profile-building fast."""
    fallback_map = {
        "asking_for": "Are you asking for yourself, or for someone else?",
        "age_group": "What age range should I keep in mind?",
        "main_goal": "What would you most like help with right now?",
        "health_context": "Any health conditions, medications, allergies, or food restrictions I should keep in mind?",
        "routine": "Any foods you avoid, budget concerns, or cooking limits that would help me in suggesting ideas?",
    }
    return fallback_map[target]


def save_and_reload_profile(user_id: str, user_message: str, *, mem, normalize_text) -> dict:
    if normalize_text(user_message):
        try:
            mem.auto_extract_and_save(user_id, user_message)
        except Exception:
            pass
    raw = mem.load_all(user_id)
    profile = {}
    for line in raw.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            profile[key.strip()] = value.strip()
    return profile


def maybe_start_profile_from_welcome(
    *,
    user_id: str,
    profile: dict,
    session: str,
    welcome_response: str,
    user_message: str,
    choose_profile_target,
    build_profile_question_fn,
    nutrition_ob_state: dict[str, dict],
) -> tuple[str, list[Button]] | None:
    """For new users, kick off profile-building directly from the welcome flow."""
    target = choose_profile_target(profile)
    if not target:
        return None
    if target != "asking_for":
        return None
    nutrition_ob_state[user_id] = {
        "target": target,
        "pending_question": "Give me practical, personalized healthy eating advice based on this user's profile.",
    }
    question = build_profile_question_fn(profile, user_message, target, session)
    return f"{welcome_response}\n\n{question}", []


def answer_saved_profile_task(
    *,
    user_id: str,
    profile: dict,
    session: str,
    state: dict,
    ai,
    main_system_prompt: str,
    format_profile_context,
    generate_buttons,
    make_buttons,
    welcome_buttons: list[dict],
) -> tuple[str, list[Button]]:
    pending_question = (state or {}).get("pending_question")
    if pending_question:
        full_query = f"[USER PROFILE]\n{format_profile_context(profile)}\n[QUESTION]\n{pending_question}"
        response = ai.ask(main_system_prompt, full_query, session)
        buttons = generate_buttons(response, session)
        return response, buttons

    response = ai.ask(
        main_system_prompt,
        f"[USER PROFILE]\n{format_profile_context(profile)}\n[QUESTION]\nGive one short, encouraging sentence that says you can now tailor nutrition help better and invites the user to ask anything.",
        session,
    ).strip()
    if not response:
        response = "Thanks, that gives me a good sense of what to tailor for you. What would you like help with first?"
    return response, make_buttons(welcome_buttons)


def continue_profile_conversation(
    *,
    user_id: str,
    user_message: str,
    session: str,
    remember_user_message,
    choose_profile_target,
    looks_like_profile_answer,
    build_profile_question_fn,
    answer_saved_profile_task_fn,
    nutrition_ob_state: dict[str, dict],
) -> tuple[str, list[Button]] | None:
    def _resolve_asking_for_value(message: str) -> str | None:
        lower = message.strip().lower()
        if re.search(r"\b(for me|myself|self|me)\b", lower):
            return "self"
        if re.search(r"\b(my child|my kid|my son|my daughter|my baby)\b", lower):
            return "child"
        if re.search(r"\b(my mom|my mother|my dad|my father|my parent)\b", lower):
            return "parent"
        if re.search(r"\b(my husband|my wife|my spouse)\b", lower):
            return "spouse"
        if "someone else" in lower:
            return "other"
        return None

    state = nutrition_ob_state.get(user_id)
    target = (state or {}).get("target")
    if not state or not target or not looks_like_profile_answer(target, user_message):
        return None

    profile = remember_user_message(user_id, user_message)
    if target == "asking_for" and not profile.get("asking_for"):
        asking_for_value = _resolve_asking_for_value(user_message)
        if asking_for_value:
            profile = dict(profile)
            profile["asking_for"] = asking_for_value
    next_target = choose_profile_target(profile)
    if next_target:
        next_state = dict(state)
        next_state["target"] = next_target
        nutrition_ob_state[user_id] = next_state
        question = build_profile_question_fn(profile, user_message, next_target, session)
        return question, []

    nutrition_ob_state.pop(user_id, None)
    return answer_saved_profile_task_fn(user_id, profile, session, state)


def maybe_append_profile_nudge(
    *,
    user_id: str,
    response: str,
    profile: dict,
    user_message: str,
    session: str,
    intent: str,
    choose_profile_target,
    normalize_text,
    build_profile_question_fn,
    nutrition_ob_state: dict[str, dict],
) -> tuple[str, list[Button]]:
    if intent != "nutrition_advice":
        return response, []
    target = choose_profile_target(profile)
    if not target:
        nutrition_ob_state.pop(user_id, None)
        return response, []
    if len(normalize_text(user_message).split()) < 3:
        return response, []
    question = build_profile_question_fn(profile, user_message, target, session)
    nutrition_ob_state[user_id] = {"target": target}
    return f"{response}\n\n{question}", []
