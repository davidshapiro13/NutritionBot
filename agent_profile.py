from wa_service_sdk import Button
import re


AGE_GROUP_BUTTONS = [
    Button(id="profile_age_0_17", title="0-17"),
    Button(id="profile_age_18_64", title="18-64"),
    Button(id="profile_age_65_plus", title="65+"),
]


def _remove_button_invite(text: str) -> str:
    """Strip menu/button CTA when we immediately ask an onboarding question."""
    cleaned = text.strip()
    patterns = [
        r"\s*Use the buttons below for eating tips, food safety, or local resources\s*[—-]\s*or type any question\.?\s*$",
        r"\s*End by inviting them to tap the buttons below or type a question\.?\s*$",
        r"\s*Tap the buttons below or type a question\.?\s*$",
        r"\s*Tap a button below or type your question\.?\s*$",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def profile_buttons_for_target(target: str) -> list[Button]:
    if target == "age_group":
        return AGE_GROUP_BUTTONS.copy()
    return []


def profile_button_value(interaction_id: str, interaction_title: str | None = None) -> str | None:
    mapping = {
        "profile_age_0_17": "0-17",
        "profile_age_18_64": "18-64",
        "profile_age_65_plus": "65+",
    }
    if interaction_id in mapping:
        return mapping[interaction_id]
    title = (interaction_title or "").strip().lower()
    reverse_titles = {
        "0-17": "0-17",
        "18-64": "18-64",
        "65+": "65+",
    }
    return reverse_titles.get(title)


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
    nutrition_ob_state[user_id] = {"target": target}
    question = build_profile_question_fn(profile, user_message, target, session)
    intro = _remove_button_invite(welcome_response)
    if intro:
        return f"{intro}\n\n{question}", profile_buttons_for_target(target)
    return question, profile_buttons_for_target(target)


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
    save_profile_value,
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

    def _resolve_age_group_value(message: str) -> str | None:
        lower = message.strip().lower()
        if re.search(r"\b(0\s*-\s*17|0 to 17)\b", lower):
            return "child"
        if re.search(r"\b(18\s*-\s*64|18 to 64)\b", lower):
            return "adult"
        if re.search(r"\b(65\s*\+|65 and up|65 or older)\b", lower):
            return "elder"
        if re.search(r"\b(child|kid|toddler|baby|infant)\b", lower):
            return "child"
        if re.search(r"\b(older adult|elder|senior)\b", lower):
            return "elder"
        if re.search(r"\b(adult)\b", lower):
            return "adult"
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
            save_profile_value(user_id, "asking_for", asking_for_value)
    if target == "age_group" and not profile.get("age_group"):
        age_group_value = _resolve_age_group_value(user_message)
        if age_group_value:
            profile = dict(profile)
            profile["age_group"] = age_group_value
            save_profile_value(user_id, "age_group", age_group_value)
    next_target = choose_profile_target(profile)
    if next_target:
        next_state = dict(state)
        next_state["target"] = next_target
        nutrition_ob_state[user_id] = next_state
        question = build_profile_question_fn(profile, user_message, next_target, session)
        return question, profile_buttons_for_target(next_target)

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
    # Keep passive profile-building rare; explicit onboarding/menu flows do the heavy lifting.
    allowed_targets = {"asking_for", "age_group"}
    if target not in allowed_targets:
        return response, []
    normalized = normalize_text(user_message)
    if len(normalized.split()) < 6:
        return response, []
    lower = normalized.lower()
    personalization_signals = (
        r"\b(for me|for my|my child|my kid|my son|my daughter|my baby|"
        r"i need|help me|meal plan|eat better|diet|weight|pregnan|"
        r"diabet|allerg|gluten|vegetarian|vegan)\b"
    )
    if not re.search(personalization_signals, lower):
        return response, []
    question = build_profile_question_fn(profile, user_message, target, session)
    nutrition_ob_state[user_id] = {"target": target}
    return f"{response}\n\n{question}", profile_buttons_for_target(target)
