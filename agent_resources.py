from wa_service_sdk import Button


def run_resources_turn(
    user_text: str,
    user_id: str,
    *,
    ai,
    rag,
    get_profile_context,
    should_retrieve_public_kb,
    parse_resources_json,
    resource_suggested_buttons,
    resources_action_type,
    wants_wic_store_by_location,
    is_synthetic_resources_hub_opener,
    resources_conversation_summary: dict[str, str],
    resources_mode_users: set[str],
    eligibility_state: set[str],
    pending_store_type: dict[str, str],
    resources_lead_system_prompt: str,
    resources_lead_json_repair_prompt: str,
    eligibility_check_prompt: str,
    main_system_prompt: str,
    user_session,
) -> tuple[str, list[Button] | str]:
    """One LLM-led Find Resources turn: JSON reply + actions + optional dynamic buttons."""
    session = user_session(user_id)
    profile_context = get_profile_context(user_id)
    summary = resources_conversation_summary.get(user_id, "(none)")
    retrieval_q = (user_text or "").strip() or "Massachusetts WIC SNAP food assistance resources"
    kb_block = ""  # Control experiment: RAG disabled; kb_block stays empty
    if False and should_retrieve_public_kb(retrieval_q, session, "resources"):  # Control experiment: RAG disabled
        try:
            ctx, _has_rel, _src = rag.get_context(retrieval_q, user_id=user_id)
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
    raw = ai.ask(resources_lead_system_prompt, query, session + "_rlead")
    data = parse_resources_json(raw)
    if not data:
        repair = ai.ask(
            resources_lead_json_repair_prompt,
            f"Invalid or missing JSON. Fix it.\n\nOriginal:\n{raw[:2000]}",
            session + "_rlead_fix",
        )
        data = parse_resources_json(repair)
    if not data:
        return (
            "I'm having trouble with that request. Could you say what you need in your own words "
            "(for example WIC, SNAP, affordable groceries, or nearby stores)?",
            [],
        )

    reply = str(data.get("reply") or "").strip()
    cs = str(data.get("conversation_summary") or "").strip()[:200]
    if cs:
        resources_conversation_summary[user_id] = cs
    actions_raw = data.get("actions") or []
    if not isinstance(actions_raw, list):
        actions_raw = []

    actions: list[dict] = []
    for action in actions_raw:
        if isinstance(action, str):
            actions.append({"type": action})
        elif isinstance(action, dict):
            actions.append(action)

    act_types = {resources_action_type(action) for action in actions}
    if wants_wic_store_by_location(user_text) and not act_types & {
        "REQUEST_WIC_LOCATION",
        "REQUEST_ALL_STORES",
    }:
        actions.append({"type": "REQUEST_WIC_LOCATION"})

    if is_synthetic_resources_hub_opener(user_text):
        actions = [
            action
            for action in actions
            if resources_action_type(action) not in ("REQUEST_WIC_LOCATION", "REQUEST_ALL_STORES")
        ]

    if any(resources_action_type(action) == "START_ELIGIBILITY" for action in actions):
        resources_mode_users.discard(user_id)
        resources_conversation_summary.pop(user_id, None)
        eligibility_state.add(user_id)
        elig_msg = ai.ask(
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
    for action in actions:
        action_type = resources_action_type(action)
        if action_type == "AFFORDABLE_OVERVIEW":
            aff_q = (
                "Tell me about affordable grocery options available to everyone in Massachusetts "
                "regardless of income or eligibility. Include Market Basket, food pantries, "
                "community fridges, and farmers markets with the HIP program. Keep it concise."
            )
            block = ai.ask(main_system_prompt, aff_q, session + "_r_aff").strip()
            if block:
                extras.append(block)
        elif action_type == "EXPLAIN_PROGRAM":
            program = str(action.get("program") or "").lower()
            if program == "wic":
                q = (
                    "In 3-4 sentences, explain who qualifies for WIC in Massachusetts: "
                    "pregnant, postpartum, breastfeeding women, or children under 5, with income under "
                    "185% of federal poverty level. End by asking if they think they qualify."
                )
            elif program == "snap":
                q = (
                    "In 3-4 sentences, explain who qualifies for SNAP in Massachusetts: "
                    "income-based, available to most low-income households, also unlocks the HIP program "
                    "for fresh produce. End by asking if they think they qualify."
                )
            else:
                continue
            block = ai.ask(main_system_prompt, q, session + "_r_exp").strip()
            if block:
                extras.append(block)
        elif action_type == "REQUEST_WIC_LOCATION":
            wants_wic_loc = True
        elif action_type == "REQUEST_ALL_STORES":
            wants_all_loc = True

    parts = [part for part in extras if part]
    if reply:
        parts.append(reply)
    combined = "\n\n".join(parts) if parts else "How can I help with local food resources today?"

    if wants_wic_loc:
        pending_store_type[user_id] = "find_wic_stores"
        loc_note = "Tap the button below to share your location — I'll list nearby WIC-authorized stores."
        combined = f"{combined}\n\n{loc_note}"
        return combined.strip(), "request_location"
    if wants_all_loc:
        pending_store_type[user_id] = "find_all_stores"
        loc_note = "Tap the button below to share your location for nearby store ideas."
        combined = f"{combined}\n\n{loc_note}"
        return combined.strip(), "request_location"

    buttons = resource_suggested_buttons(data.get("suggested_buttons"))
    return combined.strip(), buttons
