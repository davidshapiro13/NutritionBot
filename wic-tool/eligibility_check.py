"""
eligibility_check.py
====================
WIC eligibility screening flow for Massachusetts.

Steps:
    Q1  — Population criteria (pregnant / breastfeeding / child under 5)
    Q2  — Massachusetts residency
    Q3  — Auto-qualify via SNAP / MassHealth / TAFDC
    Q4a — Household size (only if Q3 = No)
    Q4b — Household income (only if Q3 = No)

Public API:
    start_eligibility_check(user_id)          → (message, buttons)
    handle_eligibility_answer(user_id, answer) → (message, buttons, done, memory_lines)
    is_in_eligibility_flow(user_id)            → bool
    clear_eligibility_session(user_id)         → None

When done=True, memory_lines is a list of "field: value" strings for UserMemory.save();
otherwise memory_lines is None.
"""

from __future__ import annotations

_INCOME_LIMITS: dict[int, int] = {
    1: 28953,
    2: 39128,
    3: 49303,
    4: 59478,
    5: 69653,
    6: 79828,
    7: 90003,
    8: 100178
}


def _income_limit_for(household_size: int) -> int:
    if household_size <= 5:
        return _INCOME_LIMITS[household_size]
    return _INCOME_LIMITS[5] + (household_size - 5) * 10175


_INTRO = (
    "I'll ask a few quick questions to see if you or someone you're "
    "helping may qualify for WIC in Massachusetts."
)

_Q1 = "Are you or someone in your household:\n\n• Pregnant\n• Breastfeeding\n• A child under age 5"
_Q1_BUTTONS = [
    {"id": "elig_q1_yes", "title": "✅ Yes"},
    {"id": "elig_q1_no",  "title": "❌ No"},
]

_Q2 = "Do you live in Massachusetts?"
_Q2_BUTTONS = [
    {"id": "elig_q2_yes", "title": "✅ Yes"},
    {"id": "elig_q2_no",  "title": "❌ No"},
]

_Q3 = (
    "Are you currently enrolled in any of these programs?\n\n"
    "• SNAP\n• MassHealth (Medicaid)\n• TAFDC / Cash Assistance"
)
_Q3_BUTTONS = [
    {"id": "elig_q3_yes", "title": "✅ Yes, I am"},
    {"id": "elig_q3_no",  "title": "❌ No"},
]

_Q4A = "How many people are in your household? (Please type a number, e.g. 3)"
_Q4A_BUTTONS = []


def _q4b_message(household_size: int) -> str:
    limit = _income_limit_for(household_size)
    return (
        f"What is your approximate yearly household income?\n\n"
        f"(WIC income limit for {household_size} "
        f"{'person' if household_size == 1 else 'people'}: "
        f"${limit:,}/year)"
    )


def _q4b_buttons(household_size: int) -> list[dict]:
    limit = _income_limit_for(household_size)
    return [
        {"id": "elig_inc_under", "title": f"Under ${limit:,}"},
        {"id": "elig_inc_over",  "title": f"${limit:,} or more"},
    ]

_NOT_ELIGIBLE_POPULATION = (
    "Based on your answers, WIC is currently available for pregnant women, "
    "breastfeeding mothers, and children under 5. It looks like no one in your "
    "household falls into these categories right now.\n\n"
    "There are other food assistance programs in Massachusetts that may help — "
    "would you like me to tell you about SNAP or local food resources?"
)

_NOT_ELIGIBLE_RESIDENCY = (
    "WIC benefits in Massachusetts are only available to Massachusetts residents. "
    "If you live in another state, please check your state's WIC program.\n\n"
    "Is there anything else I can help you with?"
)

_NOT_ELIGIBLE_INCOME = (
    "Based on your household size and income, you may not currently qualify for WIC.\n\n"
    "There are still other resources that may help — would you like me to tell you "
    "about SNAP, food pantries, or other Massachusetts food assistance programs?"
)

_ELIGIBLE_AUTO = (
    "Great news! Because you're enrolled in SNAP, MassHealth, or TAFDC, "
    "you automatically qualify for WIC in Massachusetts.\n\n"
    "You can apply at your local WIC office or visit mass.gov/wic to get started. "
    "Would you like help finding a WIC office near you?"
)

_ELIGIBLE_INCOME = (
    "Based on your answers, you may qualify for WIC in Massachusetts!\n\n"
    "You can apply at your local WIC office or visit mass.gov/wic to get started. "
    "Would you like help finding a WIC office near you?"
)

_ELIGIBLE_BUTTONS = [
    {"id": "find_wic_stores", "title": "📍 Find WIC Office"},
    {"id": "wic_apply",       "title": "ℹ️ How to Apply"},
]

_NOT_ELIGIBLE_BUTTONS = [
    {"id": "check_eligibility",  "title": "🔄 Try Again"},
]


def _mem_q1_no() -> list[str]:
    return [
        "wic_screening_population_match: no",
        "wic_screening_outcome: not_eligible_population",
    ]


def _mem_q2_no() -> list[str]:
    return [
        "wic_screening_population_match: yes",
        "wic_screening_ma_resident: no",
        "wic_screening_outcome: not_eligible_residency",
    ]


def _mem_q3_yes() -> list[str]:
    return [
        "wic_screening_population_match: yes",
        "wic_screening_ma_resident: yes",
        "wic_screening_public_assistance: yes",
        "wic_screening_household_size: not_asked",
        "wic_screening_income_vs_limit: not_asked",
        "wic_screening_outcome: eligible_auto",
    ]


def _mem_q4b(household_size: int, *, income_under: bool, eligible: bool) -> list[str]:
    outcome = "eligible_income" if eligible else "not_eligible_income"
    income_band = "under" if income_under else "over"
    return [
        "wic_screening_population_match: yes",
        "wic_screening_ma_resident: yes",
        "wic_screening_public_assistance: no",
        f"wic_screening_household_size: {household_size}",
        f"wic_screening_income_vs_limit: {income_band}",
        f"wic_screening_outcome: {outcome}",
    ]


_sessions: dict[str, dict] = {}


def start_eligibility_check(user_id: str) -> tuple[str, list[dict]]:
    _sessions[user_id] = {"step": "q1", "household_size": None}
    return f"{_INTRO}\n\n{_Q1}", _Q1_BUTTONS


def handle_eligibility_answer(
    user_id: str,
    answer_id: str,
) -> tuple[str, list[dict], bool, list[str] | None]:
    session = _sessions.get(user_id)
    if session is None:
        msg, btns = start_eligibility_check(user_id)
        return msg, btns, False, None

    step = session["step"]

    if step == "q1":
        if answer_id == "elig_q1_yes":
            session["step"] = "q2"
            return _Q2, _Q2_BUTTONS, False, None
        _sessions.pop(user_id, None)
        return _NOT_ELIGIBLE_POPULATION, _NOT_ELIGIBLE_BUTTONS, True, _mem_q1_no()

    if step == "q2":
        if answer_id == "elig_q2_yes":
            session["step"] = "q3"
            return _Q3, _Q3_BUTTONS, False, None
        _sessions.pop(user_id, None)
        return _NOT_ELIGIBLE_RESIDENCY, _NOT_ELIGIBLE_BUTTONS, True, _mem_q2_no()

    if step == "q3":
        if answer_id == "elig_q3_yes":
            _sessions.pop(user_id, None)
            return _ELIGIBLE_AUTO, _ELIGIBLE_BUTTONS, True, _mem_q3_yes()
        session["step"] = "q4a"
        return _Q4A, _Q4A_BUTTONS, False, None

    if step == "q4a":
        try:
            household_size = max(1, int(answer_id.strip()))
        except ValueError:
            return (
                "Please type a number (e.g. 2). How many people are in your household?",
                [],
                False,
                None,
            )
        session["household_size"] = household_size
        session["step"] = "q4b"
        msg = _q4b_message(household_size)
        btns = _q4b_buttons(household_size)
        return msg, btns, False, None

    if step == "q4b":
        hs = session.get("household_size") or 1
        _sessions.pop(user_id, None)
        if answer_id == "elig_inc_under":
            return (
                _ELIGIBLE_INCOME,
                _ELIGIBLE_BUTTONS,
                True,
                _mem_q4b(hs, income_under=True, eligible=True),
            )
        return (
            _NOT_ELIGIBLE_INCOME,
            _NOT_ELIGIBLE_BUTTONS,
            True,
            _mem_q4b(hs, income_under=False, eligible=False),
        )

    msg, btns = start_eligibility_check(user_id)
    return msg, btns, False, None


def is_in_eligibility_flow(user_id: str) -> bool:
    return user_id in _sessions


def clear_eligibility_session(user_id: str) -> None:
    _sessions.pop(user_id, None)
