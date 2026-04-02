from agent import NutritionAgent
from prompts import WELCOME_FALLBACK_MESSAGE, WELCOME_BUTTONS

from wa_service_sdk import (
    BaseEvent,
    TextEvent,
    InteractiveEvent,
    LocationEvent,
    create_message,
    create_buttoned_message,
    create_location_request_message,
    Button,
)

_agent = NutritionAgent()

_MAX_MESSAGE_CHARS = 900


def _truncate_preserving_sources(text: str, max_len: int = _MAX_MESSAGE_CHARS) -> str:
    """Keep trailing ``Sources:`` block when trimming long replies."""
    if len(text) <= max_len:
        return text
    marker = "\n\nSources:"
    idx = text.find(marker)
    if idx >= 0:
        body, suffix = text[:idx], text[idx:]
        budget = max_len - len(suffix)
        if budget >= 40:
            trimmed = body[:budget].rstrip()
            if len(trimmed) < len(body):
                trimmed = trimmed.rstrip(".,; ") + "…"
            return trimmed + suffix
        if len(suffix) <= max_len:
            return suffix.lstrip("\n")
    return text[: max_len - 1] + "…"


def _make_buttons(buttons_data: list[dict]) -> list[Button]:
    return [Button(id=b["id"], title=b["title"]) for b in buttons_data]


async def handle_event(event: BaseEvent):
    if isinstance(event, TextEvent):
        text, buttons = _agent.run(event.text, event.user_id)

    elif isinstance(event, InteractiveEvent):
        text, buttons = _agent.run_tool(
            event.interaction_id,
            event.user_id,
            interaction_title=event.interaction_title,
        )

    elif isinstance(event, LocationEvent):
        text, buttons = _agent.run_location(
            event.latitude, event.longitude, event.user_id
        )

    else:
        text = WELCOME_FALLBACK_MESSAGE
        buttons = _make_buttons(WELCOME_BUTTONS)

    if not text:
        text = WELCOME_FALLBACK_MESSAGE
        buttons = _make_buttons(WELCOME_BUTTONS)

    text = _truncate_preserving_sources(text, _MAX_MESSAGE_CHARS)

    if buttons == "request_location":
        return create_location_request_message(user_id=event.user_id, text=text)
    if buttons:
        return create_buttoned_message(
            user_id=event.user_id,
            text=text,
            buttons=buttons,
        )
    return create_message(user_id=event.user_id, text=text)
