from agent import NutritionAgent
from prompts import WELCOME_MESSAGE, WELCOME_BUTTONS

from wa_service_sdk import (
    BaseEvent,
    TextEvent,
    InteractiveEvent,
    LocationEvent,
    create_message,
    create_buttoned_message,
    Button,
)

_agent = NutritionAgent()


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
        text = WELCOME_MESSAGE
        buttons = _make_buttons(WELCOME_BUTTONS)

    if not text:
        text = WELCOME_MESSAGE
        buttons = _make_buttons(WELCOME_BUTTONS)

    if len(text) > 900:
        text = text[:897] + "…"

    if buttons:
        return create_buttoned_message(
            user_id=event.user_id,
            text=text,
            buttons=buttons,
        )
    return create_message(user_id=event.user_id, text=text)
