from pathlib import Path
import time

from agent import NutritionAgent
from prompts import WELCOME_FALLBACK_MESSAGE, WELCOME_BUTTONS

from wa_service_sdk import (
    BaseEvent,
    TextEvent,
    InteractiveEvent,
    LocationEvent,
    ImageEvent,
    create_message,
    create_buttoned_message,
    create_location_request_message,
    Button,
    MediaDownloadError,
    MediaExpiredError,
    MediaTooLargeError,
    MediaUnavailableError,
    download_media,
    save_media_bytes,
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
    elif isinstance(event, ImageEvent):
        if not event.media_uri:
            text = "I couldn't access that image. Please try sending it again."
            buttons = _make_buttons(WELCOME_BUTTONS)
        else:
            image_path: Path | None = None
            t0 = time.monotonic()
            try:
                t_dl = time.monotonic()
                media_bytes = download_media(event.media_uri)
                t_dl_done = time.monotonic()
                image_path = save_media_bytes(
                    media_bytes,
                    media_id=event.image_id,
                    media_uri=event.media_uri,
                    file_extension=event.file_extension,
                    mime_type=event.mime_type,
                )
                t_save_done = time.monotonic()
                text, buttons = _agent.run_image(
                    str(image_path),
                    event.user_id,
                    caption=event.caption,
                    mime_type=event.mime_type,
                )
                t_done = time.monotonic()
                print(
                    "[DEBUG] image timings "
                    f"user_id={event.user_id} "
                    f"download_ms={int((t_dl_done - t_dl) * 1000)} "
                    f"save_ms={int((t_save_done - t_dl_done) * 1000)} "
                    f"agent_ms={int((t_done - t_save_done) * 1000)} "
                    f"total_ms={int((t_done - t0) * 1000)}"
                )
            except (MediaExpiredError, MediaUnavailableError):
                text = "That image is no longer available. Please send it again."
                buttons = _make_buttons(WELCOME_BUTTONS)
            except (MediaDownloadError, MediaTooLargeError):
                text = "I couldn't download that image. Please try again with a smaller or clearer photo."
                buttons = _make_buttons(WELCOME_BUTTONS)
            except Exception:
                t_err = time.monotonic()
                print(
                    "[DEBUG] image error timing "
                    f"user_id={event.user_id} total_ms={int((t_err - t0) * 1000)}"
                )
                text = "I couldn't analyze that image right now. Please try again in a moment."
                buttons = _make_buttons(WELCOME_BUTTONS)
            finally:
                if image_path and image_path.exists():
                    image_path.unlink(missing_ok=True)
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
