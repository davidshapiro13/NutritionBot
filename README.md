# WhatsApp SDK

![WhatsApp](https://img.shields.io/badge/WhatsApp-SDK-25D366?style=flat-square&logo=whatsapp&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Ready-009688?style=flat-square&logo=fastapi&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-28%20passing-brightgreen?style=flat-square)

Lightweight toolkit to build WhatsApp service handlers locally, expose them with ngrok, and register service modes.

## Contents

- [1. Intuition](#1-intuition)
- [2. Build Your Own Service](#2-build-your-own-service)
- [3. Quick Start](#3-quick-start)
- [4. Mode and Registration Model](#4-mode-and-registration-model)
- [5. CLI Guide](#5-cli-guide)
- [6. Supported Types](#6-supported-types)
- [7. Examples](#7-examples)
- [8. Not Supported Yet](#8-not-supported-yet)
- [9. Troubleshooting](#9-troubleshooting)
- [10. Repo Layout](#10-repo-layout)
- [11. Main WhatsApp QR](#11-main-whatsapp-qr)

## 1. Intuition

Think in two layers:

1. SDK parses inbound events into typed Python objects (`TextEvent`, `ReactionEvent`, etc.).
2. Your handler returns a response envelope using SDK helpers (`create_message`, `create_list_message`, etc.).

You only write handler logic. Runtime + tunnel + registration are CLI workflows.

## 2. Build Your Own Service

Your service is a single handler function:

```python
from wa_service_sdk import BaseEvent, TextEvent, create_message

async def handle_event(event: BaseEvent):
    if isinstance(event, TextEvent):
        return create_message(user_id=event.user_id, text=f"You said: {event.text}")
    return None
```

Handler contract:

- input: one parsed SDK event object (`TextEvent`, `ReactionEvent`, `LocationEvent`, etc.)
- output: either `None` or a response dict built by SDK helpers
- path binding: done by CLI (`wa_cli run --path /your_path`), not in the handler file

Sandbox model for students:

- treat your handler as a pure service function (read event -> return response)
- run locally with `wa_cli run`
- expose publicly through ngrok automatically
- register mode + endpoint through `wa_cli register`

## 3. Quick Start

```bash
cd <repo-dir>
pip3 install -r requirements.txt
pip3 install -e .
cp .env.example .env
```

Set `.env`:

- `BASE_URL` (registration API base URL, obtain from author)
- `API_KEY` (registration API key, obtain from author)
- `NGROK_AUTH_TOKEN` (optional if already configured globally)

### 3.1 Install and Set Up ngrok (First Time Only)

If you have never used ngrok before, do this once before running the CLI.

1. Create a free account at [ngrok.com](https://ngrok.com/).
2. Download ngrok from [ngrok.com/download](https://ngrok.com/download) for your OS.
3. Install ngrok and make sure the `ngrok` command works in your terminal.
4. In the ngrok dashboard, copy your auth token.
5. Add your token to ngrok:

```bash
ngrok config add-authtoken <YOUR_NGROK_AUTH_TOKEN>
```

You can also put the same token in `.env` as `NGROK_AUTH_TOKEN`.

Quick check:

```bash
ngrok version
```

If this prints a version number, ngrok is installed correctly.

Run server + tunnel:

```bash
wa_cli run --path /webhook
```

Register your mode:

```bash
wa_cli register --mode newbot --endpoint "https://<ngrok-domain>/webhook"
```

Use the exact `Webhook URL` printed by `wa_cli run` in the previous step.

## 4. Mode and Registration Model

`mode` is the name of your WhatsApp service.

- Treat mode as unique to your service.
- Once users and integrations depend on a mode name, avoid changing it.
- If you need a rename, create a new mode and migrate traffic intentionally.

Why registration exists:

- Registration creates the routing pathway from the main WhatsApp service to your endpoint.
- Without registration, your handler will not receive traffic for that mode.

Endpoint update behavior:

- Your endpoint URL can and should be updated whenever your runtime URL changes (for example tunnel restart).
- Re-run registration with the latest `Webhook URL` to keep routing current.

## 5. CLI Guide

### Run

```bash
wa_cli run [target] --path /webhook
```

- `target` optional, defaults to `examples/simple_app.py`
- target supports:
  - file path: `examples/simple_app.py`
  - module path: `examples.simple_app`
  - handler ref: `examples.simple_app:handle_event`

### Register

```bash
wa_cli register --mode <mode> --endpoint "https://<ngrok-domain>/<path>"
wa_cli get --mode <mode>
wa_cli list
```

Registration rules:

- `mode` is your service name.
- Only one mode is allowed per team.
- You can change your mode name, but the new name must not collide with another service's mode.
- Your HTTPS endpoint can be updated any time by running `wa_cli register` again with the new URL.
- ngrok may show multiple URLs (for example public/root, webhook, health). Do not register the public/root or health URL.
- Always register the webhook URL that includes your handler path (for example `/webhook` or the custom path you passed to `wa_cli run --path ...`).

Copy the endpoint directly from the `Webhook URL` line shown when running `wa_cli run`.

If you use `--path /webhook_bus`, register endpoint with that exact path.

On successful registration, you should receive an HTTP `200` response.

## 6. Supported Types

### Inbound events (parsed by SDK)

| Type | Parsed Event Class | Key payload fields |
|---|---|---|
| `text` | `TextEvent` | `text` |
| `interactive` | `InteractiveEvent` | `interactive.type`, `interactive.<type>.id` |
| `reaction` | `ReactionEvent` | `reaction.emoji` |
| `reply` | `ReplyEvent` | `text` (or `reply.text/body`) |
| `image` | `ImageEvent` | `media.media_id` + `media.uri` (if downloadable) |
| `audio` | `AudioEvent` | `media.media_id` + `media.uri` (if downloadable) |
| `location` | `LocationEvent` | `location.latitude`, `location.longitude` (+ optional `location.name`, `location.address`, `location.url`) |

### Outbound helpers (built by SDK)

| Helper | Purpose |
|---|---|
| `create_message` | plain text message |
| `create_buttoned_message` | interactive reply buttons |
| `create_list_message` | interactive list (simple + advanced modes) |
| `create_location_request_message` | request user location |
| `create_interactive_message` | pass custom interactive payload |

## 7. Examples

### Basic text + buttons

```bash
wa_cli run examples/simple_app.py --path /webhook
```

### Media handling

```bash
wa_cli run examples/media_app.py --path /webhook_media
```

### Location request -> nearby bus-stop list

```bash
wa_cli run examples/location_bus_app.py --path /webhook_bus
```

`LocationEvent` fields available to your handler:

- `event.latitude` (float, required)
- `event.longitude` (float, required)
- `event.name` (optional)
- `event.address` (optional)
- `event.url` (optional)

### Reaction + reply context summary

```bash
wa_cli run examples/reaction_reply_app.py --path /webhook_rr
```

## 8. Not Supported Yet

Current SDK does not provide first-class parsers/helpers for these inbound message types yet:

- video
- document
- sticker
- contacts
- order/product/cart
- status/system events

You can still handle unmodeled interactive outputs through `create_interactive_message(...)`.

## 9. Troubleshooting

### `404 Not Found`

Path mismatch between `wa_cli run --path ...` and registered endpoint path.

### `422 Unsupported event type`

Inbound `message_type/type` is not in SDK registry.

### `400 Missing or invalid field`

Payload shape missing required fields for that type.

### Tunnel URL changed

Re-run `wa_cli register` with the latest ngrok URL.

## 10. Repo Layout

- `sdk/wa_service_sdk`: SDK models, parsers, response builders
- `examples/`: runnable handler examples
- `wa_cli.py`: CLI entrypoint
- `main.py`: runtime launcher + tunnel wiring
- `mode_registry.py`: mode registration client

## 11. Main WhatsApp QR

Use this QR to quickly access the main WhatsApp service for testing.

### Want to Test Your Service?

Awesome, this is the fun part. Here is the fastest way:

1. Start your service locally with `wa_cli run --path /your_path`.
2. Register your mode with `wa_cli register --mode <your_mode> --endpoint "https://<ngrok-domain>/your_path"`.
3. Scan the QR below from your phone.
4. Open the main WhatsApp chat, type `@mode`, and look at the suggestion list shown below the message box.
5. Tap your service from that list, then send a message and confirm your bot replies as expected.
6. To exit/try another service, simple resend the `@mode` command on WhatsApp

If you restart your app and the ngrok URL changes, re-run `wa_cli register` with the new HTTPS endpoint.

You can use it to:

- test your own service mode end-to-end
- try other services built by peers
- demo and compare experiences together

![Main WhatsApp Service QR](assets/main_service_qr.png)

Once your service is working, share it with your peers and have them try it.

## Author

Abdullah Bin Faisal  
abdullah@cs.tufts.edu

## License

MIT
