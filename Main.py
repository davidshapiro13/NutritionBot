import os
import sys
import logging
import socket
import importlib
import importlib.util
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from pyngrok import ngrok


ROOT = Path(__file__).resolve().parent
SDK_PATH = ROOT / "sdk"
ENV_PATH = ROOT / ".env"
ENV_EXAMPLE_PATH = ROOT / ".env.example"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SDK_PATH) not in sys.path:
    sys.path.insert(0, str(SDK_PATH))

def _ensure_env_loaded() -> None:
    if not ENV_PATH.exists() and ENV_EXAMPLE_PATH.exists():
        ENV_PATH.write_text(ENV_EXAMPLE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        raise RuntimeError("Missing .env. A template was created. Set NGROK_AUTH_TOKEN, then run again.")
    load_dotenv(dotenv_path=ENV_PATH)


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    if value.startswith("YOUR_"):
        raise RuntimeError(f"{name} is still using a placeholder value")
    return value


def _load_module_attr(ref: str) -> Any:
    if ":" not in ref:
        raise RuntimeError("Invalid handler reference. Use module:function (e.g. my_pkg.my_mod:handle_event).")
    module_name, attr_name = ref.split(":", 1)
    if not module_name or not attr_name:
        raise RuntimeError("Invalid handler reference. Use module:function (e.g. my_pkg.my_mod:handle_event).")
    module = importlib.import_module(module_name)
    if not hasattr(module, attr_name):
        raise RuntimeError(f"Handler attribute not found: {attr_name} in module {module_name}")
    return getattr(module, attr_name)


def _load_module_from_file(path: str) -> Any:
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise RuntimeError(f"Target file not found: {path}")
    module_name = f"_wa_cli_target_{file_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load target file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_app_from_handler(handler_func: Any, webhook_path: str):
    if not callable(handler_func):
        raise RuntimeError("Resolved handler is not callable")
    from wa_service_sdk import create_app

    return create_app(handler_func, path=webhook_path)


def _load_app_from_module(module: Any, webhook_path: str):
    if hasattr(module, "handle_event"):
        return _build_app_from_handler(getattr(module, "handle_event"), webhook_path), "handle_event (CLI-bound path)"
    if hasattr(module, "app"):
        return getattr(module, "app"), "app object (module-defined path)"
    raise RuntimeError("Target module must expose `app` or `handle_event`")


def _load_app(*, target: str, webhook_path: str):
    if ":" in target and not target.endswith(".py"):
        handler_func = _load_module_attr(target)
        return _build_app_from_handler(handler_func, webhook_path), f"handler: {target}"

    if target.endswith(".py") or "/" in target:
        module = _load_module_from_file(target)
        app, source = _load_app_from_module(module, webhook_path)
        return app, f"file target ({source}): {target}"

    module = importlib.import_module(target)
    app, source = _load_app_from_module(module, webhook_path)
    return app, f"module target ({source}): {target}"


def _resolve_port() -> int:
    configured = os.getenv("PORT", "").strip()
    if configured:
        return int(configured)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main(*, target: str = "examples/simple_app.py", webhook_path: str = "/webhook") -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("wa_service_sdk").setLevel(logging.INFO)
    logging.getLogger("pyngrok").setLevel(logging.WARNING)
    logging.getLogger("pyngrok.process").setLevel(logging.WARNING)
    _ensure_env_loaded()

    port = _resolve_port()
    auth_token = os.getenv("NGROK_AUTH_TOKEN", "").strip()
    if not webhook_path.startswith("/"):
        raise RuntimeError("webhook_path must start with '/'")

    # Optional: if token is not provided here, ngrok can use global config.
    if auth_token:
        if auth_token.startswith("YOUR_"):
            raise RuntimeError("NGROK_AUTH_TOKEN is still using a placeholder value")
        ngrok.set_auth_token(auth_token)

    tunnel = ngrok.connect(addr=port)

    print(f"Public URL: {tunnel.public_url}")
    print(f"Webhook URL: {tunnel.public_url}{webhook_path}")
    print("Health URL:", f"{tunnel.public_url}/health")

    app, run_mode = _load_app(target=target, webhook_path=webhook_path)
    print(f"Running app via {run_mode}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
