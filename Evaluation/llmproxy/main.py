from __future__ import annotations

import json
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


from dotenv import load_dotenv


# -----------------------
# Config & HTTP utilities
# -----------------------

@dataclass(frozen=True)
class ClientConfig:
    endpoint: str
    api_key: str
    timeout: float = 118.0  # seconds, applied to both connect & read


    @staticmethod
    def from_env() -> "ClientConfig":

        # Explicitly load .env from current working directory
        cwd_env = Path.cwd() / ".env"
        load_dotenv(dotenv_path=cwd_env, override=True)

        endpoint = os.getenv("LLMPROXY_ENDPOINT")
        api_key  = os.getenv("LLMPROXY_API_KEY")

        if not endpoint or not api_key:
            raise ValueError(
                "LLMProxy configuration error:\n"
                "Missing LLMPROXY_ENDPOINT or LLMPROXY_API_KEY.\n\n"
                "Make sure your .env file is in the SAME DIRECTORY where you run python.\n"
                "\nExample .env:\n"
                "    LLMPROXY_ENDPOINT=https://your-endpoint\n"
                "    LLMPROXY_API_KEY=your-api-key\n"
            )

        return ClientConfig(endpoint=endpoint, api_key=api_key)




def _build_session() -> requests.Session:
    """Session with retries and connection pooling."""
    s = requests.Session()
    retries = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


# -----------------------
# Core client
# -----------------------

class LLMProxy:
    def __init__(self) -> None:
        self.config = ClientConfig.from_env()
        self.session = _build_session()

    def _headers(self, request_type: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        base = {
            "x-api-key": self.config.api_key,
            "request_type": request_type,
        }
        if extra:
            base.update(extra)
        return base

    def _post_json(
        self,
        request_type: str,
        payload: Dict[str, Any],
    ) -> Dict:
        # Remove None values to avoid sending nulls unnecessarily
        clean_payload = {k: v for k, v in payload.items() if v is not None}

        try:
            resp = self.session.post(
                self.config.endpoint,
                headers=self._headers(request_type),
                json=clean_payload,
                timeout=self.config.timeout,
            )
        except requests.exceptions.RequestException as e:
            return {"error": f"Network error: {e}", "status_code": None}

        if 200 <= resp.status_code < 300:
            try:
                return resp.json()
            except ValueError:
                # JSON decode failed; return text for visibility
                return {"error": "Invalid JSON in response", "status_code": resp.status_code}
        else:
            # Try to surface server-provided error details
            detail: str
            try:
                detail = resp.json().get("error", resp.text)
            except ValueError:
                detail = resp.text
            return {"error": f"HTTP {resp.status_code}: {detail}", "status_code": resp.status_code}

    # -------- Public methods --------

    def retrieve(
        self,
        query: str,
        session_id: str,
        rag_threshold: float,
        rag_k: int,
    ) -> Dict:
        """
        Calls the retrieval endpoint. Returns server JSON
        """
        payload = {
            "query": query,
            "session_id": session_id,
            "rag_threshold": rag_threshold,
            "rag_k": rag_k,
        }
        return self._post_json("retrieve", payload)

    def model_info(self) -> Dict:
        """
        Fetches model info.
        """
        return self._post_json("model_info", {})

    def generate(
        self,
        model: str,
        system: str,
        query: str,
        websearch: bool = False,
        output_schema: Optional[Any] = None,
        temperature: Optional[float] = None,
        lastk: Optional[int] = None,
        session_id: Optional[str] = "GenericSession",
        rag_threshold: Optional[float] = 0.5,
        rag_usage: Optional[bool] = False,
        rag_k: Optional[int] = 5,
        media: Optional[List[Dict[str, str]]] = None,
    ) -> Dict:
        """
        Calls the generation endpoint.
        - Text-only call: unchanged behavior.
        - Media call: expects pre-uploaded media refs in the shape
          `{"id": "...", "type": "..."}`.
        """
        resolved_session_id = session_id or "GenericSession"
        media_refs = self._normalize_media_refs(media)
        if isinstance(media_refs, dict) and "error" in media_refs:
            return media_refs

        options = {"websearch": bool(websearch)}
        if output_schema is not None:
            model_json_schema = getattr(output_schema, "model_json_schema", None)
            if not callable(model_json_schema):
                return {
                    "error": "Invalid output_schema: expected a Pydantic model class with model_json_schema().",
                    "status_code": None,
                }
            options["output_schema"] = model_json_schema()

        payload = {
            "model": model,
            "system": system,
            "query": query,
            "options": options,
            "temperature": temperature,
            "lastk": lastk,
            "session_id": resolved_session_id,
            "rag_threshold": rag_threshold,
            "rag_usage": rag_usage,
            "rag_k": rag_k,
            "media": media_refs,
        }

        res = self._post_json("call", payload)
        if "error" in res:
            return res
        # Defensive extraction
        return res
        # result_text = res.get("result")
        # rag_context = res.get("rag_context")
        # return {"response": result_text, "rag_context": rag_context, "raw": res}

    def upload_file(
        self,
        file_path: Union[str, Path],
        session_id: str,
        mime_type: str = None,
        description: Optional[str] = None,
        strategy: Optional[str] = "smart",
    ) -> Dict:
        """
        Generic uploader for any file. Uses streaming upload and returns server JSON or error.
        """
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {path}", "status_code": None}

        if mime_type is None:
            # Minimal sniffing; caller can override
            mime_type = "application/pdf" if path.suffix.lower() == ".pdf" else "application/octet-stream"

        params = {
            "description": description,
            "session_id": session_id,
            "strategy": strategy,
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        files = {
            # Include a filename so the server can store it meaningfully
            "params": (None, json.dumps(params), "application/json"),
            "file": (None, path.open("rb"), mime_type),
        }

        try:
            resp = self.session.post(
                self.config.endpoint,
                headers=self._headers("add"),
                files=files,
                timeout=self.config.timeout,
            )
        except requests.exceptions.RequestException as e:
            return {"error": f"Network error: {e}", "status_code": None}

        if 200 <= resp.status_code < 300:
            try:
                return resp.json()
            except ValueError:
                # If server returns plain text success
                return {"message": resp.text}
        else:
            try:
                detail = resp.json().get("error", resp.text)
            except ValueError:
                detail = resp.text
            return {"error": f"HTTP {resp.status_code}: {detail}", "status_code": resp.status_code}

    def upload_init(
        self,
        content_type: str,
        session_id: str,
        size_bytes: int,
    ) -> Dict:
        """
        Step 1 of URI upload flow: request an assigned/presigned upload URI.
        """
        payload = {
            "content_type": content_type,
            "session_id": session_id,
            "size_bytes": size_bytes,
        }
        return self._post_json("upload_init", payload)

    def upload_via_uri(
        self,
        upload_url: str,
        file_path: Union[str, Path],
        content_type: Optional[str] = None,
    ) -> Dict:
        """
        Step 2 of URI upload flow: PUT file contents to assigned/presigned URI.
        """
        path = Path(file_path)
        if not path.exists():
            return {"ok": False, "error": f"File not found: {path}", "status_code": None}

        resolved_content_type = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"

        try:
            with path.open("rb") as f:
                resp = self.session.put(
                    upload_url,
                    data=f,
                    headers={"Content-Type": resolved_content_type},
                    timeout=self.config.timeout,
                )
        except requests.exceptions.RequestException as e:
            return {"ok": False, "error": f"Network error: {e}", "status_code": None}

        return {
            "ok": 200 <= resp.status_code < 300,
            "status_code": resp.status_code,
            "response_text": resp.text[:500],
            "content_type": resolved_content_type,
            "file_path": str(path),
        }

    def upload_media(
        self,
        file_path: Union[str, Path],
        session_id: str,
        content_type: str,
    ) -> Dict:
        """
        Upload media once and return a reusable media reference for `generate(...)`.
        """
        if not self._is_supported_media_type(content_type):
            return {
                "error": f"Unsupported media type: {content_type}. Only image/* and audio/* are supported",
                "status_code": None,
            }

        upload_result = self._upload_media(
            file_path=file_path,
            session_id=session_id,
            content_type=content_type,
        )
        if "error" in upload_result or not upload_result.get("ok"):
            return {
                "error": "Media upload failed",
                "status_code": upload_result.get("status_code"),
                "upload": upload_result,
            }

        media_id = self._extract_media_id(upload_result.get("upload_init", {}))
        if not media_id:
            return {
                "error": "upload_init did not return a media identifier",
                "upload": upload_result,
                "status_code": None,
            }

        return {
            "ok": True,
            "id": media_id,
            "type": content_type,
            "upload": upload_result,
        }

    def _upload_media(
        self,
        file_path: Union[str, Path],
        session_id: str,
        content_type: str,
    ) -> Dict:
        """
        Convenience wrapper for URI upload flow:
        1) request assigned URI (upload_init)
        2) upload file bytes to URI (PUT)
        """
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {path}", "status_code": None}

        if not content_type:
            return {"error": "content_type is required (for example: image/jpeg)", "status_code": None}

        resolved_content_type = content_type
        size_bytes = path.stat().st_size

        init_result = self.upload_init(
            content_type=resolved_content_type,
            session_id=session_id,
            size_bytes=size_bytes,
        )
        if "error" in init_result:
            return {"error": "upload_init failed", "upload_init": init_result}

        upload_url = self._extract_upload_url(init_result)
        if not upload_url:
            return {
                "error": "upload_init did not return an upload URL",
                "upload_init": init_result,
            }

        upload_result = self.upload_via_uri(
            upload_url=upload_url,
            file_path=path,
            content_type=resolved_content_type,
        )

        return {
            "ok": bool(upload_result.get("ok")),
            "upload_init": init_result,
            "upload_result": upload_result,
            "content_type": resolved_content_type,
            "size_bytes": size_bytes,
        }

    @staticmethod
    def _extract_upload_url(payload: Dict[str, Any]) -> Optional[str]:
        for key in ("upload_url", "uri", "signed_url", "presigned_url", "url"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    @staticmethod
    def _extract_media_id(payload: Dict[str, Any]) -> Optional[str]:
        value = payload.get("media_id")
        if isinstance(value, str) and value:
            return value
        return None

    def _normalize_media_refs(
        self,
        media: Optional[List[Dict[str, str]]],
    ) -> Union[Optional[List[Dict[str, str]]], Dict[str, Any]]:
        if not media:
            return None

        media_refs: List[Dict[str, str]] = []
        for idx, item in enumerate(media):
            media_id = item.get("id")
            content_type = item.get("type")

            if not media_id or not content_type:
                return {
                    "error": f"Invalid media item at index {idx}: expected keys `id` and `type`",
                    "status_code": None,
                }

            if not self._is_supported_media_type(content_type):
                return {
                    "error": f"Unsupported media type at index {idx}: {content_type}. Only image/* and audio/* are supported",
                    "status_code": None,
                }

            media_refs.append({"id": media_id, "type": content_type})

        return media_refs

    @staticmethod
    def _is_supported_media_type(content_type: str) -> bool:
        return content_type.startswith("image/") or content_type.startswith("audio/")

    def upload_text(
        self,
        text: str,
        session_id: str,
        description: Optional[str] = None,
        strategy: Optional[str] = "smart",
    ) -> Dict:
        """
        Uploads raw text content as a 'file' part.
        """
        params = {
            "description": description,
            "session_id": session_id,
            "strategy": strategy,
        }
        params = {k: v for k, v in params.items() if v is not None}


        files = {
            "params": (None, json.dumps(params), "application/json"),            
            "text": (None, text, "application/text"),
        }

        try:
            resp = self.session.post(
                self.config.endpoint,
                headers=self._headers("add"),
                files=files,
                timeout=self.config.timeout,
            )
        except requests.exceptions.RequestException as e:
            return {"error": f"Network error: {e}", "status_code": None}

        if 200 <= resp.status_code < 300:
            try:
                return resp.json()
            except ValueError:
                return {"message": resp.text}
        else:
            try:
                detail = resp.json().get("error", resp.text)
            except ValueError:
                detail = resp.text
            return {"error": f"HTTP {resp.status_code}: {detail}", "status_code": resp.status_code}
