import argparse
import mimetypes
from pathlib import Path

from llmproxy import LLMProxy
from prompts import image_analysis_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Test LLMProxy image processing without WhatsApp.")
    parser.add_argument("image_path", help="Path to a local image file.")
    parser.add_argument("--caption", default="", help="Optional caption to include in the prompt.")
    parser.add_argument("--session-id", default="test-image-session", help="Session id to reuse.")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise SystemExit(f"File not found: {image_path}")

    content_type = mimetypes.guess_type(image_path.name)[0]
    if not content_type or not content_type.startswith("image/"):
        raise SystemExit("Unsupported image type. Use a JPG, PNG, or HEIC file.")

    client = LLMProxy()
    upload = client.upload_media(
        file_path=str(image_path),
        session_id=args.session_id,
        content_type=content_type,
    )
    print("upload:", upload)
    if not upload.get("ok"):
        raise SystemExit("upload failed")

    media = [{"id": upload["id"], "type": upload["type"]}]
    query = f"[USER CAPTION]\n{args.caption.strip() or '(no caption provided)'}"
    res = client.generate(
        model="gpt-5-mini",
        system=image_analysis_prompt,
        query=query,
        session_id=args.session_id,
        media=media,
    )
    print("generate:", res)


if __name__ == "__main__":
    main()
