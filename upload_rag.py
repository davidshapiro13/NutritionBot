"""
One-time script to upload the public knowledge base to LLMProxy.
Run this once before starting the bot:
    python upload_rag.py
"""
import os
from pathlib import Path
from llmproxy import LLMProxy

RAG_DATA_DIR = Path(__file__).parent / "rag_data"

# Fixed session ID shared by all users for the public knowledge base
PUBLIC_SESSION = "NutritionBot_PublicKB_v1"

def upload_knowledge_base():
    client = LLMProxy()
    uploaded = 0

    for file_path in sorted(RAG_DATA_DIR.glob("*.txt")):
        print(f"Uploading: {file_path.name} ...")
        result = client.upload_file(
            file_path=file_path,
            session_id=PUBLIC_SESSION,
            mime_type="text/plain",
            description=f"NutritionBot public knowledge base: {file_path.stem}",
            strategy="smart",
        )
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  OK: {result}")
            uploaded += 1

    print(f"\nDone. {uploaded}/{len(list(RAG_DATA_DIR.glob('*.txt')))} file(s) uploaded to session '{PUBLIC_SESSION}'.")

if __name__ == "__main__":
    upload_knowledge_base()
