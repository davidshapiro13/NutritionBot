"""
User Memory for NutritionBot
=============================
Manages per-user profile and health information extracted from conversations.

Each user gets a plain-text file at user_memory/<user_id>.txt that accumulates
structured facts over time. A lightweight FAISS index is built on demand so the
most relevant memories can be retrieved for any given question.

Tracked fields (extracted automatically after each conversation turn):
  [User Profile]
    name              - user's name or the name of the person they're asking for
    age_group         - child / adult / elder
    gender            - male / female / other
    asking_for        - self / child / parent / spouse / other

  [Health & Diet]
    health_conditions - e.g. heart disease, diabetes, pregnancy, hypertension
    allergies         - e.g. peanuts, dairy, shellfish, gluten
    medications       - e.g. metformin, warfarin, statins
    dietary_restriction - e.g. vegetarian, vegan, halal, kosher, gluten-free
    disliked_foods    - e.g. broccoli, spicy food

  [Goals]
    main_goal         - e.g. lose weight, manage diabetes, eat healthier

Usage:
    from user_memory import UserMemory
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    mem = UserMemory(embed_model=model)

    # After a conversation turn:
    mem.auto_extract_and_save("user_123", "I'm vegetarian and want to lose weight")

    # At query time:
    context = mem.get_context("user_123", "what should I eat for lunch?")
"""

import re
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from llmproxy import LLMProxy

# ── Path ─────────────────────────────────────────────────────────────────────
USER_MEM_DIR = Path(__file__).parent / "user_memory"


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS  (self-contained, no dependency on rag_pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════

def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _chunk(text: str, size: int = 450, overlap: int = 50) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size].strip())
        start += size - overlap
    return [c for c in chunks if c]


def _build_chunk_store(source: str, text: str) -> list[dict]:
    return [{"source": source, "text": c} for c in _chunk(text)]


def _build_index(chunks: list[dict], model: SentenceTransformer):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def _retrieve(query: str, model: SentenceTransformer, index, chunks: list[dict], top_k: int) -> list[dict]:
    if index.ntotal == 0:
        return []
    vec = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(vec, min(top_k, index.ntotal))
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx != -1:
            results.append({**chunks[idx], "distance": float(dist)})
    return results


# ══════════════════════════════════════════════════════════════════════════════
# USER MEMORY CLASS
# ══════════════════════════════════════════════════════════════════════════════

class UserMemory:
    """
    Per-user memory store backed by plain text files + FAISS retrieval.

    Args:
        embed_model: a SentenceTransformer instance (shared with RAGPipeline
                     to avoid loading the model twice).
    """

    def __init__(self, embed_model: SentenceTransformer):
        self._model = embed_model
        self._cache: dict[str, tuple] = {}   # {user_id: (faiss_index, chunks)}
        USER_MEM_DIR.mkdir(exist_ok=True)

    # ── Read / Write ──────────────────────────────────────────────────────────

    def save(self, user_id: str, content: str) -> None:
        """
        Append a structured fact to the user's memory file.
        Invalidates the in-memory cache so the next retrieval rebuilds the index.

        Args:
            user_id : WhatsApp user_id or any unique string
            content : text to remember (e.g. "health_conditions: diabetes")
        """
        mem_file = USER_MEM_DIR / f"{user_id}.txt"
        with mem_file.open("a", encoding="utf-8") as f:
            f.write(content.strip() + "\n")
        self._cache.pop(user_id, None)

    def load_all(self, user_id: str) -> str:
        """Return the full raw memory text for a user, or empty string."""
        mem_file = USER_MEM_DIR / f"{user_id}.txt"
        return _load_txt(mem_file) if mem_file.exists() else ""

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _get_index(self, user_id: str):
        """Return (index, chunks) for a user, rebuilding from file if needed."""
        if user_id in self._cache:
            return self._cache[user_id]

        mem_file = USER_MEM_DIR / f"{user_id}.txt"
        if not mem_file.exists():
            return None, []

        text = _load_txt(mem_file)
        chunks = _build_chunk_store(source=f"memory:{user_id}", text=text)
        if not chunks:
            return None, []

        index = _build_index(chunks, self._model)
        self._cache[user_id] = (index, chunks)
        return index, chunks

    def get_context(self, user_id: str, query: str, top_k: int = 3) -> str:
        """
        Retrieve the most relevant memory snippets for a given query.

        Returns:
            Formatted string ready to inject into a prompt, or "" if no memory.
        """
        index, chunks = self._get_index(user_id)
        if index is None:
            return ""
        results = _retrieve(query, self._model, index, chunks, top_k)
        return "\n\n".join(f"[{r['source']}]\n{r['text']}" for r in results)

    # ── Structured extraction ─────────────────────────────────────────────────

    def extract(self, user_message: str) -> str | None:
        """
        Ask the LLM to extract structured user profile and health facts
        from one message.

        Tracked fields (only filled when explicitly mentioned):
          name, age_group, gender, asking_for,
          health_conditions, allergies, medications,
          dietary_restriction, disliked_foods, main_goal

        Returns:
            Formatted key: value string, or None if nothing extractable.
        """
        prompt = (
            f'Analyze this message and extract structured user profile and health information.\n\n'
            f'Message: "{user_message}"\n\n'
            f"Extract ONLY information that is explicitly mentioned into these fields:\n\n"
            f"  [User Profile]\n"
            f"  - name: (the user's name or the name of the person they're asking about)\n"
            f"  - age_group: (child / adult / elder — only if age or life stage is mentioned)\n"
            f"  - gender: (male / female / other — only if explicitly stated)\n"
            f"  - asking_for: (self / child / parent / spouse / other — who is this question for?)\n\n"
            f"  [Health & Diet]\n"
            f"  - health_conditions: (e.g. heart disease, diabetes, pregnancy, hypertension)\n"
            f"  - allergies: (e.g. peanuts, dairy, shellfish, gluten)\n"
            f"  - medications: (e.g. metformin, warfarin, blood pressure medication)\n"
            f"  - dietary_restriction: (e.g. vegetarian, vegan, halal, kosher, gluten-free)\n"
            f"  - disliked_foods: (e.g. broccoli, spicy food)\n\n"
            f"  [Goals]\n"
            f"  - main_goal: (e.g. lose weight, manage diabetes, eat healthier, build muscle)\n\n"
            f"Rules:\n"
            f"  - Only include fields that are clearly and explicitly mentioned.\n"
            f"  - Do NOT infer or guess anything not directly stated.\n"
            f"  - If NOTHING is worth saving, reply with exactly: NONE\n\n"
            f"Format (only include non-empty fields, one per line):\n"
            f"name: ...\n"
            f"age_group: ...\n"
            f"gender: ...\n"
            f"asking_for: ...\n"
            f"health_conditions: ...\n"
            f"allergies: ...\n"
            f"medications: ...\n"
            f"dietary_restriction: ...\n"
            f"disliked_foods: ...\n"
            f"main_goal: ..."
        )

        llm = LLMProxy()
        result = llm.generate(
            model="us.anthropic.claude-3-haiku-20240307-v1:0",
            system="You are a precise, minimal information extractor. Never infer beyond what is stated.",
            query=prompt,
            session_id="memory-extract",
            rag_usage=False,
            lastk=0,
        ).get("result", "").strip()

        return None if result.upper() == "NONE" else result

    def auto_extract_and_save(self, user_id: str, user_message: str) -> str | None:
        """
        Extract structured facts from a message and save them if anything found.
        Convenience method combining extract() + save().

        Returns the saved memory string, or None if nothing was extracted.
        """
        memory = self.extract(user_message)
        if memory:
            self.save(user_id, memory)
            print(f"  [Memory saved]\n{memory}")
        return memory
