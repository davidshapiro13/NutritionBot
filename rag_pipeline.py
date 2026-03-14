"""
RAG Pipeline for NutritionBot
==============================
Handles the public knowledge base (food guides, food safety documents).
User profile memory is managed separately in user_memory.py.

Pipeline steps:
  1. Load PDF / DOCX / TXT files from rag_data/
  2. Extract plain text
  3. Split text into overlapping chunks (~450 chars)
  4. Embed with a local sentence-transformer model
  5. Store embeddings in a FAISS index
  6. At query time: embed question → retrieve top-k chunks → return context

Usage:
    from rag_pipeline import RAGPipeline

    rag = RAGPipeline()
    rag.build_public_index()          # once at startup

    # get answer for a user (with optional memory)
    answer = rag.query_rag("What grains are WIC approved?", session_id="s1", user_id="u1")
"""

import re
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
import pypdf
from docx import Document as DocxDocument
from llmproxy import LLMProxy

from user_memory import UserMemory
from prompts import main_system_prompt

# ── Paths ─────────────────────────────────────────────────────────────────────
RAG_DATA_DIR = Path(__file__).parent / "rag_data"
EMBED_MODEL  = "all-MiniLM-L6-v2"


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _load_pdf(path: Path) -> str:
    reader = pypdf.PdfReader(str(path))
    return "\n".join(p.extract_text() for p in reader.pages if p.extract_text())


def _load_docx(path: Path) -> str:
    doc = DocxDocument(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":   return _load_pdf(path)
    if suffix == ".docx":  return _load_docx(path)
    if suffix == ".txt":   return _load_txt(path)
    return ""


def _load_folder(folder: Path) -> list[dict]:
    """Return [{"source": filename, "text": content}, …] for all supported files."""
    docs = []
    for file in sorted(folder.iterdir()):
        text = _load_file(file)
        if text.strip():
            docs.append({"source": file.name, "text": text})
    return docs


# ══════════════════════════════════════════════════════════════════════════════
# CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def _chunk(text: str, size: int = 450, overlap: int = 50) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size].strip())
        start += size - overlap
    return [c for c in chunks if c]


def _build_chunk_store(docs: list[dict]) -> list[dict]:
    store = []
    for doc in docs:
        for chunk in _chunk(doc["text"]):
            store.append({"source": doc["source"], "text": chunk})
    return store


# ══════════════════════════════════════════════════════════════════════════════
# FAISS INDEX
# ══════════════════════════════════════════════════════════════════════════════

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
# RAG PIPELINE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class RAGPipeline:
    """
    Public knowledge base retrieval + answer generation.
    User memory is delegated to UserMemory (user_memory.py).

    Typical lifecycle:
        rag = RAGPipeline()
        rag.build_public_index()               # once at startup

        answer = rag.query_rag(question, session_id, user_id)
    """

    def __init__(self):
        print("Loading embedding model …")
        self._model = SentenceTransformer(EMBED_MODEL)

        self._pub_chunks: list[dict] = []
        self._pub_index = None

        # UserMemory shares the same embedding model to avoid loading it twice
        self.memory = UserMemory(embed_model=self._model)

    # ── Public KB ─────────────────────────────────────────────────────────────

    def build_public_index(self) -> int:
        """
        Load all documents from rag_data/ and build the FAISS index.
        Returns the number of chunks indexed.
        """
        if not RAG_DATA_DIR.exists():
            print(f"WARNING: {RAG_DATA_DIR} does not exist. No public KB loaded.")
            return 0

        docs = _load_folder(RAG_DATA_DIR)
        self._pub_chunks = _build_chunk_store(docs)
        if self._pub_chunks:
            self._pub_index = _build_index(self._pub_chunks, self._model)
        print(f"Public KB: {len(docs)} file(s), {len(self._pub_chunks)} chunks indexed.")
        return len(self._pub_chunks)

    def get_public_context(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant passages from the public knowledge base."""
        if self._pub_index is None or self._pub_index.ntotal == 0:
            return ""
        results = _retrieve(query, self._model, self._pub_index, self._pub_chunks, top_k)
        return "\n\n".join(f"[{r['source']}]\n{r['text']}" for r in results)

    # ── Combined context ──────────────────────────────────────────────────────

    def get_context(self, query: str, user_id: str = None) -> str:
        """
        Combine public KB context and user memory context into one string
        ready to inject into a prompt.

        Args:
            query   : the user's current message
            user_id : if provided, also retrieves personal memory

        Returns:
            Combined context string, or "" if both are empty.
        """
        sections = []

        pub = self.get_public_context(query)
        if pub:
            sections.append(f"[Public Knowledge Base]\n{pub}")

        if user_id:
            usr = self.memory.get_context(user_id, query)
            if usr:
                sections.append(f"[User Memory]\n{usr}")

        return "\n\n".join(sections)

    # ── Scope check (guardrail pre-filter) ───────────────────────────────────

    def is_in_scope(self, question: str) -> bool:
        """
        Ask the LLM whether the question is within the bot's scope BEFORE
        running the full RAG pipeline.

        Returns True if in scope, False if the question should be refused.
        This is more reliable than relying on the main LLM to self-police.
        """
        check_prompt = (
            f'Question: "{question}"\n\n'
            f"Is this question about human nutrition, human food safety, "
            f"human diet, WIC-approved foods, or how long human food lasts?\n\n"
            f"Answer with exactly one word: YES or NO."
        )
        llm = LLMProxy()
        result = llm.generate(
            model="us.anthropic.claude-3-haiku-20240307-v1:0",
            system="You are a strict topic classifier. Answer only YES or NO.",
            query=check_prompt,
            session_id="scope-check",
            rag_usage=False,
            lastk=0,
        ).get("result", "YES").strip().upper()

        return result.startswith("YES")

    # ── Full RAG answer ───────────────────────────────────────────────────────

    def query_rag(self, question: str, session_id: str = "rag-demo", user_id: str = None) -> str:
        """
        Full RAG query:
          1. Scope check — refuse immediately if out of scope
          2. Retrieve public KB + user memory context
          3. Generate answer via LLMProxy
          4. Auto-extract and save any new user facts from the question

        Args:
            question   : user question
            session_id : LLMProxy session for conversation history
            user_id    : optional; enables user memory track

        Returns:
            LLM answer string
        """
        # Step 1: pre-filter — hard stop for out-of-scope questions
        if not self.is_in_scope(question):
            return (
                "I'm only able to help with human nutrition and food safety topics — "
                "such as healthy eating, WIC-approved foods, food storage times, and diet advice. "
                "Is there something food-related I can help you with?"
            )

        context = self.get_context(question, user_id=user_id)

        query_with_context = question
        if context:
            query_with_context = (
                f"Use the following context to help answer the question.\n\n"
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION:\n{question}"
            )

        llm = LLMProxy()
        response = llm.generate(
            model="us.anthropic.claude-3-haiku-20240307-v1:0",
            system=main_system_prompt,
            query=query_with_context,
            session_id=session_id,
            rag_usage=False,
            lastk=10,
        )
        answer = response.get("result", str(response))

        # Auto-extract structured user facts and save to memory
        if user_id:
            self.memory.auto_extract_and_save(user_id, question)

        return answer


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — interactive test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rag = RAGPipeline()
    rag.build_public_index()

    print("\n" + "=" * 55)
    print("  Interactive RAG test — type EXIT to quit")
    print("=" * 55)

    user_id = input("Enter a user_id (or press Enter to skip user memory): ").strip() or None

    while True:
        q = input("\nYou: ").strip()
        if not q or q.upper() == "EXIT":
            break

        # ── Debug: show retrieved chunks ──
        if rag._pub_index is not None:
            top_chunks = _retrieve(q, rag._model, rag._pub_index, rag._pub_chunks, top_k=3)
            print("\n── Retrieved chunks ──")
            for i, c in enumerate(top_chunks, 1):
                print(f"\n[{i}] Source: {c['source']}  dist={c['distance']:.3f}")
                print(c['text'])
            print("─────────────────────\n")

        answer = rag.query_rag(q, session_id="rag-test", user_id=user_id)
        print(f"Bot: {answer}")
