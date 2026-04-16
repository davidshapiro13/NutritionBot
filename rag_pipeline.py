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
import csv
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

# User-visible short tags for trailing "Sources:" (exact filename → label).
_LEAKED_LENGTH_META = re.compile(
    r"\n*\(?\s*Total\s+characters?\s*:\s*\d+\s*\)?\s*$",
    re.IGNORECASE,
)


def _strip_leaked_length_metadata(text: str) -> str:
    """Drop '(Total characters: N)' lines models sometimes add when prompts mention a length cap."""
    t = (text or "").rstrip()
    while True:
        n = _LEAKED_LENGTH_META.sub("", t).rstrip()
        if n == t:
            return n
        t = n


KB_SOURCE_DISPLAY_NAMES: dict[str, str] = {
    "food_safety_knowledge_base.txt": "food_safety",
    "Cold Food Storage Chart.pdf": "cold_storage",
    "ma_wic_approved_stores.csv": "wic_stores",
    "DGA.pdf": "dga",
}


def _kb_source_display_label(filename: str) -> str:
    if filename in KB_SOURCE_DISPLAY_NAMES:
        return KB_SOURCE_DISPLAY_NAMES[filename]
    stem = Path(filename).stem.strip().lower()
    stem = re.sub(r"\s+", "_", stem)
    return stem or filename


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


def _load_csv(path: Path) -> str:
    """Convert CSV rows to readable text lines for embedding."""
    lines = []
    with path.open(encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parts = [f"{k}: {v}" for k, v in row.items() if v and v.strip()]
            lines.append(" | ".join(parts))
    return "\n".join(lines)


def _load_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":   return _load_pdf(path)
    if suffix == ".docx":  return _load_docx(path)
    if suffix == ".txt":   return _load_txt(path)
    if suffix == ".csv":   return _load_csv(path)
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

    # L2 distance on MiniLM embeddings; chunks beyond this are not injected into the LLM prompt
    RELEVANCE_THRESHOLD = 1.25

    def get_public_context(self, query: str, top_k: int = 4) -> tuple[str, bool, list[str]]:
        """
        Retrieve passages from the public knowledge base.

        Returns (context_string, has_relevant, citation_filenames):
        - context_string: only chunks under RELEVANCE_THRESHOLD (for the model).
        - has_relevant: True if at least one such chunk exists.
        - citation_filenames: unique source files from all top-k hits (for user-visible Sources line).
        """
        if self._pub_index is None or self._pub_index.ntotal == 0:
            return "", False, []
        results = _retrieve(query, self._model, self._pub_index, self._pub_chunks, top_k)
        if not results:
            return "", False, []
        seen_cite: set[str] = set()
        citation_sources: list[str] = []
        for r in results:
            s = r["source"]
            if s not in seen_cite:
                seen_cite.add(s)
                citation_sources.append(s)
        relevant = [r for r in results if r["distance"] < self.RELEVANCE_THRESHOLD]
        if not relevant:
            return "", False, citation_sources
        context = "\n\n".join(f"[Source: {r['source']}]\n{r['text']}" for r in relevant)
        return context, True, citation_sources

    # ── Combined context ──────────────────────────────────────────────────────

    def get_context(self, query: str, user_id: str = None) -> tuple[str, bool, list[str]]:
        """
        Combine public KB context and user memory context.
        Returns (context_string, has_relevant_kb, kb_source_filenames).
        """
        sections = []

        pub, has_relevant, kb_sources = self.get_public_context(query)
        if pub:
            sections.append(f"[Public Knowledge Base]\n{pub}")

        if user_id:
            usr = self.memory.get_context(user_id, query)
            if usr:
                sections.append(f"[User Memory]\n{usr}")

        return "\n\n".join(sections), has_relevant, kb_sources

    # ── Full RAG answer ───────────────────────────────────────────────────────

    def query_rag(
        self,
        question: str,
        session_id: str = "rag-demo",
        user_id: str = None,
        memory_source_message: str | None = None,
    ) -> str:
        """
        Full RAG query:
          1. Retrieve public KB + user memory context with relevance filtering
          2. If no relevant KB context found, fall back to direct LLM answer
          3. Otherwise generate answer grounded in retrieved sources
          4. Auto-extract and save any new user facts from the question

        Args:
            question   : user question (may include profile wrapper)
            session_id : LLMProxy session for conversation history
            user_id    : optional; enables user memory tracking
            memory_source_message : if set, used for memory extraction instead of question

        Returns:
            LLM answer string, with trailing "Sources: …" when KB retrieval returned chunks
            and/or web fallback titles were used (KB filenames may appear even if chunks were
            below the relevance threshold for prompting).
        """
        context, has_relevant, kb_sources = self.get_context(question, user_id=user_id)

        llm = LLMProxy()

        length_instruction = (
            "Keep your answer under 500 characters. "
            "3-5 sentences max. Plain text only, no bullet points or markdown. "
            "Do not list source filenames or URLs in your answer; citations are added separately. "
            "Never state character counts, word counts, or how long your answer is."
        )

        web_source_labels: list[str] = []

        if not has_relevant:
            # No relevant KB chunks — try web search fallback
            web_context = ""
            try:
                from web_search import WebSearch
                results = WebSearch().search(question, max_results=3)
                for r in results or []:
                    if r.get("snippet"):
                        t = str(r.get("title", "")).strip()
                        if t:
                            web_source_labels.append(t)
                snippets = [
                    f"[{r['title']}]\n{r['snippet']}"
                    for r in results if r.get("snippet")
                ]
                web_context = "\n\n".join(snippets)
            except Exception:
                pass

            if web_context:
                query_with_context = (
                    f"{length_instruction}\n\n"
                    f"Answer using the web sources below.\n\n"
                    f"SOURCES:\n{web_context}\n\n"
                    f"QUESTION:\n{question}"
                )
            else:
                query_with_context = f"{length_instruction}\n\nQUESTION:\n{question}"
        else:
            query_with_context = (
                f"{length_instruction}\n\n"
                f"USER QUESTION: {question}\n\n"
                f"RULE: Answer ONLY the question above. "
                f"The sources may contain information about many food types — ignore everything except what directly answers the question. "
                f"Do NOT mention leftovers, dairy, chicken, or any other food category unless the user specifically asked about it.\n\n"
                f"SOURCES:\n{context}"
            )

        response = llm.generate(
            model="us.anthropic.claude-3-haiku-20240307-v1:0",
            system=main_system_prompt,
            query=query_with_context,
            session_id=session_id,
            rag_usage=False,
            lastk=10,
        )
        answer = _strip_leaked_length_metadata(response.get("result", str(response)))

        # If still over 500 chars, ask LLM to condense it
        if len(answer) > 500:
            condense_response = llm.generate(
                model="us.anthropic.claude-3-haiku-20240307-v1:0",
                system=(
                    "You are a text editor. Condense the given text to under 500 characters while "
                    "keeping all key facts accurate. Plain text only, no markdown. Never append "
                    "character counts or notes about length."
                ),
                query=f"Condense this to under 500 characters:\n\n{answer}",
                session_id=session_id + "_condense",
                rag_usage=False,
                lastk=0,
            )
            condensed = _strip_leaked_length_metadata(condense_response.get("result", answer))
            answer = condensed if len(condensed) <= 500 else condensed[:496] + "…"
        answer = _strip_leaked_length_metadata(answer)

        # Auto-extract structured user facts and save to memory
        if user_id:
            mem_src = memory_source_message if memory_source_message is not None else question
            self.memory.auto_extract_and_save(user_id, mem_src)

        unique_web = list(dict.fromkeys(web_source_labels))
        source_parts: list[str] = []
        if kb_sources:
            labels = [_kb_source_display_label(f) for f in kb_sources]
            source_parts.append("; ".join(labels))
        if unique_web:
            source_parts.append("; ".join(unique_web))
        if source_parts:
            answer = f"{answer.rstrip()}\n\nSources: {' | '.join(source_parts)}"

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
