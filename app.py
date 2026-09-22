"""
AI Document Assistant
======================
A simple Retrieval-Augmented Generation (RAG) app built with Streamlit.

Upload PDF, DOCX, TXT or MD files (or load them from Google Drive), ask
questions about them, and get answers grounded only in your documents,
with sources shown underneath every answer.

Pipeline (see the numbered sections below):
  1. Extract text from documents      -> one function per file type
  2. Split text into overlapping chunks, keeping filename/page metadata
  3. Embed chunks once with Sentence Transformers (cached, reused)
  4. Build a FAISS index for semantic (meaning-based) search
  5. Score chunks with simple keyword search
  6. Combine both into one hybrid search, ranked by a blended score
  7. Send the question + retrieved chunks to Groq and show the answer
  8. Display the sources (filename, page, chunk text) below every answer
"""

import io
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional

import faiss
import numpy as np
import streamlit as st
import fitz  # PyMuPDF
import docx  # python-docx
from sentence_transformers import SentenceTransformer
from groq import Groq

try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "openai/gpt-oss-120b"
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 150     # overlap between consecutive chunks
TOP_K = 5               # chunks retrieved per question
SEMANTIC_WEIGHT = 0.6   # weight given to semantic (FAISS) score in hybrid search
KEYWORD_WEIGHT = 0.4    # weight given to keyword score in hybrid search
SUPPORTED_EXTENSIONS = ("pdf", "docx", "txt", "md")

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "with", "and", "or", "but", "if", "then",
    "so", "as", "at", "by", "from", "that", "this", "these", "those", "it",
    "its", "what", "which", "who", "whom", "how", "when", "where", "why",
    "do", "does", "did", "can", "could", "should", "would", "will", "shall",
    "i", "you", "he", "she", "we", "they", "my", "your", "his", "her",
    "our", "their", "about", "into", "than", "also", "not", "no",
}


# ---------------------------------------------------------------------------
# Data structure for a single chunk of text
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    text: str
    filename: str
    page: Optional[int]  # None for file types with no page concept (DOCX/TXT/MD)


# ---------------------------------------------------------------------------
# 1. Document extraction — one function per file type
# ---------------------------------------------------------------------------
def extract_pdf(file_bytes: bytes, filename: str) -> List[Dict]:
    """Extract text page by page from a PDF, keeping page numbers."""
    pages = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                pages.append({"text": text, "filename": filename, "page": i})
    return pages


def extract_docx(file_bytes: bytes, filename: str) -> List[Dict]:
    """Extract text from a Word document. DOCX has no reliable page
    boundaries, so the whole document is returned as one block."""
    document = docx.Document(io.BytesIO(file_bytes))
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    if not text.strip():
        return []
    return [{"text": text, "filename": filename, "page": None}]


def extract_txt(file_bytes: bytes, filename: str) -> List[Dict]:
    """Extract text from a plain .txt file."""
    text = file_bytes.decode("utf-8", errors="ignore")
    if not text.strip():
        return []
    return [{"text": text, "filename": filename, "page": None}]


def extract_md(file_bytes: bytes, filename: str) -> List[Dict]:
    """Extract text from a Markdown file (kept as raw text)."""
    text = file_bytes.decode("utf-8", errors="ignore")
    if not text.strip():
        return []
    return [{"text": text, "filename": filename, "page": None}]


def extract_document(file_bytes: bytes, filename: str) -> List[Dict]:
    """Route a file to the correct extractor based on its extension."""
    ext = filename.lower().rsplit(".", 1)[-1]
    extractors = {
        "pdf": extract_pdf,
        "docx": extract_docx,
        "txt": extract_txt,
        "md": extract_md,
    }
    extractor = extractors.get(ext)
    return extractor(file_bytes, filename) if extractor else []


# ---------------------------------------------------------------------------
# 2. Chunking — split text into overlapping pieces, keep metadata
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping character-based chunks."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap  # step forward, keeping an overlap window
    return chunks


def build_chunks(extracted_blocks: List[Dict]) -> List[Chunk]:
    """Turn extracted page/document blocks into overlapping chunks, each
    carrying its source filename and page number."""
    all_chunks = []
    for block in extracted_blocks:
        for piece in chunk_text(block["text"]):
            all_chunks.append(Chunk(text=piece, filename=block["filename"],
                                     page=block["page"]))
    return all_chunks


# ---------------------------------------------------------------------------
# 3. Embeddings — cache the model, embed each chunk exactly once
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_chunks(chunks: List[Chunk]) -> np.ndarray:
    model = load_embedding_model()
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)  # so inner product == cosine similarity
    return embeddings.astype("float32")


# ---------------------------------------------------------------------------
# 4. FAISS vector search
# ---------------------------------------------------------------------------
def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
    index.add(embeddings)
    return index


def semantic_search(query: str, index: faiss.Index, top_k: int) -> List[Dict]:
    """Return [{'pos': chunk_index, 'semantic_score': float}, ...]."""
    model = load_embedding_model()
    query_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec.astype("float32"), top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({"pos": int(idx), "semantic_score": float(score)})
    return results


# ---------------------------------------------------------------------------
# 5. Keyword search
# ---------------------------------------------------------------------------
def extract_keywords(question: str) -> List[str]:
    words = re.findall(r"[a-zA-Z']+", question.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def keyword_search(question: str, chunks: List[Chunk]) -> Dict[int, float]:
    """Score each chunk by the fraction of question keywords it contains.
    Returns {chunk_position: score}."""
    keywords = extract_keywords(question)
    if not keywords:
        return {}
    scores = {}
    for pos, chunk in enumerate(chunks):
        chunk_lower = chunk.text.lower()
        matches = sum(1 for kw in keywords if kw in chunk_lower)
        if matches:
            scores[pos] = matches / len(keywords)
    return scores


# ---------------------------------------------------------------------------
# 6. Hybrid search — blend semantic + keyword scores, keep metadata
# ---------------------------------------------------------------------------
def hybrid_search(question: str, index: faiss.Index, chunks: List[Chunk],
                    top_k: int = TOP_K) -> List[Dict]:
    # Look at more semantic candidates than top_k, so keyword-strong chunks
    # further down the semantic ranking still get a fair chance.
    semantic_results = semantic_search(question, index, top_k=min(len(chunks), top_k * 4))
    semantic_scores = {r["pos"]: r["semantic_score"] for r in semantic_results}
    keyword_scores = keyword_search(question, chunks)

    candidate_positions = set(semantic_scores) | set(keyword_scores)
    combined = []
    for pos in candidate_positions:
        sem = semantic_scores.get(pos, 0.0)
        kw = keyword_scores.get(pos, 0.0)
        combined.append({
            "chunk": chunks[pos],
            "semantic_score": sem,
            "keyword_score": kw,
            "combined_score": SEMANTIC_WEIGHT * sem + KEYWORD_WEIGHT * kw,
        })

    combined.sort(key=lambda r: r["combined_score"], reverse=True)
    return combined[:top_k]


# ---------------------------------------------------------------------------
# 7. Groq — answer the question using only the retrieved chunks
# ---------------------------------------------------------------------------
def get_groq_client() -> Optional[Groq]:
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error(
            "GROQ_API_KEY not found. Add it in Streamlit Cloud under "
            "**Settings -> Secrets** (or in a local `.streamlit/secrets.toml`) as:\n\n"
            '`GROQ_API_KEY = "your-key-here"`'
        )
        return None
    return Groq(api_key=api_key)


def build_context(results: List[Dict]) -> str:
    parts = []
    for i, r in enumerate(results, start=1):
        chunk = r["chunk"]
        location = chunk.filename + (f", page {chunk.page}" if chunk.page else "")
        parts.append(f"[Source {i} - {location}]\n{chunk.text}")
    return "\n\n".join(parts)


def ask_groq(question: str, results: List[Dict]) -> Optional[str]:
    client = get_groq_client()
    if not client:
        return None

    context = build_context(results)
    system_prompt = (
        "You are a document assistant. Answer the user's question using ONLY "
        "the information in the context below. If the answer is not present "
        "in the context, say clearly that the information is not available "
        "in the uploaded documents. Do not use outside knowledge and do not "
        "make anything up."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Google Drive loading (files or folders, public links, no API key needed)
# ---------------------------------------------------------------------------
def parse_drive_url(url: str):
    folder_match = re.search(r"drive\.google\.com/drive/folders/([\w-]+)", url)
    if folder_match:
        return "folder", folder_match.group(1)
    file_match = (re.search(r"drive\.google\.com/file/d/([\w-]+)", url)
                  or re.search(r"[?&]id=([\w-]+)", url))
    if file_match:
        return "file", file_match.group(1)
    return None, None


def load_from_drive(url: str) -> List[str]:
    """Download a public Google Drive file or folder link and return the
    local paths of any supported (PDF/DOCX/TXT/MD) files found."""
    kind, file_id = parse_drive_url(url)
    if not kind:
        st.sidebar.error("Could not recognize that as a Google Drive file or folder link.")
        return []

    tmp_dir = tempfile.mkdtemp()
    downloaded_paths = []

    if kind == "file":
        output_path = gdown.download(id=file_id, output=tmp_dir + os.sep,
                                      quiet=True, fuzzy=True)
        if output_path:
            downloaded_paths.append(output_path)
    else:
        paths = gdown.download_folder(url, output=tmp_dir, quiet=True,
                                       use_cookies=False)
        downloaded_paths = paths or []

    supported = [p for p in downloaded_paths
                 if p.lower().rsplit(".", 1)[-1] in SUPPORTED_EXTENSIONS]
    return supported


# ---------------------------------------------------------------------------
# Processing pipeline: extract -> chunk -> embed -> index (done once per file)
# ---------------------------------------------------------------------------
def process_files(file_items: List[Dict]) -> None:
    """file_items: [{'filename': str, 'bytes': bytes}, ...]
    Skips files already processed and only embeds new chunks."""
    new_chunks: List[Chunk] = []

    for item in file_items:
        filename = item["filename"]
        if filename in st.session_state.processed_files:
            continue

        blocks = extract_document(item["bytes"], filename)
        if not blocks:
            st.sidebar.warning(f"No extractable text found in {filename}.")
            continue

        file_chunks = build_chunks(blocks)
        new_chunks.extend(file_chunks)
        st.session_state.processed_files.add(filename)
        has_pages = blocks[0]["page"] is not None
        st.session_state.doc_info.append({
            "filename": filename,
            "pages": len(blocks) if has_pages else "N/A",
            "characters": sum(len(b["text"]) for b in blocks),
            "chunks": len(file_chunks),
        })

    if not new_chunks:
        return

    with st.spinner(f"Embedding {len(new_chunks)} new chunk(s)..."):
        new_embeddings = embed_chunks(new_chunks)

    st.session_state.chunks.extend(new_chunks)
    if st.session_state.embeddings is None:
        st.session_state.embeddings = new_embeddings
    else:
        st.session_state.embeddings = np.vstack([st.session_state.embeddings, new_embeddings])

    st.session_state.faiss_index = build_faiss_index(st.session_state.embeddings)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "chunks": [],            # List[Chunk] — persists across reruns/questions
        "embeddings": None,      # np.ndarray — built once, reused for every question
        "faiss_index": None,
        "processed_files": set(),
        "doc_info": [],
        "chat_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    st.set_page_config(page_title="AI Document Assistant", page_icon="📄", layout="wide")
    init_session_state()

    st.title("📄 AI Document Assistant")
    st.caption("Upload documents, ask questions, and get answers grounded in your files — with sources.")

    # ---------------- Sidebar: add & process documents ----------------
    with st.sidebar:
        st.header("1. Add documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, TXT or MD files",
            type=list(SUPPORTED_EXTENSIONS),
            accept_multiple_files=True,
        )

        st.divider()
        st.subheader("Or load from Google Drive")
        drive_url = st.text_input("Google Drive file or folder link")
        load_drive_clicked = st.button("Load from Drive", disabled=not GDOWN_AVAILABLE)
        if not GDOWN_AVAILABLE:
            st.caption("`gdown` is not installed — Google Drive loading is disabled.")

        process_clicked = st.button("Process documents", type="primary")

        st.divider()
        st.subheader("Processed documents")
        if st.session_state.doc_info:
            for info in st.session_state.doc_info:
                st.markdown(
                    f"**{info['filename']}**  \n"
                    f"Pages: {info['pages']} · Chunks: {info['chunks']} · "
                    f"Characters: {info['characters']}"
                )
        else:
            st.caption("No documents processed yet.")

        if st.session_state.chunks:
            st.success(f"Total chunks ready for search: {len(st.session_state.chunks)}")

    # ---------------- Handle Drive loading ----------------
    if load_drive_clicked and drive_url:
        with st.spinner("Downloading from Google Drive..."):
            paths = load_from_drive(drive_url)
        if not paths:
            st.sidebar.error("No supported PDF/DOCX/TXT/MD files found at that link.")
        else:
            file_items = []
            for path in paths:
                with open(path, "rb") as f:
                    file_items.append({"filename": os.path.basename(path), "bytes": f.read()})
            process_files(file_items)
            st.rerun()

    # ---------------- Handle local upload processing ----------------
    if process_clicked:
        if not uploaded_files:
            st.sidebar.warning("Upload at least one file first.")
        else:
            file_items = [{"filename": f.name, "bytes": f.read()} for f in uploaded_files]
            process_files(file_items)
            st.rerun()

    # ---------------- Main area: ask questions ----------------
    st.header("2. Ask a question")

    if not st.session_state.chunks:
        st.info("Upload and process at least one document to start asking questions.")
        return

    question = st.text_input("Your question", key="question_input")
    ask_clicked = st.button("Ask")

    if ask_clicked and question:
        with st.spinner("Searching documents and generating an answer..."):
            results = hybrid_search(question, st.session_state.faiss_index,
                                     st.session_state.chunks)
            answer = ask_groq(question, results)
        if answer:
            st.session_state.chat_history.append(
                {"question": question, "answer": answer, "sources": results}
            )

    for entry in reversed(st.session_state.chat_history):
        st.markdown(f"**Q: {entry['question']}**")
        st.write(entry["answer"])
        with st.expander(f"Sources ({len(entry['sources'])})"):
            for r in entry["sources"]:
                chunk = r["chunk"]
                location = chunk.filename + (f" — page {chunk.page}" if chunk.page else "")
                st.markdown(f"**{location}**  (relevance: {r['combined_score']:.2f})")
                preview = chunk.text[:500] + ("..." if len(chunk.text) > 500 else "")
                st.caption(preview)
        st.divider()


if __name__ == "__main__":
    main()
