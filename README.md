# 📄 AI Document Assistant

A simple Retrieval-Augmented Generation (RAG) chatbot built with Streamlit.
Upload documents (or load them from Google Drive), ask questions in plain
English, and get answers grounded only in your documents — with sources
shown underneath every answer.

## Features

- **Multi-format extraction**: PDF, DOCX, TXT, and MD, each with its own
  extraction function. PDFs keep per-page metadata; other formats are
  extracted as a single block (they have no reliable page boundaries).
- **Overlapping chunking**: long documents are split into overlapping
  chunks so context isn't lost at chunk boundaries. Every chunk keeps its
  source filename and page number.
- **Embeddings computed once**: chunks are embedded with Sentence
  Transformers (`all-MiniLM-L6-v2`) a single time and cached in session
  state — asking more questions never re-embeds your documents.
- **Hybrid search**: combines FAISS semantic (meaning-based) search with a
  simple keyword-overlap score, so both paraphrased and keyword-heavy
  questions work well.
- **Grounded answers via Groq**: the question and retrieved chunks are
  sent to a Groq-hosted LLM, instructed to answer only from the provided
  context and say so when the answer isn't there.
- **Sources shown**: every answer is followed by the filename, page number
  (when available), and the exact retrieved chunk text.
- **Google Drive support**: paste a public Drive file or folder link and
  its supported files are downloaded and run through the same pipeline as
  local uploads.

## How it works

```
Upload / Drive link
        │
        ▼
 1. Extract text  (extract_pdf / extract_docx / extract_txt / extract_md)
        │
        ▼
 2. Chunk text    (overlapping chunks, filename + page kept on each chunk)
        │
        ▼
 3. Embed once    (Sentence Transformers, cached in session state)
        │
        ▼
 4. FAISS index   (semantic search)
        │                              5. Keyword search
        └──────────────┬───────────────────────┘
                        ▼
              6. Hybrid search (ranked, blended score)
                        │
                        ▼
              7. Groq LLM answers from retrieved chunks
                        │
                        ▼
              8. Answer + sources shown in the UI
```

## Files

- `app.py` — the entire application (extraction, chunking, embeddings,
  FAISS, hybrid search, Groq, Google Drive loading, and the Streamlit UI).
- `requirements.txt` — Python dependencies.
- `readme.md` — this file.

## Setup

### 1. Get a Groq API key

Sign up at [console.groq.com](https://console.groq.com) and create an API
key.

### 2. Set the API key as a Streamlit secret

**Never hardcode the API key in `app.py`.** Instead:

**Locally:** create `.streamlit/secrets.toml` in the project folder:

```toml
GROQ_API_KEY = "your-key-here"
```

**On Streamlit Community Cloud:** go to your app → **Settings → Secrets**
and add:

```toml
GROQ_API_KEY = "your-key-here"
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run locally

```bash
streamlit run app.py
```

### 5. Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and `readme.md` to a GitHub repo.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing to `app.py` in that repo.
3. Add `GROQ_API_KEY` under **Settings → Secrets** as shown above.
4. Deploy.

## Using the app

1. In the sidebar, upload one or more PDF/DOCX/TXT/MD files, or paste a
   public Google Drive file/folder link and click **Load from Drive**.
2. Click **Process documents**. The sidebar shows each file's page count,
   character count, and chunk count, plus the running total of chunks.
3. Type a question and click **Ask**. The answer appears with an
   expandable **Sources** section showing exactly which chunks it came
   from.
4. Add more documents any time — already-processed files are skipped, and
   only new chunks are embedded.

## Notes

- Google Drive links must be shared as **"Anyone with the link"** —
  private files can't be downloaded without extra authentication.
- DOCX, TXT, and MD files don't have a reliable concept of "pages," so
  their sources show only the filename.
- The hybrid search blend (60% semantic / 40% keyword) and chunk size
  (800 characters, 150 overlap) are set as constants near the top of
  `app.py` and can be tuned there.
