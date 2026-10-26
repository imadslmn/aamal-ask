# aamal-ask — question answering over Algerian legal texts

A small retrieval-augmented API that answers questions about Algerian legal and administrative
documents, in French, grounded only in the texts it has indexed. Built as a prototype for Solrus
clients who spend hours searching official PDFs for one article.

## How it works

```
PDFs (backend/algerian_legal_pdfs/)
   └─ text extraction → 1 200-char chunks → embeddings → Chroma (persisted locally)

POST /query
   └─ LangGraph pipeline, three steps:
        1. retrieve   — top-5 chunks by similarity
        2. synthesize — answer strictly from the retrieved context (no outside knowledge)
        3. report     — format the answer as a clean French markdown report
```

The three steps are separate graph nodes so each can be swapped or extended (a reranker before
synthesis, a citation checker after it) without touching the API.

## Run it

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example .env        # add your API key
uvicorn main_api:app --reload
```

Then build the index once and query:

```bash
curl -X POST http://localhost:8000/build-database
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "Quelles sont les conditions de création d'une EURL ?"}'
```

Interactive docs at `http://localhost:8000/docs`. Drop more PDFs into `backend/algerian_legal_pdfs/`
and rebuild the index to extend coverage.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/build-database` | Extract, chunk and embed every PDF in the directory; persist to Chroma |
| `POST` | `/query` | Run a question through retrieve → synthesize → report; returns `{ "report": "…" }` |

## Stack

FastAPI · LangChain · LangGraph · Chroma · PyPDF2 · OpenAI models (chat + embeddings)

## Scope and limits

- Answers are only as good as the indexed texts; the model is instructed to refuse outside
  knowledge, but it is not a substitute for a lawyer.
- The indexed documents are public legal texts, so nothing personal is processed. If you adapt this
  to client documents, route the model calls through a de-identifying gateway — see
  [algeria-data-protection](https://github.com/imadslmn/claude-plugins) for the pattern we use.
- Page-level citations and a reranking step are the obvious next improvements.

## License

MIT
