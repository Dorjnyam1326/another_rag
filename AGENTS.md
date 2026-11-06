# Project instructions

## What this project is

- A local, single-user Streamlit RAG chat app.
- `app.py` uses LangChain `ChatOpenAI` with OpenAI Responses API File Search and `OPENAI_VECTOR_STORE_ID`.
- The UI runs locally, but the vector store, uploaded files, retrieval, and model calls are hosted by OpenAI.
- Do not add LangSmith Fleet, LangGraph deployment, or local Chroma unless the user requests that architecture.

## Secrets and data

- Keep real credentials and IDs in `.env`; never print, commit, or replace existing `.env` values.
- `.env.example` must contain placeholders only.
- OpenAI model and File Search calls can incur API charges. Do not make live API calls unless the user asks for them.

## Local setup

- Use Python 3.12 in `.venv312` (the system Python 3.15 may lack compatible dependency wheels).
- Install with `uv pip install --python .venv312\Scripts\python.exe -r requirements.txt`.
- Run locally with `.venv312\Scripts\streamlit.exe run app.py --server.address 127.0.0.1`.
- Keep the server bound to `127.0.0.1` unless the user asks to expose it on a network.

## Chat behavior and verification

- Runtime instructions live in `prompts/rag_agent.md`; `app.py` loads that file as the chatbot system prompt.
- This `AGENTS.md` is only for coding-agent guidance and is not sent to the chatbot.
- Preserve file citations in answers and say when the vector store does not support an answer.
- After app changes, run `.venv312\Scripts\python.exe -m py_compile app.py` and the relevant Streamlit AppTest when available.
