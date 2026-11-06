# Local LangChain Chat

A small Streamlit chat app that runs locally, uses LangChain's `ChatOpenAI`, and searches an OpenAI-hosted Vector Store through the Retrieval API.

Edit `prompts/rag_agent.md` to change the chatbot's language, grounding, citation, and response-style instructions.

## Run on Windows

```powershell
uv venv --python 3.12 .venv312
uv pip install --python .venv312\Scripts\python.exe -r requirements.txt
```

Put your OpenAI API key in the existing `.env` file (do not replace any other settings):

```dotenv
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-6-luna
OPENAI_VECTOR_STORE_ID=vs_your_vector_store_id_here
QUERY_REWRITER_MODEL=gpt-4o-mini
```

Then run:

```powershell
.venv312\Scripts\streamlit.exe run app.py --server.address 127.0.0.1
```

The app is local at `http://localhost:8501`. The chat UI runs on your computer, while the Vector Store and retrieval run at OpenAI. Upload and finish indexing files in the OpenAI dashboard first, then copy that store's ID to `OPENAI_VECTOR_STORE_ID`. ChatGPT subscriptions do not include API usage.

## Chat features

- Select an example question to start, clear the current conversation, and expand cited source filenames.
- Use **Яриаг JSON-ээр татах** in the sidebar to save the current session's messages, sources, and usage metadata; internal retrieval evidence is excluded.
- Keep **Оруулсан баримтаас хайх** on for document-grounded answers. Turn it off for general chat; this skips retrieval and does not require a Vector Store ID.
- Query normalization, restriction routing, and follow-up resolution run in one `QUERY_REWRITER_MODEL` call for substantive messages (default `gpt-4o-mini`, falling back to the older `QUERY_NORMALIZER_MODEL`). It receives the last five turns plus a compact summary of older turns and returns one object per independent request, including intent, Mongolian Cyrillic canonical query, document-search route, ambiguity flag, and up to two alternatives. The same call refreshes the older-turn summary when needed; it does not add a second summary call. Reactions and replies that need no new evidence skip retrieval; ambiguous request parts are not searched and trigger a clarification. Unexpected non-Cyrillic scripts are rejected, while Latin names and identifiers from the conversation are allowed. **Олон query хувилбараар хайх** is on by default. A single-intent question retains the existing original + canonical + up-to-two-alternative search path. Compound questions search each document intent separately, concurrently, with no more than two extra variants across the turn; RRF runs per intent and final evidence is interleaved to retain coverage for each request. With multi-query off, only each intent's canonical query is searched.
- A local routing safeguard keeps recognizable workplace, company-policy, HR, benefit, and internal-system questions on the document-search path even if the rewrite model classifies them as general. Recognized live-only requests such as weather are kept out of the Vector Store. Ambiguity means unclear user intent; unavailable external data by itself is not ambiguity.
- Exact standalone greetings such as `snuu??`, `Сайн уу`, and `hello`, plus clear standalone laughter such as `hhhha`, receive a local response before any API or Vector Store call. Standalone acknowledgments such as `за ойлголоо` receive one short local closing response without an assistant echo or model/search call, and are excluded from later context and question metrics. Mixed messages still use the normal chat flow. Context-dependent replies such as `тийм ээ` and `үгүй ээ` go through the conversation router rather than being treated as standalone questions.
- The assistant receives the current Asia/Ulaanbaatar date and time at request time, so document filenames or timestamps are not treated as today's date. Payment-date calculations still require a supported schedule and holiday information.
- The sidebar has independent restriction switches for salary amounts, standalone programming questions/code requests, and standalone math problems. Salary schedules and salary-review procedures that reveal no amount are allowed; pay amounts remain restricted. All switches are on by default and apply to local prechecks and model prompts. Turning one off removes that rule for this browser session. This is an example-project control, not an access-control boundary.
- Uploaded documents may cover any subject. Document questions are matched semantically (for example, a category question can be answered from relevant examples in a source); unrelated, allowed general questions receive a clearly separated general answer. Whether standalone programming or math questions are restricted depends on the corresponding sidebar switch. Document-grounded explanations remain allowed under the active module rules.
- Explicit salary amounts, code-writing requests, and clear math exercises are caught locally; programming concepts and context-sensitive cases are classified by the structured router before retrieval. A restricted answer shows the matched module and original question, then returns its configured refusal without Vector Store search or final answer generation. Locally blocked question text is omitted from later model context and summaries.
- Each reply shows input/output token counts for answer generation and query rewriting. For `gpt-6-luna` and `gpt-4o-mini`, it also shows an approximate combined model-token cost. This estimate excludes Vector Store storage and can differ from the final invoice. Turning off multi-query reduces Vector Store searches and rewrite output; the contextual router still runs for substantive messages.
- Each query variant returns up to five passages; RRF supplies up to eight interleaved passages to answer generation while keeping evidence grouped by intent. Responses can use up to 1,200 completion tokens. The app retains the full visible transcript for the active session, sends the last five turns in full, and summarizes older turns for reference resolution. The summary is not factual evidence. Clearing the conversation clears its summary but leaves restriction switches unchanged.

## Dashboard

Use the **Dashboard / Chat** switch to see the chat or a session overview. The dashboard reports question and answer counts, input/output token totals (including query rewriting), an estimated model-token cost for supported models, citation-bearing answers, per-answer token usage, and the latest five questions with their token breakdown, rewrite/search/model/total latency in milliseconds, and estimated cost. A total row summarizes the displayed questions. It also shows whether the API key and Vector Store ID are configured without displaying their values, and whether multi-query search is enabled.

These metrics come only from the current Streamlit session. They reset when the session is cleared or expires; the dashboard does not show fabricated sample data or verify live API connectivity. To preserve metrics across sessions later, store each question, answer, usage record, and citations in PostgreSQL, then load those records for the dashboard.

## Cost and data flow

`Browser → local Streamlit app → local precheck / contextual query router → optional parallel OpenAI Vector Store search by document intent → per-intent RRF and balanced evidence → LangChain ChatOpenAI → answer and source filenames → browser`

Potential charges are model input/output tokens and Vector Store storage above the free allowance. The structured router makes one small model call for substantive messages even when document search is turned off; summary updates are part of that call. The Responses API File Search tool is not used, so its per-tool-call fee does not apply. The direct Vector Store search endpoint returns matching passages and source filenames; the app performs four or fewer searches and RRF locally. Current standard prices are $0.10 / 1M input tokens and $0.50 / 1M output tokens for `gpt-6-luna`; `gpt-4o-mini` is $0.15 / 1M input and $0.60 / 1M output. Vector Store storage is free for the first 1 GB across stores, then $0.10/GB/day. Prices can change; see the [OpenAI pricing page](https://developers.openai.com/api/docs/pricing) and [Retrieval guide](https://developers.openai.com/api/docs/guides/retrieval) for current details.

The UI, Streamlit and LangChain run locally and do not add their own usage fee. This app does not call LangSmith or use a local Chroma database. The uploaded documents and their searchable index are hosted by OpenAI, so this is a locally run interface, not fully local/private retrieval. Ingestion automatically chunks and embeds files for the Vector Store. API usage is billed separately from a ChatGPT subscription.

## Query rewrite regression cases

`tests/fixtures/query_rewrite_eval.json` contains 20 representative questions for document topics, transliteration, follow-ups, mixed requests, restricted topics, and greetings. The automated tests validate schema and routing behavior; they do not score the live model's Mongolian meaning or prove answer quality. Review these cases against approved source passages before treating the rewrite changes as a quality benchmark.
