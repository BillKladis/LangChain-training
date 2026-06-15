# LangChain Training — Agentic AI Systems

A hands-on collection of progressively more complex LangChain / LangGraph systems, from simple LLM chains to a full multi-agent RAG pipeline with a Streamlit UI.

---

## Repository Map

```
LangChain-training/
│
├── Info_Summary_Chain_Ollama/
│   └── Summary_conv.py              # Simple LCEL chain: prompt | LLM | parser
│
├── React_Under_the_hood/
│   └── LangChain_Primitives.py      # Manual ReAct loop (no prebuilt agent)
│
├── React_Search_Agent_Tavily_Pydantic/
│   ├── Search_Agent_Tavily_Pydantic.py             # Custom Tavily tool + Pydantic output
│   └── Search_Agent_Tavily_Pydantic_Tavily_Tool.py # Native TavilySearch, job-offer schema
│
├── Crawl4AI/
│   └── Crawl4AI_scrapper.py         # Standalone async web scraper
│
├── RAG_scrapping/
│   └── Vectorize_Store_Quadrant.py  # Qdrant helpers (store / search)
│
└── RAG-Langgraph/                   # Full multi-agent RAG system
    ├── state.py                     # Shared AgentState TypedDict
    ├── supervisor.py                # LLM-based routing (research / rag / END)
    ├── agents.py                    # Research + RAG ReAct agents
    ├── graph.py                     # LangGraph StateGraph wiring
    ├── tools.py                     # search_web, crawl_and_store, retrieve
    ├── vector_store.py              # Qdrant CRUD + deduplication
    ├── Crawl4AI_scrapper.py         # Enhanced scraper (timeouts, exception isolation)
    ├── query_memory_manager.py      # Cross-session JSON topic memory
    ├── main.py                      # CLI entry point
    └── ui.py                        # Streamlit chat UI
```

---

## Learning Progression

The repo is designed as a step-by-step build-up:

```
Level 1 ── Simple LLM Chain
           prompt → LLM → output parser

Level 2 ── Manual ReAct Loop
           messages → LLM → tool_calls → tools → ToolMessage → repeat

Level 3 ── Prebuilt ReAct Agent (structured output)
           create_agent(llm, tools, response_format=PydanticModel)

Level 4 ── Multi-Agent LangGraph System
           Supervisor orchestrates Research + RAG agents
           persistent vector store, cross-session memory, Streamlit UI
```

---

## Key Concepts & Math

### Text Embeddings & Similarity

The RAG system uses `sentence-transformers/all-MiniLM-L6-v2` to convert text into **384-dimensional** vectors. Embeddings are L2-normalised, which means cosine similarity reduces to a simple dot product:

```
similarity(A, B) = A · B       (since ‖A‖ = ‖B‖ = 1)
```

Qdrant uses this to rank retrieved chunks — score of 1.0 is identical content, 0.5 is unrelated.

### ReAct Loop

The manual agent in `LangChain_Primitives.py` implements the Reasoning + Acting pattern from first principles:

```
for each iteration:
    response = LLM(messages)
    if no tool_calls → return response    # done
    observation = execute(tool_calls[0])
    messages += [AI message, ToolMessage(observation)]
```

Each tool call appends two messages, giving the model a growing scratchpad. One tool per step keeps reasoning traceable.

### Structured Output with Chain-of-Thought

The Supervisor forces the LLM to reason before routing by placing `reasoning` before `next` in the Pydantic schema — the model must generate its chain-of-thought first:

```python
class SupervisorDecision(BaseModel):
    reasoning: str                            # written first
    next: Literal["research", "rag", "END"]   # decided after
```

### Content Filtering

The web scraper uses `PruningContentFilter` to discard nav bars, ads, and low-density blocks before chunking. Thresholds are tuned per domain:

| Domain | Threshold | Why |
|--------|-----------|-----|
| blogspot.com | 0.6 (fixed) | High nav/ad noise |
| medium.com | 0.3 (fixed) | Clean body text |
| wikipedia.org | 0.5 (dynamic) | Structured sections |
| default | 0.4 (dynamic) | Conservative balance |

---

## Module Summaries

### `Info_Summary_Chain_Ollama` — LCEL Basics

Introduces the pipe (`|`) operator to compose a `PromptTemplate | ChatOllama | StrOutputParser` chain. The template enforces a structured two-paragraph output (summary + interesting fact). Runs locally via Ollama with `gemma3:1b`.

---

### `React_Under_the_hood` — ReAct from Scratch

Builds the ReAct loop manually with no `AgentExecutor` or `create_react_agent`. Tools:

- `get_product_price(product)` — dict lookup
- `apply_discount(price, tier)` — `final = price × (1 − rate)` with gold/silver/bronze tiers

LangSmith tracing enabled via `@traceable`. Runs locally with `qwen3:1.7b` via Ollama.

---

### `React_Search_Agent_Tavily_Pydantic` — Structured Agent Output

Two variants of a prebuilt ReAct agent with Pydantic-enforced structured responses:

- **v1** — Custom `@tool(args_schema=Search_Format)` with `query`, `country`, `topic` fields. Final output is a plain `Ans_format(answer: str)`.
- **v2** — Uses native `TavilySearch` tool. Final output is a nested `List[JobOffer]` with description, URL, experience level, and notable tools — demonstrating richer structured extraction.

---

### `Crawl4AI` — Async Web Scraper

Crawls URLs concurrently (up to 5 at once via `asyncio.Semaphore`) using Playwright with `playwright_stealth` to bypass bot detection. Pipeline:

```
URL → Stealth Playwright → PruningContentFilter → Markdown → Header-based chunking → List[Document]
```

A Greek-character ratio check (`> 10%`) discards encoding artifacts before chunks reach the vector store.

---

### `RAG_scrapping` — Qdrant Vector Store Layer

Standalone Qdrant CRUD helpers used as a foundation before the full multi-agent system. Supports optional metadata filters alongside semantic search (filter by `source_url`, `page_title`, etc.).

---

### `RAG-Langgraph` — Full Multi-Agent System

The flagship project. A LangGraph `StateGraph` where a Supervisor routes between a Research agent and a RAG agent based on session state and cross-session memory.

#### Architecture

```
              Streamlit UI (ui.py)
                     │
              LangGraph StateGraph
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
    Supervisor ──────▶ Research Agent
         ▲                (search_web + crawl_and_store)
         │                       │
         └────────┐          Crawl4AI + DuckDuckGo
                  │
                  ▼
             RAG Agent
        (retrieve_from_vectorstore)
                  │
           Qdrant (localhost:6333)
           all-MiniLM-L6-v2, 384d, cosine
```

#### Key Design Decisions

| Decision | Why |
|----------|-----|
| Python checks before LLM routing | Deterministic rules are cheaper and more reliable |
| `iterations` counter cap (max 5) | Prevents infinite loops in the graph |
| Strip ToolMessages before agent calls | Stops ReAct agents re-issuing identical tool calls |
| URL deduplication in vector store | Same page never crawled or stored twice |
| `reasoning` field before `next` in schema | Forces chain-of-thought before routing decision |
| `ThreadPoolExecutor` for Playwright on Windows | Streamlit's SelectorEventLoop can't run subprocesses |

#### Supervisor Routing Rules

```
1. Research was done this session                           → rag
2. Query is conversational / about history / a greeting    → rag
3. Query subject noun exists in past-session memory        → rag
4. Default                                                 → research
```

Past queries are persisted in `query_memory.json` so the supervisor avoids re-crawling topics from previous sessions.

#### End-to-End Flow

```
User: "Who are the best swimmers of all time?"
  → Supervisor: no past research → "research"
  → Research agent: DuckDuckGo → 5 URLs → Crawl4AI → Qdrant (384-dim chunks stored)
  → Supervisor: research done → "rag"
  → RAG agent: similarity_search(k=4) → LLM generates cited answer
  → Supervisor: rag_completed=True → "END"
  → ui.py: streams tokens live, saves to chat_history.json, logs query to query_memory.json
```

---

## Setup & Running

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) for local-model modules
- [Qdrant](https://qdrant.tech/) on `localhost:6333` (`docker run -p 6333:6333 qdrant/qdrant`)
- OpenAI API key (RAG-Langgraph + Tavily agents)
- Tavily API key (Tavily agents)

### `.env`

```env
OPENAI_API_KEY=...
TAVILY_API_KEY=...
LANGCHAIN_API_KEY=...        # optional, LangSmith tracing
LANGCHAIN_TRACING_V2=true
```

### Running each module

```bash
python Info_Summary_Chain_Ollama/Summary_conv.py
python React_Under_the_hood/LangChain_Primitives.py
python React_Search_Agent_Tavily_Pydantic/Search_Agent_Tavily_Pydantic.py
python React_Search_Agent_Tavily_Pydantic/Search_Agent_Tavily_Pydantic_Tavily_Tool.py
python Crawl4AI/Crawl4AI_scrapper.py

# RAG-Langgraph
cd RAG-Langgraph
python main.py              # CLI
streamlit run ui.py         # Streamlit UI
```

### Install

```bash
pip install langchain langchain-openai langchain-ollama langchain-huggingface \
            langchain-qdrant langchain-tavily langgraph \
            crawl4ai playwright playwright-stealth \
            qdrant-client sentence-transformers \
            duckduckgo-search tavily-python \
            streamlit python-dotenv langsmith pydantic
playwright install
```
