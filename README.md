# LangChain Training — Agentic AI Systems

A hands-on collection of progressively more complex LangChain / LangGraph systems, spanning simple LLM chains, ReAct agents built from scratch, structured Pydantic output agents, and a full multi-agent RAG pipeline with a Streamlit UI.

---

## Table of Contents

1. [Repository Map](#repository-map)
2. [System Design Overview](#system-design-overview)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Module Deep-Dives](#module-deep-dives)
   - [Info\_Summary\_Chain\_Ollama](#1-info_summary_chain_ollama)
   - [React\_Under\_the\_hood](#2-react_under_the_hood)
   - [React\_Search\_Agent\_Tavily\_Pydantic](#3-react_search_agent_tavily_pydantic)
   - [Crawl4AI (standalone)](#4-crawl4ai-standalone)
   - [RAG\_scrapping](#5-rag_scrapping)
   - [RAG-Langgraph (flagship)](#6-rag-langgraph-flagship)
5. [Component Architecture](#component-architecture)
6. [Data Flow — End-to-End](#data-flow--end-to-end)
7. [Setup & Running](#setup--running)

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
    ├── state.py
    ├── supervisor.py
    ├── agents.py
    ├── graph.py
    ├── tools.py
    ├── vector_store.py
    ├── Crawl4AI_scrapper.py
    ├── query_memory_manager.py
    ├── main.py
    └── ui.py
```

---

## System Design Overview

The repo follows a learning progression from primitive to production-grade:

```
Level 1 ── Simple LLM Chain
           prompt → LLM → output parser
                       ↓
Level 2 ── Manual ReAct Loop
           messages → LLM → tool_calls → tools → ToolMessage → repeat
                       ↓
Level 3 ── Prebuilt ReAct Agent (structured output)
           create_agent(llm, tools, response_format=PydanticModel)
                       ↓
Level 4 ── Multi-Agent LangGraph System
           Supervisor orchestrates Research + RAG agents
           persistent vector store, cross-session memory, Streamlit UI
```

### Core Design Principles

| Principle | Implementation |
|-----------|----------------|
| Separation of concerns | Crawler / VectorStore / Agent / Graph / UI are decoupled |
| Idempotent ingestion | URLs tracked in Qdrant; re-crawling skipped automatically |
| Finite state control | `iterations` counter prevents infinite LangGraph loops |
| Deterministic routing | Supervisor reasons in Python first, LLM only resolves ambiguity |
| Clean context passing | ToolMessages and tool-call AI messages stripped before passing history to agents |

---

## Mathematical Foundations

### 1. Text Embeddings

The system uses `sentence-transformers/all-MiniLM-L6-v2`, which maps a sentence of arbitrary length to a fixed **384-dimensional** dense vector:

```
f(text) → v ∈ ℝ³⁸⁴
```

`encode_kwargs={"normalize_embeddings": True}` applies L2-normalisation:

```
v̂ = v / ‖v‖₂     where ‖v̂‖₂ = 1
```

### 2. Cosine Similarity (Qdrant COSINE distance)

For two documents A and B, semantic similarity is:

```
cos(θ) = (A · B) / (‖A‖ · ‖B‖)
```

Because embeddings are L2-normalised (`‖A‖ = ‖B‖ = 1`), this simplifies to a pure dot product:

```
cos(θ) = A · B = Σᵢ Aᵢ Bᵢ
```

Qdrant internally remaps to a [0, 1] score for ranking:

```
score = (1 + cos(θ)) / 2
```

A score of **1.0** means identical content; **0.5** means orthogonal (unrelated).

### 3. Pruning Content Filter

`PruningContentFilter(threshold=t, threshold_type="fixed"|"dynamic")` scores each content block by its information density relative to the page. Blocks whose score falls below `t` are discarded:

```
keep_block ⟺ density_score(block) ≥ t
```

Domain-tuned thresholds:

| Domain | Threshold | Type | Why |
|--------|-----------|------|-----|
| blogspot.com | 0.6 | fixed | High nav/ad noise |
| medium.com | 0.3 | fixed | Clean body text |
| wikipedia.org | 0.5 | dynamic | Structured sections |
| default | 0.4 | dynamic | Conservative balance |

### 4. Greek-Character Noise Filter

After chunking, blocks with a Greek-character ratio above 10% are discarded. This catches pages where the Latin-script parser produces encoding artifacts:

```
greek_ratio = |{c ∈ text : c ∈ [U+0370–U+03FF] ∪ [U+1F00–U+1FFF]}| / |text|
is_noise ⟺ greek_ratio > 0.1
```

### 5. ReAct (Reasoning + Acting) Loop

The manual loop in `LangChain_Primitives.py` implements the original ReAct paper algorithm:

```
messages = [SystemMessage, HumanMessage]
for i in 1..MAX_ITERATIONS:
    response = LLM(messages)
    if response.tool_calls == []:
        return response          # final answer
    tool_call = response.tool_calls[0]
    observation = execute(tool_call)
    messages += [response, ToolMessage(observation)]
```

Each tool call adds two messages (AI message + ToolMessage), giving the model a growing "scratchpad". One tool is called per iteration to keep reasoning steps traceable.

### 6. Structured Output with Chain-of-Thought

The Supervisor uses:

```python
class SupervisorDecision(BaseModel):
    reasoning: str   # generated first
    next: Literal["research", "rag", "END"]
```

By placing `reasoning` **before** `next` in the schema, the model is forced to write its chain-of-thought before committing to a routing decision—a direct application of "think before you answer" forcing via JSON field ordering.

---

## Module Deep-Dives

### 1. `Info_Summary_Chain_Ollama`

**File:** `Summary_conv.py`

**Purpose:** Introduces LangChain Expression Language (LCEL) with the pipe (`|`) operator.

**Architecture:**

```
PromptTemplate(template) | ChatOllama(gemma3:1b) | StrOutputParser()
```

The template enforces structured two-paragraph output (summary + interesting fact). The `information` variable is injected at `chain.invoke({"information": text})`.

**Key concept:** LCEL builds a **Runnable** pipeline where each component implements `.invoke()`, `.stream()`, and `.batch()`. The pipe operator composes them left-to-right via `RunnableSequence`.

---

### 2. `React_Under_the_hood`

**File:** `LangChain_Primitives.py`

**Purpose:** Implements the ReAct agent loop from first principles — no `create_react_agent`, no `AgentExecutor`.

**Tools:**

| Tool | Input | Output | Logic |
|------|-------|--------|-------|
| `get_product_price` | `product: str` | `float` | Dict lookup |
| `apply_discount` | `price: float, tier: str` | `float` | `final = price × (1 − discount_rate)` |

Discount rates: gold=20%, silver=10%, bronze=5%.

**Agent Loop:**

```python
llm_tools = llm.bind_tools(tools)   # tools schemas injected into system prompt
for i in range(MAX_ITERATIONS):
    ai_message = llm_tools.invoke(messages)
    if not ai_message.tool_calls:
        return ai_message.content    # done
    # execute exactly one tool, append result as ToolMessage
```

**LangSmith tracing** is enabled via `@traceable(name="LangChain Agent Loop")`, which logs the entire message sequence per run.

**Model:** `qwen3:1.7b` via Ollama (local inference, no API key required).

---

### 3. `React_Search_Agent_Tavily_Pydantic`

#### `Search_Agent_Tavily_Pydantic.py`

**Purpose:** Prebuilt ReAct agent with a custom Tavily search tool and Pydantic structured final output.

**Tool Schema (Pydantic-enforced):**

```python
class Search_Format(BaseModel):
    query: str
    country: str
    topic: Literal["general", "news", "finance"]
```

`@tool(args_schema=Search_Format)` tells the agent exactly what fields to produce for each search call, preventing hallucinated arguments.

**Output Schema:**

```python
class Ans_format(BaseModel):
    answer: str
```

`create_agent(model, tools, response_format=Ans_format)` wraps the final generation step in a structured output call, guaranteeing the response is always a valid `Ans_format` object.

#### `Search_Agent_Tavily_Pydantic_Tavily_Tool.py`

**Purpose:** Extends the pattern with a richer nested Pydantic output and the native `TavilySearch` tool.

**Output Schema:**

```python
class JobOffer(BaseModel):
    job_description: str
    url: str
    experience_level: str
    notable_tools: List[str]

class Ans_format(BaseModel):
    offers: List[JobOffer]
```

The agent searches for AI engineering jobs and deserialises each result into a typed `JobOffer` object with description, URL, experience level, and tools required.

---

### 4. `Crawl4AI` (standalone)

**File:** `Crawl4AI/Crawl4AI_scrapper.py`

**Purpose:** Standalone async web scraper for feeding documents into a vector store.

**Pipeline:**

```
URL list
  → asyncio.Semaphore(5)       # max 5 concurrent pages
  → Playwright + Stealth       # bot-detection bypass
  → PruningContentFilter       # remove nav/ads
  → DefaultMarkdownGenerator   # HTML → Markdown
  → MarkdownHeaderTextSplitter # split on H1/H2/H3
  → Greek-noise filter         # discard encoding artifacts
  → List[Document]
```

**Stealth hook:**

```python
async def on_page_context_created(page, ...):
    await stealth.apply_stealth_async(page)
```

`playwright_stealth` patches browser fingerprints (WebGL, navigator.plugins, etc.) before any page script runs, preventing Cloudflare / Distil fingerprinting.

**Concurrency model:** `asyncio.Semaphore(5)` limits open browser pages. Tasks are gathered with `asyncio.gather(*tasks)`, so crawling is concurrent, not sequential.

---

### 5. `RAG_scrapping`

**File:** `Vectorize_Store_Quadrant.py`

**Purpose:** Low-level Qdrant vector store CRUD layer, used as the foundation before the full LangGraph system.

**Collection config:**
- Embedding dimension: **384** (all-MiniLM-L6-v2)
- Distance metric: **COSINE**
- Collection name: `Scrapping_for_RAG`

**Similarity search with optional metadata filter:**

```python
Filter(must=[FieldCondition(key="metadata.field", match=MatchValue(value=v))])
```

This enables filtering by `source_url`, `page_title`, or `description` alongside semantic similarity — a hybrid retrieval pattern.

---

### 6. `RAG-Langgraph` (flagship)

The centrepiece of the repo: a multi-agent system orchestrated by LangGraph.

#### Architecture Diagram

```
                    ┌──────────────────────────────────────────┐
                    │              Streamlit UI                │
                    │  (ui.py — conversation persistence,      │
                    │   streaming token render, sidebar nav)   │
                    └─────────────────┬────────────────────────┘
                                      │ graph_inputs (messages, query, ...)
                                      ▼
                    ┌──────────────────────────────────────────┐
                    │           LangGraph StateGraph           │
                    │                                          │
                    │  ┌──────────┐     ┌──────────────────┐  │
                    │  │Supervisor│────▶│  Research Agent  │  │
                    │  │          │◀────│  (search_web +   │  │
                    │  │ routes:  │     │  crawl_and_store)│  │
                    │  │ research │     └──────────────────┘  │
                    │  │ rag      │                            │
                    │  │ END      │     ┌──────────────────┐  │
                    │  └──────────┘────▶│    RAG Agent     │  │
                    │       ▲    ◀──────│  (vectorstore    │  │
                    │       │           │   retrieval)     │  │
                    │       └───────────└──────────────────┘  │
                    └──────────────────────────────────────────┘
                              │                  │
              ┌───────────────┘                  └─────────────────┐
              ▼                                                     ▼
  ┌──────────────────────┐                          ┌──────────────────────────┐
  │   DuckDuckGo (DDGS)  │                          │  Qdrant Vector Store     │
  │   (web URL search)   │                          │  (localhost:6333)        │
  └──────────────────────┘                          │  all-MiniLM-L6-v2 384d  │
              │                                     │  cosine similarity       │
              ▼                                     └──────────────────────────┘
  ┌──────────────────────┐
  │  Crawl4AI + Stealth  │
  │  (async web crawl)   │
  └──────────────────────┘
```

---

#### `state.py` — Shared State

```python
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    urls: List[str]
    crawled_urls: List[str]
    next: str           # supervisor routing decision
    iterations: int     # loop guard
    rag_completed: bool # prevents duplicate RAG runs
```

`add_messages` is a LangGraph reducer that merges new messages into the list rather than replacing it, enabling multi-turn message accumulation across nodes.

---

#### `supervisor.py` — Routing Logic

The supervisor applies **deterministic Python rules first**, only asking the LLM to resolve genuinely ambiguous cases:

```
Rule 1: research_done_this_session → rag       (Python check on message content)
Rule 2: conversational/greeting/history → rag  (LLM judgement)
Rule 3: subject noun in past research topics → rag  (LLM judgement, strict)
Rule 4: default → research
```

**Past-session memory** is loaded from `query_memory.json` and injected as context, giving the supervisor awareness of topics researched in previous runs without re-crawling.

**Safety gate:** `iterations >= 5` forces `END` regardless of LLM output, preventing infinite loops.

---

#### `graph.py` — LangGraph StateGraph

```
Entry: supervisor
  │
  ├─ "research" ──▶ run_research_agent ──▶ supervisor
  ├─ "rag"      ──▶ run_rag_agent      ──▶ supervisor
  └─ "END"      ──▶ END
```

Both agent nodes strip `ToolMessage` and AI tool-call messages from the history before passing it to the agent:

```python
clean_history = [
    m for m in state["messages"]
    if not isinstance(m, ToolMessage)
    and not getattr(m, "tool_calls", None)
]
```

This prevents the ReAct agents from seeing old search results and re-issuing identical tool calls.

---

#### `tools.py` — Tool Implementations

| Tool | Description | Returns |
|------|-------------|---------|
| `search_web(query)` | DuckDuckGo text search, max 5 results | `List[str]` (URLs) |
| `crawl_and_store(urls)` | Crawls new URLs via Crawl4AI, stores chunks in Qdrant | Status string |
| `retrieve_from_vectorstore(query)` | Top-4 cosine-similar chunks with source metadata | Formatted context string |

`crawl_and_store` checks `get_crawled_urls()` before crawling, making it **idempotent** — the same URL is never crawled twice.

**Windows compatibility:** Crawling runs in a `ThreadPoolExecutor` worker so a `ProactorEventLoop` can be set before Playwright starts (Streamlit's default `SelectorEventLoop` doesn't support subprocesses on Windows).

---

#### `vector_store.py` — Qdrant Layer

**Singleton pattern:** `_embeddings` and `_client` are module-level singletons, initialised once per process. The `HuggingFaceEmbeddings` model download happens at import time.

**Duplicate detection:**

```python
def get_crawled_urls() -> set:
    results = _client.scroll(collection_name=..., limit=10000)
    return {point.payload["metadata"]["source_url"] for point in results[0]}
```

Before storing, `store_documents` computes `new_docs = [d for d in documents if d.metadata["source_url"] not in existing_urls]`, ensuring deduplication at the document level.

---

#### `Crawl4AI_scrapper.py` (RAG-Langgraph version)

Enhanced version of the standalone scrapper with two additions:

1. **Per-page timeout:**
   ```python
   content = await asyncio.wait_for(crawler.arun(url, config=run_cfg), timeout=25)
   ```
   Prevents a single slow page from blocking the semaphore slot indefinitely.

2. **Exception isolation:**
   ```python
   results = await asyncio.gather(*tasks, return_exceptions=True)
   results = [r for r in results if not isinstance(r, Exception)]
   ```
   One failed crawl does not abort the entire batch.

---

#### `query_memory_manager.py` — Cross-Session Memory

Simple JSON-backed persistence for topic tracking:

```json
["What is baseball and who are the top 10 best players of all time?", ...]
```

On each successful RAG response, `add_query(query)` appends the query string. On the next session, the supervisor loads this list and uses it to avoid redundant research crawls for already-known topics.

---

#### `ui.py` — Streamlit Chat Interface

**Conversation persistence:** Each conversation is a UUID-keyed entry in `chat_history.json`. `datetime` objects are serialised to ISO format strings and deserialised on load.

**Streaming architecture:**

```python
for chunk, metadata in app.stream(graph_inputs, stream_mode="messages"):
    node = metadata.get("langgraph_node", "")
    if node == "rag":
        content = getattr(chunk, "content", "")
        full_response += content
        answer_slot.markdown(full_response + "▌")   # live cursor
```

`stream_mode="messages"` emits individual `BaseMessage` chunks as they are generated. Only chunks from the `"rag"` node that are plain AI text (no tool-call payloads) are rendered, avoiding flickering from internal tool messages.

**Auto-naming:** A conversation is named after its first user prompt (truncated to 35 characters) as soon as the first assistant reply arrives.

---

## Component Architecture

### Dependency Graph

```
ui.py
 └── graph.py
      ├── state.py
      ├── supervisor.py
      │    └── query_memory_manager.py
      └── agents.py
           └── tools.py
                ├── Crawl4AI_scrapper.py
                │    └── [crawl4ai, playwright_stealth, langchain_text_splitters]
                └── vector_store.py
                     └── [qdrant_client, langchain_huggingface, langchain_qdrant]
```

### LLM Usage by Component

| Component | Model | Provider | Why |
|-----------|-------|----------|-----|
| Summary chain | gemma3:1b | Ollama (local) | No API key, offline demo |
| ReAct primitives | qwen3:1.7b | Ollama (local) | Low-cost tool-calling test |
| Tavily agents | gpt-4.1-nano | OpenAI | Strong instruction-following |
| Supervisor | gpt-4.1-nano | OpenAI | Reliable structured output |
| Research agent | gpt-4.1-nano | OpenAI | Multi-step tool chaining |
| RAG agent | gpt-4.1-nano | OpenAI | Citation-aware generation |

---

## Data Flow — End-to-End

A full query through the RAG-Langgraph system:

```
1. User types: "Who are the best swimmers of all time?"
      │
2. ui.py → graph_inputs = {messages, query, iterations=0, rag_completed=False}
      │
3. supervisor.py
   ├── No research this session, topic not in memory → decision: "research"
   └── {"next": "research"}
      │
4. run_research_agent (graph.py)
   ├── research_agent calls search_web("best swimmers of all time")
   │     └── DuckDuckGo → 5 URLs
   ├── research_agent calls crawl_and_store([url1, url2, ...])
   │     ├── Crawl4AI fetches pages (max 5 concurrent, 25s timeout each)
   │     ├── PruningContentFilter + MarkdownHeaderTextSplitter
   │     ├── Greek-noise filter
   │     └── QdrantVectorStore.add_documents(chunks) → 384-dim cosine index
   └── "Crawled and stored N chunks from M URLs"
      │
5. supervisor.py (2nd call)
   └── research_done_this_session=True → decision: "rag"
      │
6. run_rag_agent (graph.py)
   ├── rag_agent calls retrieve_from_vectorstore("best swimmers of all time")
   │     └── vectorstore.similarity_search(query, k=4)
   │           score = A·B  (normalised dot product)
   └── rag_agent generates answer citing page_title + source_url
      │
7. supervisor.py (3rd call)
   └── rag_completed=True → decision: "END"
      │
8. ui.py
   ├── Streams tokens from node="rag" to answer_slot (live cursor ▌)
   ├── Saves assistant reply to chat_history.json
   └── add_query("Who are the best swimmers of all time?") → query_memory.json
```

---

## Setup & Running

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) (for local model modules)
- [Qdrant](https://qdrant.tech/) running on `localhost:6333`
- OpenAI API key (for RAG-Langgraph)
- Tavily API key (for Tavily agents)

### Environment Variables (`.env`)

```env
OPENAI_API_KEY=...
TAVILY_API_KEY=...
LANGCHAIN_API_KEY=...          # optional, for LangSmith tracing
LANGCHAIN_TRACING_V2=true      # optional
```

### Qdrant (Docker)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Module-by-module

```bash
# 1. Simple summary chain (Ollama)
python Info_Summary_Chain_Ollama/Summary_conv.py

# 2. Manual ReAct agent (Ollama)
python React_Under_the_hood/LangChain_Primitives.py

# 3. Tavily search agent
python React_Search_Agent_Tavily_Pydantic/Search_Agent_Tavily_Pydantic.py
python React_Search_Agent_Tavily_Pydantic/Search_Agent_Tavily_Pydantic_Tavily_Tool.py

# 4. Standalone web scraper
python Crawl4AI/Crawl4AI_scrapper.py

# 5. RAG-Langgraph — CLI
cd RAG-Langgraph && python main.py

# 6. RAG-Langgraph — Streamlit UI
cd RAG-Langgraph && streamlit run ui.py
```

### Install Dependencies

```bash
pip install langchain langchain-openai langchain-ollama langchain-huggingface \
            langchain-qdrant langchain-tavily langgraph \
            crawl4ai playwright playwright-stealth \
            qdrant-client sentence-transformers \
            duckduckgo-search tavily-python \
            streamlit python-dotenv langsmith pydantic
playwright install
```
