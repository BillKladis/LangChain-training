from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from pydantic import BaseModel
from typing import Literal

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

class SupervisorDecision(BaseModel):
    reasoning: str          # generated first — forces the model to reason before committing
    next: Literal["research", "rag", "END"]

def supervisor(state):
    from query_memory_manager import load_query_memory, add_query

    messages = state["messages"]
    query = state["query"]
    iterations = state.get("iterations", 0)
    rag_completed = state.get("rag_completed", False)

    if iterations >= 5:
        print("Max iterations reached, forcing END")
        return {"next": "END", "iterations": iterations + 1}

    if rag_completed:
        print("Supervisor → END: RAG already completed")
        add_query(query)
        return {"next": "END", "iterations": iterations + 1}

    # Determine in Python whether crawling happened this session — do NOT ask the LLM to infer it.
    research_done_this_session = any(
        "Crawled and stored" in str(getattr(m, "content", ""))
        for m in messages
    )

    previous_queries = load_query_memory()
    memory_context = (
        f"Topics researched in past sessions: {previous_queries}"
        if previous_queries else "No past research."
    )

    # Pass only clean Q&A turns to keep the context short and unambiguous.
    clean_turns = [
        m for m in messages
        if not isinstance(m, ToolMessage) and not getattr(m, "tool_calls", None)
    ]

    decision = llm.with_structured_output(SupervisorDecision).invoke([
        SystemMessage("""You are a routing supervisor. Think step by step, then output the decision.

        Rules (apply in order, stop at first match):
        1. Research was already done this session → rag
        2. Query is conversational, a greeting, asks about chat history, or is a personal opinion → rag
        3. The core SUBJECT NOUN of the query (e.g. "swimmers", "ballet", "F1") is explicitly present in the previously researched topics list → rag
           WARNING: question format similarity ("top 10") does NOT count. Only the subject noun matters.
        4. Default → research

        Be strict about rule 3: if you have any doubt that the exact subject was researched, apply rule 4.
        """),
        HumanMessage(content=f"""Current query: "{query}"

Research done this session: {"YES — data is in the vector DB" if research_done_this_session else "NO"}
{memory_context}

Recent conversation:
{clean_turns[-6:] if len(clean_turns) > 6 else clean_turns}

Next step?""")
    ])

    print(f"Supervisor → {decision.next}: {decision.reasoning}")
    return {"next": decision.next, "iterations": iterations + 1}