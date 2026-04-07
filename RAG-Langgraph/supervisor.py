from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from typing import Literal

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

class SupervisorDecision(BaseModel):
    next: Literal["research", "rag", "END"]
    reasoning: str

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
        add_query(query)  # save only here, after RAG actually ran
        return {"next": "END", "iterations": iterations + 1}

    previous_queries = load_query_memory()
    memory_context = f"Previously researched: {previous_queries}" if previous_queries else "No previous research."

    decision = llm.with_structured_output(SupervisorDecision).invoke([
        SystemMessage("""You are a supervisor managing two agents:
        - research: searches the web and stores content in the vector DB.
        - rag: retrieves from vector DB and generates an answer.
        - END: the task is complete.

        Decision rules:
        - If the query is covered by previously researched topics → rag directly, skip research
        - If no relevant previous research exists → research first
        - If research was attempted but 0 chunks stored → research again
        - If you see 'Crawled and stored' in messages AND rag has not run yet → rag
        - If rag has produced a final answer → END

        CRITICAL: Only call rag ONCE. After rag answers, return END.
        """),
        HumanMessage(content=f"""Query: {query}

{memory_context}

Conversation so far:
{messages}

What should we do next?""")
    ])

    print(f"Supervisor → {decision.next}: {decision.reasoning}")
    return {"next": decision.next, "iterations": iterations + 1}