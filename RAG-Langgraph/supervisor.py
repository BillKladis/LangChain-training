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
    messages = state["messages"]
    query = state["query"]
    iterations = state.get("iterations", 0)

    if iterations >= 5:
        print("Max iterations reached, forcing END")
        return {"next": "END", "iterations": iterations + 1}

    decision = llm.with_structured_output(SupervisorDecision).invoke([
        SystemMessage("""You are a supervisor managing two agents:
        - research: searches the web and stores content in the vector DB.
        - rag: retrieves from vector DB and generates an answer.
        - END: the task is complete.

        Decision rules:
        - If no research has been attempted yet → research
        - If research was attempted but returned empty URLs or 0 chunks → research again
        - If you see a message containing 'Crawled and stored' with chunks > 0 AND rag has NOT run yet → rag
        - If rag has run and produced a final answer → END immediately

        CRITICAL: Only route to rag ONCE. After rag produces any answer, return END.
        CRITICAL: Do NOT route to rag unless you see 'Crawled and stored' in the messages.
        """),
        HumanMessage(content=f"Query: {query}\n\nConversation so far:\n{messages}")
    ])

    print(f"Supervisor → {decision.next}: {decision.reasoning}")
    return {"next": decision.next, "iterations": iterations + 1}