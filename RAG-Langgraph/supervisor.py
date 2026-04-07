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

    decision = llm.with_structured_output(SupervisorDecision).invoke([
        SystemMessage("""You are a supervisor managing two agents:
        - research: searches the web and stores content in the vector DB. Call this FIRST.
        - rag: retrieves from vector DB and generates an answer. Call this AFTER research.
        - END: the task is complete.

        Decision rules:
        - If no research has been done yet → research
        - If research is complete → rag
        - If rag has generated an answer → END
        """),
        HumanMessage(content=f"Query: {query}\n\nConversation so far:\n{messages}")
    ])

    print(f"Supervisor → {decision.next}: {decision.reasoning}")
    return {"next": decision.next}