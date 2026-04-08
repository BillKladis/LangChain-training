from langgraph.graph import StateGraph, END
from state import AgentState
from supervisor import supervisor
from agents import research_agent, rag_agent
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

def run_research_agent(state, config: RunnableConfig = None):
    # Same clean-history filter as the RAG agent: strip ToolMessages and AI
    # tool-call messages so the ReAct loop doesn't see old search/crawl results
    # and repeat them as new calls.
    clean_history = [
        m for m in state["messages"]
        if not isinstance(m, ToolMessage)
        and not getattr(m, "tool_calls", None)
    ]
    result = research_agent.invoke({"messages": clean_history}, config=config)
    return {"messages": result["messages"]}

def run_rag_agent(state, config: RunnableConfig = None):
    # Pass clean conversation history (no ToolMessages, no AI tool-call messages)
    # so the RAG agent has full Q&A context for follow-ups and history questions.
    clean_history = [
        m for m in state["messages"]
        if not isinstance(m, ToolMessage)
        and not getattr(m, "tool_calls", None)
    ]
    result = rag_agent.invoke({"messages": clean_history}, config=config)
    return {"messages": result["messages"], "rag_completed": True}

def route(state):
    return state["next"]

graph = StateGraph(AgentState)

graph.add_node("supervisor", supervisor)
graph.add_node("research", run_research_agent)
graph.add_node("rag", run_rag_agent)

graph.set_entry_point("supervisor")

graph.add_conditional_edges("supervisor", route, {
    "research": "research",
    "rag": "rag",
    "END": END
})

graph.add_edge("research", "supervisor")
graph.add_edge("rag", "supervisor")

app = graph.compile()