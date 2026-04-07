from langgraph.graph import StateGraph, END
from state import AgentState
from supervisor import supervisor
from agents import research_agent, rag_agent
from langchain_core.messages import HumanMessage

def run_research_agent(state):
    result = research_agent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"]}

def run_rag_agent(state):
    # pass query explicitly so RAG doesn't get confused by conversation history
    result = rag_agent.invoke({
        "messages": [HumanMessage(content=state["query"])]
    })
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