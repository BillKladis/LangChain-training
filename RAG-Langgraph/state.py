from typing import Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    urls: List[str]
    next: str