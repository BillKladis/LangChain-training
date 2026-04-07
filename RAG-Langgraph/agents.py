from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from tools import search_web, crawl_and_store, retrieve_from_vectorstore

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

research_agent = create_react_agent(
    model=llm,
    tools=[search_web, crawl_and_store],
    state_modifier=SystemMessage("""You are a research agent. Your job is:
    1. Search the web for URLs relevant to the user query using search_web
    2. Crawl those URLs and store their content using crawl_and_store
    Once both steps are done, report back that research is complete.""")
)

rag_agent = create_react_agent(
    model=llm,
    tools=[retrieve_from_vectorstore],
    state_modifier=SystemMessage("""You are a RAG agent. Your job is:
    1. Retrieve relevant information from the vector database using retrieve_from_vectorstore
    2. Generate a clear, accurate answer based on the retrieved context
    Always cite your sources by mentioning the page title and URL.""")
)