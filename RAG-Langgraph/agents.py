from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from tools import search_web, crawl_and_store, retrieve_from_vectorstore

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

research_agent = create_react_agent(
    model=llm,
    tools=[search_web, crawl_and_store],
    prompt=SystemMessage("""You are a research agent. You MUST follow these steps in order:
    1. Call search_web with the user query to get URLs
    2. Call crawl_and_store with the URLs returned from step 1
    You MUST call BOTH tools. Do not stop after search_web alone.
    Only report complete after crawl_and_store has been called and returned a result.""")
)

rag_agent = create_react_agent(
    model=llm,
    tools=[retrieve_from_vectorstore],
    prompt=SystemMessage("""You are a RAG agent. You MUST follow these steps:
    1. Call retrieve_from_vectorstore with the user query
    2. Generate a clear answer based on the retrieved context
    Always cite sources by mentioning page title and URL.
    After generating the answer, stop immediately.""")
)