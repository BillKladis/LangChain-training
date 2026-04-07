import asyncio
from typing import List
from langchain.tools import tool
from duckduckgo_search import DDGS
from Crawl4AI import crawl_web
from vector_store import store_documents, search_vectorstore

@tool
def search_web(query: str) -> List[str]:
    """Search the web using DuckDuckGo and return a list of URLs relevant to the query."""
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)
        urls = [r["href"] for r in results]
        print(f"Found URLs: {urls}")
        return urls

@tool
def crawl_and_store(urls: List[str]) -> str:
    """Crawl a list of URLs, extract content, and store in the vector database."""
    docs = asyncio.run(crawl_web(urls))
    store_documents(docs)
    return f"Crawled and stored {len(docs)} chunks from {len(urls)} URLs"

@tool
def retrieve_from_vectorstore(query: str) -> str:
    """Search the vector database for chunks relevant to the query."""
    results = search_vectorstore(query=query, k=4)
    if not results:
        return "No relevant information found in the knowledge base."
    context = "\n\n---\n\n".join(
        f"Source: {r.metadata.get('page_title', 'Unknown')}\n"
        f"URL: {r.metadata.get('source_url', '')}\n\n"
        f"{r.page_content}"
        for r in results
    )
    return context