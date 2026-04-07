import asyncio
from typing import List
from langchain.tools import tool
from ddgs import DDGS
from Crawl4AI_scrapper import crawl_web
from vector_store import store_documents, search_vectorstore, get_crawled_urls

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
    """Crawl a list of URLs, extract content, and store in the vector database. Skips already crawled URLs."""
    existing = get_crawled_urls()
    new_urls = [u for u in urls if u not in existing]

    if not new_urls:
        return f"All {len(urls)} URLs already in vector store, skipping crawl"

    print(f"Skipping {len(urls) - len(new_urls)} already crawled URLs, crawling {len(new_urls)} new ones")
    docs = asyncio.run(crawl_web(new_urls))
    store_documents(docs)
    return f"Crawled and stored {len(docs)} chunks from {len(new_urls)} new URLs ({len(urls) - len(new_urls)} skipped)"

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