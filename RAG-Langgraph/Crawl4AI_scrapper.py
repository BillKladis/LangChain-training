import asyncio
import re
from typing import List, Tuple
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, PruningContentFilter, DefaultMarkdownGenerator
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from playwright_stealth import Stealth

stealth = Stealth()

DOMAIN_CONFIGS = {
    "blogspot.com": {"threshold": 0.6, "threshold_type": "fixed", "min_chunk_length": 500},
    "medium.com": {"threshold": 0.3, "threshold_type": "fixed", "min_chunk_length": 300},
    "wikipedia.org": {"threshold": 0.5, "threshold_type": "dynamic", "min_chunk_length": 200},
}

DEFAULT_CONFIG = {"threshold": 0.4, "threshold_type": "dynamic", "min_chunk_length": 300}

def get_domain_config(url: str) -> dict:
    for domain, cfg in DOMAIN_CONFIGS.items():
        if domain in url:
            return cfg
    return DEFAULT_CONFIG

def is_noise(doc: Document) -> bool:
    text = doc.page_content
    greek_chars = sum(1 for c in text if '\u0370' <= c <= '\u03ff' or '\u1f00' <= c <= '\u1fff')
    greek_ratio = greek_chars / len(text) if text else 0
    return greek_ratio > 0.1

async def crawl_web(urls: List[str]) -> List[Document]:

    async def on_page_context_created(*args, **kwargs):
        page = args[0]
        await stealth.apply_stealth_async(page)

    bsr_cfg = BrowserConfig(
        headless=True,
        viewport_height=720,
        viewport_width=1280,
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    strategy = AsyncPlaywrightCrawlerStrategy(browser_config=bsr_cfg)
    strategy.set_hook("on_page_context_created", on_page_context_created)

    semaphore = asyncio.Semaphore(5)
    documents = []

    async def fetch_and_return(url: str, crawler) -> Tuple:
        async with semaphore:
            cfg = get_domain_config(url)
            print(f"Crawling {url} with config: {cfg}")

            md_generator = DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(
                    threshold=cfg["threshold"],
                    threshold_type=cfg["threshold_type"]
                )
            )
            run_cfg = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                word_count_threshold=200,
                markdown_generator=md_generator,
                exclude_external_links=True,
                page_timeout=20000,   # 20 s per page load
            )

            try:
                content = await asyncio.wait_for(
                    crawler.arun(url, config=run_cfg),
                    timeout=25,        # 25 s hard cap per URL
                )
            except asyncio.TimeoutError:
                print(f"Timeout crawling {url}, skipping")
                return None
            if not content.success:
                print(f"Failed {url} with {content.error_message}")
                return None
            return content, url, cfg["min_chunk_length"]

    async with AsyncWebCrawler(crawler_strategy=strategy) as crawler:
        tasks = [fetch_and_return(u, crawler) for u in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        results = [r for r in results if not isinstance(r, Exception)]

        headers_to_split_on = [("#", "Header1"), ("##", "Header2"), ("###", "Header3")]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        for res_tuple in results:
            if res_tuple is None:
                continue
            content, url, min_chunk_length = res_tuple

            if content.markdown:
                splits = splitter.split_text(content.markdown)
                splits = [s for s in splits if len(s.page_content) > min_chunk_length and not is_noise(s)]
                for split in splits:
                    split.metadata["source_url"] = url
                    split.metadata["page_title"] = content.metadata.get("title", "Unknown")
                    split.metadata["description"] = content.metadata.get("description", "")
                documents.extend(splits)
                print(f"Extracted {len(splits)} chunks from {url}")

    return documents