from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from typing import List

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "Scrapping_for_RAG"
QDRANT_URL = "http://localhost:6333"

_embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
_client = QdrantClient(url=QDRANT_URL)

def create_or_fetch_collection(embedding_dim: int = 384):
    col = [c.name for c in _client.get_collections().collections]
    if COLLECTION_NAME not in col:
        _client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
        )
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Using existing collection: {COLLECTION_NAME}")

create_or_fetch_collection()

def get_vectorstore() -> QdrantVectorStore:
    return QdrantVectorStore(
        client=_client,
        collection_name=COLLECTION_NAME,
        embedding=_embeddings,
    )

def get_crawled_urls() -> set:
    try:
        results = _client.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=True,
            limit=10000
        )
        urls = set()
        for point in results[0]:
            url = point.payload.get("metadata", {}).get("source_url")
            if url:
                urls.add(url)
        return urls
    except Exception:
        return set()

def store_documents(documents: List[Document]):
    if not documents:
        print("No documents to store")
        return None

    existing_urls = get_crawled_urls()
    new_docs = [d for d in documents if d.metadata.get("source_url") not in existing_urls]

    if not new_docs:
        print("All URLs already in vector store, skipping")
        return get_vectorstore()

    vectorstore = get_vectorstore()
    vectorstore.add_documents(new_docs)
    print(f"Stored {len(new_docs)} new chunks ({len(documents) - len(new_docs)} duplicates skipped)")
    return vectorstore

def search_vectorstore(query: str, filter: dict = None, k: int = 3):
    vectorstore = get_vectorstore()
    if filter is not None:
        qdrant_filter = Filter(
            must=[
                FieldCondition(key=f"metadata.{field}", match=MatchValue(value=value))
                for field, value in filter.items()
            ]
        )
        return vectorstore.similarity_search(query=query, filter=qdrant_filter, k=k)
    return vectorstore.similarity_search(query=query, k=k)