from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from typing import List

EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME="Scrapping_for_RAG"
QDRANT_URL = "http://localhost:6333"


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True})
    
def create_or_fetch_collection(client, embedding_dim=384):
    col = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in col:
        client.create_collection(collection_name=COLLECTION_NAME, vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE))
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Using existing collection: {COLLECTION_NAME}")
        
def get_vectorstore() -> QdrantVectorStore:
    embeddings = get_embeddings()
    client = QdrantClient(url=QDRANT_URL)
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )   
    
def store_documents(documents: List[Document]):
    vectorstore = get_vectorstore()
    create_or_fetch_collection(vectorstore.client)
    vectorstore.add_documents(documents)
    print(f"Stored {len(documents)} chunks into Qdrant")
    return vectorstore

def search_vectorstore(query: str, filter: dict = None, k: int = 3):
    vectorstore = get_vectorstore()
    if filter is not None:
        search_filter = Filter(
            must=[
                FieldCondition(key=f"metadata.{field}", match=MatchValue(value=value))
                for field, value in filter.items()
            ]
        )
        return vectorstore.similarity_search(query=query, filter=search_filter, k=k)
    return vectorstore.similarity_search(query=query, k=k)
    

    