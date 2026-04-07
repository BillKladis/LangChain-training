import json
import os

MEMORY_FILE = "query_memory.json"

def load_query_memory() -> list:
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def save_query_memory(queries: list):
    with open(MEMORY_FILE, "w") as f:
        json.dump(queries, f, indent=2)

def add_query(query: str):
    queries = load_query_memory()
    if query not in queries:
        queries.append(query)
        save_query_memory(queries)
        print(f"Saved query to memory: {query}")