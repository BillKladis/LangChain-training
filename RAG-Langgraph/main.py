from dotenv import load_dotenv
load_dotenv()

from graph import app
from langchain_core.messages import HumanMessage

def main():
    queries = [
        "What is baseball and who are the top 10 best players of all time?",
        "Tell me more about baseball history",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print('='*60)

        result = app.invoke({
            "messages": [HumanMessage(content=query)],
            "query": query,
            "urls": [],
            "crawled_urls": [],
            "next": "",
            "iterations": 0,
            "rag_completed": False
        })

        print("\n--- FINAL ANSWER ---")
        print(result["messages"][-1].content)

if __name__ == "__main__":
    main()