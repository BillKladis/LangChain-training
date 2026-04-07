from dotenv import load_dotenv
load_dotenv()

from graph import app
from langchain_core.messages import HumanMessage

def main():
    query = "What is Formula One and who is Lewis Hamilton?"
    result = app.invoke({
        "messages": [HumanMessage(content=query)],
        "query": query,
        "urls": [],
        "next": "",
        "iterations": 0
    })
    print("\n--- FINAL ANSWER ---")
    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()