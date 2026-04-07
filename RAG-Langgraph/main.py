from dotenv import load_dotenv
from graph import app
from langchain_core.messages import HumanMessage

load_dotenv()

def main():
    query = "What is football, and who are the top 10 best players"
    result = app.invoke({
        "messages": [HumanMessage(content=query)],
        "query": query,
        "urls": [],
        "next": ""
    })
    print("\n--- FINAL ANSWER ---")
    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()