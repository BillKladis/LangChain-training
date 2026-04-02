from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from tavily import TavilyClient
from langchain_tavily import TavilySearch

from typing import Literal

load_dotenv()
tavily=TavilyClient()

class Ans_format(BaseModel):
    answer: str = Field(description="The final answer to the user's question")
  
  

class Search_Format(BaseModel):
    query: str =Field(description="Query that will be used by the search engine")
    country: str= Field(description="Country for the search to be limited to")
    topic: Literal["general", "news", "finance"] = Field(
          
        description="The search category."
    )
    
    

    
#@tool(args_schema=Search_Format)
"""def search(query: str, country: str, topic: str) -> str:
    ""Tool that searches the internet with specific filters.""
    
    print(f"DEBUG: Searching for {query} in {country}")
    
    # Pass the individual arguments directly to Tavily
    result = tavily.search(
        query=query,
        topic=topic,
        country=country,
        search_depth="basic"
    )
    return result"""
def main():
     
    llm=ChatOpenAI(temperature=0, model="gpt-4.1-nano")
    tools=[TavilySearch()]
    agent=create_agent(model=llm, tools=tools, response_format=Ans_format)
    answer=agent.invoke({"messages": [HumanMessage(content="Search for AI engineer jobs in Athens, Greece. Return me a brief description of the jobs found in different paragraphs and the link to the offer.")]})
    print(answer["structured_response"].answer)
if __name__=="__main__":
    main()
