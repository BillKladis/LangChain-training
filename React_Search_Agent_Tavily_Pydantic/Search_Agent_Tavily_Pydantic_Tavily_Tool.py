from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from tavily import TavilyClient
from langchain_tavily import TavilySearch

from typing import List, Literal

load_dotenv()
tavily=TavilyClient()

class JobOffer(BaseModel):
    job_description: str = Field(description="A brief summary of this specific job")
    url: str = Field(description="The direct link to this specific job offer")
    experience_level: str = Field(default="Not_Specified", description="Required experience (e.g., Junior, Senior)")
    notable_tools: List[str] = Field(default="Not_Specified", description="List of specific tools like Python, PyTorch, etc.")

class Ans_format(BaseModel):
    offers: List[JobOffer] = Field(description="A list of all unique job offers found")
    
  
  

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
    for job in answer["structured_response"].offers:
        print(f"Job: {job.job_description}")
        print(f"Link: {job.url}")
if __name__=="__main__":
    main()
