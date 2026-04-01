import os
from typing import Dict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
load_dotenv()

def summarize_person(info_text: str) -> str:
    template = """Text to analyze: {information}

    Provide the summary below using this exact format in two paragraphs:

    Information
    [A one paragraph summary of the person's life]
    
    Something interesting is:
    [One unique fact] 

    Output:
    Information"""
    
    prompt = PromptTemplate.from_template(template)
    
    '''llm = ChatOpenAI(
        model="gpt-4.1-nano", 
        temperature=0.2,
        max_retries=2
    )
    '''
    llm=ChatOllama(temperature=0, model="gemma3:1b")
    chain = prompt | llm | StrOutputParser()
    
    
    response=chain.invoke({"information": info_text})
    
    return response
    

if __name__ == "__main__":
    information = """Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, X, and xAI. Musk has been the wealthiest person in the world since 2025; as of February 2026, Forbes estimates his net worth to be around US$852 billion.
    
    Born into a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.
    In 2002, Musk founded the space technology company SpaceX, becoming its CEO and chief engineer; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined the automaker Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, he co-founded OpenAI to advance artificial intelligence (AI) research, but later left; growing discontent with the organization's direction and leadership in the AI boom in the 2020s led him to establish xAI, which became a subsidiary of SpaceX in 2026. In 2022, he acquired the social network Twitter, implementing significant changes, and rebranding it as X in 2023. His other businesses include the neurotechnology company Neuralink, which he co-founded in 2016, and the tunneling company the Boring Company, which he founded in 2017. In November 2025, a Tesla pay package worth $1 trillion for Musk was approved, which he is to receive over 10 years if he meets specific goals."""
    

    print(summarize_person(information))
    
    