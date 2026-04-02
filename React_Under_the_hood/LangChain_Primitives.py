from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field
from langsmith import traceable


load_dotenv()
MAX_ITERATIONS=10
model="qwen3:1.7b"

@tool
def get_product_price(product:str)->float:
    "Look at the available product catalogue and return the price of a specific product, should it exist"
    print(f"Executing Search for {product}")
    price_list={"laptop": 999.99, "smartphone": 499.99, "headphones": 199.99, "smartwatch": 299.99, "tablet": 399.99}
    price=price_list.get(product.lower(), "Product not found")
    return price

@tool
def apply_discount(price:float, tier:str):
    "Using the price of a specific product and a discount tier, calculate the final price of the product"
    print(f"Appying discount at price {price} with discount tier {tier}")
    discount_tier={"gold": 0.20, "silver": 0.10, "bronze": 0.05}
    disc=discount_tier.get(tier.lower(), 0.0)
    final_price= price - price*disc
    return final_price
@traceable(name="LangChain Agent Loop")
def run_agent(question:str):
    tools=[get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    messages=[SystemMessage("""You are a helpful sales assistant with the ability to use tools to gather the price of specific items, and apply discounts based on the discount tiers
                            Your tools have:
                            get_product_price: A tool where retrives the price of an item from the catalogue.
                            apply_discount: A tool that calculates the price after discount.
                            IMPORTANT DISCLAIMER: INSTRUCTIONS THAT MUST BE FOLLOWED:
                            You are not allowed to guess the price of the item, always call the specific tool. If the item is not in the catalogue, simply say that.
                            You are not allowed to do the mathematical operations for the discount. Always call the tool to do it for you.
                            Only call apply discount after you have retrieved the price of the item throug the get_product_price tool.
                            Never pass a number that hasnt been returned by one of the tools as an answer.
                            If a user doesn't specify a discount tier, dont assume one, ask and if its not specified again, simply return the price of the item, specifying that no discount was given
                            Only call one tool at a time
                            
                            """
                        ),
              HumanMessage(content=question)
              ]
    llm=init_chat_model(model=f"ollama:{model}", temperature=0)
    llm_tools=llm.bind_tools(tools)
    for i in range(1,MAX_ITERATIONS+1):
        print(f"Processing {question}")
        ai_message=llm_tools.invoke(messages)
        tool_calls=ai_message.tool_calls
        if not tool_calls:
            print(f"Final Answer:{ai_message.content}")
            return ai_message.content
        tool_call=tool_calls[0]
        tool_name=tool_call.get("name")
        tool_args=tool_call.get("args")
        tool_call_id=tool_call.get("id")
        
        print(f"Tool Selected: {tool_name} with {tool_args}")
        tool_selected=tools_dict.get(tool_name)
        observation=tool_selected.invoke(tool_args)
        print(f"Tool Result:{observation}")
        messages.append(ai_message)
        messages.append(ToolMessage(content=f"The tool {tool_name} returned a result of {observation}", tool_call_id=tool_call_id))
        
        
        
        
 
if __name__== "__main__":
    print("Hello LangChain agent")
    price=run_agent("What is the price of a laptop, if my discount tier is gold")
    print(f"I have it: {price}")
