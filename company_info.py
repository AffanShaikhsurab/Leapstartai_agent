import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser

app = FastAPI()

# Initialize the language model
llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", google_api_key="AIzaSyBUspiUwI4SB_gHkU2Jd_dbb6CD3EHgxug", temperature=0.0)

# Define the Pydantic models
class IndustrySector(BaseModel):
    industry_desc: str
    industry_valuation: int

class StartUp(BaseModel):
    name: str
    industry_sector: IndustrySector
    description: str
    website: str
    valuation: int
    number_of_employees: int
    sales: int
    revenue: int
    profit: int

def get_startup_info(market_niche: str):
    parser = PydanticOutputParser(pydantic_object=StartUp)
    prompt = PromptTemplate(
        template="""
        Provide detailed information for a startup in the market niche: {market_niche}.
        Include the following fields:
        - Name
        - Industry Sector
          - Industry Description
          - Industry Valuation
        - Startup Description
        - Startup Website
        - Startup Valuation
        - Number of Employees
        - Sales
        - Revenue
        - Profit as json format
        """,
        input_variables=["market_niche"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    prompt_and_model = prompt | llm
    output_json = prompt_and_model.invoke({"market_niche": market_niche})
    transformed_data = output_json.content.split("```json")[1].replace("```", "")

    try:
        output = json.loads(transformed_data)
        print(output)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        print(f"Output JSON: {output_json.content}")
        raise HTTPException(status_code=500, detail="Invalid JSON response from language model")

    return output


