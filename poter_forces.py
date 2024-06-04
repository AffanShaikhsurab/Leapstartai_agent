import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for this example
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],
)

# Replace with your actual API key
llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", google_api_key="AIzaSyBUspiUwI4SB_gHkU2Jd_dbb6CD3EHgxug", temperature=0.0)

class MarketInfo(BaseModel):
    threat_of_new_entrants: str
    threat_of_substitutes: str
    bargaining_power_of_suppliers: str
    bargaining_power_of_buyers: str
    competitive_rivalry: str
    summary: str

def analyze_market( market_description: str):
    parser = PydanticOutputParser(pydantic_object=MarketInfo)

    prompt = PromptTemplate(
        template="""
        Analyze the market for the startup, given the following description: {market_description}

        Provide the analysis using Porter's Five Forces model with the following structure:
        1. Threat of New Entrants: (text)
        2. Threat of Substitutes: (text)
        3. Bargaining Power of Suppliers: (text)
        4. Bargaining Power of Buyers: (text)
        5. Competitive Rivalry: (text)
        Summary: (text)

        Give data as json
        """,
        input_variables=[ "market_description"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    prompt_and_model = prompt | llm
    output_json = prompt_and_model.invoke({
        "market_description": market_description
    })

    # Debugging: Print the raw response from the language model
    print("Raw response from language model:")
    transformed_data = output_json.content.split("```json")[1].replace("```", "")
    print(transformed_data)

    try:
        output = json.loads(transformed_data)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        print(f"Output JSON: {output_json.content}")
        raise HTTPException(status_code=500, detail="Invalid JSON response from language model")

    return output



