import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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

class MVPInfo(BaseModel):
    core_features: str
    market_valuation: str
    marketing_strategy: str
    timeline_and_milestones: str
    budget_and_allocation: str
    performance_measurement: str

def analyze_mvp(market_niche: str):
    parser = PydanticOutputParser(pydantic_object=MVPInfo)

    prompt = PromptTemplate(
        template="""
        Analyze the market for a startup in the following niche: {market_niche}

        Provide a detailed path to MVP with the following structure:
        1. Core Features: (text)
        2. Market Valuation: (text)
        3. Marketing Strategy: (text)
        4. Timeline and Milestones: (text)
        5. Budget and Allocation: (text)
        6. Performance Measurement: (text)

        Give data as json
        """,
        input_variables=["market_niche"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    prompt_and_model = prompt | llm
    output_json = prompt_and_model.invoke({"market_niche": market_niche})

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



