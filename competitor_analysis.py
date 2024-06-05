import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from typing import List

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

class Statement(BaseModel):
    stat: str

class Competitor(BaseModel):
    strength: List[Statement]
    weakness: List[Statement]
    market_share: str

class Competitors(BaseModel):
    list_of_competitor: List[Competitor]
    list_of_indirect_competitor : List[str]

def analyze_competitors(market_niche: str):
    parser = PydanticOutputParser(pydantic_object=Competitors)

    prompt = PromptTemplate(
        template="""
        Analyze the competitors in the market niche: {market_niche}

        Provide information about each competitor with the following structure:
        - Strengths:
            - Statement 1
            - Statement 2
            - ...
        - Weaknesses:
            - Statement 1
            - Statement 2
            - ...
        - Market Share: (percentage or description)

        - list of indirect competitors:
            - Statement 1
            - Statement 2
            - ...
        Give data as JSON.
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



analyze_competitors("edtech")