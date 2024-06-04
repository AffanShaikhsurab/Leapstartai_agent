import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from typing import List

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", google_api_key="AIzaSyBUspiUwI4SB_gHkU2Jd_dbb6CD3EHgxug", temperature=0.0)

class Feature(BaseModel):
    feature: str

class Features(BaseModel):
    lists: List[Feature]

class Plot(BaseModel):
    x_features: str
    y_value: str

class Competitor(BaseModel):
    name: str

class Competitors(BaseModel):
    competitors: List[str]

class Startup(BaseModel):
    name: str
    plots: List[Plot]

class Startups(BaseModel):
    startups: List[Startup]

def get_competitors(market_niche: str):
    parser = PydanticOutputParser(pydantic_object=Competitors)
    prompt = PromptTemplate(
        template="""
        Identify top 4 competitors in the given Indian market niche: {market_niche} as json.
        """,
        input_variables=["market_niche"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    prompt_and_model = prompt | llm
    output_json = prompt_and_model.invoke({"market_niche": market_niche})

    # Debugging: Print the raw response from the language model
    print("Raw response from language model for competitors:")
    print(output_json.content)

    try:
        transformed_data = output_json.content.split("```json")[1].replace("```", "").strip()
        output = json.loads(transformed_data)
        competitors = output["competitors"]
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error: {e}")
        print(f"Output JSON: {output_json.content}")
        raise HTTPException(status_code=500, detail="Invalid JSON response from language model")

    return competitors

def get_features(market_niche: str):
    parser = PydanticOutputParser(pydantic_object=Features)
    prompt = PromptTemplate(
        template="""
        Identify 4 key features for startups in the market niche: {market_niche}.
        Provide each feature as a JSON object with the key "feature".
        """,
        input_variables=["market_niche"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    prompt_and_model = prompt | llm
    output_json = prompt_and_model.invoke({"market_niche": market_niche})

    # Debugging: Print the raw response from the language model
    print("Raw response from language model for features:")
    print(output_json.content)

    try:
        transformed_data = output_json.content.split("```json")[1].replace("```", "").strip()
        features = json.loads(transformed_data)
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error: {e}")
        print(f"Output JSON: {output_json.content}")
        raise HTTPException(status_code=500, detail="Invalid JSON response from language model")

    return features

def generate_startup_data(features: List[str], market_niche: str, competitors: List[str]):
    parser = PydanticOutputParser(pydantic_object=Startups)
    prompt = PromptTemplate(
        template="""
        Based on the features: {features} for the market niche: {market_niche},
        generate startup data (top 4 competitors in the market: {competitors}) with their values to plot on a graph.
        Provide the output as a JSON object with the keys "startups", where "startups" is a list of startups,
        each startup has the keys "name" and "plots", and "plots" is a list of JSON objects with keys "x_features" and "y_value".
        """,
        input_variables=["features", "market_niche", "competitors"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    prompt_and_model = prompt | llm
    output_json = prompt_and_model.invoke({"features": features, "market_niche": market_niche, "competitors": competitors})

    # Debugging: Print the raw response from the language model
    print("Raw response from language model for startup data:")
    print(output_json.content)

    try:
        transformed_data = output_json.content.split("```json")[1].replace("```", "").strip()
        output = json.loads(transformed_data)
        startups = output["startups"]
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error: {e}")
        print(f"Output JSON: {output_json.content}")
        raise HTTPException(status_code=500, detail="Invalid JSON response from language model")

    return startups

