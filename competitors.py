import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from typing import List
from bing_image_urls import bing_image_urls

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", google_api_key="AIzaSyBUspiUwI4SB_gHkU2Jd_dbb6CD3EHgxug", temperature=0.0)

class CompetitorInfo(BaseModel):
    name: str
    short_description: str
    logo: str

class Competitors(BaseModel):
    competitors: List[CompetitorInfo]

# Dummy implementation of bing_image_urls for this example
# Replace this with the actual implementation or import
def bing_image_url(query, limit=1):
    return bing_image_urls(query , limit=1)[0]


def analyze_competitors(market_niche: str):
    parser = PydanticOutputParser(pydantic_object=Competitors)
    prompt = PromptTemplate(
        template="""
        Analyze the competitors in the market niche: {market_niche}
        Provide information about each competitor with the following structure:
        - Name: {name}
        - Short Description: {short_description}
        """,
        input_variables=["market_niche"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    prompt_and_model = prompt | llm
    output_json = prompt_and_model.invoke({"market_niche": market_niche})
    transformed_data = output_json.content.split("```json")[1].replace("```", "")
    
    try:
        output = json.loads(transformed_data)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        print(f"Output JSON: {output_json.content}")
        raise HTTPException(status_code=500, detail="Invalid JSON response from language model")

    competitors_info = []
    for competitor in output["competitors"]:
        try:
            image_url = bing_image_url(f"{competitor['name']} Company Logo", limit=1)[0]
        except Exception as e:
            print(f"Error retrieving image for {competitor['name']}: {e}")
            image_url = ""
        
        competitors_info.append(CompetitorInfo(
            name=competitor["name"],
            short_description=competitor["short_description"],
            logo=image_url
        ))

    return {"competitors": competitors_info}

