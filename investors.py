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

class Investor(BaseModel):
    investor_name: str

class Investors(BaseModel):
    investors: List[Investor]

def analyze_investors(market_niche: str):
    parser = PydanticOutputParser(pydantic_object=Investors)
    prompt = PromptTemplate(
        template="""
        List the top 6 investors for startups in the market niche: {market_niche}
        Provide the name of each investor.
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

    investors_info = [Investor(investor_name=inv["investor_name"]) for inv in output["investors"]]
    return {"investors": investors_info}




