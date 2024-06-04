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
    target_audience: str
    competitive_landscape: str
    market_opportunities: str
    market_challenges: str
    market_summary: str

def analyze_market(startupMarket: str):
    parser = PydanticOutputParser(pydantic_object=MarketInfo)

    prompt = PromptTemplate(
        template="""
        Analyze the following market information for a startup, specializing in {startupMarket}:

        Target Audience:
        Competitive Landscape: 
        Market Opportunities: 
        Market Challenges: 
        Market Summary:

        Provide a concise analysis based on the information. Also, summarize the key insights and potential areas of focus for {startupMarket}.
        give data as json """,
        input_variables=["startupMarket"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    prompt_and_model = prompt | llm
    output_json = prompt_and_model.invoke({"startupMarket": startupMarket})

    # Debugging: Print the raw response from the language model
    print("Raw response from language model:")
    transformed_data = output_json.content.split("```json")[1].replace("```" , "")
    print(transformed_data)

    try:
        output = json.loads(transformed_data)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        print(f"Output JSON: {output_json.content}")
        raise HTTPException(status_code=500, detail="Invalid JSON response from language model")

    return output


