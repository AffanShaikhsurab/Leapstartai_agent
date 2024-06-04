import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
import path_to_mvp
import poter_forces
import target_market
import competitor_analysis
import competitors
import investors
import graph
import business_generator
import company_info

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

class GoToMarketStrategy(BaseModel):
    target_market_and_segments: str
    value_proposition_and_positioning: str
    pricing_strategy: str
    marketing_and_communication_plan: str
    sales_strategy: str
    kpis: str
    summary: str

def generate_go_to_market_strategy(market_niche: str):
    parser = PydanticOutputParser(pydantic_object=GoToMarketStrategy)

    prompt = PromptTemplate(
        template="""
        Analyze the market for a startup in the following niche: {market_niche}

        Provide a detailed go-to-market strategy with the following structure:
        1. 🎯 Defining the Target Market and Customer Segments: (text)
        2. 🌟 Developing a Unique Value Proposition and Positioning: (text)
        3. 💰 Setting an Optimal Pricing Strategy: (text)
        4. 📣 Creating a Marketing and Communication Plan: (text)
        5. 🎯 Designing a Tailored Sales Strategy: (text)
        6. 📈 Tracking Success with Key Performance Indicators (KPIs): (text)
        7. Summary: (text)

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

@app.post("/generate_go_to_market_strategy")
async def generate_go_to_market_strategy_info(market_niche: str):
    try:
        analysis = generate_go_to_market_strategy(market_niche)
        return {"go_to_market_strategy": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/analyze_mvp")
async def analyze_mvp_info(market_niche: str):
    try:
        analysis = path_to_mvp.analyze_mvp(market_niche)
        return {"mvp_analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze_market")
async def analyze_market_info(market_niche: str):
    try:
        analysis = poter_forces.analyze_market( market_niche)
        return {"market_analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_investors")
async def analyze_investors_info(market_niche: str):
    try:
        analysis = investors.analyze_investors(market_niche)
        return {"investor_analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze_target_market")
async def analyze_market_info(market_niche: str):
    try:
        analysis = target_market. analyze_market(market_niche)
        return {"market_analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_competitors")
async def analyze_competitors_info(market_niche: str):
    try:
        analysis = competitor_analysis.analyze_competitors(market_niche)
        return {"competitor_analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_competitors")
async def analyze_competitors_info(market_niche: str):
    try:
        analysis = competitors.analyze_competitors(market_niche)
        return {"competitor_analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/generate_graph")
async def analyze_market_info(market_niche: str):
    try:
        competitors = graph.get_competitors(market_niche)
        features = graph.get_features(market_niche)
        feature_list = [f["feature"] for f in features]
        startups = graph.generate_startup_data(feature_list, market_niche, competitors)
        return {"features": features, "startup_data": startups}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_pdf")
def generate_pdf(market_niche: str ):
    report = business_generator.generate_comprehensive_report_from_llm(market_niche)
    html = business_generator.generate_html(report)
    # Load the Jinja2 template
    print(html)
    file_path = "./sample.html"

# Write the HTML string to the file
    with open(file_path, "w") as file:
        file.write(html)

    # Convert HTML to PDF
    path = os.path.abspath('sample.html')
    business_generator.converter.convert(f'file:///{path}', 'sample.pdf')
    # generate_pdf_from_html(html, pdf_file_path)

    print("PDF generated successfully!")
    # Return the PDF file as response
    return FileResponse("sample.pdf", media_type='application/pdf')

@app.post("/get_startup_info")
async def analyze_market_info(market_niche: str):
    try:
        startup_info = company_info.get_startup_info(market_niche)
        return startup_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
