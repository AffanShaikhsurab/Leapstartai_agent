import json
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
import os
from pyhtml2pdf import converter

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

class ExecutiveSummary(BaseModel):
    overview: str
    objectives: str
    key_findings: str

class MarketAnalysis(BaseModel):
    target_market: str
    market_need: str
    market_size_growth: str
    competitive_analysis: str
    market_trends: str

class ProductDescription(BaseModel):
    product_features: str
    unique_selling_proposition: str
    development_roadmap: str

class TechnicalFeasibility(BaseModel):
    technology_requirements: str
    development_process: str
    technical_challenges: str

class OperationalFeasibility(BaseModel):
    operational_plan: str
    resource_requirements: str
    regulatory_legal_considerations: str

class FinancialFeasibility(BaseModel):
    cost_estimates: str
    revenue_projections: str
    break_even_analysis: str
    funding_requirements: str

class RiskAnalysis(BaseModel):
    swot_analysis: str
    risk_mitigation_strategies: str

class ProductFeasibilityAnalysis(BaseModel):
    market_analysis: MarketAnalysis
    product_description: ProductDescription
    technical_feasibility: TechnicalFeasibility
    operational_feasibility: OperationalFeasibility
    financial_feasibility: FinancialFeasibility
    risk_analysis: RiskAnalysis

class ReliabilityRequirements(BaseModel):
    performance_standards: str
    user_expectations: str
    regulatory_requirements: str

class DesignAnalysis(BaseModel):
    design_simplicity: str
    component_selection: str
    redundancy_fail_safes: str

class ReliabilityPredictionMethods(BaseModel):
    historical_data: str
    analytical_methods: str
    accelerated_life_testing: str

class TestingValidation(BaseModel):
    test_plan: str
    environmental_testing: str
    stress_testing: str
    usage_simulation: str
    beta_testing: str

class DataCollectionAnalysis(BaseModel):
    data_sources: str
    failure_data_analysis: str
    statistical_methods: str

class RiskAssessmentMitigation(BaseModel):
    risk_identification: str
    risk_assessment: str
    mitigation_strategies: str

class ReliabilityAnalysis(BaseModel):
    reliability_requirements: ReliabilityRequirements
    design_analysis: DesignAnalysis
    reliability_prediction_methods: ReliabilityPredictionMethods
    testing_validation: TestingValidation
    data_collection_analysis: DataCollectionAnalysis
    risk_assessment_mitigation: RiskAssessmentMitigation

class CostStructure(BaseModel):
    fixed_costs: str
    variable_costs: str

class PricingStrategy(BaseModel):
    pricing_model: str
    price_per_unit: str

class BreakEvenCalculation(BaseModel):
    formula: str
    break_even_units: str
    break_even_sales: str

class SensitivityAnalysis(BaseModel):
    variable_changes: str
    scenarios: str
    impact_analysis: str

class ContributionMargin(BaseModel):
    contribution_margin_per_unit: str
    contribution_margin_ratio: str

class GraphicalRepresentation(BaseModel):
    break_even_chart: str

class BreakEvenAnalysis(BaseModel):
    cost_structure: CostStructure
    pricing_strategy: PricingStrategy
    break_even_calculation: BreakEvenCalculation
    sensitivity_analysis: SensitivityAnalysis
    contribution_margin: ContributionMargin
    graphical_representation: GraphicalRepresentation

class ConclusionRecommendations(BaseModel):
    feasibility_summary: str
    reliability_summary: str
    break_even_summary: str
    next_steps: str

class Appendices(BaseModel):
    supporting_documents: str
    detailed_financial_statements: str
    assumptions: str
    test_results_data: str

class ComprehensiveReport(BaseModel):
    executive_summary: ExecutiveSummary
    product_feasibility_analysis: ProductFeasibilityAnalysis
    reliability_analysis: ReliabilityAnalysis
    break_even_analysis: BreakEvenAnalysis
    conclusion_recommendations: ConclusionRecommendations
    appendices: Appendices


class GoToMarketStrategy(BaseModel):
    target_market_and_segments: str
    value_proposition_and_positioning: str
    pricing_strategy: str
    marketing_and_communication_plan: str
    sales_strategy: str
    kpis: str
    summary: str

class htmlCode(BaseModel):
    html : str

def generate_comprehensive_report_from_llm(market_niche: str):
    parser = PydanticOutputParser(pydantic_object=GoToMarketStrategy)

    prompt = PromptTemplate(
        template="""
        Generate a comprehensive report for the given market description: {market_niche}

        The report should include the following sections with detailed information:
        1. Executive Summary
        2. Product Feasibility Analysis
            - Market Analysis
            - Product Description
            - Technical Feasibility
            - Operational Feasibility
            - Financial Feasibility
            - Risk Analysis
        3. Reliability Analysis
        4. Break-Even Analysis
        5. Conclusion and Recommendations
        6. Appendices

        Provide the output as JSON with the structure corresponding to the defined Pydantic models.
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


def generate_html(complete: str):
    parser = PydanticOutputParser(pydantic_object=htmlCode)

    prompt = PromptTemplate(
        template="""
        Craft a sleek and contemporary medium  like article  website (HTML file with inline [tailwind CSS]) use minimilistic gradient effects , background color , shadows , animation , padding , text-alignment , and so much more...[ use minimilistic colors and use emojies]  tailored to the provided data [make sure to include complete data ]: {complete}""",
        input_variables=["complete"],
                partial_variables={"format_instructions": parser.get_format_instructions()},

    )
    prompt_and_model = prompt | llm
    output_json = prompt_and_model.invoke({"complete": complete})

    # Debugging: Print the raw response from the language model
    print("Raw response from language model:")
    transformed_data = output_json.content.split("```html")[1].replace("```", "")


    return transformed_data
async def generate_pdf_from_html(html_content, pdf_path):
    browser = await launch()
    page = await browser.newPage()
        
    await page.setContent(html_content)
        
    await page.pdf({'path': pdf_path, 'format': 'A4'})
        
    await browser.close()

@app.post("/generate_pdf")
def generate_pdf(market_niche: str ):
    report = generate_comprehensive_report_from_llm(market_niche)
    html = generate_html(report)
    # Load the Jinja2 template
    print(html)
    file_path = "./sample.html"

# Write the HTML string to the file
    with open(file_path, "w") as file:
        file.write(html)

    # Convert HTML to PDF
    path = os.path.abspath('sample.html')
    converter.convert(f'file:///{path}', 'sample.pdf')
    # generate_pdf_from_html(html, pdf_file_path)

    print("PDF generated successfully!")
    # Return the PDF file as response
    return FileResponse("sample.pdf", media_type='application/pdf')


