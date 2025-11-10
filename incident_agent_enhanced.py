from typing import TypedDict, Annotated, List, Optional, Literal
from typing_extensions import NotRequired
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import operator
import json
import os

# Load environment variables
load_dotenv()

# State Schema
class IncidentState(TypedDict):
    date: str
    company: NotRequired[str]
    incident_description: NotRequired[str]
    search_keywords: Annotated[List[str], operator.add]
    search_results: Annotated[List[dict], operator.add]
    incident_report: NotRequired[str]
    missing_info: Annotated[List[str], operator.add]
    review_result: NotRequired[dict]
    iteration_count: int

# Initialize LLM
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), 
    temperature=0,
    base_url=os.getenv("OPENAI_BASE_URL")
)

# Enhanced Tools
@tool
def generate_search_keywords(date: str, company: str = "", description: str = "") -> List[str]:
    """Generate comprehensive search keywords for incident research"""
    prompt = ChatPromptTemplate.from_template("""
    Generate 5-7 specific search keywords for finding incident reports about:
    Date: {date}
    Company: {company}
    Description: {description}
    
    Include variations like "outage", "incident", "postmortem", "downtime".
    Return as comma-separated list.
    """)
    
    result = llm.invoke(prompt.format(date=date, company=company, description=description))
    return result.content.split(", ")

@tool
def web_search_tool(query: str) -> dict:
    """Search web for incident information"""
    # Replace with actual web search API (Tavily, SerpAPI, etc.)
    return {
        "query": query,
        "results": [
            {
                "title": f"Incident Report: {query}",
                "url": f"https://example.com/incident/{hash(query)}",
                "content": f"Sample incident content for {query}. This would contain actual incident details, timeline, root cause analysis, and resolution steps."
            }
        ]
    }

@tool
def generate_incident_report(search_data: str, template_type: str = "root_cause") -> dict:
    """Generate structured incident report from search data"""
    prompt = ChatPromptTemplate.from_template("""
    Create a comprehensive incident report using this template:
    
    ## Incident Summary
    - Date/Time:
    - Duration:
    - Services Affected:
    - Impact:
    
    ## Timeline
    - Detection:
    - Response:
    - Resolution:
    
    ## Root Cause Analysis
    - Primary Cause:
    - Contributing Factors:
    
    ## Resolution
    - Immediate Actions:
    - Long-term Fixes:
    
    ## Lessons Learned
    - What Went Well:
    - Areas for Improvement:
    
    Based on this search data: {search_data}
    
    Also identify any missing information needed for a complete report.
    Return as JSON with 'report' and 'missing_info' fields.
    """)
    
    result = llm.invoke(prompt.format(search_data=search_data))
    try:
        return json.loads(result.content)
    except:
        return {
            "report": result.content,
            "missing_info": ["Unable to parse structured response"]
        }

@tool
def review_incident_report(report: str) -> dict:
    """Review incident report for quality, accuracy, and completeness"""
    prompt = ChatPromptTemplate.from_template("""
    Review this incident report for:
    1. Factual accuracy (0-10)
    2. Completeness (0-10)
    3. Format compliance (0-10)
    4. Technical depth (0-10)
    
    Report: {report}
    
    Provide specific feedback and determine if approved (score >= 7 in all areas).
    Return as JSON with scores, feedback, and approved boolean.
    """)
    
    result = llm.invoke(prompt.format(report=report))
    try:
        return json.loads(result.content)
    except:
        return {
            "accuracy": 5,
            "completeness": 5,
            "format": 5,
            "technical_depth": 5,
            "feedback": ["Unable to parse review"],
            "approved": False
        }

# Node Functions
def search_node(state: IncidentState) -> dict:
    """Generate keywords and search for incident information"""
    # Generate keywords if not enough
    if len(state.get("search_keywords", [])) < 3:
        keywords = generate_search_keywords.invoke({
            "date": state["date"],
            "company": state.get("company", ""),
            "description": state.get("incident_description", "")
        })
    else:
        # Use existing keywords but modify for iteration
        keywords = [f"{kw} detailed" for kw in state["search_keywords"][:3]]
    
    # Perform searches
    results = []
    for keyword in keywords[:3]:  # Limit to 3 searches
        result = web_search_tool.invoke({"query": keyword})
        results.append(result)
    
    return {
        "search_keywords": keywords,
        "search_results": results
    }

def summarize_node(state: IncidentState) -> dict:
    """Generate incident report from search results"""
    search_content = json.dumps(state["search_results"], indent=2)
    result = generate_incident_report.invoke({"search_data": search_content})
    
    return {
        "incident_report": result.get("report", ""),
        "missing_info": result.get("missing_info", [])
    }

def review_node(state: IncidentState) -> dict:
    """Review the generated incident report"""
    result = review_incident_report.invoke({"report": state["incident_report"]})
    
    return {
        "review_result": result,
        "iteration_count": state.get("iteration_count", 0) + 1
    }

def should_continue(state: IncidentState) -> Literal["search_node", END]:
    """Determine if more iterations are needed"""
    # Max iterations check
    if state.get("iteration_count", 0) >= 3:
        return END
    
    # Quality check
    review = state.get("review_result", {})
    if review.get("approved", False):
        return END
    
    # Continue if quality is insufficient
    return "search_node"

# Create Agent
def create_incident_agent():
    """Create and compile the incident analysis agent"""
    workflow = StateGraph(IncidentState)
    
    # Add nodes
    workflow.add_node("search", search_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("review", review_node)
    
    # Define flow
    workflow.add_edge(START, "search")
    workflow.add_edge("search", "summarize")
    workflow.add_edge("summarize", "review")
    workflow.add_conditional_edges(
        "review",
        should_continue,
        ["search", END]
    )
    
    return workflow.compile()

# Example usage
if __name__ == "__main__":
    # Ensure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    agent = create_incident_agent()
    
    # Test case
    initial_state = {
        "date": "November 2024",
        "company": "AWS",
        "incident_description": "S3 service disruption affecting multiple regions",
        "search_keywords": [],
        "search_results": [],
        "missing_info": [],
        "iteration_count": 0
    }
    
    print("Starting incident analysis...")
    result = agent.invoke(initial_state)
    
    print(f"\n=== FINAL INCIDENT REPORT ===")
    print(result["incident_report"])
    print(f"\nIterations completed: {result['iteration_count']}")
    print(f"Missing information: {result.get('missing_info', [])}")
    
    if result.get("review_result"):
        print(f"Final review: {result['review_result']}")
