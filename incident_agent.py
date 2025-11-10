from typing import TypedDict, Annotated, List, Optional, Literal
from typing_extensions import NotRequired
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
import operator
import json

# State Schema
class IncidentState(TypedDict):
    # Input
    date: str
    company: NotRequired[str]
    incident_description: NotRequired[str]
    
    # Search state
    search_keywords: Annotated[List[str], operator.add]
    search_results: Annotated[List[dict], operator.add]
    
    # Report state
    incident_report: NotRequired[str]
    missing_info: Annotated[List[str], operator.add]
    
    # Review state
    review_result: NotRequired[dict]
    iteration_count: int

# Tools
@tool
def web_search(keywords: str) -> dict:
    """Search web for incident reports using keywords"""
    # Mock implementation - replace with actual web search
    return {
        "query": keywords,
        "results": [
            {"title": f"Sample incident report for {keywords}", 
             "content": f"Mock content about {keywords} incident"}
        ]
    }

@tool
def generate_report(search_data: str, template: str = "root_cause") -> dict:
    """Generate incident report from search data"""
    # Mock implementation - replace with LLM call
    return {
        "report": f"Generated report based on: {search_data}",
        "missing": ["timeline", "impact_assessment"]
    }

@tool
def review_report(report: str) -> dict:
    """Review incident report for quality and completeness"""
    # Mock implementation - replace with LLM call
    return {
        "quality_score": 7,
        "fact_check": True,
        "format_compliance": True,
        "needs_improvement": ["more technical details"],
        "approved": False
    }

# Node Functions
def search_node(state: IncidentState) -> dict:
    """Generate search keywords and perform web search"""
    base_keywords = [state["date"]]
    if state.get("company"):
        base_keywords.append(state["company"])
    if state.get("incident_description"):
        base_keywords.extend(state["incident_description"].split()[:3])
    
    # Add incident-specific terms
    keywords = base_keywords + ["outage", "incident", "postmortem"]
    
    results = []
    for keyword_combo in [" ".join(keywords[:3]), " ".join(keywords[1:4])]:
        result = web_search.invoke({"keywords": keyword_combo})
        results.append(result)
    
    return {
        "search_keywords": keywords,
        "search_results": results
    }

def summarize_node(state: IncidentState) -> dict:
    """Generate incident report from search results"""
    search_content = json.dumps(state["search_results"])
    result = generate_report.invoke({"search_data": search_content})
    
    return {
        "incident_report": result["report"],
        "missing_info": result["missing"]
    }

def review_node(state: IncidentState) -> dict:
    """Review the incident report"""
    result = review_report.invoke({"report": state["incident_report"]})
    
    return {
        "review_result": result,
        "iteration_count": state.get("iteration_count", 0) + 1
    }

def should_continue(state: IncidentState) -> Literal["search_node", END]:
    """Decide whether to continue iterating or end"""
    if state.get("iteration_count", 0) >= 3:  # Max 3 iterations
        return END
    
    review = state.get("review_result", {})
    if review.get("approved", False):
        return END
    
    return "search_node"

# Build Graph
def create_incident_agent():
    workflow = StateGraph(IncidentState)
    
    # Add nodes
    workflow.add_node("search_node", search_node)
    workflow.add_node("summarize_node", summarize_node)
    workflow.add_node("review_node", review_node)
    
    # Add edges
    workflow.add_edge(START, "search_node")
    workflow.add_edge("search_node", "summarize_node")
    workflow.add_edge("summarize_node", "review_node")
    workflow.add_conditional_edges(
        "review_node",
        should_continue,
        ["search_node", END]
    )
    
    return workflow.compile()

# Usage
if __name__ == "__main__":
    agent = create_incident_agent()
    
    # Test input
    initial_state = {
        "date": "2024-11",
        "company": "AWS",
        "incident_description": "S3 service outage",
        "search_keywords": [],
        "search_results": [],
        "missing_info": [],
        "iteration_count": 0
    }
    
    result = agent.invoke(initial_state)
    print(f"Final report: {result['incident_report']}")
    print(f"Iterations: {result['iteration_count']}")
