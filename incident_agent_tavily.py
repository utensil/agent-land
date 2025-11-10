from typing import TypedDict, Annotated, List, Optional, Literal
from typing_extensions import NotRequired
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from dotenv import load_dotenv
import operator
import json
import os
import hashlib
import pickle
from pathlib import Path

# Load environment variables
load_dotenv()

# Cache setup
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# Initialize clients
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), 
    temperature=0,
    base_url=os.getenv("OPENAI_BASE_URL")
)
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

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

def get_cache_key(query: str) -> str:
    """Generate cache key for search query"""
    return hashlib.md5(query.encode()).hexdigest()

def get_cached_result(query: str) -> Optional[dict]:
    """Get cached search result"""
    cache_file = CACHE_DIR / f"{get_cache_key(query)}.pkl"
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None

def cache_result(query: str, result: dict):
    """Cache search result"""
    cache_file = CACHE_DIR / f"{get_cache_key(query)}.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)

@tool
def tavily_search(query: str) -> dict:
    """Search using Tavily with caching"""
    # Check cache first
    cached = get_cached_result(query)
    if cached:
        return {"query": query, "results": cached["results"], "cached": True}
    
    # Perform search
    try:
        response = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=3,
            include_domains=["github.com", "status.aws.amazon.com", "blog.cloudflare.com"]
        )
        
        result = {
            "query": query,
            "results": [
                {
                    "title": r["title"],
                    "url": r["url"],
                    "content": r["content"][:500]  # Truncate for cost
                }
                for r in response.get("results", [])
            ],
            "cached": False
        }
        
        # Cache the result
        cache_result(query, result)
        return result
        
    except Exception as e:
        return {"query": query, "results": [], "error": str(e), "cached": False}

@tool
def generate_incident_report(search_data: str) -> dict:
    """Generate incident report from search data"""
    prompt = ChatPromptTemplate.from_template("""
    Create a concise incident report:
    
    ## Summary
    - Date/Time:
    - Services Affected:
    - Impact:
    
    ## Root Cause
    - Primary Cause:
    
    ## Resolution
    - Actions Taken:
    
    Based on: {search_data}
    
    Return JSON with 'report' and 'missing_info' fields.
    """)
    
    result = llm.invoke(prompt.format(search_data=search_data))
    try:
        return json.loads(result.content)
    except:
        return {"report": result.content, "missing_info": []}

@tool
def review_report(report: str) -> dict:
    """Review incident report quality"""
    prompt = ChatPromptTemplate.from_template("""
    Rate this incident report (0-10):
    - Completeness: 
    - Technical accuracy:
    
    Report: {report}
    
    Return JSON with scores and 'approved' boolean (>7 = approved).
    """)
    
    result = llm.invoke(prompt.format(report=report))
    try:
        return json.loads(result.content)
    except:
        return {"completeness": 5, "accuracy": 5, "approved": False}

# Nodes
def search_node(state: IncidentState) -> dict:
    """Search for incident information using Tavily"""
    base_query = f"{state['date']} incident"
    if state.get("company"):
        base_query += f" {state['company']}"
    if state.get("incident_description"):
        base_query += f" {state['incident_description']}"
    
    # Generate search variations
    queries = [
        base_query,
        f"{base_query} outage postmortem",
        f"{base_query} root cause analysis"
    ]
    
    results = []
    keywords = []
    
    for query in queries:
        result = tavily_search.invoke({"query": query})
        results.append(result)
        keywords.extend(query.split())
    
    return {"search_keywords": keywords, "search_results": results}

def summarize_node(state: IncidentState) -> dict:
    """Generate incident report"""
    search_content = json.dumps(state["search_results"], indent=2)
    result = generate_incident_report.invoke({"search_data": search_content})
    
    return {
        "incident_report": result.get("report", ""),
        "missing_info": result.get("missing_info", [])
    }

def review_node(state: IncidentState) -> dict:
    """Review report quality"""
    result = review_report.invoke({"report": state["incident_report"]})
    
    return {
        "review_result": result,
        "iteration_count": state.get("iteration_count", 0) + 1
    }

def should_continue(state: IncidentState) -> Literal["search", END]:
    """Check if more iterations needed"""
    if state.get("iteration_count", 0) >= 3:
        return END
    
    review = state.get("review_result", {})
    if review.get("approved", False):
        return END
    
    return "search"

def create_incident_agent():
    """Create Tavily-powered incident agent"""
    workflow = StateGraph(IncidentState)
    
    workflow.add_node("search", search_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("review", review_node)
    
    workflow.add_edge(START, "search")
    workflow.add_edge("search", "summarize")
    workflow.add_edge("summarize", "review")
    workflow.add_conditional_edges("review", should_continue, ["search", END])
    
    return workflow.compile()

if __name__ == "__main__":
    if not os.getenv("TAVILY_API_KEY"):
        print("Please set TAVILY_API_KEY environment variable")
        exit(1)
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    agent = create_incident_agent()
    
    initial_state = {
        "date": "November 2024",
        "company": "AWS",
        "incident_description": "S3 outage",
        "search_keywords": [],
        "search_results": [],
        "missing_info": [],
        "iteration_count": 0
    }
    
    print("Starting Tavily-powered incident analysis...")
    result = agent.invoke(initial_state)
    
    print(f"\n=== INCIDENT REPORT ===")
    print(result["incident_report"])
    print(f"\nIterations: {result['iteration_count']}")
    
    # Show cache usage
    cached_count = sum(1 for r in result["search_results"] if r.get("cached"))
    total_searches = len(result["search_results"])
    print(f"Cache hits: {cached_count}/{total_searches}")
