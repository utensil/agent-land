from typing import TypedDict, Annotated, List, Optional, Literal
from typing_extensions import NotRequired
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from datetime import datetime
import operator
import json
import os
import hashlib
import pickle
from pathlib import Path
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache setup
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# Pydantic Models for Structured Output
class IncidentMetadata(BaseModel):
    start_time: Optional[str] = Field(description="Incident start time")
    end_time: Optional[str] = Field(description="Incident end time")
    affected_services: List[str] = Field(description="List of affected services")
    severity: Optional[str] = Field(description="Incident severity level")
    status: Optional[str] = Field(description="Current incident status")

class TimelineEvent(BaseModel):
    timestamp: str = Field(description="Event timestamp")
    event: str = Field(description="Event description")
    source: str = Field(description="Information source")

class IncidentReport(BaseModel):
    summary: str = Field(description="Incident summary")
    timeline: List[TimelineEvent] = Field(description="Chronological timeline")
    root_cause: str = Field(description="Root cause analysis")
    impact: str = Field(description="Impact assessment")
    resolution: str = Field(description="Resolution steps")
    missing_info: List[str] = Field(description="Missing information")

class ProgressStatus(BaseModel):
    search_complete: bool = False
    metadata_extracted: bool = False
    timeline_built: bool = False
    report_generated: bool = False
    review_complete: bool = False
    quality_acceptable: bool = False
    current_step: str = "starting"

# State Schema
class IncidentState(TypedDict):
    date: str
    company: NotRequired[str]
    incident_description: NotRequired[str]
    search_keywords: Annotated[List[str], operator.add]
    search_results: Annotated[List[dict], operator.add]
    metadata: NotRequired[IncidentMetadata]
    timeline: Annotated[List[TimelineEvent], operator.add]
    incident_report: NotRequired[str]
    missing_info: Annotated[List[str], operator.add]
    review_result: NotRequired[dict]
    iteration_count: int
    progress: NotRequired[ProgressStatus]
    errors: Annotated[List[str], operator.add]

# Initialize clients
try:
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), 
        temperature=0,
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
except Exception as e:
    logger.error(f"Failed to initialize clients: {e}")
    raise

# Cache functions
def get_cache_key(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()

def get_cached_result(query: str) -> Optional[dict]:
    cache_file = CACHE_DIR / f"{get_cache_key(query)}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
    return None

def cache_result(query: str, result: dict):
    cache_file = CACHE_DIR / f"{get_cache_key(query)}.pkl"
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
    except Exception as e:
        logger.warning(f"Cache write error: {e}")

# Enhanced Tools with Error Handling
@tool
def tavily_search_with_fallback(query: str) -> dict:
    """Search using Tavily with caching and error handling"""
    try:
        # Check cache first
        cached = get_cached_result(query)
        if cached:
            return {"query": query, "results": cached["results"], "cached": True}
        
        # Perform search
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
                    "content": r["content"][:500]
                }
                for r in response.get("results", [])
            ],
            "cached": False
        }
        
        cache_result(query, result)
        return result
        
    except Exception as e:
        logger.error(f"Search failed for '{query}': {e}")
        return {"query": query, "results": [], "error": str(e), "cached": False}

@tool
def extract_incident_metadata(content: str) -> IncidentMetadata:
    """Extract structured incident metadata from content"""
    try:
        llm_with_structure = llm.with_structured_output(IncidentMetadata)
        prompt = ChatPromptTemplate.from_template("""
        Extract incident metadata from this content:
        {content}
        
        Focus on: start/end times, affected services, severity, status.
        If information is missing, leave fields empty.
        """)
        
        result = llm_with_structure.invoke(prompt.format(content=content))
        return result
        
    except Exception as e:
        logger.error(f"Metadata extraction failed: {e}")
        return IncidentMetadata(
            affected_services=[],
            severity="unknown"
        )

@tool
def build_timeline(search_results: List[dict]) -> List[TimelineEvent]:
    """Build chronological timeline from search results"""
    try:
        llm_with_structure = llm.with_structured_output(List[TimelineEvent])
        content = json.dumps(search_results, indent=2)
        
        prompt = ChatPromptTemplate.from_template("""
        Build a chronological timeline from these search results:
        {content}
        
        Extract key events with timestamps. Sort chronologically.
        Include source for each event.
        """)
        
        timeline = llm_with_structure.invoke(prompt.format(content=content))
        return timeline or []
        
    except Exception as e:
        logger.error(f"Timeline construction failed: {e}")
        return []

@tool
def generate_structured_report(search_data: str, metadata: dict, timeline: List[dict]) -> IncidentReport:
    """Generate structured incident report"""
    try:
        llm_with_structure = llm.with_structured_output(IncidentReport)
        
        prompt = ChatPromptTemplate.from_template("""
        Create a comprehensive incident report using:
        
        Search Data: {search_data}
        Metadata: {metadata}
        Timeline: {timeline}
        
        Generate complete sections for summary, root cause, impact, and resolution.
        List any missing information needed for completeness.
        """)
        
        report = llm_with_structure.invoke(prompt.format(
            search_data=search_data,
            metadata=json.dumps(metadata),
            timeline=json.dumps(timeline)
        ))
        
        return report
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return IncidentReport(
            summary="Report generation failed",
            timeline=[],
            root_cause="Unable to determine",
            impact="Unknown",
            resolution="Unable to determine",
            missing_info=["All information due to generation error"]
        )

@tool
def review_report_quality(report: str) -> dict:
    """Review report quality with structured assessment"""
    try:
        prompt = ChatPromptTemplate.from_template("""
        Review this incident report and rate (0-10):
        - completeness: How complete is the information?
        - accuracy: How accurate does the technical content appear?
        - clarity: How clear and well-structured is the report?
        
        Report: {report}
        
        Return JSON with scores and 'approved' boolean (average >= 7).
        """)
        
        result = llm.invoke(prompt.format(report=report))
        review = json.loads(result.content)
        
        # Calculate approval
        scores = [review.get("completeness", 0), review.get("accuracy", 0), review.get("clarity", 0)]
        avg_score = sum(scores) / len(scores) if scores else 0
        review["approved"] = avg_score >= 7
        
        return review
        
    except Exception as e:
        logger.error(f"Report review failed: {e}")
        return {"completeness": 0, "accuracy": 0, "clarity": 0, "approved": False, "error": str(e)}

def update_progress(state: IncidentState, step: str, **updates) -> dict:
    """Update progress tracking"""
    current_progress = state.get("progress", ProgressStatus())
    
    # Update specific fields
    for key, value in updates.items():
        if hasattr(current_progress, key):
            setattr(current_progress, key, value)
    
    current_progress.current_step = step
    logger.info(f"Progress: {step} - {updates}")
    
    return {"progress": current_progress}

# Enhanced Node Functions
def search_node(state: IncidentState) -> dict:
    """Enhanced search with progress tracking and error handling"""
    try:
        # Generate search queries
        base_query = f"{state['date']} incident"
        if state.get("company"):
            base_query += f" {state['company']}"
        if state.get("incident_description"):
            base_query += f" {state['incident_description']}"
        
        queries = [
            base_query,
            f"{base_query} outage postmortem",
            f"{base_query} root cause analysis"
        ]
        
        results = []
        keywords = []
        errors = []
        
        for query in queries:
            result = tavily_search_with_fallback.invoke({"query": query})
            results.append(result)
            keywords.extend(query.split())
            
            if "error" in result:
                errors.append(f"Search error: {result['error']}")
        
        progress_update = update_progress(state, "search_complete", search_complete=True)
        
        return {
            "search_keywords": keywords,
            "search_results": results,
            "errors": errors,
            **progress_update
        }
        
    except Exception as e:
        logger.error(f"Search node failed: {e}")
        return {
            "errors": [f"Search node error: {str(e)}"],
            **update_progress(state, "search_failed")
        }

def extract_node(state: IncidentState) -> dict:
    """Extract metadata and build timeline"""
    try:
        # Extract metadata from search results
        all_content = " ".join([
            " ".join([r.get("content", "") for r in result.get("results", [])])
            for result in state["search_results"]
        ])
        
        metadata = extract_incident_metadata.invoke({"content": all_content})
        timeline = build_timeline.invoke({"search_results": state["search_results"]})
        
        progress_update = update_progress(
            state, 
            "extraction_complete", 
            metadata_extracted=True, 
            timeline_built=True
        )
        
        return {
            "metadata": metadata,
            "timeline": timeline,
            **progress_update
        }
        
    except Exception as e:
        logger.error(f"Extract node failed: {e}")
        return {
            "errors": [f"Extraction error: {str(e)}"],
            **update_progress(state, "extraction_failed")
        }

def generate_node(state: IncidentState) -> dict:
    """Generate structured incident report"""
    try:
        search_content = json.dumps(state["search_results"], indent=2)
        metadata = state.get("metadata", {})
        timeline = state.get("timeline", [])
        
        report = generate_structured_report.invoke({
            "search_data": search_content,
            "metadata": metadata,
            "timeline": timeline
        })
        
        progress_update = update_progress(state, "report_generated", report_generated=True)
        
        return {
            "incident_report": report.summary + "\n\n" + report.root_cause + "\n\n" + report.resolution,
            "missing_info": report.missing_info,
            **progress_update
        }
        
    except Exception as e:
        logger.error(f"Generate node failed: {e}")
        return {
            "errors": [f"Generation error: {str(e)}"],
            **update_progress(state, "generation_failed")
        }

def review_node(state: IncidentState) -> dict:
    """Review report with enhanced quality assessment"""
    try:
        result = review_report_quality.invoke({"report": state["incident_report"]})
        
        progress_update = update_progress(
            state, 
            "review_complete", 
            review_complete=True,
            quality_acceptable=result.get("approved", False)
        )
        
        return {
            "review_result": result,
            "iteration_count": state.get("iteration_count", 0) + 1,
            **progress_update
        }
        
    except Exception as e:
        logger.error(f"Review node failed: {e}")
        return {
            "errors": [f"Review error: {str(e)}"],
            **update_progress(state, "review_failed")
        }

def should_continue(state: IncidentState) -> Literal["search", END]:
    """Enhanced continuation logic with error handling"""
    # Check for critical errors
    errors = state.get("errors", [])
    if len(errors) > 5:  # Too many errors
        logger.warning("Too many errors, stopping iteration")
        return END
    
    # Max iterations check
    if state.get("iteration_count", 0) >= 3:
        return END
    
    # Quality check
    review = state.get("review_result", {})
    if review.get("approved", False):
        return END
    
    return "search"

def create_incident_agent():
    """Create production-ready incident agent"""
    workflow = StateGraph(IncidentState)
    
    workflow.add_node("search", search_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("review", review_node)
    
    workflow.add_edge(START, "search")
    workflow.add_edge("search", "extract")
    workflow.add_edge("extract", "generate")
    workflow.add_edge("generate", "review")
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
        "timeline": [],
        "missing_info": [],
        "iteration_count": 0,
        "errors": [],
        "progress": ProgressStatus()
    }
    
    print("Starting incident analysis...")
    result = agent.invoke(initial_state)
    
    print(f"\n=== FINAL REPORT ===")
    print(result["incident_report"])
    print(f"\nIterations: {result['iteration_count']}")
    print(f"Errors encountered: {len(result.get('errors', []))}")
    
    progress = result.get("progress", {})
    print(f"Final progress: {progress}")
    
    if result.get("metadata"):
        print(f"Extracted metadata: {result['metadata']}")
    
    timeline_count = len(result.get("timeline", []))
    print(f"Timeline events: {timeline_count}")
