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
import requests
import time

# Tencent Cloud SDK imports
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.wsa.v20250508 import wsa_client, models

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache setup
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# Rate limiting for Brave search (1 request per second)
last_brave_request_time = 0

def mask_api_key(key: str) -> str:
    """Mask API key showing only first and last 5 characters"""
    if not key or len(key) < 10:
        return "***INVALID***"
    return f"{key[:5]}...{key[-5:]}"

def extract_json_from_response(response_text: str) -> str:
    """Extract JSON from markdown code blocks or plain text"""
    import re
    
    # Try to find JSON in markdown code blocks (objects or arrays)
    json_match = re.search(r'```(?:json)?\s*([{\[].*?[}\]])\s*```', response_text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    # Try to find JSON array directly
    json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    # Try to find JSON object directly
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    # Return original if no JSON found
    return response_text

def log_api_keys():
    """Log masked API keys for debugging"""
    openai_key = os.getenv("OPENAI_API_KEY", "")
    brave_key = os.getenv("BRAVE_API_KEY", "")
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    tc_id = os.getenv("TC_SECRET_ID", "")
    
    logger.info(f"API Keys - OpenAI: {mask_api_key(openai_key)}")
    logger.info(f"API Keys - TC: {mask_api_key(tc_id)}")
    logger.info(f"API Keys - Brave: {mask_api_key(brave_key)}")
    logger.info(f"API Keys - Tavily: {mask_api_key(tavily_key)}")
    logger.info(f"Search Config - USE_BRAVE_SEARCH: {os.getenv('USE_BRAVE_SEARCH', 'true')}")

# Log API keys on startup
log_api_keys()

# Pydantic Models for Structured Output
class InputParsing(BaseModel):
    date: str = Field(description="Extracted date or time period")
    company: Optional[str] = Field(description="Company or service name")
    incident_description: Optional[str] = Field(description="Brief incident description")

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
    # Input (can be natural language or structured)
    natural_input: NotRequired[str]
    date: str
    company: NotRequired[str]
    incident_description: NotRequired[str]
    
    # Search & Processing
    search_keywords: Annotated[List[str], operator.add]
    search_results: Annotated[List[dict], operator.add]
    metadata: NotRequired[IncidentMetadata]
    timeline: Annotated[List[TimelineEvent], operator.add]
    
    # Output & Quality
    incident_report: NotRequired[str]
    missing_info: Annotated[List[str], operator.add]
    review_result: NotRequired[dict]
    
    # Control & Monitoring
    iteration_count: int
    progress: NotRequired[ProgressStatus]
    errors: Annotated[List[str], operator.add]

# Initialize clients
try:
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    logger.info(f"OpenAI config - Model: {openai_model}, Base URL: {openai_base_url}")
    logger.info(f"OpenAI API key: {mask_api_key(openai_key)}")
    
    llm = ChatOpenAI(
        model=openai_model, 
        temperature=0,
        base_url=openai_base_url
    )
    
    # Initialize search clients based on configuration
    use_brave = os.getenv("USE_BRAVE_SEARCH", "true").lower() == "true"
    
    if os.getenv("TC_SECRET_ID") and os.getenv("TC_SECRET_KEY"):
        logger.info("Using TC Search API")
    elif use_brave and os.getenv("BRAVE_API_KEY"):
        brave_api_key = os.getenv("BRAVE_API_KEY")
        logger.info("Using Brave Search API")
    elif os.getenv("TAVILY_API_KEY"):
        tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        logger.info("Using Tavily Search API")
    else:
        logger.warning("No search API keys found")
        
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

def brave_search_with_rate_limit(query: str) -> dict:
    """Search using Brave API with rate limiting (1 req/sec)"""
    global last_brave_request_time
    
    logger.info(f"Brave search starting for: {query}")
    
    # Rate limiting: ensure 1 second between requests
    current_time = time.time()
    time_since_last = current_time - last_brave_request_time
    if time_since_last < 1.0:
        sleep_time = 1.0 - time_since_last
        logger.info(f"Rate limiting: sleeping {sleep_time:.2f}s")
        time.sleep(sleep_time)
    
    try:
        api_key = os.getenv("BRAVE_API_KEY")
        logger.info(f"Using Brave API key: {mask_api_key(api_key)}")
        
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key
        }
        
        params = {
            "q": query,
            "count": 3,
            "search_lang": "en",
            "country": "US",
            "safesearch": "moderate"
        }
        
        logger.info(f"Brave API request: {params}")
        
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params,
            timeout=10
        )
        
        last_brave_request_time = time.time()
        
        logger.info(f"Brave API response: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            results = []
            
            for item in data.get("web", {}).get("results", [])[:3]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("description", "")[:500]
                })
            
            logger.info(f"Brave search success: {len(results)} results")
            return {
                "query": query,
                "results": results,
                "cached": False,
                "provider": "brave"
            }
        else:
            error_text = response.text[:200] if response.text else "No error details"
            logger.error(f"Brave API error {response.status_code}: {error_text}")
            return {"query": query, "results": [], "error": f"API error {response.status_code}: {error_text}", "provider": "brave"}
            
    except Exception as e:
        logger.error(f"Brave search exception: {e}")
        return {"query": query, "results": [], "error": str(e), "provider": "brave"}

def tc_search(query: str) -> dict:
    """Search using TC Cloud API with official SDK"""
    try:
        logger.info(f"TC search starting for: {query}")
        
        secret_id = os.getenv("TC_SECRET_ID")
        secret_key = os.getenv("TC_SECRET_KEY")
        
        if not secret_id or not secret_key:
            logger.warning("TC API credentials not found")
            return {"query": query, "results": [], "error": "Missing credentials", "provider": "tc"}
        
        # Initialize credentials
        cred = credential.Credential(secret_id, secret_key)
        
        # Configure HTTP profile
        httpProfile = HttpProfile()
        httpProfile.endpoint = "wsa.tencentcloudapi.com"
        
        # Configure client profile
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        
        # Initialize client
        client = wsa_client.WsaClient(cred, "", clientProfile)
        
        # Create request
        req = models.SearchProRequest()
        params = {
            "Query": f"{query} site:github.com OR site:status.aws.amazon.com OR site:reddit.com",
            "Mode": 0  # Natural search
        }
        req.from_json_string(json.dumps(params))
        
        # Execute search
        resp = client.SearchPro(req)
        response_data = json.loads(resp.to_json_string())
        
        # Parse results
        results = []
        pages = response_data.get("Pages")
        
        if pages is None:
            logger.warning("TC search returned no Pages field")
            return {"query": query, "results": [], "error": "No results", "provider": "tc"}
        
        for page_json in pages:
            try:
                page = json.loads(page_json)
                results.append({
                    "title": page.get("title", ""),
                    "url": page.get("url", ""),
                    "content": page.get("passage", ""),
                    "score": page.get("score", 0)
                })
            except json.JSONDecodeError:
                continue
        
        logger.info(f"TC search success: {len(results)} results")
        return {"query": query, "results": results, "provider": "tc"}
        
    except TencentCloudSDKException as e:
        logger.error(f"TC SDK exception: {e}")
        return {"query": query, "results": [], "error": str(e), "provider": "tc"}
    except Exception as e:
        logger.error(f"TC search exception: {e}")
        return {"query": query, "results": [], "error": str(e), "provider": "tc"}

def tavily_search_fallback(query: str) -> dict:
    """Fallback to Tavily search"""
    try:
        logger.info(f"Tavily search starting for: {query}")
        api_key = os.getenv("TAVILY_API_KEY")
        logger.info(f"Using Tavily API key: {mask_api_key(api_key)}")
        
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
            "cached": False,
            "provider": "tavily"
        }
        
        logger.info(f"Tavily search success: {len(result['results'])} results")
        return result
        
    except Exception as e:
        logger.error(f"Tavily search exception: {e}")
        return {"query": query, "results": [], "error": str(e), "provider": "tavily"}

# Enhanced Tools with Error Handling
@tool
def parse_natural_input(input_text: str) -> InputParsing:
    """Parse natural language input to extract date, company, and incident description"""
    try:
        logger.info(f"Parsing input: {input_text}")
        
        prompt = ChatPromptTemplate.from_template("""
        Extract structured information from this natural language input:
        "{input_text}"
        
        Return ONLY a JSON object with these fields:
        - date: Any date, time period, or relative time mentioned
        - company: Company, service, or platform name (null if not found)
        - incident_description: Brief description of what happened (null if not found)
        
        Example: {{"date": "2025-10-29", "company": "Azure", "incident_description": "outage"}}
        """)
        
        response = llm.invoke(prompt.format(input_text=input_text))
        json_text = extract_json_from_response(response.content)
        
        try:
            parsed_data = json.loads(json_text)
            result = InputParsing(**parsed_data)
            logger.info(f"Parsing result: date={result.date}, company={result.company}, description={result.incident_description}")
            return result
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"JSON parsing failed: {e}, raw response: {response.content[:200]}")
            raise e
        
    except Exception as e:
        logger.error(f"Input parsing failed: {e}")
        return InputParsing(
            date=input_text,  # Fallback to original input
            company=None,
            incident_description=None
        )

@tool
def search_with_fallback(query: str) -> dict:
    """Search using TC (primary), Brave, or Tavily with caching and error handling"""
    try:
        # Check cache first
        cached = get_cached_result(query)
        if cached:
            return {"query": query, "results": cached["results"], "cached": True, "provider": cached.get("provider", "cache")}
        
        # Try TC search first
        if os.getenv("TC_SECRET_ID") and os.getenv("TC_SECRET_KEY"):
            result = tc_search(query)
            if result.get("results"):  # Success
                cache_result(query, result)
                return result
            else:
                logger.warning("TC search failed, trying Brave fallback")
        
        # Try Brave search as first fallback (if enabled and API key available)
        use_brave = os.getenv("USE_BRAVE_SEARCH", "true").lower() == "true"
        
        if use_brave and os.getenv("BRAVE_API_KEY"):
            result = brave_search_with_rate_limit(query)
            if result.get("results"):  # Success
                cache_result(query, result)
                return result
            else:
                logger.warning("Brave search failed, trying Tavily fallback")
        
        # Fallback to Tavily
        if os.getenv("TAVILY_API_KEY"):
            result = tavily_search_fallback(query)
            if result.get("results"):
                cache_result(query, result)
                return result
        
        # No results from any provider
        return {"query": query, "results": [], "error": "No search providers available", "cached": False}
        
    except Exception as e:
        logger.error(f"Search failed for '{query}': {e}")
        return {"query": query, "results": [], "error": str(e), "cached": False}

@tool
def extract_incident_metadata(content: str) -> IncidentMetadata:
    """Extract structured incident metadata from content"""
    try:
        prompt = ChatPromptTemplate.from_template("""
        Extract incident metadata from this content:
        {content}
        
        Return ONLY a JSON object with these fields:
        - start_time: Incident start time (null if not found)
        - end_time: Incident end time (null if not found)  
        - affected_services: Array of affected services (empty array if none)
        - severity: Incident severity level (null if not found)
        - status: Current incident status (null if not found)
        
        Example: {{"start_time": null, "end_time": null, "affected_services": ["Azure"], "severity": "high", "status": "resolved"}}
        """)
        
        response = llm.invoke(prompt.format(content=content))
        json_text = extract_json_from_response(response.content)
        
        try:
            parsed_data = json.loads(json_text)
            result = IncidentMetadata(**parsed_data)
            return result
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Metadata JSON parsing failed: {e}, raw response: {response.content[:200]}")
            raise e
        
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
        content = json.dumps(search_results, indent=2)
        
        prompt = ChatPromptTemplate.from_template("""
        Build a chronological timeline from these search results:
        {content}
        
        Return ONLY a JSON array of timeline events. Each event should have:
        - timestamp: Event timestamp
        - event: Event description  
        - source: Information source
        
        Example: [{{"timestamp": "2025-10-29 10:00", "event": "Outage detected", "source": "Azure Status"}}]
        """)
        
        response = llm.invoke(prompt.format(content=content))
        json_text = extract_json_from_response(response.content)
        
        try:
            # Handle incomplete JSON by trying to fix common issues
            if not json_text.strip().endswith(']') and json_text.strip().startswith('['):
                # Try to close incomplete array
                brace_count = 0
                quote_count = 0
                in_string = False
                last_complete = -1
                
                for i, char in enumerate(json_text):
                    if char == '"' and (i == 0 or json_text[i-1] != '\\'):
                        in_string = not in_string
                    elif not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                last_complete = i
                
                if last_complete > 0 and not json_text.strip().endswith(']'):
                    json_text = json_text[:last_complete + 1] + ']'
            
            parsed_data = json.loads(json_text)
            timeline = [TimelineEvent(**event) for event in parsed_data] if isinstance(parsed_data, list) else []
            return timeline
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Timeline JSON parsing failed: {e}, raw response: {response.content[:200]}")
            return []
        
    except Exception as e:
        logger.error(f"Timeline construction failed: {e}")
        return []

@tool
def generate_structured_report(search_data: str, metadata: dict, timeline: List[dict]) -> IncidentReport:
    """Generate structured incident report"""
    try:
        prompt = ChatPromptTemplate.from_template("""
        Create a comprehensive incident report using:
        
        Search Data: {search_data}
        Metadata: {metadata}
        Timeline: {timeline}
        
        Return ONLY a JSON object with these fields:
        - summary: Incident summary
        - timeline: Array of timeline events (can reuse provided timeline)
        - root_cause: Root cause analysis
        - impact: Impact assessment
        - resolution: Resolution steps
        - missing_info: Array of missing information
        
        Example: {{"summary": "Azure outage", "timeline": [], "root_cause": "Network issue", "impact": "Service disruption", "resolution": "Fixed routing", "missing_info": ["Duration"]}}
        """)
        
        response = llm.invoke(prompt.format(
            search_data=search_data,
            metadata=json.dumps(metadata),
            timeline=json.dumps(timeline)
        ))
        
        json_text = extract_json_from_response(response.content)
        
        try:
            parsed_data = json.loads(json_text)
            # Convert timeline data if present
            if 'timeline' in parsed_data and isinstance(parsed_data['timeline'], list):
                parsed_data['timeline'] = [TimelineEvent(**event) if isinstance(event, dict) else event for event in parsed_data['timeline']]
            
            report = IncidentReport(**parsed_data)
            return report
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Report JSON parsing failed: {e}, raw response: {response.content[:200]}")
            raise e
        
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
def parse_node(state: IncidentState) -> dict:
    """Parse natural language input into structured fields"""
    try:
        natural_input = state.get("natural_input")
        if natural_input:
            parsed = parse_natural_input.invoke({"input_text": natural_input})
            
            progress_update = update_progress(state, "input_parsed")
            
            return {
                "date": parsed.date,
                "company": parsed.company,
                "incident_description": parsed.incident_description,
                **progress_update
            }
        else:
            # Already structured input, just update progress
            return update_progress(state, "input_ready")
            
    except Exception as e:
        logger.error(f"Parse node failed: {e}")
        return {
            "errors": [f"Parsing error: {str(e)}"],
            **update_progress(state, "parsing_failed")
        }

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
            result = search_with_fallback.invoke({"query": query})
            results.append(result)
            keywords.extend(query.split())
            
            if "error" in result:
                errors.append(f"Search error: {result['error']}")
            
            # Log search provider used
            provider = result.get("provider", "unknown")
            cached = result.get("cached", False)
            logger.info(f"Search: {query[:50]}... | Provider: {provider} | Cached: {cached}")
        
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
        
        # Convert IncidentMetadata to dict if needed
        if hasattr(metadata, '__dict__'):
            metadata = metadata.__dict__
        
        # Convert TimelineEvent objects to dicts
        timeline_dicts = []
        for event in timeline:
            if hasattr(event, '__dict__'):
                timeline_dicts.append(event.__dict__)
            else:
                timeline_dicts.append(event)
        
        report = generate_structured_report.invoke({
            "search_data": search_content,
            "metadata": metadata,
            "timeline": timeline_dicts
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
    """Enhanced continuation logic with error handling - limited to 1 iteration"""
    # Check for critical errors
    errors = state.get("errors", [])
    if len(errors) > 5:  # Too many errors
        logger.warning("Too many errors, stopping iteration")
        return END
    
    # Max iterations check - reduced to 1 for token saving
    if state.get("iteration_count", 0) >= 1:
        logger.info("Reached max iterations (1), stopping to save tokens")
        return END
    
    # Quality check
    review = state.get("review_result", {})
    if review.get("approved", False):
        return END
    
    return "search"

def create_incident_agent():
    """Create production-ready incident agent"""
    workflow = StateGraph(IncidentState)
    
    workflow.add_node("parse", parse_node)
    workflow.add_node("search", search_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("review", review_node)
    
    workflow.add_edge(START, "parse")
    workflow.add_edge("parse", "search")
    workflow.add_edge("search", "extract")
    workflow.add_edge("extract", "generate")
    workflow.add_edge("generate", "review")
    workflow.add_conditional_edges("review", should_continue, ["search", END])
    
    return workflow.compile()

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    # Check for command line argument
    import sys
    if len(sys.argv) > 1:
        natural_input = " ".join(sys.argv[1:])
        initial_state = {
            "natural_input": natural_input,
            "date": "",  # Will be filled by parsing
            "search_keywords": [],
            "search_results": [],
            "timeline": [],
            "missing_info": [],
            "iteration_count": 0,
            "errors": [],
            "progress": ProgressStatus()
        }
        print(f"Analyzing: {natural_input}")
    else:
        # Default example
        initial_state = {
            "natural_input": "Azure outage on 2025-10-29",
            "date": "",
            "search_keywords": [],
            "search_results": [],
            "timeline": [],
            "missing_info": [],
            "iteration_count": 0,
            "errors": [],
            "progress": ProgressStatus()
        }
        print("Using default example: Azure outage on 2025-10-29")
    
    agent = create_incident_agent()
    
    print("Starting incident analysis...")
    result = agent.invoke(initial_state)
    
    print(f"\n=== PARSED INPUT ===")
    print(f"Date: {result.get('date', 'N/A')}")
    print(f"Company: {result.get('company', 'N/A')}")
    print(f"Description: {result.get('incident_description', 'N/A')}")
    
    print(f"\n=== FINAL REPORT ===")
    print(result["incident_report"])
    print(f"\nIterations: {result['iteration_count']}")
    print(f"Errors encountered: {len(result.get('errors', []))}")
    
    # Show search provider usage
    search_results = result.get("search_results", [])
    if search_results:
        providers = [r.get("provider", "unknown") for r in search_results]
        cached_count = sum(1 for r in search_results if r.get("cached"))
        print(f"Search providers used: {set(providers)}")
        print(f"Cache hits: {cached_count}/{len(search_results)}")
    
    progress = result.get("progress", {})
    print(f"Final progress: {progress}")
    
    if result.get("metadata"):
        print(f"Extracted metadata: {result['metadata']}")
    
    timeline_count = len(result.get("timeline", []))
    print(f"Timeline events: {timeline_count}")
