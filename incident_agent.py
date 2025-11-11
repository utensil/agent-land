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

# Suppress httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)

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
    
    logger.debug(f"API Keys - OpenAI: {mask_api_key(openai_key)}")
    logger.debug(f"API Keys - TC: {mask_api_key(tc_id)}")
    logger.debug(f"API Keys - Brave: {mask_api_key(brave_key)}")
    logger.debug(f"API Keys - Tavily: {mask_api_key(tavily_key)}")
    logger.debug(f"Search Config - USE_BRAVE_SEARCH: {os.getenv('USE_BRAVE_SEARCH', 'true')}")

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
    incident_name: str = Field(description="Incident name in format: {company} on YYYY-MM-DD due to [cause] affecting [services]")
    company_product: str = Field(description="Company and product information")
    incident_time: str = Field(description="Incident start time in YYYY-MM-DD HH:MM:SS format")
    report_links: List[str] = Field(description="Original report links")
    impact_description: str = Field(description="Geographic, user, service scope and duration impact in markdown")
    incident_process: str = Field(description="Timeline with trigger, spread, and recovery phases in markdown")
    root_cause: str = Field(description="Direct and root causes in markdown")
    improvement_measures: str = Field(description="Prevention and response measures in markdown")

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
        logger.debug("Using TC Search API")
    elif use_brave and os.getenv("BRAVE_API_KEY"):
        brave_api_key = os.getenv("BRAVE_API_KEY")
        logger.debug("Using Brave Search API")
    elif os.getenv("TAVILY_API_KEY"):
        tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        logger.debug("Using Tavily Search API")
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
    
    logger.debug(f"Brave search starting for: {query}")
    
    # Rate limiting: ensure 1 second between requests
    current_time = time.time()
    time_since_last = current_time - last_brave_request_time
    if time_since_last < 1.0:
        sleep_time = 1.0 - time_since_last
        logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
        time.sleep(sleep_time)
    
    last_brave_request_time = time.time()
    
    try:
        brave_api_key = os.getenv("BRAVE_API_KEY")
        logger.debug(f"Using Brave API key: {mask_api_key(brave_api_key)}")
        
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": brave_api_key
        }
        
        params = {
            "q": query,
            "count": 3,
            "search_lang": "en",
            "country": "US",
            "safesearch": "moderate"
        }
        
        logger.debug(f"Brave API request: {params}")
        
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params,
            timeout=10
        )
        
        logger.debug(f"Brave API response: {response.status_code}")
        response.raise_for_status()
        
        data = response.json()
        
        if "web" in data and "results" in data["web"]:
            results = []
            for item in data["web"]["results"]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("description", "")
                })
            
            logger.info(f"📊 Brave Search: {len(results)} results found")
            return {
                "query": query,
                "results": results,
                "provider": "brave"
            }
        else:
            logger.warning("Brave API returned unexpected format")
            return {"query": query, "results": [], "error": "Unexpected response format", "provider": "brave"}
            
    except Exception as e:
        logger.error(f"Brave search exception: {e}")
        return {"query": query, "results": [], "error": str(e), "provider": "brave"}

# Search keyword pools by priority (incident-focused order)
INCIDENT_KEYWORDS = [
    "incident", "outage", "down", "disruption", "failure", 
    "unavailable", "postmortem", "service degradation", "issue",
    "maintenance", "problem", "error"
]

RELEVANCE_THRESHOLD = 4.0  # More reasonable threshold for incident relevance

def log_llm_response(response_content: str, context: str = "LLM"):
    """Log a one-line summary of LLM response"""
    try:
        # Try to extract JSON content first
        import re
        json_match = re.search(r'"reasoning":\s*"([^"]+)"', response_content)
        if json_match:
            reasoning = json_match.group(1)
            logger.info(f"🤖 {context}: {reasoning}")
            return
        
        # Try to extract overall_score
        score_match = re.search(r'"overall_score":\s*(\d+(?:\.\d+)?)', response_content)
        if score_match:
            score = score_match.group(1)
            logger.info(f"🤖 {context}: Score {score}/10")
            return
        
        # Fallback to first meaningful line (no truncation)
        lines = [line.strip() for line in response_content.strip().split('\n') if line.strip()]
        for line in lines:
            if not line.startswith('```') and len(line) > 10:
                logger.info(f"🤖 {context}: {line}")
                return
        
        logger.info(f"🤖 {context}: Response received")
    except Exception as e:
        logger.info(f"🤖 {context}: Response received (parse error)")

def evaluate_search_results_strict(query: str, results: List[dict]) -> dict:
    """Comprehensive multi-angle evaluation of incident search results in one LLM call"""
    try:
        if not results:
            return {"overall_relevant": False, "average_score": 0, "results_scores": [], "reason": "No results"}
        
        # Prepare all results for single LLM evaluation
        results_content = ""
        for i, result in enumerate(results):
            content_sample = f"Title: {result.get('title', '')}\nURL: {result.get('url', '')}\nContent: {result.get('content', '')[:400]}..."
            results_content += f"\n--- RESULT {i+1} ---\n{content_sample}\n"
        
        prompt = ChatPromptTemplate.from_template("""
        Analyze ALL search results for incident completeness: {query}
        
        {results_content}
        
        Multi-angle evaluation framework for EACH result:
        
        1. INCIDENT PROCESS COMPLETENESS (0-10):
           - Timeline clarity (start/end times, key milestones)
           - Impact components (affected services, systems)
           - Event sequence (what happened when)
        
        2. ROOT CAUSE DEPTH (0-10):
           - Direct cause identification
           - Root cause analysis depth
           - Technical detail accuracy
        
        3. IMPACT SCOPE DETAIL (0-10):
           - User impact quantification
           - Service scope (which services affected)
           - Geographic/duration specificity
        
        4. ACTIONABLE MEASURES (0-10):
           - Prevention measures (before)
           - Response measures (during)
           - Recovery measures (after)
        
        5. SOURCE CREDIBILITY (0-10):
           - Official status pages/postmortems
           - Technical blogs/documentation
           - News reliability and technical depth
        
        Return JSON array with evaluation for each result:
        [
          {{
            "result_index": 0,
            "incident_completeness": 0-10,
            "root_cause_depth": 0-10,
            "impact_scope": 0-10,
            "actionable_measures": 0-10,
            "source_credibility": 0-10,
            "overall_score": 0-10,
            "incident_type": "availability/security/news/other",
            "is_duplicate": true/false,
            "missing_info": ["list", "of", "missing", "elements"],
            "reasoning": "brief explanation of scoring"
          }},
          ...
        ]
        
        STRICT CRITERIA: Only score 8+ for comprehensive incident reports with clear timeline, root cause, and actionable details.
        FOCUS: System availability incidents only, exclude security breaches and general news.
        """)
        
        response = llm.invoke(prompt.format(query=query, results_content=results_content))
        
        # Extract individual reasoning from the response for logging
        result_json = extract_json_from_response(response.content)
        
        if result_json:
            import json
            results_evaluation = json.loads(result_json)
            
            # Log individual evaluations
            for i, evaluation in enumerate(results_evaluation):
                reasoning = evaluation.get("reasoning", "No reasoning provided")
                score = evaluation.get("overall_score", 0)
                incident_completeness = evaluation.get("incident_completeness", 0)
                root_cause_depth = evaluation.get("root_cause_depth", 0)
                
                # Calculate relevance and comprehensiveness
                relevance_score = (score + incident_completeness) / 2
                comprehensiveness_score = (incident_completeness + root_cause_depth) / 2
                
                log_llm_response(f"Result {i+1}: Match:{relevance_score:.1f} Depth:{comprehensiveness_score:.1f} Overall:{score}/10 | {reasoning}", f"Eval-{i+1}")
            
            # Add URL and title to each evaluation
            for i, evaluation in enumerate(results_evaluation):
                if i < len(results):
                    evaluation["url"] = results[i].get("url", "")
                    evaluation["title"] = results[i].get("title", "")
        else:
            log_llm_response("Failed to parse evaluation results", "EvalAll")
            # Fallback to default scores if parsing fails
            results_evaluation = []
            for i in range(len(results)):
                results_evaluation.append({
                    "result_index": i, "overall_score": 2, "reasoning": "Could not evaluate",
                    "incident_completeness": 2, "root_cause_depth": 2, "impact_scope": 2,
                    "actionable_measures": 2, "source_credibility": 2, "incident_type": "other",
                    "is_duplicate": False, "missing_info": ["evaluation_failed"],
                    "url": results[i].get("url", ""), "title": results[i].get("title", "")
                })
        
        # Calculate comprehensive metrics
        scores = [r.get("overall_score", 0) for r in results_evaluation]
        average_score = sum(scores) / len(scores) if scores else 0
        max_score = max(scores) if scores else 0
        
        # Filter for availability incidents only
        availability_incidents = [r for r in results_evaluation if r.get("incident_type") == "availability"]
        high_quality_count = sum(1 for s in scores if s >= 8)
        recommended_reports = sorted([r for r in results_evaluation if r.get("overall_score", 0) >= 8], 
                                   key=lambda x: x.get("overall_score", 0), reverse=True)
        
        # Overall relevance based on more reasonable criteria
        overall_relevant = (average_score >= 5.0 or  # Raised threshold to trigger more keyword extraction
                          (max_score >= 7 and high_quality_count >= 1) or  # Higher individual threshold
                          (len(availability_incidents) >= 2 and max_score >= 5) or  # Multiple availability incidents
                          (len(availability_incidents) >= 1 and max_score >= 6))  # Single good availability incident
        
        return {
            "overall_relevant": overall_relevant,
            "average_score": round(average_score, 1),
            "max_score": max_score,
            "relevant_count": high_quality_count,
            "total_results": len(results),
            "availability_incidents": len(availability_incidents),
            "recommended_reports": recommended_reports,
            "results_scores": results_evaluation,
            "reason": f"Avg: {average_score:.1f}, Max: {max_score}, Quality: {high_quality_count}/{len(results)}, Availability: {len(availability_incidents)}"
        }
        
    except Exception as e:
        logger.error(f"Comprehensive evaluation failed: {e}")
        return {"overall_relevant": False, "average_score": 0, "results_scores": [], "reason": f"Evaluation error: {e}"}

def extract_keywords_from_results(query: str, results: List[dict]) -> List[str]:
    """Extract new keywords from search results for further searches"""
    try:
        if not results:
            return []
        
        # Sample content from results
        content_sample = "\n".join([
            f"Title: {r.get('title', '')}\nContent: {r.get('content', '')[:200]}..."
            for r in results[:3]
        ])
        
        prompt = ChatPromptTemplate.from_template("""
        Extract specific technical keywords from these search results that could help find more comprehensive incident information for: {query}
        
        Results:
        {content}
        
        Extract keywords that are:
        1. Technical terms (DNS, API, service names, error codes)
        2. Incident-specific terms (postmortem, RCA, timeline, impact)
        3. Company/service specific terms (service names, regions, components)
        
        Return JSON array of 3-5 new keywords that are NOT generic (avoid: outage, down, incident, disruption, failure)
        Focus on specific technical terms that could find detailed incident reports.
        
        Example: ["DNS resolution", "postmortem", "us-east-1", "lambda timeout", "RCA"]
        """)
        
        response = llm.invoke(prompt.format(query=query, content=content_sample))
        log_llm_response(response.content, "KeywordExtract")
        result_json = extract_json_from_response(response.content)
        
        if result_json:
            import json
            keywords = json.loads(result_json)
            # Filter out existing keywords
            new_keywords = [k for k in keywords if k.lower() not in [existing.lower() for existing in INCIDENT_KEYWORDS]]
            logger.info(f"🔍 Extracted new keywords: {new_keywords}")
            return new_keywords[:3]  # Limit to 3 new keywords
        
        return []
        
    except Exception as e:
        logger.error(f"Keyword extraction failed: {e}")
        return []

def smart_search_with_keywords(query_base: str, provider_func, max_attempts: int = 6) -> dict:
    """Search with keyword pool and dynamic keyword extraction"""
    date_company = query_base  # e.g., "2025-10-20 amazon"
    best_result = None
    best_evaluation = {"average_score": 0}
    used_keywords = []
    
    # First try standard keywords
    for i, keyword in enumerate(INCIDENT_KEYWORDS[:max_attempts]):
        search_query = f"{date_company} {keyword}"
        logger.info(f"🔍 Trying '{keyword}' keyword ({i+1}/{max_attempts})")
        used_keywords.append(keyword)
        
        result = provider_func(search_query)
        
        # If search failed or empty, try next keyword
        if result.get("error") or not result.get("results"):
            logger.info(f"❌ No results for '{keyword}', trying next")
            continue
        
        # Show search results concisely
        results = result.get("results", [])
        logger.info(f"📋 Found {len(results)} results:")
        for j, r in enumerate(results[:3]):  # Show first 3
            title = r.get("title", "No title")[:60] + "..." if len(r.get("title", "")) > 60 else r.get("title", "No title")
            logger.info(f"   {j+1}. {title}")
        
        # Always evaluate results (even if from cache)
        evaluation = evaluate_search_results_strict(search_query, result["results"])
        avg_score = evaluation.get("average_score", 0)
        max_score = evaluation.get("max_score", 0)
        relevant_count = evaluation.get("relevant_count", 0)
        total_count = evaluation.get("total_results", 0)
        availability_count = evaluation.get("availability_incidents", 0)
        recommended_count = len(evaluation.get("recommended_reports", []))
        
        logger.info(f"⭐ Evaluation: Avg={avg_score:.1f}, Max={max_score}, Quality={relevant_count}/{total_count}, Availability={availability_count}, Recommended={recommended_count}")
        
        # Show top recommended reports
        recommended = evaluation.get("recommended_reports", [])
        if recommended:
            logger.info(f"🏆 Top Reports:")
            for j, report in enumerate(recommended[:2]):  # Show top 2
                title = report.get("title", "No title")[:50] + "..." if len(report.get("title", "")) > 50 else report.get("title", "No title")
                score = report.get("overall_score", 0)
                logger.info(f"   {j+1}. [{score:.1f}/10] {title}")
        
        # Track best result even if not good enough
        if avg_score > best_evaluation.get("average_score", 0):
            best_result = result
            best_result["keyword_used"] = keyword
            best_result["evaluation"] = evaluation
            best_evaluation = evaluation
        
        # If good enough, use these results
        if evaluation.get("overall_relevant", False):
            logger.info(f"✅ Excellent results with '{keyword}' - using these!")
            result["keyword_used"] = keyword
            result["evaluation"] = evaluation
            return result
        
        # If relevant but not comprehensive, try extracting new keywords
        if avg_score >= 1.5 and availability_count >= 1:  # Lower threshold to trigger more often
            logger.info(f"🔄 Results relevant but not comprehensive - extracting new keywords")
            new_keywords = extract_keywords_from_results(search_query, results)
            
            # Try new keywords (limited attempts)
            for new_keyword in new_keywords[:2]:  # Try max 2 new keywords
                if len(used_keywords) >= max_attempts + 2:  # Don't exceed total limit
                    break
                    
                new_search_query = f"{date_company} {new_keyword}"
                logger.info(f"🆕 Trying extracted keyword: '{new_keyword}'")
                used_keywords.append(new_keyword)
                
                new_result = provider_func(new_search_query)
                
                if new_result.get("results") and not new_result.get("error"):
                    new_evaluation = evaluate_search_results_strict(new_search_query, new_result["results"])
                    new_avg_score = new_evaluation.get("average_score", 0)
                    
                    logger.info(f"⭐ New keyword result: Avg={new_avg_score:.1f}")
                    
                    if new_evaluation.get("overall_relevant", False):
                        logger.info(f"✅ Excellent results with new keyword '{new_keyword}' - using these!")
                        new_result["keyword_used"] = new_keyword
                        new_result["evaluation"] = new_evaluation
                        return new_result
                    
                    # Update best if better
                    if new_avg_score > best_evaluation.get("average_score", 0):
                        best_result = new_result
                        best_result["keyword_used"] = new_keyword
                        best_result["evaluation"] = new_evaluation
                        best_evaluation = new_evaluation
        
        logger.info(f"⚠️  Not comprehensive enough, trying next keyword")
    
    # All keywords exhausted - return best result with evaluation details
    if best_result:
        logger.warning(f"🔄 Keywords exhausted. Best: avg={best_evaluation.get('average_score'):.1f} (threshold={RELEVANCE_THRESHOLD})")
        return best_result
    else:
        logger.warning("❌ All keywords exhausted, no results found")
        return {"query": query_base, "results": [], "error": "Keywords exhausted", "provider": "unknown"}
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
    """Search using TC Cloud API with smart site targeting"""
    try:
        logger.debug(f"TC search starting for: {query}")
        
        secret_id = os.getenv("TC_SECRET_ID")
        secret_key = os.getenv("TC_SECRET_KEY")
        
        if not secret_id or not secret_key:
            logger.debug("TC API credentials not found")
            return {"query": query, "results": [], "error": "Missing credentials", "provider": "tc"}
        
        # Initialize credentials and client
        cred = credential.Credential(secret_id, secret_key)
        httpProfile = HttpProfile()
        httpProfile.endpoint = "wsa.tencentcloudapi.com"
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        client = wsa_client.WsaClient(cred, "", clientProfile)
        
        # Try with site restrictions first
        site_queries = [
            f"{query} site:aws.amazon.com OR site:status.aws.amazon.com OR site:github.com OR site:reddit.com OR site:news.ycombinator.com",
            query  # Fallback without site restrictions
        ]
        
        for i, search_query in enumerate(site_queries):
            logger.debug(f"TC search attempt {i+1}: {'with sites' if i == 0 else 'without sites'}")
            
            req = models.SearchProRequest()
            params = {"Query": search_query, "Mode": 0}
            req.from_json_string(json.dumps(params))
            
            resp = client.SearchPro(req)
            response_data = json.loads(resp.to_json_string())
            
            pages = response_data.get("Pages")
            if pages is None:
                continue
                
            results = []
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
            
            # If we got decent results, use them
            if len(results) >= 3:
                logger.info(f"📊 TC Search: {len(results)} results found ({'with sites' if i == 0 else 'no restrictions'})")
                return {"query": query, "results": results, "provider": "tc"}
        
        # No good results from either approach
        logger.info(f"📊 TC Search: No results found")
        return {"query": query, "results": [], "error": "No results", "provider": "tc"}
        
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
        log_llm_response(response.content, "Parse")
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
    """Search using TC (primary), Brave, or Tavily with strict evaluation and provider fallback"""
    try:
        # Check cache first
        cached = get_cached_result(query)
        if cached:
            logger.info(f"💾 Using cached search results")
            cached_result = {"query": query, "results": cached["results"], "cached": True, "provider": cached.get("provider", "cache")}
            
            # Evaluate cached results
            logger.info(f"📋 Found {len(cached['results'])} cached results:")
            for j, r in enumerate(cached["results"][:3]):  # Show first 3
                title = r.get("title", "No title")[:60] + "..." if len(r.get("title", "")) > 60 else r.get("title", "No title")
                logger.info(f"   {j+1}. {title}")
            
            evaluation = evaluate_search_results_strict(query, cached["results"])
            cached_result["evaluation"] = evaluation
            return cached_result
        
        # Try TC search first with smart keywords
        if os.getenv("TC_SECRET_ID") and os.getenv("TC_SECRET_KEY"):
            logger.info("🔍 Trying TC Search...")
            result = smart_search_with_keywords(query, tc_search, max_attempts=4)
            
            if result.get("results") and not result.get("error"):
                evaluation = result.get("evaluation", {})
                avg_score = evaluation.get("average_score", 0)
                if evaluation.get("overall_relevant", False):
                    logger.info(f"✅ TC Search: Excellent results (avg: {avg_score:.1f})")
                    cache_result(query, result)
                    return result
                elif avg_score >= 3.0:  # Lower threshold
                    logger.info(f"✅ TC Search: Good results (avg: {avg_score:.1f})")
                    cache_result(query, result)
                    return result
                else:
                    logger.info(f"⚠️  TC Search: Below threshold (avg: {avg_score:.1f} < 3.0)")
            elif result.get("error") == "Keywords exhausted":
                logger.info("⚠️  TC Search: All keywords exhausted")
            else:
                logger.info("❌ TC Search: Failed")
        
        # Try Brave search as first fallback
        use_brave = os.getenv("USE_BRAVE_SEARCH", "true").lower() == "true"
        
        if use_brave and os.getenv("BRAVE_API_KEY"):
            logger.info("🔍 Trying Brave Search...")
            result = smart_search_with_keywords(query, brave_search_with_rate_limit, max_attempts=4)
            
            if result.get("results") and not result.get("error"):
                evaluation = result.get("evaluation", {})
                avg_score = evaluation.get("average_score", 0)
                if evaluation.get("overall_relevant", False):
                    logger.info(f"✅ Brave Search: Excellent results (avg: {avg_score:.1f})")
                    cache_result(query, result)
                    return result
                elif avg_score >= 3.0:  # Lower threshold
                    logger.info(f"✅ Brave Search: Good results (avg: {avg_score:.1f})")
                    cache_result(query, result)
                    return result
                else:
                    logger.info(f"⚠️  Brave Search: Below threshold (avg: {avg_score:.1f} < 3.0)")
            elif result.get("error") == "Keywords exhausted":
                logger.info("⚠️  Brave Search: All keywords exhausted")
            else:
                logger.info("❌ Brave Search: Failed")
        
        # Fallback to Tavily
        if os.getenv("TAVILY_API_KEY"):
            logger.info("🔍 Trying Tavily Search (final attempt)...")
            result = smart_search_with_keywords(query, tavily_search_fallback, max_attempts=4)
            
            if result.get("results") and not result.get("error"):
                evaluation = result.get("evaluation", {})
                avg_score = evaluation.get("average_score", 0)
                logger.info(f"📊 Tavily Search: Final results (avg: {avg_score:.1f})")
                cache_result(query, result)
                return result
            elif result.get("error") == "Keywords exhausted":
                logger.info("⚠️  Tavily Search: All keywords exhausted")
        
        # No relevant results from any provider
        return {"query": query, "results": [], "error": "No relevant results from any provider", "cached": False}
        
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
        - "incident_name": Format as "{{company}} on YYYY-MM-DD due to [cause] affecting [services]"
        - "company_product": Simple string with company and product name (e.g., "CrowdStrike Falcon Sensor")
        - "incident_time": Start time in YYYY-MM-DD HH:MM:SS format (use available data or estimate)
        - "report_links": Array of original report URLs from search results
        - "impact_description": Geographic regions, user count, service scope, duration in markdown format
        - "incident_process": Timeline with three phases in markdown format
        - "root_cause": Direct and root causes in markdown format  
        - "improvement_measures": Prevention and response measures in markdown format
        
        Use markdown formatting for multi-line fields. Extract URLs from search data for report_links.
        """)
        
        response = llm.invoke(prompt.format(
            search_data=search_data,
            metadata=json.dumps(metadata),
            timeline=json.dumps(timeline)
        ))
        log_llm_response(response.content, "Report")
        
        json_text = extract_json_from_response(response.content)
        
        try:
            parsed_data = json.loads(json_text)
            result = IncidentReport(**parsed_data)
            logger.info(f"Generated structured report: {result.incident_name}")
            return result
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Report JSON parsing failed: {e}, raw response: {response.content[:200]}")
            return IncidentReport(
                incident_name="Unknown incident",
                company_product="Unknown",
                incident_time="Unknown",
                report_links=[],
                impact_description="Unable to determine impact details",
                incident_process="Unable to determine incident process",
                root_cause="Unable to determine root cause",
                improvement_measures="Unable to determine improvement measures"
            )
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return IncidentReport(
            incident_name="Report generation failed",
            company_product="Unknown",
            incident_time="Unknown", 
            report_links=[],
            impact_description="Error occurred during report generation",
            incident_process="Error occurred during report generation",
            root_cause="Error occurred during report generation",
            improvement_measures="Error occurred during report generation"
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
    """Enhanced search with keyword strategy and relevance evaluation"""
    try:
        # Generate base search query (date + company + description)
        base_query = f"{state['date']}"
        if state.get("company"):
            base_query += f" {state['company']}"
        if state.get("incident_description"):
            base_query += f" {state['incident_description']}"
        
        logger.info(f"🎯 Starting incident search: {base_query}")
        
        # Single smart search with keyword strategy
        result = search_with_fallback.invoke({"query": base_query})
        
        results = [result]
        keywords = base_query.split()
        errors = []
        
        if "error" in result:
            errors.append(f"Search error: {result['error']}")
        
        # Log final search summary
        provider = result.get("provider", "unknown")
        cached = result.get("cached", False)
        keyword_used = result.get("keyword_used", "none")
        evaluation = result.get("evaluation", {})
        
        if cached:
            logger.info(f"💾 Used cached results from {provider}")
        else:
            avg_score = evaluation.get("average_score", 0)
            relevant_count = evaluation.get("relevant_count", 0)
            total_count = evaluation.get("total_results", 0)
            availability_count = evaluation.get("availability_incidents", 0)
            recommended_count = len(evaluation.get("recommended_reports", []))
            
            # Determine relevance and comprehensiveness
            relevance = "High" if avg_score >= 6 else "Medium" if avg_score >= 3 else "Low"
            comprehensiveness = "Complete" if recommended_count > 0 else "Partial" if availability_count > 0 else "Limited"
            
            logger.info(f"📊 Final Results: {provider} | '{keyword_used}' | Score: {avg_score:.1f} | Relevance: {relevance} | Comprehensiveness: {comprehensiveness}")
            logger.info(f"📈 Details: Quality: {relevant_count}/{total_count} | Availability: {availability_count} | Recommended: {recommended_count}")
            
            # Show final recommended reports
            recommended = evaluation.get("recommended_reports", [])
            if recommended:
                logger.info(f"📋 Recommended Reports ({len(recommended)}):")
                for j, report in enumerate(recommended[:3]):  # Show top 3
                    title = report.get("title", "No title")[:60] + "..." if len(report.get("title", "")) > 60 else report.get("title", "No title")
                    score = report.get("overall_score", 0)
                    incident_type = report.get("incident_type", "unknown")
                    logger.info(f"   {j+1}. [{score:.1f}/10] [{incident_type}] {title}")
        
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
            "incident_report": f"# {report.incident_name}\n\n**Company/Product:** {report.company_product}\n\n**Incident Time:** {report.incident_time}\n\n## Impact\n{report.impact_description}\n\n## Incident Process\n{report.incident_process}\n\n## Root Cause\n{report.root_cause}\n\n## Improvement Measures\n{report.improvement_measures}",
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
