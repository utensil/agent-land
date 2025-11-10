#!/usr/bin/env python3
"""Test script for the incident analysis agent"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_agent():
    """Test the incident agent"""
    print("Testing incident agent...")
    
    # Check for required API keys
    if not os.getenv("TAVILY_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Missing API keys - testing with mock mode")
        print("   Set TAVILY_API_KEY and OPENAI_API_KEY in .env for full testing")
        return
    
    try:
        from incident_agent import create_incident_agent, ProgressStatus
        
        agent = create_incident_agent()
        
        test_case = {
            "name": "AWS S3 Outage",
            "input": {
                "date": "2024-11",
                "company": "AWS", 
                "incident_description": "S3 service outage",
                "search_keywords": [],
                "search_results": [],
                "timeline": [],
                "missing_info": [],
                "iteration_count": 0,
                "errors": [],
                "progress": ProgressStatus()
            }
        }
        
        print(f"\n--- Testing: {test_case['name']} ---")
        result = agent.invoke(test_case["input"])
        print(f"✓ Success - Iterations: {result['iteration_count']}")
        print(f"  Report length: {len(result.get('incident_report', ''))}")
        print(f"  Keywords found: {len(result.get('search_keywords', []))}")
        print(f"  Search results: {len(result.get('search_results', []))}")
        print(f"  Timeline events: {len(result.get('timeline', []))}")
        print(f"  Errors: {len(result.get('errors', []))}")
        
        if result.get("review_result"):
            approved = result["review_result"].get("approved", False)
            print(f"  Review approved: {approved}")
    
    except Exception as e:
        print(f"✗ Failed: {e}")

if __name__ == "__main__":
    print("Incident Analysis Agent Test Suite")
    print("=" * 40)
    
    test_agent()
    
    print("\n" + "=" * 40)
    print("Test completed!")
    print("\nTo run with real APIs:")
    print("1. Copy .env.example to .env")
    print("2. Add your OPENAI_API_KEY and TAVILY_API_KEY")
    print("3. Run: just run")
