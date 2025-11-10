#!/usr/bin/env python3
"""Test script for the incident analysis agent"""

from incident_agent import create_incident_agent

def test_basic_agent():
    """Test the basic incident agent with mock data"""
    print("Testing basic incident agent...")
    
    agent = create_incident_agent()
    
    test_cases = [
        {
            "name": "AWS S3 Outage",
            "input": {
                "date": "2024-11",
                "company": "AWS",
                "incident_description": "S3 service outage",
                "search_keywords": [],
                "search_results": [],
                "missing_info": [],
                "iteration_count": 0
            }
        },
        {
            "name": "Generic Incident",
            "input": {
                "date": "October 2024",
                "incident_description": "database connection issues",
                "search_keywords": [],
                "search_results": [],
                "missing_info": [],
                "iteration_count": 0
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- Testing: {test_case['name']} ---")
        try:
            result = agent.invoke(test_case["input"])
            print(f"✓ Success - Iterations: {result['iteration_count']}")
            print(f"  Report length: {len(result.get('incident_report', ''))}")
            print(f"  Keywords found: {len(result.get('search_keywords', []))}")
            print(f"  Search results: {len(result.get('search_results', []))}")
            
            if result.get("review_result"):
                approved = result["review_result"].get("approved", False)
                print(f"  Review approved: {approved}")
        
        except Exception as e:
            print(f"✗ Failed: {e}")

def print_agent_structure():
    """Print the agent's graph structure"""
    print("\n=== Agent Structure ===")
    agent = create_incident_agent()
    
    # Get graph representation
    try:
        graph_dict = agent.get_graph().to_json()
        print("Nodes:", list(graph_dict.get("nodes", {}).keys()))
        print("Edges:", len(graph_dict.get("edges", [])))
    except Exception as e:
        print(f"Could not display graph structure: {e}")

if __name__ == "__main__":
    print("Incident Analysis Agent Test Suite")
    print("=" * 40)
    
    print_agent_structure()
    test_basic_agent()
    
    print("\n" + "=" * 40)
    print("Test completed!")
    print("\nTo use the enhanced version with real LLM:")
    print("1. Set OPENAI_API_KEY environment variable")
    print("2. Run: python incident_agent_enhanced.py")
