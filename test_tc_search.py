#!/usr/bin/env python3
"""
Simple test for TC search integration
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_tc_search():
    """Test TC search function"""
    # Import after loading env vars
    from incident_agent import tc_search
    
    # Check if credentials are available
    secret_id = os.getenv("TC_SECRET_ID")
    secret_key = os.getenv("TC_SECRET_KEY")
    
    if not secret_id or not secret_key:
        print("❌ TC credentials not found in environment")
        print("Please set TC_SECRET_ID and TC_SECRET_KEY in .env file")
        return False
    
    print(f"✅ Found TC credentials: {secret_id[:8]}...")
    
    # Test search
    print("🔍 Testing TC search with query: 'AWS outage 2024'")
    result = tc_search("AWS outage 2024")
    
    print(f"Provider: {result.get('provider')}")
    print(f"Results count: {len(result.get('results', []))}")
    
    if result.get('error'):
        print(f"❌ Error: {result['error']}")
        return False
    
    if result.get('results'):
        print("✅ Search successful!")
        for i, r in enumerate(result['results'][:2]):  # Show first 2 results
            print(f"  {i+1}. {r.get('title', 'No title')}")
            print(f"     URL: {r.get('url', 'No URL')}")
            print(f"     Content: {r.get('content', 'No content')[:100]}...")
        return True
    else:
        print("⚠️  No results returned")
        return False

def test_search_fallback():
    """Test the complete search fallback system"""
    from incident_agent import search_with_fallback
    
    print("\n🔍 Testing complete search fallback system...")
    result = search_with_fallback.invoke({"query": "GitHub outage 2024"})
    
    print(f"Provider: {result.get('provider')}")
    print(f"Cached: {result.get('cached', False)}")
    print(f"Results count: {len(result.get('results', []))}")
    
    if result.get('error'):
        print(f"❌ Error: {result['error']}")
        return False
    
    if result.get('results'):
        print("✅ Fallback search successful!")
        return True
    else:
        print("⚠️  No results from any provider")
        return False

if __name__ == "__main__":
    print("🧪 Testing TC Search Integration\n")
    
    # Test individual TC search
    tc_success = test_tc_search()
    
    # Test complete fallback system
    fallback_success = test_search_fallback()
    
    print(f"\n📊 Test Results:")
    print(f"  TC Search: {'✅ PASS' if tc_success else '❌ FAIL'}")
    print(f"  Search Fallback: {'✅ PASS' if fallback_success else '❌ FAIL'}")
