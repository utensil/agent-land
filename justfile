default:
    just --list

init:
    uv init || true
    uv add langgraph langchain-core
    uv add 'langgraph-cli[inmem]'

push:
    jj bookmark track lg-dev@origin
    jj bs lg-dev -r @-
    jj psb lg-dev

# https://docs.aws.amazon.com/amazonq/latest/qdeveloper-ug/command-line-mcp-config-CLI.html
# 
qmcp:
    hx ~/.aws/amazonq/mcp.json

# Install dependencies
install:
    uv sync

# Run basic incident agent test
test:
    uv run python test_incident_agent.py

# Run basic incident agent
run:
    uv run python incident_agent.py

# Run enhanced incident agent (requires OPENAI_API_KEY)
run-enhanced:
    uv run --env-file .env python incident_agent_enhanced.py

# Run Tavily-powered agent (requires TAVILY_API_KEY + OPENAI_API_KEY)
run-tavily:
    uv run --env-file .env python incident_agent_tavily.py

# Run production agent with all features (requires TAVILY_API_KEY + OPENAI_API_KEY)
run-production:
    uv run --env-file .env python incident_agent_production.py

# Clean up generated files
clean:
    rm -rf .venv __pycache__ *.pyc .cache

