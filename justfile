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

# Run incident agent (requires TAVILY_API_KEY + OPENAI_API_KEY)
run:
    uv run --env-file .env python incident_agent.py

# Clean up generated files
clean:
    rm -rf .venv __pycache__ *.pyc .cache

