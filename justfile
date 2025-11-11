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

# Test TC search integration
test-tc:
    uv run python test_tc_search.py

incident input:
    uv run --env-file .env python incident_agent.py {{input}}

# Clean up generated files
clean:
    rm -rf .venv __pycache__ *.pyc .cache

