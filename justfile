default:
    just --list

init:
    uv init || true
    uv add langgraph langchain-core
