default:
    just --list

init:
    uv init || true
    uv add langgraph langchain-core

push:
    jj bookmark track lg-dev@origin
    jj bs lg-dev -r @-
    jj psb lg-dev

