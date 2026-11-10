"""Check that the selected framework integrations import in the project environment."""

from __future__ import annotations


def main() -> None:
    import fastapi
    import httpx
    import langchain
    import langchain_chroma
    import langchain_groq
    import langchain_huggingface
    import langchain_openai
    import langgraph
    import matplotlib
    import numpy
    import pandas
    import pydantic
    import pypdf
    import sentence_transformers
    import seaborn
    import yaml

    from langchain_core.prompts import ChatPromptTemplate
    from langgraph.graph import END, START, StateGraph
    from langchain_groq import ChatGroq
    from typing import TypedDict

    class SmokeState(TypedDict):
        value: str

    graph = StateGraph(SmokeState)
    graph.add_node("identity", lambda state: state)
    graph.add_edge(START, "identity")
    graph.add_edge("identity", END)
    compiled_graph = graph.compile()
    prompt = ChatPromptTemplate.from_messages(
        [("system", "Reply briefly."), ("human", "{question}")]
    )
    assert compiled_graph.invoke({"value": "ok"}) == {"value": "ok"}
    assert prompt.format_messages(question="smoke")

    modules = (
        fastapi,
        httpx,
        langchain,
        langchain_chroma,
        langchain_groq,
        langchain_huggingface,
        langchain_openai,
        langgraph,
        matplotlib,
        numpy,
        pandas,
        pydantic,
        pypdf,
        sentence_transformers,
        seaborn,
        yaml,
    )
    print("Imported", len(modules), "selected modules and core framework symbols.")
    print("ChatGroq:", ChatGroq.__name__)
    print("ChatPromptTemplate:", ChatPromptTemplate.__name__)
    print("StateGraph:", StateGraph.__name__, "compiled graph and prompt smoke check passed.")


if __name__ == "__main__":
    main()
