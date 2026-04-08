"""LangGraph 기반 PKM Agent 그래프 정의"""
from langchain_ollama import ChatOllama
# from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from tools import load_all_tools

SYSTEM_PROMPT = """You are a personal knowledge management assistant. Respond in Korean.

Notes are saved as markdown files in: data/notes/
Knowledge is stored in a vector database at: data/vectordb/

When the user asks to save or organize something:
1. Call save_note once with a title and content
2. Call add_to_knowledge once with the same content
3. Reply with a short confirmation including the file path (data/notes/<title>.md) and stop

When the user asks where something was saved, tell them the exact path.

When the user asks to search or read:
1. Call the appropriate tool once
2. Reply with the result and stop

Do NOT call the same tool twice. Do NOT loop. After completing the task, always give a final answer."""


def build_graph(model: str = "llama3.2"):
    llm = ChatOllama(model=model, temperature=0)
    # llm = ChatOpenAI(model=model, temperature=0)  # OpenAI 사용 시
    graph = create_react_agent(
        model=llm,
        tools=load_all_tools(),
        prompt=SYSTEM_PROMPT,
    )
    return graph
