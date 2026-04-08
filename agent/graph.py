"""LangGraph 기반 PKM Agent 그래프 정의"""
from langchain_ollama import ChatOllama
# from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from tools.note_tool import save_note, read_note, list_notes
from tools.retriever_tool import add_to_knowledge, search_knowledge

TOOLS = [save_note, read_note, list_notes, add_to_knowledge, search_knowledge]

SYSTEM_PROMPT = """당신은 개인 지식 관리(PKM) 어시스턴트입니다.
사용자가 지식을 저장하고, 정리하고, 나중에 다시 찾을 수 있도록 돕습니다.

사용 가능한 도구:
- save_note: 노트를 마크다운 파일로 저장
- read_note: 저장된 노트 읽기
- list_notes: 저장된 노트 목록 조회
- add_to_knowledge: 내용을 벡터DB에 저장 (의미 기반 검색용)
- search_knowledge: 저장된 지식에서 관련 내용 검색

중요한 정보나 학습 내용은 노트 저장과 벡터DB 저장을 모두 수행하세요."""


def build_graph(model: str = "llama3.2"):
    llm = ChatOllama(model=model, temperature=0)
    # llm = ChatOpenAI(model=model, temperature=0)  # OpenAI 사용 시
    graph = create_react_agent(
        model=llm,
        tools=TOOLS,
        prompt=SYSTEM_PROMPT,
    )
    return graph
