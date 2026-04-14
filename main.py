"""PKM Agent 실행 진입점"""
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

from agent.graph import build_graph


def chat(graph, user_input: str) -> str:
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"recursion_limit": 50},
    )

    content = result["messages"][-1].content
    return content.replace("STOP", "").strip()


def main():
    print("PKM Agent 시작 (종료: 'quit')")
    graph = build_graph(model="qwen2.5:14b")    # Ollama 사용 시
    # graph = build_graph(model="gpt-4o-mini")  # OpenAI 사용 시

    while True:
        user_input = input("\n나: ").strip()
        if user_input.lower() in ("quit", "exit", "종료"):
            break
        if not user_input:
            continue

        try:
            response = chat(graph, user_input)
            print(f"\nAgent: {response}")
        except Exception as e:
            print(f"\nAgent 오류: {e}")


if __name__ == "__main__":
    main()
