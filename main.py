"""PKM Agent 실행 진입점"""
import sys
import threading
import itertools
import time
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv()

from agent.graph import build_graph

FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# 각 단계별 목표 진행률 (%)
STEP_PROGRESS = {
    "생각 중":       (0,  40),
    "Tool 호출":     (40, 65),
    "결과 처리 중":  (65, 85),
    "응답 생성 중":  (85, 98),
}
BAR_WIDTH = 30


def _render_bar(pct: float) -> str:
    filled = int(BAR_WIDTH * pct / 100)
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    return f"[{bar}] {pct:5.1f}%"


def _spinner(stop_event: threading.Event, status: dict):
    """경과 시간 + 진행률 바 + 현재 작업 상태 표시"""
    start = time.time()
    current_pct = 0.0

    for frame in itertools.cycle(FRAMES):
        if stop_event.is_set():
            # 완료 시 100% 표시
            elapsed = time.time() - start
            line = f"\r✓ {_render_bar(100.0)} | 완료 ({elapsed:.1f}s)"
            sys.stdout.write(line.ljust(80))
            sys.stdout.flush()
            break

        elapsed = time.time() - start
        step = status.get("step", "생각 중")

        # 현재 단계의 목표 진행률 범위
        low, high = STEP_PROGRESS.get(step, (0, 98))

        # 목표치를 향해 서서히 증가 (최대 high까지)
        if current_pct < high:
            # 목표까지 남은 거리의 5%씩 증가 (점점 느려지는 효과)
            current_pct += (high - current_pct) * 0.05
        current_pct = max(current_pct, float(low))

        detail = status.get("detail", "")
        bar = _render_bar(current_pct)
        info = f"{step} {detail}".strip()
        line = f"\r{frame} {bar} | {info} ({elapsed:.1f}s)"
        sys.stdout.write(line.ljust(100))
        sys.stdout.flush()
        time.sleep(0.1)

    # 스피너 줄 완전히 지우고 새 줄로 이동
    sys.stdout.write("\r" + " " * 100 + "\r")
    sys.stdout.write("\n")
    sys.stdout.flush()


def _summarize_args(args: dict) -> str:
    """Tool 인자를 간결하게 요약"""
    parts = []
    for k, v in args.items():
        val = str(v)
        if len(val) > 30:
            val = val[:30] + "…"
        parts.append(f"{k}={val}")
    return ", ".join(parts)


def _preview(text: str, limit: int = 40) -> str:
    text = text.strip().replace("\n", " ")
    return text[:limit] + "…" if len(text) > limit else text


# Tool 결과를 그대로 반환할 읽기 전용 툴 목록
READ_ONLY_TOOLS = {
    "read_file_full", "read_file_structured", "read_image",
    "find_file", "list_directory", "scan_files",
    "find_connections", "detect_duplicates",
    "search_knowledge", "read_note", "list_notes",
}


def chat(graph, user_input: str) -> str:
    stop_event = threading.Event()
    status = {"step": "생각 중", "detail": ""}
    result_holder = {}

    def run():
        try:
            last_tool_name = None
            last_tool_result = None

            for chunk in graph.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config={"recursion_limit": 50},
                stream_mode="values",
            ):
                messages = chunk.get("messages", [])
                if not messages:
                    continue
                last = messages[-1]

                if isinstance(last, AIMessage) and last.tool_calls:
                    for tc in last.tool_calls:
                        args_str = _summarize_args(tc.get("args", {}))
                        last_tool_name = tc["name"]
                        status["step"] = f"Tool 호출: {tc['name']}"
                        status["detail"] = f"({args_str})" if args_str else ""

                elif isinstance(last, ToolMessage):
                    preview = _preview(str(last.content))
                    status["step"] = f"결과 수신: {last.name}"
                    status["detail"] = f"→ {preview}"
                    last_tool_name = last.name
                    last_tool_result = str(last.content)

                elif isinstance(last, AIMessage) and last.content:
                    status["step"] = "응답 생성 중"
                    status["detail"] = ""
                    content = last.content
                    if isinstance(content, list):
                        content = "\n".join(
                            block.get("text", "") if isinstance(block, dict) else str(block)
                            for block in content
                        )
                    result_holder["content"] = content
                    result_holder["last_tool"] = last_tool_name
                    result_holder["last_tool_result"] = last_tool_result

        except Exception as e:
            result_holder["error"] = str(e)

    agent_thread = threading.Thread(target=run, daemon=True)
    spinner_thread = threading.Thread(
        target=_spinner, args=(stop_event, status), daemon=True
    )

    agent_thread.start()
    spinner_thread.start()

    try:
        agent_thread.join()
    except KeyboardInterrupt:
        stop_event.set()
        spinner_thread.join()
        return None

    stop_event.set()
    spinner_thread.join()

    if "error" in result_holder:
        raise Exception(result_holder["error"])

    # 읽기 전용 툴이면 LLM 응답 대신 Tool 결과를 바로 반환
    last_tool = result_holder.get("last_tool")
    if last_tool in READ_ONLY_TOOLS and result_holder.get("last_tool_result"):
        return result_holder["last_tool_result"].strip()

    content = result_holder.get("content", "")
    content = content.replace("STOP", "")
    content = content.replace("\\n", "\n")
    return content.strip()


def main():
    print("PKM Agent 시작 (종료: 'quit' / 중단: Ctrl+C)")
    # graph = build_graph(model="qwen2.5:14b")  # Ollama 사용 시
    graph = build_graph(model="gpt-4o-mini")    # OpenAI 사용 시

    while True:
        try:
            user_input = input("\n나: ").strip()
        except KeyboardInterrupt:
            print("\n종료합니다.")
            break

        if user_input.lower() in ("quit", "exit", "종료"):
            break
        if not user_input:
            continue

        try:
            response = chat(graph, user_input)
            if response is None:
                print("\n[Agent 중단됨] 새 질문을 입력하세요.")
                continue
            sys.stdout.write(f"\nAgent:\n{response}\n")
            sys.stdout.flush()
        except Exception as e:
            print(f"\nAgent 오류: {e}")


if __name__ == "__main__":
    main()
