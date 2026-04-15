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
BAR_WIDTH = 30

# 각 단계 실제 발생 시 즉시 반영되는 진행률
STEP_PCT = {
    "생각 중":      5,
    "Tool 호출":   40,
    "결과 수신":   65,
    "응답 생성 중": 85,
}


def _render_bar(pct: float) -> str:
    filled = int(BAR_WIDTH * pct / 100)
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    return f"[{bar}] {pct:5.1f}%"


def _spinner(stop_event: threading.Event, status: dict):
    """실제 이벤트 기반 진행률 + 경과 시간 표시"""
    start = time.time()
    current_pct = 0.0

    for frame in itertools.cycle(FRAMES):
        if stop_event.is_set():
            elapsed = time.time() - start
            line = f"\r✓ {_render_bar(100.0)} | 완료 ({elapsed:.1f}s)"
            sys.stdout.write(line.ljust(110))
            sys.stdout.flush()
            break

        elapsed = time.time() - start
        step = status.get("step", "생각 중")
        detail = status.get("detail", "")
        tokens = status.get("tokens", 0)

        # 실제 이벤트 발생 시 즉시 해당 % 로 점프
        target = STEP_PCT.get(step.split(":")[0].strip(), current_pct)
        if target > current_pct:
            current_pct = target

        # 응답 생성 중에는 토큰 수 표시
        if step == "응답 생성 중" and tokens > 0:
            detail = f"({tokens} tokens)"

        info = f"{step} {detail}".strip()
        line = f"\r{frame} {_render_bar(current_pct)} | {info} ({elapsed:.1f}s)"
        sys.stdout.write(line.ljust(110))
        sys.stdout.flush()
        time.sleep(0.1)

    sys.stdout.write("\r" + " " * 110 + "\r")
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
            from langchain_core.messages import AIMessageChunk

            last_tool_name = None
            last_tool_result = None
            response_tokens = []

            for chunk, metadata in graph.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config={"recursion_limit": 50},
                stream_mode="messages",
            ):
                # Tool 호출 청크
                if isinstance(chunk, AIMessageChunk) and chunk.tool_call_chunks:
                    for tc in chunk.tool_call_chunks:
                        if tc.get("name"):
                            last_tool_name = tc["name"]
                            status["step"] = f"Tool 호출: {tc['name']}"
                            status["detail"] = ""
                            status["tokens"] = 0
                            response_tokens.clear()

                # Tool 결과
                elif isinstance(chunk, ToolMessage):
                    preview = _preview(str(chunk.content))
                    status["step"] = f"결과 수신: {chunk.name}"
                    status["detail"] = f"→ {preview}"
                    last_tool_name = chunk.name
                    last_tool_result = str(chunk.content)

                # 응답 생성 — 토큰 단위로 스트리밍
                elif isinstance(chunk, AIMessageChunk) and chunk.content:
                    content = chunk.content
                    if isinstance(content, list):
                        content = "".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in content
                        )
                    if content:
                        status["step"] = "응답 생성 중"
                        status["detail"] = ""
                        response_tokens.append(content)
                        status["tokens"] = len(response_tokens)

            # 최종 응답 조합
            if response_tokens:
                result_holder["content"] = "".join(response_tokens)
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
