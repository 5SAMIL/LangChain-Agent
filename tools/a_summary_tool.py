"""A 역할 - 요약 Tool"""

from langchain_core.tools import tool
from tools.a_feature_common import run_a_analysis


@tool
def a_generate_summary(
    content: str = "",
    note_title: str = "",
    source_name: str = "",
    source_type: str = "",
) -> str:
    """A 역할: 문서 요약을 생성한다. content 또는 note_title을 입력받는다."""
    try:
        analysis = run_a_analysis(
            content=content,
            note_title=note_title,
            source_name=source_name,
            source_type=source_type,
        )
    except Exception as e:
        return f"A 요약 실패: {e}"

    summary = analysis.get("summary", {})
    return (
        "A 요약 결과\n"
        f"analysis_id: {analysis.get('analysis_id')}\n"
        f"source: {analysis.get('source_name')} ({analysis.get('source_type')})\n"
        f"short: {summary.get('short', '')}\n"
        f"detailed: {summary.get('detailed', '')}"
    )
