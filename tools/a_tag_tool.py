"""A 역할 - 태그 Tool"""

from langchain_core.tools import tool
from tools.a_feature_common import run_a_analysis


@tool
def a_generate_tags(
    content: str = "",
    note_title: str = "",
    source_name: str = "",
    source_type: str = "",
    top_k: int = 10,
) -> str:
    """A 역할: 문서 태그를 생성한다. content 또는 note_title을 입력받는다."""
    try:
        analysis = run_a_analysis(
            content=content,
            note_title=note_title,
            source_name=source_name,
            source_type=source_type,
        )
    except Exception as e:
        return f"A 태그 생성 실패: {e}"

    top_k = max(1, min(top_k, 30))
    tags = analysis.get("tags", [])[:top_k]

    lines = [
        "A 태그 결과",
        f"analysis_id: {analysis.get('analysis_id')}",
        f"source: {analysis.get('source_name')} ({analysis.get('source_type')})",
        f"top_k: {top_k}",
    ]

    if not tags:
        lines.append("- 태그 없음")
        return "\n".join(lines)

    for item in tags:
        lines.append(f"- {item.get('name')} (confidence={item.get('confidence')})")
    return "\n".join(lines)
