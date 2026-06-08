# PKM Agent

LangChain · LangGraph 기반 **개인 지식 관리(Personal Knowledge Management) Agent**입니다.
노트 저장부터 문서 읽기(PDF/이미지/Office), AI 분석(요약·키워드·태그·분류), 벡터 검색, 자동 분류·정리까지 하나의 대화형 에이전트로 처리합니다.

> 📄 발표 코멘트 · 코드 리뷰 가이드 · 구현 정리를 한 곳에 모은 통합 문서: [`PKM_Agent.html`](PKM_Agent.html)

## 주요 기능

- **노트 관리** — 마크다운 노트 저장/조회/목록 (`data/notes/`)
- **문서 읽기** — PDF, 이미지(OCR), Word(`.docx`), PowerPoint(`.pptx`) 텍스트 추출 및 노트 변환
- **AI 분석 (A-role)** — 요약, 키워드 추출, 태그 생성, 자동 문서 분류, 정리 제안
- **벡터 검색 (RAG)** — Chroma 벡터 DB 기반 지식 검색 및 관련 노트 연결(`find_connections`)
- **중복 탐지** — 노트/텍스트 간 유사도 기반 중복·유사 문서 탐지
- **자동 분류 및 정리 (C-role)** — `taxonomy.json` 규칙에 따라 카테고리별 폴더로 자동 분류·이동·재정리·메타데이터 수정·배치 정리
- **대화형 CLI** — 실시간 진행률 바 + 토큰 스트리밍 표시

## 아키텍처

```
main.py                 CLI 진입점 (스트리밍 출력 + 진행률 표시)
agent/
  graph.py              LangGraph create_react_agent + 시스템 프롬프트
  state.py              상태 정의
tools/                  @tool 단위로 정의된 기능 모듈 (자동 로드)
data/
  notes/                마크다운 노트 저장소
  vectordb/             Chroma 벡터 DB
  taxonomy.json         자동 분류 카테고리/키워드 규칙
tests/                  pytest 테스트
```

- **LLM**: OpenAI `gpt-4o-mini` (temperature=0), 임베딩 `text-embedding-3-small`
- **에이전트**: LangGraph `create_react_agent` (ReAct 패턴) + `MemorySaver` 체크포인터로 대화 맥락 유지
- **툴 로딩**: `tools/load_all_tools()`가 `tools/` 패키지 내 모든 `BaseTool` 인스턴스를 자동 수집 (현재 43개 툴)
- **직접 출력**: 읽기·조회 성격의 툴은 `DIRECT_OUTPUT_TOOLS`로 지정해 LLM 가공 없이 결과를 그대로 출력

## 팀 구성 및 담당 영역

| 담당 | 영역 | 주요 구현 |
| --- | --- | --- |
| 박가은 | 파일 처리 · 연결 탐색 | LangChain 환경 구축, Agent 안정화, 파일 읽기/스캔, 벡터DB, 중복·연결 탐지 |
| 김보민 | 문서 분석 · AI 처리 | 요약, 키워드, 태그, 자동 분류, 정리 제안 (A-role) |
| 이해원 | 정리 실행 · 사용자 상호작용 | 이동·이름 변경·폴더 생성, 미리보기·승인, 메타데이터 수정, 배치 재정리 (C-role) |

## 설치

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

## 환경 변수

`.env` 파일을 만들고 OpenAI API 키를 설정합니다.

```bash
OPENAI_API_KEY=sk-...
```

## 실행

```bash
python main.py
```

```
PKM Agent 시작 (종료: 'quit' / 중단: Ctrl+C)

나: langchain 정리 노트 저장해줘
나: data/notes/공부_정리.md 요약해줘
나: 이 노트 자동 분류해서 저장해줘
나: 머신러닝 관련 노트 찾아줘
```

종료는 `quit` / `exit` / `종료`, 작업 중단은 `Ctrl+C` 입니다.

## 툴 목록

| 분류 | 툴 |
| --- | --- |
| 노트 | `save_note`, `read_note`, `list_notes` |
| 파일 | `list_directory`, `count_files`, `scan_files`, `find_file`, `search_in_files`, `read_file_full`, `file_to_note`, `create_file`, `move_file`, `copy_file`, `duplicate_note`, `delete_file`, `delete_folder` |
| 문서 읽기 | `read_file_structured`, `read_image` |
| AI 분석 (A) | `a_generate_summary`, `a_extract_keywords`, `a_generate_tags`, `a_classify_document`, `a_suggest_organization` |
| 검색/연결 | `search_knowledge`, `add_to_knowledge`, `find_connections` |
| 중복 탐지 | `check_note_duplicates`, `detect_duplicates` |
| 인덱싱 | `index_all_notes`, `index_note`, `index_file`, `index_folder` |
| 자동 정리 (C) | `preview_organized_path`, `organize_and_save_note`, `preview_file_organized_path`, `organize_file_and_save_note`, `move_note_to_category`, `rename_note_file`, `rename_organized_folder`, `preview_reorganized_note`, `reorganize_existing_note`, `update_note_metadata`, `batch_reorganize_notes` |

## 테스트

```bash
pytest
```

## 기술 스택

LangChain · LangGraph · Chroma · OpenAI · EasyOCR · pypdf / pdfplumber / PyMuPDF · python-docx · python-pptx
