import os
import shutil
import tempfile
import unittest

from tools.batch_organizer_tool import batch_reorganize_notes
from tools.metadata_tool import update_note_metadata
from tools.move_tool import move_note_to_category
from tools.rename_tool import rename_note_file
from tools.reorganize_tool import preview_reorganized_note, reorganize_existing_note


def _write_note(path: str, title: str, body: str, metadata: dict[str, str] | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    metadata = metadata or {}
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        for key, value in metadata.items():
            f.write(f"*{key}: {value}*\n")
        if metadata:
            f.write("\n")
        f.write(body)


class ManagementToolTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        os.environ["PKM_NOTES_DIR"] = self.temp_dir

    def tearDown(self):
        os.environ.pop("PKM_NOTES_DIR", None)
        shutil.rmtree(self.temp_dir)

    def test_move_note_to_category_moves_file_and_updates_metadata(self):
        note_path = os.path.join(self.temp_dir, "inbox", "2026-04", "회의_메모.md")
        _write_note(note_path, "회의 메모", "회의 안건과 action item", {"분류 폴더": "inbox"})

        result = move_note_to_category.invoke(
            {"current_path": "회의 메모", "new_category": "projects"}
        )

        moved_path = os.path.join(self.temp_dir, "projects", "2026-04", "회의_메모.md")
        self.assertIn("노트 이동 완료", result)
        self.assertTrue(os.path.isfile(moved_path))
        with open(moved_path, "r", encoding="utf-8") as f:
            self.assertIn("*분류 폴더: projects*", f.read())

    def test_rename_note_file_changes_filename_and_optionally_keeps_heading(self):
        note_path = os.path.join(self.temp_dir, "study", "2026-04", "old_name.md")
        _write_note(note_path, "Old Name", "body")

        result = rename_note_file.invoke(
            {"current_path": "study/2026-04/old_name.md", "new_name": "새 제목", "update_title": False}
        )

        renamed_path = os.path.join(self.temp_dir, "study", "2026-04", "새_제목.md")
        self.assertIn("노트 이름 변경 완료", result)
        self.assertTrue(os.path.isfile(renamed_path))
        with open(renamed_path, "r", encoding="utf-8") as f:
            self.assertIn("# Old Name", f.read())

    def test_preview_and_reorganize_existing_note_move_based_on_rules(self):
        note_path = os.path.join(self.temp_dir, "inbox", "2026-04", "강의_메모.md")
        _write_note(
            note_path,
            "강의 메모",
            "LangChain 강의와 학습 정리",
            {"문서 유형": "document", "분류 폴더": "inbox"},
        )

        preview = preview_reorganized_note.invoke({"current_path": "inbox/2026-04/강의_메모.md"})
        self.assertIn("분류: study", preview)

        result = reorganize_existing_note.invoke(
            {"current_path": "data/notes/inbox/2026-04/강의_메모.md"}
        )

        reorganized_path = os.path.join(self.temp_dir, "study", "2026-04", "강의_메모.md")
        self.assertIn("재분류 완료", result)
        self.assertTrue(os.path.isfile(reorganized_path))

    def test_update_note_metadata_updates_requested_fields(self):
        note_path = os.path.join(self.temp_dir, "projects", "2026-04", "api_정리.md")
        _write_note(note_path, "API 정리", "body", {"분류 폴더": "projects"})

        result = update_note_metadata.invoke(
            {
                "current_path": "projects/2026-04/api_정리.md",
                "source_name": "notion",
                "source_type": "document",
                "tags": "api, backend",
                "saved_at": "2026-04-15 15:00",
            }
        )

        self.assertIn("메타데이터 수정 완료", result)
        with open(note_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("*출처: notion*", content)
        self.assertIn("*문서 유형: document*", content)
        self.assertIn("*태그: api, backend*", content)
        self.assertIn("*저장 시각: 2026-04-15 15:00*", content)

    def test_rename_note_file_supports_note_title_reference(self):
        note_path = os.path.join(self.temp_dir, "study", "2026-04", "요약_노트.md")
        _write_note(note_path, "요약 노트", "body")

        result = rename_note_file.invoke(
            {"current_path": "요약 노트", "new_name": "정리 완료", "update_title": True}
        )

        renamed_path = os.path.join(self.temp_dir, "study", "2026-04", "정리_완료.md")
        self.assertIn("노트 이름 변경 완료", result)
        self.assertTrue(os.path.isfile(renamed_path))
        with open(renamed_path, "r", encoding="utf-8") as f:
            self.assertIn("# 정리 완료", f.read())

    def test_batch_reorganize_notes_supports_dry_run_limit_and_execute(self):
        _write_note(
            os.path.join(self.temp_dir, "inbox", "2026-04", "회의1.md"),
            "회의1",
            "회의 agenda 정리",
            {"분류 폴더": "inbox"},
        )
        _write_note(
            os.path.join(self.temp_dir, "inbox", "2026-04", "공부1.md"),
            "공부1",
            "학습 내용과 강의 정리",
            {"분류 폴더": "inbox"},
        )

        dry_run = batch_reorganize_notes.invoke(
            {"source_folder": "inbox/2026-04", "dry_run": True, "limit": 1}
        )
        self.assertIn("배치 재정리 미리보기", dry_run)
        self.assertIn("처리 제한", dry_run)

        result = batch_reorganize_notes.invoke(
            {"source_folder": "inbox/2026-04", "dry_run": False, "limit": 10}
        )
        self.assertIn("배치 재정리 완료", result)
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, "meetings", "2026-04", "회의1.md")))
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, "study", "2026-04", "공부1.md")))

    def test_tools_return_not_found_messages_for_missing_files(self):
        move_result = move_note_to_category.invoke(
            {"current_path": "missing/path.md", "new_category": "projects"}
        )
        rename_result = rename_note_file.invoke(
            {"current_path": "missing/path.md", "new_name": "renamed"}
        )
        reorganize_result = reorganize_existing_note.invoke(
            {"current_path": "missing/path.md"}
        )
        metadata_result = update_note_metadata.invoke(
            {"current_path": "missing/path.md", "tags": "x"}
        )

        self.assertIn("노트를 찾을 수 없습니다", move_result)
        self.assertIn("노트를 찾을 수 없습니다", rename_result)
        self.assertIn("노트를 찾을 수 없습니다", reorganize_result)
        self.assertIn("노트를 찾을 수 없습니다", metadata_result)


if __name__ == "__main__":
    unittest.main()
