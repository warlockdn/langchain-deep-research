import json
from importlib import util
from pathlib import Path

from langchain_core.documents import Document

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "pdf_chunk_count.py"
)


def load_script_module():
    spec = util.spec_from_file_location("pdf_chunk_count", SCRIPT_PATH)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_loader_kwargs_uses_pypdf_page_mode():
    module = load_script_module()
    args = module.parse_args([])

    kwargs = module.build_loader_kwargs(args)

    assert kwargs == {
        "password": None,
        "headers": None,
        "extract_images": False,
        "mode": "page",
        "images_parser": None,
        "images_inner_format": "text",
        "pages_delimiter": "\n\f",
        "extraction_mode": "plain",
        "extraction_kwargs": None,
    }


def test_parse_args_leaves_splitting_disabled_by_default():
    module = load_script_module()

    args = module.parse_args([])

    assert args.split is False
    assert args.pgvector is False
    assert args.chunk_size == 1000
    assert args.chunk_overlap == 200


def test_resolve_output_path_defaults_under_input_dir():
    module = load_script_module()
    args = module.parse_args(["--input-dir", "/tmp/example-docs"])

    output_path = module.resolve_output_path(args)

    assert output_path == Path("/tmp/example-docs/pdf_chunks.jsonl")


def test_main_prints_per_file_and_total_chunk_counts(monkeypatch, capsys, tmp_path):
    module = load_script_module()
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    pdf_a.write_bytes(b"%PDF-1.4")
    pdf_b.write_bytes(b"%PDF-1.4")

    captured: dict[str, object] = {"file_paths": []}

    class FakeLoader:
        def __init__(self, file_path, **kwargs):
            captured["file_paths"].append(file_path)
            captured["kwargs"] = kwargs

        def lazy_load(self):
            if Path(captured["file_paths"][-1]).name == "a.pdf":
                return iter(
                    [
                        Document(page_content="a1", metadata={"filename": "a.pdf"}),
                        Document(page_content="a2", metadata={"filename": "a.pdf"}),
                    ]
                )
            return iter([Document(page_content="b1", metadata={"filename": "b.pdf"})])

    monkeypatch.setattr(module, "PyPDFLoader", FakeLoader)

    exit_code = module.main(["--input-dir", str(tmp_path)])

    output = capsys.readouterr()
    output_file = tmp_path / "pdf_chunks.jsonl"
    lines = output_file.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]

    assert exit_code == 0
    assert captured["file_paths"] == [str(pdf_a), str(pdf_b)]
    assert "a.pdf: 2 chunks" in output.out
    assert "b.pdf: 1 chunks" in output.out
    assert "TOTAL: 2 files, 3 chunks" in output.out
    assert "Saved chunks to" in output.out
    assert len(records) == 3
    assert records[0]["source_file"] == "a.pdf"
    assert records[0]["chunk_index"] == 1
    assert records[0]["page_content"] == "a1"
    assert records[0]["metadata"] == {"source": "a.pdf"}


def test_main_falls_back_to_source_basename(monkeypatch, capsys, tmp_path):
    module = load_script_module()
    pdf_a = tmp_path / "nested-name.pdf"
    pdf_a.write_bytes(b"%PDF-1.4")

    class FakeLoader:
        def __init__(self, file_path, **kwargs):
            self.file_path = file_path

        def lazy_load(self):
            return iter(
                [
                    Document(
                        page_content="a1",
                        metadata={"source": str(self.file_path)},
                    )
                ]
            )

    monkeypatch.setattr(module, "PyPDFLoader", FakeLoader)

    exit_code = module.main(["--input-dir", str(tmp_path)])

    output = capsys.readouterr()

    assert exit_code == 0
    assert "nested-name.pdf: 1 chunks" in output.out


def test_main_returns_non_zero_when_loader_skips_a_discovered_pdf(
    monkeypatch, capsys, tmp_path
):
    module = load_script_module()
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    pdf_a.write_bytes(b"%PDF-1.4")
    pdf_b.write_bytes(b"%PDF-1.4")

    class FakeLoader:
        def __init__(self, file_path, **kwargs):
            self.file_path = file_path

        def lazy_load(self):
            if Path(self.file_path).name == "a.pdf":
                return iter(
                    [
                        Document(page_content="a1", metadata={"filename": "a.pdf"}),
                    ]
                )
            return iter([])

    monkeypatch.setattr(module, "PyPDFLoader", FakeLoader)

    exit_code = module.main(["--input-dir", str(tmp_path)])

    output = capsys.readouterr()

    assert exit_code == 1
    assert "a.pdf: 1 chunks" in output.out
    assert "b.pdf: 0 chunks" in output.out
    assert "Missing chunk output for: b.pdf" in output.err


def test_main_returns_non_zero_when_no_pdfs_are_found(capsys, tmp_path):
    module = load_script_module()

    exit_code = module.main(["--input-dir", str(tmp_path)])

    output = capsys.readouterr()

    assert exit_code == 1
    assert "No PDF files matched" in output.err


def test_main_splits_documents_only_when_split_flag_is_set(
    monkeypatch, capsys, tmp_path
):
    module = load_script_module()
    pdf_a = tmp_path / "a.pdf"
    pdf_a.write_bytes(b"%PDF-1.4")

    class FakeLoader:
        def __init__(self, file_path, **kwargs):
            self.file_path = file_path

        def lazy_load(self):
            return iter(
                [
                    Document(
                        page_content="abcdefghij",
                        metadata={"filename": Path(self.file_path).name, "page": 0},
                    )
                ]
            )

    monkeypatch.setattr(module, "PyPDFLoader", FakeLoader)

    exit_code = module.main(
        [
            "--input-dir",
            str(tmp_path),
            "--split",
            "--chunk-size",
            "4",
            "--chunk-overlap",
            "0",
        ]
    )

    output = capsys.readouterr()
    output_file = tmp_path / "pdf_chunks.jsonl"
    records = [
        json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()
    ]

    assert exit_code == 0
    assert "a.pdf: 3 chunks" in output.out
    assert "TOTAL: 1 files, 3 chunks" in output.out
    assert [record["page_content"] for record in records] == ["abcd", "efgh", "ij"]
    assert [record["chunk_index"] for record in records] == [1, 2, 3]
    assert all(record["metadata"] == {"source": "a.pdf", "page": 0} for record in records)


def test_main_whitelists_only_expected_pdf_metadata(monkeypatch, capsys, tmp_path):
    module = load_script_module()
    pdf_a = tmp_path / "a.pdf"
    pdf_a.write_bytes(b"%PDF-1.4")

    class FakeLoader:
        def __init__(self, file_path, **kwargs):
            self.file_path = file_path

        def lazy_load(self):
            return iter(
                [
                    Document(
                        page_content="a1",
                        metadata={
                            "producer": "Adobe PDF Library 17.0",
                            "creator": "Adobe InDesign 19.0 (Macintosh)",
                            "creationdate": "2023-12-18T08:12:53-06:00",
                            "source": str(self.file_path),
                            "total_pages": 58,
                            "page": 6,
                            "page_label": "7",
                            "filename": Path(self.file_path).name,
                        },
                    )
                ]
            )

    monkeypatch.setattr(module, "PyPDFLoader", FakeLoader)

    exit_code = module.main(["--input-dir", str(tmp_path)])

    output = capsys.readouterr()
    output_file = tmp_path / "pdf_chunks.jsonl"
    records = [
        json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()
    ]

    assert exit_code == 0
    assert "a.pdf: 1 chunks" in output.out
    assert records[0]["metadata"] == {
        "source": str(pdf_a),
        "total_pages": 58,
        "page": 6,
        "page_label": "7",
    }


def test_main_persists_final_documents_to_pgvector_when_flag_is_set(
    monkeypatch, capsys, tmp_path
):
    module = load_script_module()
    pdf_a = tmp_path / "a.pdf"
    pdf_a.write_bytes(b"%PDF-1.4")
    captured: dict[str, object] = {}

    class FakeLoader:
        def __init__(self, file_path, **kwargs):
            self.file_path = file_path

        def lazy_load(self):
            return iter(
                [
                    Document(
                        page_content="abcdefghij",
                        metadata={"filename": Path(self.file_path).name, "page": 0},
                    )
                ]
            )

    def fake_store_documents_in_pgvector(documents):
        captured["documents"] = documents
        return {"collection_name": "research_docs", "count": len(documents)}

    monkeypatch.setattr(module, "PyPDFLoader", FakeLoader)
    monkeypatch.setattr(module, "store_documents_in_pgvector", fake_store_documents_in_pgvector)

    exit_code = module.main(
        [
            "--input-dir",
            str(tmp_path),
            "--split",
            "--chunk-size",
            "4",
            "--chunk-overlap",
            "0",
            "--pgvector",
        ]
    )

    output = capsys.readouterr()

    assert exit_code == 0
    assert [document.page_content for document in captured["documents"]] == [
        "abcd",
        "efgh",
        "ij",
    ]
    assert all(document.metadata == {"source": "a.pdf", "page": 0} for document in captured["documents"])
    assert "Saved 3 chunks to PGVector collection research_docs" in output.out


def test_main_returns_non_zero_when_pgvector_flag_is_set_without_required_env(
    monkeypatch, capsys, tmp_path
):
    module = load_script_module()
    pdf_a = tmp_path / "a.pdf"
    pdf_a.write_bytes(b"%PDF-1.4")

    class FakeLoader:
        def __init__(self, file_path, **kwargs):
            self.file_path = file_path

        def lazy_load(self):
            return iter(
                [
                    Document(
                        page_content="a1",
                        metadata={"filename": Path(self.file_path).name},
                    )
                ]
            )

    def fake_store_documents_in_pgvector(documents):
        raise RuntimeError("Document ingestion is unavailable: CONNECTION_STRING and COLLECTION_NAME must both be set.")

    monkeypatch.setattr(module, "PyPDFLoader", FakeLoader)
    monkeypatch.setattr(module, "store_documents_in_pgvector", fake_store_documents_in_pgvector)

    exit_code = module.main(["--input-dir", str(tmp_path), "--pgvector"])

    output = capsys.readouterr()

    assert exit_code == 1
    assert "Document ingestion is unavailable: CONNECTION_STRING and COLLECTION_NAME must both be set." in output.err
