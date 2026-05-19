"""Index project documentation and source files into the RAG vector store."""

from __future__ import annotations

import argparse
from pathlib import Path

from .document_loader import DocumentLoader
from .embedding import embedding_service
from .milvus_client import MilvusManager
from .milvus_writer import MilvusWriter
from .parent_chunk_store import ParentChunkStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INCLUDE_DIRS = ("backend", "frontend")
DEFAULT_INCLUDE_FILES = ("README.md", "docker-compose.yml", "pyproject.toml")
DEFAULT_SUFFIXES = {".md", ".py", ".js", ".css", ".html", ".yml", ".yaml", ".toml"}
DEFAULT_EXCLUDED_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def _iter_project_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for filename in DEFAULT_INCLUDE_FILES:
        path = root / filename
        if path.is_file():
            files.append(path)

    for dirname in DEFAULT_INCLUDE_DIRS:
        base = root / dirname
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if any(part in DEFAULT_EXCLUDED_NAMES for part in path.parts):
                continue
            if path.suffix.lower() in DEFAULT_SUFFIXES:
                files.append(path)

    return sorted(set(files), key=lambda item: item.relative_to(root).as_posix())


def _remove_existing_file(
    filename: str,
    milvus_manager: MilvusManager,
    parent_chunk_store: ParentChunkStore,
) -> None:
    rows = milvus_manager.query_all(
        filter_expr=f'filename == "{filename}"',
        output_fields=["text"],
    )
    texts = [row.get("text") or "" for row in rows]
    embedding_service.increment_remove_documents(texts)
    try:
        milvus_manager.delete(f'filename == "{filename}"')
    except Exception:
        pass
    parent_chunk_store.delete_by_filename(filename)


def build_project_documents(root: Path) -> list[dict]:
    loader = DocumentLoader()
    documents: list[dict] = []
    for path in _iter_project_files(root):
        rel_path = path.relative_to(root).as_posix()
        docs = loader.load_text_document(str(path), rel_path)
        for doc in docs:
            doc["file_path"] = str(path)
        documents.extend(docs)
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Index project files into Milvus for RAG queries.")
    parser.add_argument("--root", default=str(PROJECT_ROOT), help="Project root to index.")
    parser.add_argument("--batch-size", type=int, default=32, help="Vector write batch size.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Project root not found: {root}")

    milvus_manager = MilvusManager()
    parent_chunk_store = ParentChunkStore()
    writer = MilvusWriter(embedding_service=embedding_service, milvus_manager=milvus_manager)

    milvus_manager.init_collection()
    source_files = _iter_project_files(root)
    for path in source_files:
        _remove_existing_file(path.relative_to(root).as_posix(), milvus_manager, parent_chunk_store)

    documents = build_project_documents(root)
    parent_docs = [doc for doc in documents if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
    leaf_docs = [doc for doc in documents if int(doc.get("chunk_level", 0) or 0) == 3]
    if not leaf_docs:
        print("No project chunks generated.")
        return

    parent_chunk_store.upsert_documents(parent_docs)
    writer.write_documents(leaf_docs, batch_size=max(1, args.batch_size))

    print(
        f"Indexed {len(source_files)} files: "
        f"{len(parent_docs)} parent chunks, {len(leaf_docs)} leaf chunks."
    )


if __name__ == "__main__":
    main()
