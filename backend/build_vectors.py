"""从 medical_new_2.json 生成向量数据写入 Milvus。"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - 环境未安装 tqdm 时退化为普通迭代
    def tqdm(iterable, **_kwargs):
        return iterable

from .milvus_writer import MilvusWriter



PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_PATH = PROJECT_ROOT / "backend" / "data" / "medical_new_2.json"


def _read_records(source_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with source_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip().rstrip(",")
            if not line or line in {"[", "]"}:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                record = ast.literal_eval(line)
            if isinstance(record, dict):
                records.append(record)
    return records


def _stringify_list(values: Any) -> str:
    if not values:
        return ""
    if isinstance(values, str):
        return values.strip()
    if not isinstance(values, list):
        return str(values)
    cleaned: list[str] = []
    for item in values:
        if isinstance(item, list):
            item = item[0] if item else ""
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return "、".join(cleaned)


def _truncate_text(text: str, max_length: int = 1500) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "…"


def _build_document(record: dict[str, Any], index: int, source_name: str, source_path: Path) -> dict[str, Any]:
    name = str(record.get("name", "")).strip() or f"unknown-{index}"

    sections = [
        f"疾病: {name}",
        f"简介: {str(record.get('desc', '')).strip()[:300]}",
    ]

    symptom = _stringify_list(record.get("symptom", []))
    cure_department = _stringify_list(record.get("cure_department", []))
    common_drug = _stringify_list(record.get("common_drug", []))

    if symptom:
        sections.append(f"症状: {symptom[:150]}")
    if cure_department:
        sections.append(f"科室: {cure_department}")
    if common_drug:
        sections.append(f"药物: {common_drug[:150]}")

    text = _truncate_text("\n".join(section for section in sections if section.strip()))
    chunk_id = f"{source_name}::{index}::{name}"

    return {
        "text": text,
        "filename": source_name,
        "file_type": "medical_json",
        "file_path": str(source_path),
        "page_number": 0,
        "chunk_idx": index,
        "chunk_id": chunk_id,
        "parent_chunk_id": "",
        "root_chunk_id": chunk_id,
        "chunk_level": 3,
    }


def build_documents(source_path: Path) -> list[dict[str, Any]]:
    records = _read_records(source_path)
    source_name = source_path.name
    documents = []
    for index, record in enumerate(tqdm(records, desc="解析疾病数据", unit="条"), start=1):
        documents.append(_build_document(record, index, source_name, source_path))
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="从 medical_new_2.json 生成向量数据并写入 Milvus")
    parser.add_argument(
        "--source",
        type=str,
        default=str(DEFAULT_SOURCE_PATH),
        help="medical_new_2.json 的路径",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Milvus 分批写入大小",
    )
    args = parser.parse_args()

    source_path = Path(args.source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"找不到数据文件: {source_path}")

    documents = build_documents(source_path)
    if not documents:
        print("未解析到可写入的疾病数据")
        return

    writer = MilvusWriter()
    writer.write_documents(documents, batch_size=max(1, args.batch_size))
    print(f"已写入 {len(documents)} 条疾病向量数据到 Milvus")


if __name__ == "__main__":
    main()