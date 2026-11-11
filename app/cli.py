"""命令行入口：python -m app.cli <db-status|ingest|ask>"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.config import get_settings
from app.pipeline import RagPipeline
from app.store import ensure_ready


def main() -> int:
    parser = argparse.ArgumentParser(description="Python + pgvector RAG 命令行")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("db-status", help="检查数据库连通性、pgvector 扩展与切片数")
    p_ingest = sub.add_parser("ingest", help="清洗并入库一个文本文件")
    p_ingest.add_argument("path")
    p_ingest.add_argument("--mode", choices=["char", "paragraph", "section"], default=None,
                          help="切片方式，默认取 CHUNK_MODE")
    p_ingest.add_argument("--size", type=int, choices=[100, 500, 1000], default=None,
                          help="按字数切片时的块大小，默认取 CHUNK_SIZE")
    p_ask = sub.add_parser("ask", help="对知识库提问")
    p_ask.add_argument("question")
    p_ask.add_argument("--similarity", choices=["precise", "broad"], default=None,
                       help="相似度模式：precise 0.7 / broad 0.6，默认取 SIMILARITY_MODE")

    args = parser.parse_args()
    s = get_settings()

    if args.cmd == "db-status":
        print(json.dumps(ensure_ready(s), ensure_ascii=False, indent=2))
        return 0

    if not s.embedding_ready:
        print("缺少 OPENAI_API_KEY / EMBEDDING_MODEL，无法调用向量模型", file=sys.stderr)
        return 2

    rag = RagPipeline(s)

    if args.cmd == "ingest":
        path = Path(args.path)
        if not path.exists():
            print(f"文件不存在：{path}", file=sys.stderr)
            return 2
        raw = path.read_text(encoding="utf-8", errors="ignore")
        print(
            json.dumps(
                rag.ingest(raw, path.name, args.mode, args.size), ensure_ascii=False, indent=2
            )
        )
        return 0

    if args.cmd == "ask":
        print(json.dumps(rag.ask(args.question, args.similarity), ensure_ascii=False, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
