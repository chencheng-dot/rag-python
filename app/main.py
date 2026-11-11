"""FastAPI 入口：入库、问答、数据库自检与健康检查。"""
from __future__ import annotations

import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.chunker import MODES, SIZE_PRESETS
from app.config import get_settings
from app.pipeline import RagPipeline
from app.store import ensure_ready

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG（Python + LangChain + pgvector）", version="1.0.0")

_pipeline: RagPipeline | None = None


def pipeline() -> RagPipeline:
    """惰性构建，避免缺配置时应用直接起不来。"""
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline()
    return _pipeline


class IngestRequest(BaseModel):
    text: str = Field(..., description="原始正文，允许含 HTML/Markdown，入库前会自动清洗")
    source: str = Field("manual", description="来源标识，用于溯源与引用展示")
    chunkMode: str | None = Field(None, description="切片方式：char / paragraph / section，默认取 CHUNK_MODE")
    chunkSize: int | None = Field(None, description="按字数切片的块大小：100 / 500 / 1000")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    similarityMode: str | None = Field(None, description="相似度模式：precise(0.7) / broad(0.6)")


def _validate_chunk(chunk_mode: str | None, chunk_size: int | None) -> None:
    if chunk_mode is not None and chunk_mode not in MODES:
        raise HTTPException(status_code=400, detail=f"chunkMode 只能是 {list(MODES)}")
    if chunk_size is not None and chunk_size not in SIZE_PRESETS:
        raise HTTPException(status_code=400, detail=f"chunkSize 只能是 {list(SIZE_PRESETS)}")


@app.get("/health")
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok" if (s.pg_ready and s.embedding_ready) else "degraded",
        "pgReady": s.pg_ready,
        "embeddingReady": s.embedding_ready,
        "rerankReady": s.rerank_ready,
        "chatModel": s.chat_model,
        "embeddingModel": s.embedding_model,
        "collection": s.collection_name,
        "chunkMode": s.chunk_mode,
        "chunkSize": s.chunk_size,
        "similarityMode": s.similarity_mode,
    }


@app.get("/db/status")
def db_status() -> dict:
    try:
        return ensure_ready()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"数据库不可用：{exc}") from exc


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text 不能为空")
    _validate_chunk(req.chunkMode, req.chunkSize)
    try:
        return pipeline().ingest(req.text, req.source, req.chunkMode, req.chunkSize)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"入库失败：{exc}") from exc


@app.post("/ingest/file")
async def ingest_file(
    file: UploadFile = File(...),
    source: str | None = None,
    chunkMode: str | None = None,
    chunkSize: int | None = None,
) -> dict:
    name = file.filename or "upload"
    if not name.lower().endswith((".txt", ".md", ".markdown")):
        raise HTTPException(status_code=415, detail="仅支持 .txt / .md 文本文件")
    _validate_chunk(chunkMode, chunkSize)
    raw = (await file.read()).decode("utf-8", errors="ignore")
    try:
        return pipeline().ingest(raw, source or name, chunkMode, chunkSize)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"入库失败：{exc}") from exc


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    try:
        return pipeline().ask(req.question, req.similarityMode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"问答失败：{exc}") from exc
