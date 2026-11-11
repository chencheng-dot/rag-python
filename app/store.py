"""pgvector 向量库：连接、建表、健康检查。

用 LangChain 官方 `PGVector`，距离策略显式设为 EUCLIDEAN（L2）。
表由 PGVector 在首次写入时自动创建：
  langchain_pg_collection（集合元数据）/ langchain_pg_embedding（切片 + 向量）
"""
from __future__ import annotations

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy
from sqlalchemy import create_engine, text

from app.config import Settings, get_settings

_EMBEDDING_TABLE = "langchain_pg_embedding"


def build_embeddings(settings: Settings | None = None) -> OpenAIEmbeddings:
    settings = settings or get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        # Dashscope 等非 OpenAI 端点必须关闭 tiktoken 分词，否则会发 token 数组而不是原始字符串
        check_embedding_ctx_length=False,
        chunk_size=10,  # 单批条数，避免超出厂商批量上限
    )


def build_store(settings: Settings | None = None) -> PGVector:
    settings = settings or get_settings()
    if not settings.pg_ready:
        raise RuntimeError("PostgreSQL 配置不完整，请检查 .env 中的 PG_HOST / PG_DB")
    return PGVector(
        embeddings=build_embeddings(settings),
        connection=settings.pg_dsn,
        embedding_length=settings.embedding_dims,
        collection_name=settings.collection_name,
        # 欧氏距离：检索按 L2 升序，距离越小越相近
        distance_strategy=DistanceStrategy.EUCLIDEAN,
        use_jsonb=True,
        create_extension=True,
    )


def ensure_ready(settings: Settings | None = None) -> dict:
    """连通性 + pgvector 扩展 + 现有切片数自检。"""
    settings = settings or get_settings()
    engine = create_engine(settings.pg_dsn, pool_pre_ping=True, connect_args={"connect_timeout": 5})
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            version = conn.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            ).scalar()
            exists = conn.execute(
                text("SELECT to_regclass(:t)"), {"t": _EMBEDDING_TABLE}
            ).scalar()
            count = (
                conn.execute(text(f"SELECT count(*) FROM {_EMBEDDING_TABLE}")).scalar()
                if exists
                else 0
            )
        return {
            "connected": True,
            "pgvectorVersion": version,
            "tableReady": bool(exists),
            "chunks": int(count or 0),
        }
    finally:
        engine.dispose()
