"""集中读取环境变量。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import quote_plus

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else int(raw)


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# 相似度两档：精确 / 广泛；数值是重排分的门限（DashScope 为 0–1）
SIMILARITY_PRESETS = {"precise": 0.7, "broad": 0.6}
DEFAULT_SIMILARITY_MODE = "precise"


def similarity_threshold(mode: str) -> float:
    if mode not in SIMILARITY_PRESETS:
        raise ValueError(f"不支持的相似度模式：{mode}，可选 {list(SIMILARITY_PRESETS)}")
    return SIMILARITY_PRESETS[mode]


@dataclass(frozen=True)
class Settings:
    # PostgreSQL + pgvector
    pg_host: str
    pg_port: int
    pg_db: str
    pg_user: str
    pg_password: str
    collection_name: str

    # 文本模型 / 向量模型（OpenAI 兼容协议）
    openai_base_url: str
    openai_api_key: str
    chat_model: str
    embedding_model: str
    embedding_dims: int

    # 重排
    rerank_enabled: bool
    rerank_base_url: str
    rerank_api_key: str
    rerank_model: str

    # 查询改写（仅模糊问题触发，结果只用于 embedding）
    rewrite_enabled: bool

    # 生成约束：相似度模式（precise 0.7 / broad 0.6）与单块截断
    similarity_mode: str
    chunk_max_chars: int

    # 切片与检索
    chunk_mode: str
    chunk_size: int
    chunk_overlap: int
    retrieve_top_n: int
    rerank_top_k: int
    max_context_chars: int

    @property
    def pg_dsn(self) -> str:
        """SQLAlchemy + psycopg3 连接串。"""
        user = quote_plus(self.pg_user)
        pwd = quote_plus(self.pg_password)
        return f"postgresql+psycopg://{user}:{pwd}@{self.pg_host}:{self.pg_port}/{self.pg_db}?connect_timeout=5"

    @property
    def pg_ready(self) -> bool:
        return bool(self.pg_host.strip() and self.pg_db.strip())

    @property
    def embedding_ready(self) -> bool:
        return bool(self.openai_api_key.strip() and self.embedding_model.strip())

    @property
    def rerank_ready(self) -> bool:
        return self.rerank_enabled and bool(self.rerank_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings(
        pg_host=os.getenv("PG_HOST", "127.0.0.1"),
        pg_port=_int("PG_PORT", 5432),
        pg_db=os.getenv("PG_DB", "rag_db"),
        pg_user=os.getenv("PG_USER", "rag"),
        pg_password=os.getenv("PG_PASSWORD", "rag_pwd"),
        collection_name=os.getenv("PG_COLLECTION", "kb_chunks"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        chat_model=os.getenv("CHAT_MODEL", "qwen-plus"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding-flash"),
        embedding_dims=_int("EMBEDDING_DIMS", 1024),
        rerank_enabled=_bool("RERANK_ENABLED", True),
        rerank_base_url=os.getenv("RERANK_BASE_URL", "https://dashscope.aliyuncs.com"),
        rerank_api_key=os.getenv("RERANK_API_KEY", "") or os.getenv("OPENAI_API_KEY", ""),
        rerank_model=os.getenv("RERANK_MODEL", "qwen3.7-text-rerank"),
        rewrite_enabled=_bool("REWRITE_ENABLED", True),
        similarity_mode=os.getenv("SIMILARITY_MODE", DEFAULT_SIMILARITY_MODE),
        chunk_max_chars=_int("CHUNK_MAX_CHARS", 800),
        chunk_mode=os.getenv("CHUNK_MODE", "char"),
        chunk_size=_int("CHUNK_SIZE", 500),
        chunk_overlap=_int("CHUNK_OVERLAP", 80),
        retrieve_top_n=_int("RETRIEVE_TOP_N", 20),
        rerank_top_k=_int("RERANK_TOP_K", 5),
        max_context_chars=_int("MAX_CONTEXT_CHARS", 6000),
    )
