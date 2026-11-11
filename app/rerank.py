"""重排：调用 Dashscope text-rerank 做真正的二次排序。

Spring AI 2.x 没有 rerank 模块，Python 侧的 LangChain 也没有把它做成通用组件，
所以这里按 LangChain 的 BaseDocumentCompressor 接口实现一个，保持调用方式标准。
"""
from __future__ import annotations

import logging
from typing import Sequence

import httpx
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor

logger = logging.getLogger(__name__)

_PATH = "/api/v1/services/rerank/text-rerank/text-rerank"


class DashScopeRerank(BaseDocumentCompressor):
    """text-rerank 二次排序；调用失败时原样返回候选，不阻断问答。"""

    model: str = "qwen3.7-text-rerank"
    base_url: str = "https://dashscope.aliyuncs.com"
    api_key: str = ""
    top_n: int = 5
    timeout: float = 30.0
    # 最近一次调用是否真正成功；失败会降级为向量距离顺序
    last_error: str | None = None

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        if not documents:
            return []
        payload = {
            "model": self.model,
            "input": {"query": query, "documents": [d.page_content for d in documents]},
            "parameters": {
                "return_documents": False,
                "top_n": min(self.top_n, len(documents)),
            },
        }
        try:
            resp = httpx.post(
                self.base_url.rstrip("/") + _PATH,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            results = resp.json()["output"]["results"]
        except Exception as exc:  # noqa: BLE001 —— 降级优先，不因重排失败让问答整体 500
            self.last_error = str(exc)
            logger.warning("rerank 调用失败，降级为向量距离顺序：%s", exc)
            return list(documents)[: self.top_n]

        self.last_error = None

        ranked: list[Document] = []
        for item in sorted(results, key=lambda x: x.get("relevance_score", 0.0), reverse=True):
            index = item["index"]
            if not 0 <= index < len(documents):
                continue
            doc = documents[index]
            doc.metadata = {**doc.metadata, "rerank_score": item.get("relevance_score")}
            ranked.append(doc)
        return ranked or list(documents)[: self.top_n]
