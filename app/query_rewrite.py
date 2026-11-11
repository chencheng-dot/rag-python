"""问题向量化前的查询改写：把口语化、省略口径的提问补成一条可直接检索的明确查询。

改写结果只用于 embedding；重排仍按用户原问题打分（见 pipeline.ask），
避免「改写漂移」把排序也带偏。任何一步异常都回退原问题，不阻断问答。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import Settings

logger = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = """你是检索查询改写器（query rewriter）。用户提问常省略口径，你要把它补全成一条可直接用于向量检索的明确查询。

【规则】
1. 保留原意：不换品类、不换时间范围、不加用户没提的筛选（品牌、地区、渠道）。
2. 模糊词只按【可用口径】展开；口径里没有的阈值、周期、数字一律不编，写进 unconfirmed。
3. 改写结果要自解释：没看过原话的人读这一句，知道查什么、按什么条件。
4. 60 字以内；实体（商品名、指标、时间）原样保留，不用代词。
5. 同时输出 keywords（关键词召回）与 filters（结构化条件，没有就空数组）。
6. 只输出 JSON，不要解释。

【输出】
{"rewritten_query":"...","keywords":["..."],"filters":[{"field":"...","op":"lte|gte|eq|between","value":...}],"unconfirmed":["..."],"reason":"..."}"""

MAX_QUERY_CHARS = 60
_VAGUE_MAX_CHARS = 12
_VAGUE_WORDS = (
    "这个", "那个", "它", "这些", "那些", "最近", "刚才", "上面", "之前",
    "那种", "这种", "怎么样", "有没有", "是不是",
)
_DIGITS = re.compile(r"\d+")


@dataclass
class RewriteResult:
    original: str
    rewritten_query: str
    keywords: list[str] = field(default_factory=list)
    filters: list[dict] = field(default_factory=list)
    unconfirmed: list[str] = field(default_factory=list)
    reason: str = ""
    used: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "rewrittenQuery": self.rewritten_query,
            "keywords": self.keywords,
            "filters": self.filters,
            "unconfirmed": self.unconfirmed,
            "reason": self.reason,
            "used": self.used,
            "error": self.error,
        }


def is_vague(question: str) -> bool:
    """是否需要改写：短问句或含指代/模糊词。判定偏保守，宁可多改一次。"""
    q = (question or "").strip()
    if not q:
        return False
    if len(q) <= _VAGUE_MAX_CHARS:
        return True
    return any(word in q for word in _VAGUE_WORDS)


def entities_preserved(original: str, rewritten: str) -> bool:
    """原问题里的数字必须原样出现在改写句里，防止「≤10 件」被改成「20 件」。"""
    return all(number in rewritten for number in _DIGITS.findall(original))


def _strip_fence(raw: str) -> str:
    text = (raw or "").strip()
    if not text.startswith("```"):
        return text
    text = text.split("\n", 1)[1] if "\n" in text else ""
    text = text.strip()
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _str_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def parse_rewrite(raw: str, question: str) -> RewriteResult:
    """解析模型输出；任何不合规（非 JSON / 缺字段 / 超长 / 丢实体）都回退原问题。"""
    fallback = RewriteResult(original=question, rewritten_query=question)
    try:
        data = json.loads(_strip_fence(raw))
    except Exception:  # noqa: BLE001 —— 模型偶尔会输出解释性文字，按失败处理
        fallback.error = "改写结果不是合法 JSON"
        return fallback
    if not isinstance(data, dict):
        fallback.error = "改写结果不是 JSON 对象"
        return fallback

    rewritten = str(data.get("rewritten_query") or "").strip()
    if not rewritten:
        fallback.error = "改写结果缺少 rewritten_query"
        return fallback
    if len(rewritten) > MAX_QUERY_CHARS:
        fallback.error = f"改写结果超过 {MAX_QUERY_CHARS} 字"
        return fallback
    if not entities_preserved(question, rewritten):
        fallback.error = "改写丢失了原问题中的数字实体"
        return fallback

    raw_filters = data.get("filters")
    return RewriteResult(
        original=question,
        rewritten_query=rewritten,
        keywords=_str_list(data.get("keywords")),
        filters=[item for item in (raw_filters or []) if isinstance(item, dict)],
        unconfirmed=_str_list(data.get("unconfirmed")),
        reason=str(data.get("reason") or "").strip(),
        used=True,
    )


class QueryRewriter:
    """只在模糊问题触发改写；keywords 不拼进 embedding 文本，留给关键词召回用。"""

    def __init__(self, settings: Settings):
        self.chat = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0.0,
            timeout=20,
        )

    def rewrite(self, question: str, available_filters: dict | None = None) -> RewriteResult:
        question = (question or "").strip()
        if not is_vague(question):
            return RewriteResult(original=question, rewritten_query=question)

        messages = [
            SystemMessage(content=REWRITE_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"【用户原话】{question}\n"
                    f"【可用口径】{json.dumps(available_filters or {}, ensure_ascii=False)}\n"
                    f"【当前时间】{datetime.now():%Y-%m-%d}"
                )
            ),
        ]
        try:
            raw = self.chat.invoke(messages).content
        except Exception as exc:  # noqa: BLE001 —— 改写失败只降级，不影响问答
            logger.warning("查询改写调用失败，回退原问题：%s", exc)
            fallback = RewriteResult(original=question, rewritten_query=question)
            fallback.error = f"调用失败：{exc}"
            return fallback

        result = parse_rewrite(str(raw), question)
        if result.error:
            logger.warning("查询改写被丢弃（%s），回退原问题", result.error)
        return result
