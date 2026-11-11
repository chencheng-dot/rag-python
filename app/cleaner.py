"""数据清洗：把抓来的原始文本处理成适合切片的干净正文。"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

_SCRIPT_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HTML_ENTITY = re.compile(r"&(?:nbsp|amp|lt|gt|quot|#\d+);", re.IGNORECASE)
_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u2028\u2029\ufeff]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE = re.compile(r"[ \t\u00a0\u3000]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
# 纯符号/纯数字页码类噪声行
_PUNCT_ONLY = re.compile(r"^[\W_]+$", re.UNICODE)
_PAGE_NO = re.compile(r"^(?:[-—\s]*)?(?:第\s*)?\d{1,4}(?:\s*页)?(?:[-—\s]*)?$")

_ENTITIES = {"&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"'}


@dataclass
class CleanStats:
    raw_chars: int
    clean_chars: int
    dropped_lines: int
    deduped_lines: int

    def to_dict(self) -> dict:
        return asdict(self)


def _strip_markup(text: str) -> str:
    text = _SCRIPT_STYLE.sub(" ", text)
    text = _MD_IMAGE.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _TAG.sub(" ", text)
    for entity, char in _ENTITIES.items():
        text = text.replace(entity, char)
    text = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), text)
    return text


def clean_text(raw: str, *, min_line_len: int = 1, dedupe: bool = True) -> tuple[str, CleanStats]:
    """清洗正文。

    规则：去脚本/标签/Markdown 图片、链接只留文字、去零宽与控制字符、
    逐行去首尾空白、丢弃纯符号行与页码行、去完全重复的行、
    连续空行折叠为一个（保留段落边界）、折叠多余空格。
    """
    if raw is None:
        raw = ""
    raw_len = len(raw)

    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _strip_markup(text)
    text = _ZERO_WIDTH.sub("", text)
    text = _CONTROL.sub("", text)

    kept: list[str] = []
    seen: set[str] = set()
    dropped = 0
    deduped = 0

    for line in text.split("\n"):
        line = _MULTI_SPACE.sub(" ", line).strip()
        if not line:
            # 空行只保留一个，作为段落边界；否则「按段落切」会退化成按字数切
            if kept and kept[-1] != "":
                kept.append("")
            continue
        if _PUNCT_ONLY.match(line) or _PAGE_NO.match(line):
            dropped += 1
            continue
        if len(line) < min_line_len:
            dropped += 1
            continue
        if dedupe:
            if line in seen:
                deduped += 1
                continue
            seen.add(line)
        kept.append(line)

    cleaned = "\n".join(kept)
    cleaned = _MULTI_NEWLINE.sub("\n\n", cleaned).strip()

    return cleaned, CleanStats(
        raw_chars=raw_len,
        clean_chars=len(cleaned),
        dropped_lines=dropped,
        deduped_lines=deduped,
    )
