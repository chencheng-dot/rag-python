"""切片：三种方式——按字数（预设 100/500/1000）、按段落、按章节。

段落/章节切出来的超长片段会逐级回退到按字数切，保证单块不超过 chunk_size。
"""
from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 从粗到细的分隔符；中文标点优先，最后退化到单字符
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""]

# 三种切片方式与三个字数预设
MODES = ("char", "paragraph", "section")
SIZE_PRESETS = (100, 500, 1000)
DEFAULT_MODE = "char"

# Markdown 章节标题（# ~ ######）
_HEADING = re.compile(r"^#{1,6}\s+\S")


def _split_by_chars(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        keep_separator=True,
        length_function=len,
    )
    return [piece.strip() for piece in splitter.split_text(text) if piece.strip()]


def _split_by_paragraph(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    pieces: list[str] = []
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        pieces.extend(
            _split_by_chars(para, chunk_size, chunk_overlap) if len(para) > chunk_size else [para]
        )
    return pieces


def _split_by_section(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    sections: list[list[str]] = []
    current: list[str] = []
    for line in text.split("\n"):
        # 标题行开启新章节；文档开头就有标题时不会产生空章节
        if _HEADING.match(line) and current:
            sections.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append(current)

    pieces: list[str] = []
    for lines in sections:
        section = "\n".join(lines).strip()
        if not section:
            continue
        # 一节就是一块；只有整节超过 chunk_size 才退化为按段落/按字数切
        if len(section) <= chunk_size:
            pieces.append(section)
        else:
            pieces.extend(_split_by_paragraph(section, chunk_size, chunk_overlap))
    return pieces


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    mode: str = DEFAULT_MODE,
) -> list[str]:
    if not text or not text.strip():
        return []
    if mode not in MODES:
        raise ValueError(f"不支持的切片方式：{mode}，可选 {list(MODES)}")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    if mode == "paragraph":
        pieces = _split_by_paragraph(text, chunk_size, chunk_overlap)
    elif mode == "section":
        pieces = _split_by_section(text, chunk_size, chunk_overlap)
    else:
        pieces = _split_by_chars(text, chunk_size, chunk_overlap)
    return [piece for piece in pieces if piece.strip()]
