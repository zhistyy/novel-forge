"""
提示词管理 — 模板、版本、编排

功能:
  - 从 DB 读写提示词模板
  - 变量替换渲染
  - 提示词链（多个模板组合）
  - 版本管理
"""

from __future__ import annotations

import json, os
from pathlib import Path
from typing import Optional

from .db import get_conn
from .crud import (
    list_prompts, get_active_prompt, create_prompt, render_prompt,
)

# ── 默认提示词注册 ──

DEFAULT_PROMPTS = {
    "planner": {
        "role": "planner",
        "description": "章节规划师 — 为本章制定写作规划",
        "file": "system_planner.md",
    },
    "writer": {
        "role": "writer",
        "description": "正文写手 — 写出完整一章正文",
        "file": "system_writer.md",
    },
    "reviewer": {
        "role": "reviewer",
        "description": "质检员 — 检查章节质量和设定一致性",
        "file": "system_reviewer.md",
    },
    "refiner": {
        "role": "refiner",
        "description": "文风润色师 — 优化正文文学表现力",
        "file": "system_refiner.md",
    },
    "entry_manager": {
        "role": "entry_manager",
        "description": "设定管理员 — 识别条目变化并更新",
        "file": "system_entry_manager.md",
    },
    "consistency_checker": {
        "role": "consistency_checker",
        "description": "一致性审查官 — 检查条目间矛盾",
        "file": "system_consistency_checker.md",
    },
}


def ensure_default_prompts(force_reload: bool = False):
    """
    确保默认提示词已注册到数据库。

    参数：
        force_reload: 强制重新加载所有提示词（升级版本号）

    自动升级机制：
        若文件内容与当前活跃版本的 system_text 不同，自动 deactivate 旧版本
        并创建新版本（版本号 +1，is_active=1）。
    """
    prompts_dir = Path(__file__).resolve().parent.parent.parent / "novel_agent" / "prompts"
    for name, info in DEFAULT_PROMPTS.items():
        file_path = prompts_dir / info["file"]
        if not file_path.exists():
            continue
        text = file_path.read_text(encoding="utf-8")

        existing = get_active_prompt(name)
        if not existing:
            # 首次注册
            create_prompt(name, info["role"], text, description=info["description"])
            continue

        # 已存在 → 检查内容是否变化
        if force_reload or existing.system_text != text:
            # 内容变化 → 升级版本
            from .crud import deactivate_prompt
            deactivate_prompt(name)
            create_prompt(name, info["role"], text, description=info["description"])


# ── 提示词编排 ──

def build_writing_context(project_name: str, event_num: int, chapter_num: int,
                          prev_chapter_tail: str = "") -> dict:
    """构建写作上下文变量字典"""
    from .crud import get_project_by_name, get_event, get_chapters_for_event, list_entries

    proj = get_project_by_name(project_name)
    if not proj:
        return {}

    evt = get_event(proj.id, event_num)
    if not evt:
        return {}

    entries = list_entries(proj.id)
    entries_text = "\n\n".join(
        f"**{e.name}** ({e.category})\n{e.one_line}\n{e.content[:500]}"
        for e in entries
    ) if entries else "（无关联条目）"

    chapters = get_chapters_for_event(evt.id)
    prev_chapter = next((c for c in chapters if c.num == chapter_num - 1), None)
    prev_tail = ""
    if prev_chapter:
        prev_tail = (prev_chapter.refined_content or prev_chapter.content or "")[-500:]

    return {
        "project_name": project_name,
        "genre": proj.genre,
        "tone": proj.tone or "（未设置）",
        "plot": (proj.plot or "")[:3000],
        "words_per_chapter": str(proj.words_per_chapter),
        "event_num": str(event_num),
        "chapter_num": str(chapter_num),
        "event_plan": evt.plan or "（无事件纲）",
        "entries": entries_text,
        "prev_chapter_tail": prev_tail or "（本章是第一章）",
        "style_features": "",
    }


def render_planner_prompt(project_name: str, event_num: int, chapter_num: int) -> str:
    """渲染规划师提示词"""
    ctx = build_writing_context(project_name, event_num, chapter_num)
    return render_prompt("planner", **ctx)


def render_writer_prompt(project_name: str, event_num: int, chapter_num: int,
                         chapter_plan: str) -> str:
    """渲染写作提示词"""
    ctx = build_writing_context(project_name, event_num, chapter_num)
    ctx["chapter_plan"] = chapter_plan or "（无规划）"
    return render_prompt("writer", **ctx)


def render_reviewer_prompt(project_name: str, event_num: int, chapter_num: int,
                           chapter_content: str) -> str:
    """渲染质检提示词"""
    ctx = build_writing_context(project_name, event_num, chapter_num)
    ctx["chapter_content"] = chapter_content
    return render_prompt("reviewer", **ctx)


def render_refiner_prompt(project_name: str, chapter_content: str,
                          style_features: str = "") -> str:
    """渲染润色提示词"""
    from .crud import get_project_by_name
    proj = get_project_by_name(project_name)
    ctx = {
        "style_features": style_features or proj.tone[:500] if proj else "",
        "chapter_content": chapter_content,
    }
    text = render_prompt("refiner", **ctx)
    # 追加用户内容
    return text + f"\n\n# 需要润色的正文\n\n{chapter_content}"


def get_prompt_chain(name: str) -> list[str]:
    """获取提示词链（编排顺序）"""
    CHAINS = {
        "write_chapter": ["planner", "writer", "reviewer", "refiner"],
        "manage_entries": ["entry_manager", "consistency_checker"],
        "quick_write": ["writer", "refiner"],
    }
    return CHAINS.get(name, [name])
