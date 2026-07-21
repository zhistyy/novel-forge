"""
项目创建 — 从零创建项目：建文件夹 + 模板文件 + 导入数据库
从 BrainOrchestrator._tool_create_project 抽出，脱离 Brain 系统依赖。
"""
from __future__ import annotations

import json as _json
import re
import shutil
from pathlib import Path

from . import crud, prompt_manager
from novel_agent.utils.file_io import get_project_dir


# ── 从 plot 提取初始条目（多分类） ──────────────────────

# 段落标题 → 分类映射
_PLOT_HEADING_MAP = [
    (r"##\s*主要人物设定[^\n]*|##\s*主要人物[^\n]*|##\s*人物设定[^\n]*|##\s*核心人物[^\n]*", "人物设定"),
    (r"##\s*主要势力设定[^\n]*|##\s*主要势力[^\n]*|##\s*势力设定[^\n]*|##\s*组织设定[^\n]*|##\s*阵营设定[^\n]*", "势力设定"),
    (r"##\s*核心概念设定[^\n]*|##\s*核心概念[^\n]*|##\s*概念设定[^\n]*|##\s*关键术语[^\n]*|##\s*术语表[^\n]*", "概念设定"),
    (r"##\s*关键道具设定[^\n]*|##\s*关键道具[^\n]*|##\s*道具设定[^\n]*|##\s*重要物品[^\n]*|##\s*关键物品[^\n]*", "道具设定"),
    (r"##\s*关键地点设定[^\n]*|##\s*关键地点[^\n]*|##\s*地点设定[^\n]*|##\s*主要场所[^\n]*|##\s*主要地点[^\n]*", "地点设定"),
    (r"##\s*世界观设定[^\n]*|##\s*世界观[^\n]*|##\s*世界设定[^\n]*", "其他设定"),
]

# 非条目段落（已用作其他用途）
_NON_ENTRY_HEADINGS = {
    "大主线", "主线核心", "事件分布", "主线硬约束", "硬约束", "事件规划",
    "概述", "章节规划", "关键人物", "关键势力", "关键概念", "关键道具", "关键地点",
    "爽点布局", "注意事项",
}


def _extract_person_entries(plot: str) -> list[dict]:
    """[保留兼容] 从 plot 提取人物条目"""
    return _extract_all_entries(plot).get("人物设定", [])


def _extract_all_entries(plot: str) -> dict[str, list[dict]]:
    """从 plot 多分类提取条目。

    返回 {category: [{"name":..., "one_line":..., "content":...}]}
    """
    result: dict[str, list[dict]] = {}
    if not plot:
        return result

    # 找到所有 ## 段落标题及其位置
    headings = []  # [(start, end, title_text, category)]
    for pat, cat in _PLOT_HEADING_MAP:
        for m in re.finditer(pat, plot):
            headings.append((m.start(), m.end(), m.group(0), cat))
    if not headings:
        return result

    # 按位置排序
    headings.sort()

    for i, (start, end, title, cat) in enumerate(headings):
        # 段落正文：从当前标题后到下一个 ## 段落
        rest = plot[end:]
        # 跳过同分类的连续标题（已合并）
        next_h2 = re.search(r"\n##\s+", rest)
        section = rest[:next_h2.start()] if next_h2 else rest

        entries = []
        for line in section.splitlines():
            line = line.strip()
            if not line.startswith("-"):
                continue
            body = line[1:].strip()
            # 匹配 **名字**[（注释）]**：描述  或  名字（注释）：描述  或  名字：描述
            m = re.match(
                r"\*{0,2}\s*([^\s*（(：:]{1,10})\s*[\*]{0,2}\s*"
                r"(?:[（(]([^）)]*)[）)])?\s*[\*]{0,2}\s*[：:]\s*(.+)",
                body,
            )
            if not m:
                continue
            name = m.group(1).strip().strip("*")
            note = (m.group(2) or "").strip()
            desc = m.group(3).strip()
            if not name or not desc:
                continue
            if name in _NON_ENTRY_HEADINGS:
                continue
            if len(name) > 10:
                continue
            one_line = desc[:100]
            full = f"{name}：{desc}"
            if note:
                full = f"{name}（{note}）：{desc}"
            entries.append({"name": name, "one_line": one_line, "content": full})

        if entries:
            result.setdefault(cat, []).extend(entries)

    return result


def create_project(
    name: str,
    genre: str,
    tone: str = "",
    plot: str = "",
    total_events: int = 3,
    total_chapters: int = 0,
    words_per_chapter: int = 1000,
) -> dict:
    """从零创建项目。

    返回 {"success": bool, "text": str, "error": str, "project_id": int}
    """
    if not name:
        return _err("需要项目名称（name）")
    if not genre:
        return _err("需要小说类型（genre）")

    # 防止路径注入
    if any(c in name for c in '/\\:*?"<>|'):
        return _err(f"项目名含非法字符：{name}")

    if not total_chapters:
        total_chapters = total_events * 3

    # 检查重名
    if crud.get_project_by_name(name):
        return _err(f"项目「{name}」已存在")

    root = get_project_dir(name)
    if root.exists():
        return _err(f"目录已存在：{root}，请先删除或换名")

    try:
        chapters_per_event = max(1, total_chapters // total_events) if total_events > 0 else total_chapters

        # 创建目录结构（条目目录按需在提取/扫描时创建）
        root.mkdir(parents=True, exist_ok=False)
        for i in range(1, total_events + 1):
            evt_dir = root / f"事件{i}"
            evt_dir.mkdir()
            (evt_dir / "正文").mkdir()
            ch_start = (i - 1) * chapters_per_event + 1
            ch_end = total_chapters if i == total_events else i * chapters_per_event
            # 空事件纲模板
            (evt_dir / "事件纲.md").write_text(
                f"# 事件{i}\n\n## 概述\n（待填写）\n\n## 章节规划（第{ch_start}-{ch_end}章）\n"
                + "\n".join(
                    f"- 第{ch}章：待规划"
                    for ch in range(ch_start, min(ch_end + 1, ch_start + 3))
                )
                + "\n",
                encoding="utf-8",
            )

        # project.json
        (root / "project.json").write_text(
            _json.dumps({
                "name": name,
                "genre": genre,
                "pen_name": "",
                "events_count": total_events,
                "total_chapters": total_chapters,
                "words_per_chapter": words_per_chapter,
                "tone": tone,
                "plot": plot,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 全书剧情.md（如果用户给了 plot，写入实质内容；否则用空模板）
        if plot:
            (root / "全书剧情.md").write_text(plot.strip() + "\n", encoding="utf-8")
        else:
            (root / "全书剧情.md").write_text(
                f"# {name} 全书剧情\n\n## 大主线\n（待填写）\n\n## 事件分布\n"
                + "\n".join(
                    f"- 事件{i}（第{(i-1)*chapters_per_event + 1}-{total_chapters if i == total_events else i*chapters_per_event}章）：待规划"
                    for i in range(1, total_events + 1)
                )
                + "\n",
                encoding="utf-8",
            )

        # 基调.md（如果用户给了 tone）
        if tone:
            # 避免重复前缀：如果 tone 已含 "# 基调" 标题，直接用
            tone_text = tone if tone.strip().startswith("# 基调") else f"# 基调\n\n{tone}"
            (root / "基调.md").write_text(tone_text.rstrip() + "\n", encoding="utf-8")

        # 从 plot 自动提取各类条目并写入对应目录
        if plot:
            from novel_agent.state import ENTRY_CATEGORIES
            all_entries = _extract_all_entries(plot)
            for cat in ENTRY_CATEGORIES:
                items = all_entries.get(cat, [])
                if not items:
                    continue
                cat_dir = root / cat
                cat_dir.mkdir(parents=True, exist_ok=True)
                for item in items:
                    import json as _json2
                    fm_lines = [
                        "---",
                        f"name: {item['name']}",
                        f"category: {cat}",
                        # one_line 加引号避免冒号解析问题
                        f'one_line: "{item["one_line"].replace(chr(34), "")}"',
                        "version: 1",
                        f"appears_in: {', '.join(f'事件{i}' for i in range(1, total_events + 1))}",
                        "status: active",
                        "change_history: []",
                        "---",
                        "",
                        item["content"],
                    ]
                    (cat_dir / f"{item['name']}.md").write_text(
                        "\n".join(fm_lines), encoding="utf-8"
                    )

        # 导入数据库
        pid = crud.import_project_from_fs(name)
        if not pid:
            return _err(f"目录已建但数据库导入失败：{root}")

        prompt_manager.ensure_default_prompts()
        return {
            "success": True,
            "project_id": pid,
            "text": (
                f"项目「{name}」已创建并导入数据库 (id={pid})。\n"
                f"  类型：{genre}\n  事件数：{total_events}\n  总章数：{total_chapters}\n"
                f"  每章字数：{words_per_chapter}\n  目录：{root}"
            ),
            "error": "",
        }
    except Exception as e:
        # 失败时清理半成品目录
        try:
            if root.exists():
                shutil.rmtree(root)
        except Exception:
            pass
        return _err(f"创建项目失败：{e}")


def _err(msg: str) -> dict:
    return {"success": False, "text": msg, "error": msg, "project_id": 0}
