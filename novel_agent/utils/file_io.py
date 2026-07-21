"""
文件 I/O — 独立实现，不依赖旧 scripts/
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

from novel_agent.state import NovelState, EntryPool, EntryState, EventState, ChapterState


def read_env():
    """读取 .env 文件"""
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_project_dir(project_name: str) -> Path:
    """获取项目目录（基于文件位置，不依赖 CWD）"""
    base = Path(__file__).resolve().parent.parent.parent  # novel_agent/../.. = 项目根
    return base / "projects" / project_name


def read_file(project_dir: Path, *subpaths: str) -> str:
    """读取项目内文件"""
    fpath = project_dir.joinpath(*subpaths)
    if not fpath.exists():
        return ""
    return fpath.read_text(encoding="utf-8")


def file_exists(project_dir: Path, *subpaths: str) -> bool:
    """检查文件是否存在且非空"""
    fpath = project_dir.joinpath(*subpaths)
    return fpath.exists() and fpath.stat().st_size > 0


def save_file(project_dir: Path, content: str, *subpaths: str):
    """写入项目内文件"""
    fpath = project_dir.joinpath(*subpaths)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fpath.write_text(content, encoding="utf-8")


def read_project_json(project_dir: Path) -> dict:
    """读取 project.json"""
    p = project_dir / "project.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def write_project_json(project_dir: Path, meta: dict):
    """写入 project.json"""
    (project_dir / "project.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_frontmatter(text: str) -> dict:
    """简易 frontmatter 解析"""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.DOTALL)
    if not m:
        return {"frontmatter": {}, "content": text.strip()}

    fm_text = m.group(1)
    content = m.group(2).strip()

    frontmatter = {}
    for line in fm_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            frontmatter[key.strip()] = value.strip()

    return {"frontmatter": frontmatter, "content": content}


def load_project_to_state(project_name: str) -> NovelState:
    """从 projects/ 目录加载项目到 NovelState"""
    project_dir = get_project_dir(project_name)
    if not project_dir.exists():
        return NovelState(project_name=project_name)

    meta = read_project_json(project_dir)

    state = NovelState(
        project_name=project_name,
        genre=meta.get("genre", ""),
        pen_name=meta.get("pen_name", ""),
        total_chapters=meta.get("total_chapters", 166),
        total_events=meta.get("events_count", 9),
        words_per_chapter=meta.get("words_per_chapter", 1000),
    )

    # 基调 + 全书剧情
    state.tone = read_file(project_dir, "基调.md")
    state.plot = read_file(project_dir, "全书剧情.md")

    # 加载所有条目
    import json as _json
    from novel_agent.state import ENTRY_CATEGORIES
    for cat in ENTRY_CATEGORIES:
        cat_dir = project_dir / cat
        if not cat_dir.exists():
            continue
        for f in sorted(cat_dir.glob("*.md")):
            text = f.read_text(encoding="utf-8")
            parsed = parse_frontmatter(text)
            fm = parsed["frontmatter"]
            appears_in_raw = fm.get("appears_in", "")
            appears_list = [a.strip() for a in appears_in_raw.split(",") if a.strip()] if appears_in_raw else []

            # one_line 去掉首尾引号（保存时加了引号）
            one_line = fm.get("one_line", "")
            if one_line.startswith('"') and one_line.endswith('"'):
                one_line = one_line[1:-1].replace('\\"', '"')

            # change_history JSON 反序列化
            ch_raw = fm.get("change_history", "")
            change_history = []
            if ch_raw and ch_raw not in ("[]", ""):
                try:
                    change_history = _json.loads(ch_raw) if isinstance(ch_raw, str) else (ch_raw or [])
                except Exception:
                    change_history = []

            entry = EntryState(
                name=f.stem,
                category=cat,
                one_line=one_line,
                content=parsed["content"],
                version=int(fm.get("version", 1)),
                appears_in=appears_list,
                change_history=change_history,
            )
            state.entries.upsert(cat, entry)

    # 加载已有事件
    for i in range(1, state.total_events + 1):
        evt_state = EventState(event_num=i)
        evt_dir = project_dir / f"事件{i}"

        # 事件纲
        plan_content = read_file(evt_dir, "事件纲.md")
        if plan_content:
            evt_state.plan = plan_content
            evt_state.status = "planned"
            # 章节范围正则：优先匹配"## 章节规划（第X-Y章）"标题，避免误匹配概述里的"承接第1-2章"
            m = re.search(r"章节规划\s*[（(]\s*第\s*(\d+)\s*-\s*(\d+)\s*章\s*[）)]", plan_content)
            if m:
                evt_state.chapter_range = (int(m.group(1)), int(m.group(2)))
            else:
                # 回退：取最后一个"第X-Y章"匹配（章节规划通常在文档后部）
                all_matches = re.findall(r"第\s*(\d+)\s*-\s*(\d+)\s*章", plan_content)
                if all_matches:
                    evt_state.chapter_range = (int(all_matches[-1][0]), int(all_matches[-1][1]))
            if evt_state.chapter_range == (0, 0):
                # 兜底：从 total_chapters / total_events 推算（避免 chapter_range=(0,0)
                # 导致 _get_next_step_type 无法推进事件）
                _cpe = max(1, state.total_chapters // state.total_events) if state.total_events > 0 else state.total_chapters
                _start = (i - 1) * _cpe + 1
                _end = state.total_chapters if i == state.total_events else i * _cpe
                evt_state.chapter_range = (_start, _end)

        # 已有的正文
        body_dir = evt_dir / "正文"
        if body_dir.exists():
            for f in sorted(body_dir.glob("第*章.md")):
                m = re.search(r"(\d+)", f.stem)
                if not m:
                    continue
                ch_num = int(m.group(1))
                content = f.read_text(encoding="utf-8")
                ch_state = ChapterState(
                    chapter_num=ch_num,
                    content=content,
                    word_count=len(re.findall(r"[\u4e00-\u9fff]", content)),
                )
                # 已写到磁盘的章节视为已确认（用户若想重写，由 Brain 工具或 step_engine.modify 触发）
                ch_state.status = "human_confirmed"
                ch_state.refined_content = content
                evt_state.chapters[ch_num] = ch_state

        state.events[i] = evt_state

    return state


def save_chapter_to_file(state: NovelState, evt_num: int, ch_num: int):
    """将一章的最终内容写回文件"""
    project_dir = get_project_dir(state.project_name)
    evt_dir = project_dir / f"事件{evt_num}"
    body_dir = evt_dir / "正文"
    body_dir.mkdir(parents=True, exist_ok=True)

    ch_state = state.events[evt_num].chapters.get(ch_num)
    if not ch_state:
        return

    content = ch_state.refined_content or ch_state.content
    fpath = body_dir / f"第{str(ch_num).zfill(3)}章.md"
    fpath.write_text(content, encoding="utf-8")


def save_entry_to_file(state: NovelState, category: str, entry: EntryState):
    """将条目写回文件（含 change_history 序列化）"""
    project_dir = get_project_dir(state.project_name)
    cat_dir = project_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    # one_line 中可能含冒号，需加引号；content 含换行用 block scalar
    lines = ["---"]
    lines.append(f"name: {entry.name}")
    lines.append(f"category: {category}")
    # one_line 用双引号包住，避免 frontmatter 解析冒号出错
    one_line_quoted = entry.one_line.replace('"', '\\"')
    lines.append(f'one_line: "{one_line_quoted}"')
    lines.append(f"version: {entry.version}")
    lines.append(f"appears_in: {', '.join(entry.appears_in)}")
    lines.append("status: active")
    # change_history 序列化为单行 JSON（frontmatter 不支持多行值）
    if entry.change_history:
        import json as _json
        lines.append(f"change_history: {_json.dumps(entry.change_history, ensure_ascii=False)}")
    else:
        lines.append("change_history: []")
    lines.append("---")
    lines.append("")
    lines.append(entry.content)

    fpath = cat_dir / f"{entry.name}.md"
    fpath.write_text("\n".join(lines), encoding="utf-8")
