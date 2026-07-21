"""
CRUD 操作 — 全量增删改查

每个实体提供: create / get / get_all / update / delete
"""

from __future__ import annotations

import json, re
from typing import Optional

from .db import (
    get_conn, transaction, row_to_dataclass,
    ProjectDB, EventDB, ChapterDB, EntryDB, PromptDB, WorkflowDB,
)


# ════════════════════════════════════════════
# 项目
# ════════════════════════════════════════════

def create_project(name: str, genre="", pen_name="", tone="", plot="",
                   total_events=9, total_chapters=166, words_per_chapter=1000) -> ProjectDB:
    with transaction() as conn:
        cur = conn.execute(
            "INSERT INTO projects (name,genre,pen_name,tone,plot,total_events,total_chapters,words_per_chapter) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (name, genre, pen_name, tone, plot, total_events, total_chapters, words_per_chapter),
        )
        pid = cur.lastrowid
        # 按章节数均匀分配事件范围（最后一个事件吸收余数）
        chapters_per_event = max(1, total_chapters // total_events) if total_events > 0 else total_chapters
        for i in range(1, total_events + 1):
            start = (i - 1) * chapters_per_event + 1
            if i == total_events:
                end = total_chapters  # 最后一个事件吸收余数
            else:
                end = i * chapters_per_event
            conn.execute(
                "INSERT INTO events (project_id,num,ch_range_start,ch_range_end) VALUES (?,?,?,?)",
                (pid, i, start, end),
            )
        # 创建工作流
        conn.execute(
            "INSERT INTO workflow (project_id) VALUES (?)", (pid,),
        )
    return get_project_by_name(name)

def get_project_by_name(name: str) -> Optional[ProjectDB]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM projects WHERE name=?", (name,)).fetchone()
    return row_to_dataclass(row, ProjectDB) if row else None

def get_project_by_id(pid: int) -> Optional[ProjectDB]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    return row_to_dataclass(row, ProjectDB) if row else None

def list_projects() -> list[ProjectDB]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM projects ORDER BY name").fetchall()
    return [row_to_dataclass(r, ProjectDB) for r in rows]

def update_project(pid: int, **kwargs) -> Optional[ProjectDB]:
    allowed = {"genre","pen_name","tone","plot","total_events","total_chapters","words_per_chapter"}
    updates = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    if not updates:
        return get_project_by_id(pid)
    # updated_at 直接用 SQL 函数设值，避免传 None 导致字段变 NULL（DEFAULT 仅 INSERT 时生效）
    set_clause = ", ".join(f"{k}=?" for k in updates) + ", updated_at=datetime('now','localtime')"
    values = list(updates.values())
    with transaction() as conn:
        conn.execute(f"UPDATE projects SET {set_clause} WHERE id=?", (*values, pid))
    return get_project_by_id(pid)

def delete_project(pid: int) -> bool:
    with transaction() as conn:
        # 不能用 conn.total_changes：它是连接级累计值，本线程之前有任何成功操作就 >0。
        # 用 cur.rowcount 获取本条 DELETE 实际影响的行数。
        cur = conn.execute("DELETE FROM projects WHERE id=?", (pid,))
        return cur.rowcount > 0


# ════════════════════════════════════════════
# 事件
# ════════════════════════════════════════════

def get_events_for_project(pid: int) -> list[EventDB]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM events WHERE project_id=? ORDER BY num", (pid,)).fetchall()
    return [row_to_dataclass(r, EventDB) for r in rows]

def get_event(pid: int, num: int) -> Optional[EventDB]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM events WHERE project_id=? AND num=?", (pid, num)).fetchone()
    return row_to_dataclass(row, EventDB) if row else None

def update_event(eid: int, **kwargs) -> Optional[EventDB]:
    allowed = {"plan","status","ch_range_start","ch_range_end"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return None
    set_clause = ", ".join(f"{k}=?" for k in updates) + ", updated_at=datetime('now','localtime')"
    values = list(updates.values())
    with transaction() as conn:
        conn.execute(f"UPDATE events SET {set_clause} WHERE id=?", (*values, eid))
    conn = get_conn()
    row = conn.execute("SELECT * FROM events WHERE id=?", (eid,)).fetchone()
    return row_to_dataclass(row, EventDB)


# ════════════════════════════════════════════
# 章节
# ════════════════════════════════════════════

def get_chapters_for_event(eid: int) -> list[ChapterDB]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM chapters WHERE event_id=? ORDER BY num", (eid,)).fetchall()
    return [row_to_dataclass(r, ChapterDB) for r in rows]

def get_chapter(eid: int, num: int) -> Optional[ChapterDB]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM chapters WHERE event_id=? AND num=?", (eid, num)).fetchone()
    return row_to_dataclass(row, ChapterDB) if row else None

def create_chapter(eid: int, num: int, plan="", status="pending") -> ChapterDB:
    with transaction() as conn:
        # INSERT OR IGNORE：row 已存在时不报错。用 cur.rowcount 判断是否真的插入：
        # rowcount=1 表示新插入，rowcount=0 表示被 IGNORE（已存在）。
        # 不能用 cur.lastrowid is None：lastrowid 在 IGNORE 时是上一条成功 INSERT 的 rowid
        # （或初始值），并非 None，判断不可靠。
        cur = conn.execute(
            "INSERT OR IGNORE INTO chapters (event_id,num,plan,status) VALUES (?,?,?,?)",
            (eid, num, plan, status),
        )
        if cur.rowcount == 0:
            # 已存在，更新 plan 和 status
            conn.execute("UPDATE chapters SET plan=?,status=? WHERE event_id=? AND num=?",
                        (plan, status, eid, num))
    return get_chapter(eid, num)

def update_chapter(ch_id: int, **kwargs) -> Optional[ChapterDB]:
    allowed = {"plan","content","refined_content","review_feedback","status","word_count"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return None
    set_clause = ", ".join(f"{k}=?" for k in updates) + ", updated_at=datetime('now','localtime')"
    values = list(updates.values())
    with transaction() as conn:
        conn.execute(f"UPDATE chapters SET {set_clause} WHERE id=?", (*values, ch_id))
    conn = get_conn()
    row = conn.execute("SELECT * FROM chapters WHERE id=?", (ch_id,)).fetchone()
    return row_to_dataclass(row, ChapterDB)

def delete_chapter(eid: int, num: int) -> bool:
    with transaction() as conn:
        cur = conn.execute("DELETE FROM chapters WHERE event_id=? AND num=?", (eid, num))
        return cur.rowcount > 0


# ════════════════════════════════════════════
# 条目 (增删改查核心)
# ════════════════════════════════════════════

def list_entries(pid: int, category: str = "") -> list[EntryDB]:
    conn = get_conn()
    if category:
        rows = conn.execute(
            "SELECT * FROM entries WHERE project_id=? AND category=? ORDER BY name",
            (pid, category),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM entries WHERE project_id=? ORDER BY category, name", (pid,),
        ).fetchall()
    return [row_to_dataclass(r, EntryDB) for r in rows]

def get_entry(pid: int, name: str) -> Optional[EntryDB]:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM entries WHERE project_id=? AND name=?", (pid, name),
    ).fetchone()
    return row_to_dataclass(row, EntryDB) if row else None

def create_entry(pid: int, name: str, category: str, one_line="", content="",
                 appears_in=None, version=1) -> EntryDB:
    appears_json = json.dumps(appears_in or [], ensure_ascii=False)
    with transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO entries (project_id,name,category,one_line,content,version,appears_in) "
            "VALUES (?,?,?,?,?,?,?)",
            (pid, name, category, one_line, content, version, appears_json),
        )
    return get_entry(pid, name)

def update_entry(pid: int, name: str, **kwargs) -> Optional[EntryDB]:
    allowed = {"category","one_line","content","version","appears_in"}
    updates = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    if not updates:
        return get_entry(pid, name)
    if "appears_in" in updates and isinstance(updates["appears_in"], (list, tuple)):
        updates["appears_in"] = json.dumps(updates["appears_in"], ensure_ascii=False)
    set_clause = ", ".join(f"{k}=?" for k in updates) + ", updated_at=datetime('now','localtime')"
    values = list(updates.values())
    with transaction() as conn:
        conn.execute(f"UPDATE entries SET {set_clause} WHERE project_id=? AND name=?", (*values, pid, name))
    return get_entry(pid, name)

def delete_entry(pid: int, name: str) -> bool:
    with transaction() as conn:
        cur = conn.execute("DELETE FROM entries WHERE project_id=? AND name=?", (pid, name))
        return cur.rowcount > 0


def upsert_entry(pid: int, name: str, category: str, one_line: str = "",
                 content: str = "", version: int = 1, appears_in: str = "",
                 **extra) -> Optional[EntryDB]:
    """存在则更新，不存在则创建"""
    existing = get_entry(pid, name)
    if existing:
        updates = {
            "category": category,
            "one_line": one_line,
            "content": content,
            "version": version,
            "appears_in": appears_in,
        }
        updates.update(extra)
        # updated_at 用 SQL 函数设值，避免传 None 导致字段变 NULL（DEFAULT 仅 INSERT 时生效）
        set_clause = ", ".join(f"{k}=?" for k in updates) + ", updated_at=datetime('now','localtime')"
        values = list(updates.values())
        with transaction() as conn:
            conn.execute(
                f"UPDATE entries SET {set_clause} WHERE project_id=? AND name=?",
                (*values, pid, name),
            )
        return get_entry(pid, name)
    else:
        # appears_in 已是 JSON 字符串，传给 create_entry 时需先反序列化为 list
        try:
            appears_list = json.loads(appears_in) if isinstance(appears_in, str) else (appears_in or [])
        except (json.JSONDecodeError, TypeError):
            appears_list = []
        return create_entry(
            pid, name, category,
            one_line=one_line, content=content,
            appears_in=appears_list, version=version,
        )


# ════════════════════════════════════════════
# 提示词 (带版本管理)
# ════════════════════════════════════════════

def list_prompts(role: str = "") -> list[PromptDB]:
    conn = get_conn()
    if role:
        rows = conn.execute(
            "SELECT * FROM prompts WHERE role=? ORDER BY name, version DESC", (role,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM prompts ORDER BY role, name, version DESC",
        ).fetchall()
    return [row_to_dataclass(r, PromptDB) for r in rows]

def get_active_prompt(name: str) -> Optional[PromptDB]:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM prompts WHERE name=? AND is_active=1 ORDER BY version DESC LIMIT 1",
        (name,),
    ).fetchone()
    return row_to_dataclass(row, PromptDB) if row else None

def create_prompt(name: str, role: str, system_text: str,
                  template_vars=None, description="") -> PromptDB:
    vars_json = json.dumps(template_vars or [], ensure_ascii=False)
    with transaction() as conn:
        # 获取下一版本号
        row = conn.execute(
            "SELECT MAX(version) as mv FROM prompts WHERE name=?", (name,),
        ).fetchone()
        version = (row["mv"] or 0) + 1
        conn.execute(
            "INSERT INTO prompts (name,role,version,system_text,template_vars,is_active,description) "
            "VALUES (?,?,?,?,?,1,?)",
            (name, role, version, system_text, vars_json, description),
        )
    return get_active_prompt(name)

def update_prompt_text(prompt_id: int, system_text: str) -> bool:
    with transaction() as conn:
        cur = conn.execute("UPDATE prompts SET system_text=?,updated_at=datetime('now','localtime') WHERE id=?",
                    (system_text, prompt_id))
        return cur.rowcount > 0

def deactivate_prompt(name: str) -> bool:
    with transaction() as conn:
        cur = conn.execute("UPDATE prompts SET is_active=0 WHERE name=?", (name,))
        return cur.rowcount > 0

def render_prompt(name: str, **vars) -> str:
    """渲染提示词模板：用传入变量替换 {var} 占位符"""
    p = get_active_prompt(name)
    if not p:
        return ""
    text = p.system_text
    for k, v in vars.items():
        text = text.replace(f"{{{k}}}", str(v))
    return text


# ════════════════════════════════════════════
# 工作流
# ════════════════════════════════════════════

def get_workflow(pid: int) -> Optional[WorkflowDB]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM workflow WHERE project_id=?", (pid,)).fetchone()
    return row_to_dataclass(row, WorkflowDB) if row else None

def upsert_workflow(pid: int, **kwargs) -> WorkflowDB:
    allowed = {"current_event","current_chapter","pending_review","pending_refine",
               "pending_confirm","status","extra"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return get_workflow(pid)
    if "extra" in updates and isinstance(updates["extra"], dict):
        updates["extra"] = json.dumps(updates["extra"], ensure_ascii=False)
    for key in ("pending_review","pending_refine","pending_confirm"):
        if key in updates and isinstance(updates[key], (list, tuple)):
            updates[key] = json.dumps(updates[key], ensure_ascii=False)
    set_clause = ", ".join(f"{k}=?" for k in updates) + ", updated_at=datetime('now','localtime')"
    values = list(updates.values())
    with transaction() as conn:
        # 先确保 row 存在。不能用 conn.total_changes 判断 UPDATE 是否命中：
        # total_changes 是连接级累计值，只要本连接之前有过任何 INSERT/UPDATE 成功就 >0，
        # 会导致 row 不存在时 INSERT 分支永远不触发（imported 项目工作流状态丢失）。
        conn.execute("INSERT OR IGNORE INTO workflow (project_id) VALUES (?)", (pid,))
        conn.execute(f"UPDATE workflow SET {set_clause} WHERE project_id=?", (*values, pid))
    return get_workflow(pid)


# ════════════════════════════════════════════
# 导入工具 (从文件系统导入到数据库)
# ════════════════════════════════════════════

def import_project_from_fs(project_name: str) -> Optional[int]:
    """从 projects/ 目录导入项目到数据库"""
    from novel_agent.utils.file_io import load_project_to_state
    state = load_project_to_state(project_name)
    if not state:
        return None

    # 创建或更新项目
    existing = get_project_by_name(project_name)
    if existing:
        pid = existing.id
        update_project(pid, genre=state.genre, pen_name=state.pen_name,
                      tone=state.tone or "", plot=state.plot or "",
                      total_events=state.total_events,
                      total_chapters=state.total_chapters,
                      words_per_chapter=state.words_per_chapter)
    else:
        pj = create_project(project_name, genre=state.genre, pen_name=state.pen_name,
                          tone=state.tone or "", plot=state.plot or "",
                          total_events=state.total_events,
                          total_chapters=state.total_chapters,
                          words_per_chapter=state.words_per_chapter)
        pid = pj.id

    # 导入事件
    for evt_num, evt in state.events.items():
        db_evt = get_event(pid, evt_num)
        if not db_evt:
            conn = get_conn()
            cur = conn.execute(
                "INSERT INTO events (project_id,num,plan,status,ch_range_start,ch_range_end) VALUES (?,?,?,?,?,?)",
                (pid, evt_num, evt.plan or "", evt.status,
                 evt.chapter_range[0], evt.chapter_range[1]),
            )
            conn.commit()
            db_evt = get_event(pid, evt_num)
        if db_evt:
            update_event(db_evt.id, plan=evt.plan or "", status=evt.status,
                         ch_range_start=evt.chapter_range[0],
                         ch_range_end=evt.chapter_range[1])
            # 导入章节
            for ch_num, ch in evt.chapters.items():
                create_chapter(db_evt.id, ch_num,
                             plan=ch.plan or "", status=ch.status)
                update_chapter_by_event(db_evt.id, ch_num,
                                      content=ch.content or "",
                                      refined_content=ch.refined_content or "",
                                      review_feedback=ch.review_feedback or "",
                                      word_count=ch.word_count,
                                      status=ch.status)

    # 导入条目（全 6 分类）
    from novel_agent.state import ENTRY_CATEGORIES
    for cat in ENTRY_CATEGORIES:
        pool = getattr(state.entries, cat, {})
        for name, entry in pool.items():
            create_entry(pid, name, cat, one_line=entry.one_line,
                       content=entry.content, version=entry.version,
                       appears_in=entry.appears_in or [])

    # 工作流
    upsert_workflow(pid, current_event=state.current_event,
                   current_chapter=state.current_chapter,
                   status=state.status)

    return pid


def update_chapter_by_event(eid: int, ch_num: int, **kwargs):
    """通过 event_id + chapter_num 更新章节"""
    ch = get_chapter(eid, ch_num)
    if ch:
        update_chapter(ch.id, **kwargs)
