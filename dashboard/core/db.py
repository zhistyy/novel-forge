"""
数据库层 — SQLite + 简易 ORM

表结构:
  projects    — 项目元数据
  events      — 事件与事件纲
  chapters    — 章节正文与状态
  entries     — 设定条目（人物/概念/势力/其他）
  prompts     — 提示词模板（带版本管理）
  agent_logs  — 代理执行历史
  workflow    — 工作流状态持久化
"""

from __future__ import annotations

import json, os, sqlite3, threading, re
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DB_DIR / "novel_forge.db"

_thread_local = threading.local()


def get_conn() -> sqlite3.Connection:
    """获取线程级数据库连接"""
    if not hasattr(_thread_local, "conn") or _thread_local.conn is None:
        DB_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _thread_local.conn = conn
    return _thread_local.conn


@contextmanager
def transaction():
    """事务上下文"""
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ── 初始化 ──

def init_db():
    """创建所有表"""
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS projects (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT NOT NULL UNIQUE,
        genre       TEXT DEFAULT '',
        pen_name    TEXT DEFAULT '',
        tone        TEXT DEFAULT '',
        plot        TEXT DEFAULT '',
        total_events INTEGER DEFAULT 9,
        total_chapters INTEGER DEFAULT 166,
        words_per_chapter INTEGER DEFAULT 1000,
        created_at  TEXT DEFAULT (datetime('now','localtime')),
        updated_at  TEXT DEFAULT (datetime('now','localtime'))
    );

    CREATE TABLE IF NOT EXISTS events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        num         INTEGER NOT NULL,
        plan        TEXT DEFAULT '',
        status      TEXT DEFAULT 'planned',
        ch_range_start INTEGER DEFAULT 1,
        ch_range_end   INTEGER DEFAULT 18,
        created_at  TEXT DEFAULT (datetime('now','localtime')),
        updated_at  TEXT DEFAULT (datetime('now','localtime')),
        UNIQUE(project_id, num)
    );

    CREATE TABLE IF NOT EXISTS chapters (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id        INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
        num             INTEGER NOT NULL,
        plan            TEXT DEFAULT '',
        content         TEXT DEFAULT '',
        refined_content TEXT DEFAULT '',
        review_feedback TEXT DEFAULT '',
        status          TEXT DEFAULT 'pending',
        word_count      INTEGER DEFAULT 0,
        created_at      TEXT DEFAULT (datetime('now','localtime')),
        updated_at      TEXT DEFAULT (datetime('now','localtime')),
        UNIQUE(event_id, num)
    );

    CREATE TABLE IF NOT EXISTS entries (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        name        TEXT NOT NULL,
        category    TEXT NOT NULL,
        one_line    TEXT DEFAULT '',
        content     TEXT DEFAULT '',
        version     INTEGER DEFAULT 1,
        appears_in  TEXT DEFAULT '[]',
        created_at  TEXT DEFAULT (datetime('now','localtime')),
        updated_at  TEXT DEFAULT (datetime('now','localtime')),
        UNIQUE(project_id, name)
    );

    CREATE TABLE IF NOT EXISTS prompts (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT NOT NULL,
        role        TEXT NOT NULL,
        version     INTEGER DEFAULT 1,
        system_text TEXT NOT NULL,
        template_vars TEXT DEFAULT '[]',
        is_active   INTEGER DEFAULT 1,
        description TEXT DEFAULT '',
        created_at  TEXT DEFAULT (datetime('now','localtime')),
        updated_at  TEXT DEFAULT (datetime('now','localtime')),
        UNIQUE(name, version)
    );

    CREATE TABLE IF NOT EXISTS agent_logs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id  INTEGER REFERENCES projects(id) ON DELETE SET NULL,
        agent_name  TEXT NOT NULL,
        action      TEXT NOT NULL,
        input       TEXT DEFAULT '',
        output      TEXT DEFAULT '',
        duration_ms INTEGER DEFAULT 0,
        tokens_in   INTEGER DEFAULT 0,
        tokens_out  INTEGER DEFAULT 0,
        status      TEXT DEFAULT 'success',
        error       TEXT DEFAULT '',
        created_at  TEXT DEFAULT (datetime('now','localtime'))
    );

    CREATE TABLE IF NOT EXISTS workflow (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        current_event INTEGER DEFAULT 1,
        current_chapter INTEGER DEFAULT 1,
        pending_review   TEXT DEFAULT '[]',
        pending_refine   TEXT DEFAULT '[]',
        pending_confirm  TEXT DEFAULT '[]',
        status      TEXT DEFAULT 'idle',
        extra       TEXT DEFAULT '{}',
        updated_at  TEXT DEFAULT (datetime('now','localtime'))
    );

    CREATE INDEX IF NOT EXISTS idx_chapters_event ON chapters(event_id);
    CREATE INDEX IF NOT EXISTS idx_entries_project ON entries(project_id);
    CREATE INDEX IF NOT EXISTS idx_events_project ON events(project_id);
    CREATE INDEX IF NOT EXISTS idx_agent_logs_project ON agent_logs(project_id);

    CREATE TABLE IF NOT EXISTS agent_memory (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id  INTEGER REFERENCES projects(id) ON DELETE CASCADE,
        category    TEXT NOT NULL,
        key         TEXT NOT NULL,
        value       TEXT DEFAULT '',
        metadata    TEXT DEFAULT '{}',
        created_at  TEXT DEFAULT (datetime('now','localtime')),
        updated_at  TEXT DEFAULT (datetime('now','localtime')),
        UNIQUE(project_id, category, key)
    );
    CREATE INDEX IF NOT EXISTS idx_agent_memory_project ON agent_memory(project_id);
    CREATE INDEX IF NOT EXISTS idx_agent_memory_category ON agent_memory(category);

    -- ── P4: Agent 会话持久化 ──
    CREATE TABLE IF NOT EXISTS agent_sessions (
        id          TEXT PRIMARY KEY,
        project_id  INTEGER REFERENCES projects(id) ON DELETE CASCADE,
        title       TEXT DEFAULT '',
        summary     TEXT DEFAULT '',
        created_at  TEXT DEFAULT (datetime('now','localtime')),
        updated_at  TEXT DEFAULT (datetime('now','localtime'))
    );
    CREATE INDEX IF NOT EXISTS idx_agent_sessions_project ON agent_sessions(project_id);

    CREATE TABLE IF NOT EXISTS agent_messages (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  TEXT NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
        role        TEXT NOT NULL,
        content     TEXT DEFAULT '',
        tool_calls  TEXT DEFAULT '',
        tool_call_id TEXT DEFAULT '',
        meta        TEXT DEFAULT '{}',
        created_at  TEXT DEFAULT (datetime('now','localtime'))
    );
    CREATE INDEX IF NOT EXISTS idx_agent_messages_session ON agent_messages(session_id);

    CREATE TABLE IF NOT EXISTS agent_runs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  TEXT NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
        user_message TEXT NOT NULL,
        assistant_text TEXT DEFAULT '',
        success     INTEGER DEFAULT 1,
        iterations  INTEGER DEFAULT 0,
        duration_ms INTEGER DEFAULT 0,
        tokens_in   INTEGER DEFAULT 0,
        tokens_out  INTEGER DEFAULT 0,
        summarized  INTEGER DEFAULT 0,
        compressed_count INTEGER DEFAULT 0,
        finish_reason TEXT DEFAULT '',
        error       TEXT DEFAULT '',
        created_at  TEXT DEFAULT (datetime('now','localtime'))
    );
    CREATE INDEX IF NOT EXISTS idx_agent_runs_session ON agent_runs(session_id);

    CREATE TABLE IF NOT EXISTS agent_steps (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id      INTEGER NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
        step_idx    INTEGER NOT NULL,
        tool_name   TEXT NOT NULL,
        arguments   TEXT DEFAULT '{}',
        result_preview TEXT DEFAULT '',
        success     INTEGER DEFAULT 1,
        error       TEXT DEFAULT '',
        duration_ms INTEGER DEFAULT 0,
        created_at  TEXT DEFAULT (datetime('now','localtime'))
    );
    CREATE INDEX IF NOT EXISTS idx_agent_steps_run ON agent_steps(run_id);
    """)
    conn.commit()


# ── 数据类 ──

@dataclass
class ProjectDB:
    id: int = 0
    name: str = ""
    genre: str = ""
    pen_name: str = ""
    tone: str = ""
    plot: str = ""
    total_events: int = 9
    total_chapters: int = 166
    words_per_chapter: int = 1000
    created_at: str = ""
    updated_at: str = ""

@dataclass
class EventDB:
    id: int = 0
    project_id: int = 0
    num: int = 0
    plan: str = ""
    status: str = "planned"
    ch_range_start: int = 1
    ch_range_end: int = 18
    created_at: str = ""
    updated_at: str = ""

@dataclass
class ChapterDB:
    id: int = 0
    event_id: int = 0
    num: int = 0
    plan: str = ""
    content: str = ""
    refined_content: str = ""
    review_feedback: str = ""
    status: str = "pending"
    word_count: int = 0
    created_at: str = ""
    updated_at: str = ""

@dataclass
class EntryDB:
    id: int = 0
    project_id: int = 0
    name: str = ""
    category: str = ""
    one_line: str = ""
    content: str = ""
    version: int = 1
    appears_in: str = "[]"
    created_at: str = ""
    updated_at: str = ""

@dataclass
class PromptDB:
    id: int = 0
    name: str = ""
    role: str = ""
    version: int = 1
    system_text: str = ""
    template_vars: str = "[]"
    is_active: int = 1
    description: str = ""
    created_at: str = ""
    updated_at: str = ""

@dataclass
class AgentLogDB:
    id: int = 0
    project_id: int = 0
    agent_name: str = ""
    action: str = ""
    input: str = ""
    output: str = ""
    duration_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    status: str = "success"
    error: str = ""
    created_at: str = ""

@dataclass
class WorkflowDB:
    id: int = 0
    project_id: int = 0
    current_event: int = 1
    current_chapter: int = 1
    pending_review: str = "[]"
    pending_refine: str = "[]"
    pending_confirm: str = "[]"
    status: str = "idle"
    extra: str = "{}"
    updated_at: str = ""

@dataclass
class AgentMemoryDB:
    id: int = 0
    project_id: int = 0
    category: str = ""
    key: str = ""
    value: str = ""
    metadata: str = "{}"
    created_at: str = ""
    updated_at: str = ""


@dataclass
class AgentSessionDB:
    id: str = ""
    project_id: int = 0
    title: str = ""
    summary: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass
class AgentMessageDB:
    id: int = 0
    session_id: str = ""
    role: str = ""
    content: str = ""
    tool_calls: str = ""
    tool_call_id: str = ""
    meta: str = "{}"
    created_at: str = ""


@dataclass
class AgentRunDB:
    id: int = 0
    session_id: str = ""
    user_message: str = ""
    assistant_text: str = ""
    success: int = 1
    iterations: int = 0
    duration_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    summarized: int = 0
    compressed_count: int = 0
    finish_reason: str = ""
    error: str = ""
    created_at: str = ""


@dataclass
class AgentStepDB:
    id: int = 0
    run_id: int = 0
    step_idx: int = 0
    tool_name: str = ""
    arguments: str = "{}"
    result_preview: str = ""
    success: int = 1
    error: str = ""
    duration_ms: int = 0
    created_at: str = ""


def row_to_dataclass(row: sqlite3.Row, cls):
    """将 sqlite3.Row 转为 dataclass"""
    if row is None:
        return None
    cols = row.keys()  # sqlite3.Row.keys() 返回列名列表
    kwargs = {col: row[col] for col in cols if col in cls.__dataclass_fields__}
    return cls(**kwargs)
