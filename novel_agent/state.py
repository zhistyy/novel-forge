"""
State Schema — 整个 Agent 的"上帝白板"

所有 Node 读写同一份 State，调度器根据 State 做路由决策。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Literal


# ── 条目相关 ─────────────────────────────────────────

# 条目分类常量（统一引用，避免散落字符串）
ENTRY_CATEGORIES = ("人物设定", "势力设定", "概念设定", "道具设定", "地点设定", "其他设定")


@dataclass
class EntryState:
    """单个条目的运行时状态"""
    name: str
    category: str              # 见 ENTRY_CATEGORIES
    one_line: str              # 一句话介绍
    content: str               # 完整正文
    version: int = 1
    appears_in: list[str] = field(default_factory=list)   # ["事件1", "事件3"] 或 ["事件1/第3章"]
    change_history: list[dict] = field(default_factory=list)  # [{"v":2,"event":2,"chapter":7,"reason":"...","ts":"..."}]

    def to_prompt_block(self) -> str:
        return f"### {self.name} (v{self.version})\n{self.content}"

    def update(self, reason: str, event: int = 0, chapter: int = 0,
               new_content: str = None, new_one_line: str = None,
               extra_appears_in: str = None) -> None:
        """条目更新：版本+1，写历史，按需更新 content/one_line/appears_in。

        参数：
          reason: 变更原因（必填，简短一句话）
          event/chapter: 触发更新的位置（用于追溯）
          new_content: 如提供则替换 content
          new_one_line: 如提供则替换 one_line
          extra_appears_in: 如提供则追加到 appears_in（如 "事件3" 或 "事件2/第7章"）
        """
        from datetime import datetime
        self.version += 1
        if new_content is not None and new_content.strip():
            self.content = new_content.strip()
        if new_one_line is not None and new_one_line.strip():
            self.one_line = new_one_line.strip()[:120]
        if extra_appears_in and extra_appears_in not in self.appears_in:
            self.appears_in.append(extra_appears_in)
        self.change_history.append({
            "v": self.version,
            "event": event,
            "chapter": chapter,
            "reason": reason,
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    def touch_appears_in(self, event: int = 0, chapter: int = 0) -> None:
        """仅追加 appears_in 标记，不改版本（用于按需加载时标记条目被引用）"""
        tag = f"事件{event}" if event and not chapter else (
            f"事件{event}/第{chapter}章" if event and chapter else ""
        )
        if tag and tag not in self.appears_in:
            # 只记事件级，避免 appears_in 爆炸
            evt_tag = f"事件{event}" if event else ""
            if evt_tag and evt_tag not in self.appears_in:
                self.appears_in.append(evt_tag)


@dataclass
class EntryPool:
    """全部条目的容器，按分类组织"""
    人物设定: dict[str, EntryState] = field(default_factory=dict)
    势力设定: dict[str, EntryState] = field(default_factory=dict)
    概念设定: dict[str, EntryState] = field(default_factory=dict)
    道具设定: dict[str, EntryState] = field(default_factory=dict)
    地点设定: dict[str, EntryState] = field(default_factory=dict)
    其他设定: dict[str, EntryState] = field(default_factory=dict)

    def get(self, category: str, name: str) -> Optional[EntryState]:
        pool = getattr(self, category, None)
        if pool is None:
            return None
        return pool.get(name)

    def upsert(self, category: str, entry: EntryState):
        pool = getattr(self, category, None)
        if pool is None:
            raise ValueError(f"未知分类: {category}")
        pool[entry.name] = entry

    def all_entries(self) -> list[EntryState]:
        """返回所有条目"""
        result = []
        for cat in ENTRY_CATEGORIES:
            pool = getattr(self, cat, {})
            result.extend(pool.values())
        return result

    def get_entries_for_event(self, event_name: str) -> list[EntryState]:
        """按需加载：只返回 appears_in 包含该事件的条目"""
        result = []
        for cat in ENTRY_CATEGORIES:
            pool = getattr(self, cat, {})
            for entry in pool.values():
                if event_name in entry.appears_in:
                    result.append(entry)
        return result

    def get_entries_for_chapter(self, event_name: str, chapter_num: int) -> list[EntryState]:
        """按需加载：返回该事件相关的所有条目（章节级粒度太细，仍按事件级）"""
        return self.get_entries_for_event(event_name)

    def all_entries_summary(self) -> str:
        """所有条目的一句话摘要（用于上下文构建）"""
        lines = []
        for cat in ENTRY_CATEGORIES:
            pool = getattr(self, cat, {})
            if pool:
                lines.append(f"## {cat}")
                for e in pool.values():
                    lines.append(f"- {e.name} (v{e.version})：{e.one_line}")
        return "\n".join(lines)

    def entries_summary_for_event(self, event_name: str) -> str:
        """仅当前事件相关条目的摘要"""
        lines = []
        for cat in ENTRY_CATEGORIES:
            pool = getattr(self, cat, {})
            relevant = [e for e in pool.values() if event_name in e.appears_in]
            if relevant:
                lines.append(f"## {cat}")
                for e in relevant:
                    lines.append(f"- {e.name} (v{e.version})：{e.one_line}")
        return "\n".join(lines)


# ── 章节相关 ─────────────────────────────────────────

ChapterStatus = Literal["draft", "reviewed", "refined", "human_confirmed"]


@dataclass
class ChapterState:
    """单章状态"""
    chapter_num: int
    content: str = ""
    status: ChapterStatus = "draft"
    plan: str = ""                     # 本章规划（写之前生成的）
    review_feedback: str = ""          # 质检反馈
    refined_content: str = ""          # 润色后正文
    word_count: int = 0
    entries_updated: bool = False      # 是否已跑过 update_entries


# ── 事件相关 ─────────────────────────────────────────

EventStatus = Literal["planned", "writing", "entries_loaded", "completed"]


@dataclass
class EventState:
    """单个事件的状态"""
    event_num: int
    plan: str = ""                     # 事件纲正文
    status: EventStatus = "planned"
    chapters: dict[int, ChapterState] = field(default_factory=dict)
    chapter_range: tuple[int, int] = (0, 0)  # (start, end)

    @property
    def event_name(self) -> str:
        return f"事件{self.event_num}"

    @property
    def progress(self) -> str:
        written = sum(1 for c in self.chapters.values() if c.status != "draft")
        total = len(self.chapters) or 1
        return f"{written}/{total}"


# ── 主状态 ───────────────────────────────────────────

@dataclass
class NovelState:
    """Agent 的完整状态板"""

    # ── 项目信息 ──
    project_name: str = ""
    genre: str = ""
    pen_name: str = ""

    # ── 核心内容 ──
    tone: str = ""                     # 基调.md 全文
    plot: str = ""                     # 全书剧情.md 全文

    # ── 条目池 ──
    entries: EntryPool = field(default_factory=EntryPool)

    # ── 进度 ──
    current_event: int = 1
    total_events: int = 9
    current_chapter: int = 1
    total_chapters: int = 166
    words_per_chapter: int = 1000

    # ── 事件状态 ──
    events: dict[int, EventState] = field(default_factory=dict)

    # ── 队列（调度器用） ──
    pending_refinement: list[int] = field(default_factory=list)        # 待润色的章号
    pending_entry_updates: list[dict] = field(default_factory=list)    # 待更新的条目
    pending_human_review: list[int] = field(default_factory=list)      # 等待人工确认的章节
    pending_events_to_plan: list[int] = field(default_factory=list)    # 待生成事件纲的事件号

    # ── 写作特征（持续学习） ──
    writing_style_features: str = ""

    # ── 修改意见（用户中断后提的意见，节点读取后清空） ──
    modification_hint: str = ""

    # ── 元数据 ──
    status: str = "initialized"        # initialized / running / waiting_for_human / completed
    last_error: str = ""
    consistency_check_needed: bool = False
    consistency_score: float = 1.0     # 最近一次一致性评分 (0.0~1.0)
    chapter_counter: int = 1           # 全局累计章节数

    # ── 序列化辅助 ──
    def to_checkpoint_dict(self) -> dict:
        """转为 dict（使用 dataclasses.asdict，保留 int 键）"""
        return asdict(self)

    @classmethod
    def from_checkpoint_dict(cls, data: dict) -> "NovelState":
        """从 checkpoint 恢复，兼容 dict 和已有对象混合的情况"""
        d = dict(data)  # 不修改原始 dict

        # 重建 EntryPool
        entries_data = d.get("entries", {})
        if isinstance(entries_data, EntryPool):
            pool = entries_data
        elif isinstance(entries_data, dict):
            pool = EntryPool()
            for cat in ENTRY_CATEGORIES:
                cat_data = entries_data.get(cat, {})
                if isinstance(cat_data, dict):
                    pool.__dict__[cat] = {
                        k: EntryState(**v) if isinstance(v, dict) else v
                        for k, v in cat_data.items()
                    }
        else:
            pool = EntryPool()

        # 重建 events（兼容已有 EventState 对象 和 dict）
        events_data = d.get("events", {})
        events = {}
        for k, v in events_data.items():
            if isinstance(v, EventState):
                events[int(k)] = v
            elif isinstance(v, dict):
                ch_data = v.get("chapters", {})
                chapters = {}
                for ck, cv in ch_data.items():
                    if isinstance(cv, ChapterState):
                        chapters[int(ck)] = cv
                    elif isinstance(cv, dict):
                        chapters[int(ck)] = ChapterState(**cv)
                v = dict(v)
                v["chapters"] = chapters
                # chapter_range 经 asdict 序列化后变成 list，转回 tuple
                # 避免 == (0, 0) 等比较因类型不一致而失败
                cr = v.get("chapter_range")
                if isinstance(cr, list):
                    v["chapter_range"] = tuple(cr)
                events[int(k)] = EventState(**v)

        d["entries"] = pool
        d["events"] = events
        # 过滤掉 NovelState 不认识的字段（如 node_wrapper merge 进来的
        # entry_updates / llm_output / last_error 等节点返回值），
        # 避免 cls(**d) 因意外关键字参数 TypeError
        import dataclasses as _dc
        valid_keys = {f.name for f in _dc.fields(cls)}
        d = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**d)
