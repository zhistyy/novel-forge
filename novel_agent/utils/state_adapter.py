"""
状态适配器 — LangGraph dict ↔ NovelState dataclass 互转

LangGraph 内部以 dict 管理状态，而 NovelState 是 dataclass。
每个 Node 需要用此适配器做转换。
"""

from __future__ import annotations

import functools
from typing import Callable, Any

from novel_agent.state import NovelState, EntryPool, EntryState, EventState, ChapterState, ENTRY_CATEGORIES


def _ensure_dataclass_state(state: NovelState) -> NovelState:
    """确保 state 内的嵌套 dict 都被转回 dataclass 对象"""
    # 1. 修复 entries: dict → EntryPool（全 6 分类）
    if isinstance(state.entries, dict):
        pool = EntryPool()
        entries_dict = state.entries
        for cat in ENTRY_CATEGORIES:
            cat_data = entries_dict.get(cat, {})
            if isinstance(cat_data, dict):
                pool.__dict__[cat] = {
                    k: EntryState(**v) if isinstance(v, dict) else v
                    for k, v in cat_data.items()
                }
        state.entries = pool

    # 2. 修复 events 中的 dict → EventState
    for evt_key, evt_val in list(state.events.items()):
        if isinstance(evt_val, dict):
            ch_dict = evt_val.get("chapters", {})
            for ch_key, ch_val in list(ch_dict.items()):
                if isinstance(ch_val, dict):
                    ch_dict[ch_key] = ChapterState(**ch_val)
            evt_val["chapters"] = ch_dict
            # chapter_range 经序列化可能变成 list，转回 tuple 保持类型一致
            cr = evt_val.get("chapter_range")
            if isinstance(cr, list):
                evt_val["chapter_range"] = tuple(cr)
            state.events[evt_key] = EventState(**evt_val)
    return state


def node_wrapper(func: Callable) -> Callable:
    """
    装饰器：将 Node 函数的输入从 dict 转为 NovelState，
    执行后再将返回值 merge 回 dict 格式。
    """
    @functools.wraps(func)
    def wrapper(state_dict: dict, *args, **kwargs) -> dict:
        # dict → NovelState
        state = NovelState.from_checkpoint_dict(state_dict) if isinstance(state_dict, dict) else state_dict
        # 二次确保嵌套对象正确
        state = _ensure_dataclass_state(state)
        # 执行节点逻辑
        result = func(state, *args, **kwargs)
        # 将 state 再转回 dict 用于 LangGraph 合并
        # 返回 result 和 state 的全部字段
        merged = state.to_checkpoint_dict()
        if isinstance(result, dict):
            merged.update(result)
        return merged

    return wrapper
