"""
节点：规划本章
基于事件纲和当前进度，为下一章制定详细写作规划。
展示：Node / State Management
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from novel_agent.state import NovelState, ChapterState
from novel_agent.utils.llm import call_llm
from novel_agent.utils.state_adapter import node_wrapper
from novel_agent.utils.prompt_context import PromptContextBuilder
from novel_agent.nodes.review_chapter import _parse_review_verdict


@node_wrapper
def plan_chapter(state: NovelState) -> dict:
    """为当前事件当前章节生成写作规划。
    信任 state.current_chapter（由 step_engine 推进），不自己决定返工。"""
    evt = state.events.get(state.current_event)
    if not evt:
        return {"last_error": f"事件{state.current_event} 不存在"}

    # 章号来源：优先用 state.current_chapter（step_engine 已推进）；
    # 若该章已存在且有规划，直接跳过；
    # 若 current_chapter 落在事件范围外，回退到事件起始章。
    start_ch, end_ch = evt.chapter_range
    ch_num = state.current_chapter or start_ch or 1
    if start_ch > 0 and not (start_ch <= ch_num <= end_ch):
        ch_num = start_ch

    existing = evt.chapters
    ch = existing.get(ch_num)

    # 已有规划且无返工信号 → 跳过
    if ch and ch.plan and not (
        ch.review_feedback and not _parse_review_verdict(ch.review_feedback) and ch.status == "draft"
    ):
        # 已规划过，无需重复
        if ch.plan:
            return {}

    # 使用统一的上下文构建器
    builder = PromptContextBuilder(state)
    hint = (state.modification_hint or "").strip()
    user_prompt = builder.for_chapter_plan(ch_num, hint=hint)

    system_prompt = open(
        os.path.join(os.path.dirname(__file__), "..", "prompts", "system_planner.md"),
        encoding="utf-8",
    ).read()

    result = call_llm(system_prompt, user_prompt, temperature=0.7, max_tokens=2048)

    # 创建或更新章节状态
    if ch:
        ch.plan = result["output"]
    else:
        ch = ChapterState(
            chapter_num=ch_num,
            plan=result["output"],
        )
        evt.chapters[ch_num] = ch
    state.current_chapter = ch_num
    state.chapter_counter = max(state.chapter_counter, ch_num)

    return {}
