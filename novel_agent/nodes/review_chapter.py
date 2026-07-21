"""
节点：质检
检查刚写完的章节质量。
展示：Multi-Agent / Conditional Edge（通过 → 继续，不通过 → 返工循环）
"""

from __future__ import annotations
import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from novel_agent.state import NovelState
from novel_agent.utils.llm import call_llm
from novel_agent.utils.state_adapter import node_wrapper
from novel_agent.utils.prompt_context import PromptContextBuilder
from novel_agent.utils.consistency_checker import ConsistencyChecker


def _parse_review_verdict(text: str) -> bool:
    """从 review 输出中解析是否通过。
    优先匹配首行的 PASS/FAIL 标记，回退到"通过：是/否"格式。
    """
    if not text:
        return False
    # 去掉 markdown 加粗和空白
    head = text.lstrip()[:200].upper()
    # 优先：首行 PASS/FAIL
    first_line = text.lstrip().split("\n", 1)[0].strip().strip("*#-").upper()
    if first_line.startswith("PASS"):
        return True
    if first_line.startswith("FAIL"):
        return False
    # 回退：兼容旧格式 "通过：是" / "通过:否" / "**通过：** 是"
    # 先检测"不通过"或"通过：否"
    if re.search(r"不通过|通过[：:]\s*\**\s*否\b|FAIL", head):
        return False
    if re.search(r"通过[：:]\s*\**\s*是\b|PASS", head):
        return True
    return False


@node_wrapper
def review_chapter(state: NovelState) -> dict:
    """质检刚写完的章节"""
    evt = state.events.get(state.current_event)
    if not evt:
        return {"last_error": f"事件{state.current_event} 不存在"}

    ch = evt.chapters.get(state.current_chapter)
    if not ch or not ch.content:
        return {"last_error": "没有可质检的章节"}

    # 如果已经通过质检，跳过
    if ch.status == "reviewed" and ch.review_feedback and _parse_review_verdict(ch.review_feedback):
        return {}

    system_prompt = open(
        os.path.join(os.path.dirname(__file__), "..", "prompts", "system_reviewer.md"),
        encoding="utf-8",
    ).read()
    system_prompt = system_prompt.replace("{words_per_chapter}", str(state.words_per_chapter))

    # 使用统一的上下文构建器（包含条目池+事件纲+进度摘要）
    builder = PromptContextBuilder(state)
    user_prompt = builder.for_chapter_review(state.current_chapter)

    # 跨事件一致性校验（规则匹配，零成本）
    checker = ConsistencyChecker(state)
    cons_ok, cons_reason = checker.check_chapter_draft(ch.content, state.current_chapter)

    result = call_llm(system_prompt, user_prompt, temperature=0.3, max_tokens=2048)

    # 如果 LLM 判定通过但一致性校验失败，强制改为 FAIL
    llm_passed = _parse_review_verdict(result["output"])
    if llm_passed and not cons_ok:
        forced_feedback = (
            f"FAIL\n\n"
            f"LLM 原始判定：PASS\n"
            f"但一致性校验失败：{cons_reason}\n\n"
            f"--- LLM 原始反馈 ---\n{result['output']}"
        )
        ch.review_feedback = forced_feedback
        return {}

    ch.review_feedback = result["output"]

    # 判断是否通过
    if not _parse_review_verdict(result["output"]):
        # 不通过 → 返回信号，写作子图的条件边会路由回 plan 节点
        # review_feedback 保留，plan_chapter 会检测到并复用章号
        return {}

    # 通过 → 标记已审查，等待润色
    ch.status = "reviewed"
    if state.current_chapter not in state.pending_refinement:
        state.pending_refinement.append(state.current_chapter)

    return {}
