"""
节点：起草正文
按照规划写出完整一章正文。
展示：Node / Multi-Agent / Streaming
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from novel_agent.state import NovelState
from novel_agent.utils.llm import call_llm
from novel_agent.utils.state_adapter import node_wrapper
from novel_agent.utils.prompt_context import PromptContextBuilder
from novel_agent.utils.consistency_checker import ConsistencyChecker


@node_wrapper
def draft_chapter(state: NovelState) -> dict:
    """根据规划写出正文"""
    evt = state.events.get(state.current_event)
    if not evt:
        return {"last_error": f"事件{state.current_event} 不存在"}

    ch = evt.chapters.get(state.current_chapter)
    if not ch:
        return {"last_error": f"第{state.current_chapter}章未规划"}

    # 人工确认过的章节，永远不重写（除非用户手动清除状态）
    if ch.status == "human_confirmed":
        return {}

    if ch.content and not ch.review_feedback:
        # 除非有用户修改意见，否则已起草过就跳过
        if not (state.modification_hint or "").strip():
            return {}
        # 有修改意见：强制返工
        ch.content = ""
        ch.review_feedback = ""
        ch.word_count = 0
    if ch.content and ch.review_feedback:
        # 返工：清空旧内容和质检反馈
        ch.content = ""
        ch.review_feedback = ""
        ch.word_count = 0

    # 使用统一的上下文构建器
    builder = PromptContextBuilder(state)
    hint = (state.modification_hint or "").strip()
    user_prompt = builder.for_chapter_draft(state.current_chapter, hint=hint)

    system_prompt = open(
        os.path.join(os.path.dirname(__file__), "..", "prompts", "system_writer.md"),
        encoding="utf-8",
    ).read()

    # 字数硬性区间：目标 ±10%
    target_wc = state.words_per_chapter
    wc_min = int(target_wc * 0.9)
    wc_max = int(target_wc * 1.1)

    result = call_llm(system_prompt, user_prompt, temperature=0.8, max_tokens=8192)
    output = _strip_title(result["output"])

    # 字数超区间时重试，最多 2 次；每次都按"更接近目标"策略选用
    def _cn_count(s: str) -> int:
        return len([c for c in s if '\u4e00' <= c <= '\u9fff'])

    wc = _cn_count(output)
    for attempt in range(2):
        if wc_min <= wc <= wc_max:
            break
        direction = "少于最低" if wc < wc_min else "多于最高"
        if wc < wc_min:
            advice = f"必须扩写到 {wc_min}-{wc_max} 字，可增加对话、心理活动、环境描写、动作细节"
        else:
            advice = f"必须压缩到 {wc_min}-{wc_max} 字，删除冗余对话、合并场景、精简描写，保留核心情节"
        retry_prompt = user_prompt + f"\n\n# 字数警告（第{attempt+1}次重试）\n\n你上次的输出有 {wc} 字（{direction}要求 {wc_min}-{wc_max} 字）。{advice}。重新输出完整正文，确保字数在 {wc_min}-{wc_max} 之间。"
        result = call_llm(system_prompt, retry_prompt, temperature=0.8, max_tokens=8192)
        new_output = _strip_title(result["output"])
        new_wc = _cn_count(new_output)
        # 采用"更接近目标"的结果（即使仍超区间）
        if abs(new_wc - target_wc) < abs(wc - target_wc):
            output = new_output
            wc = new_wc

    # 兜底：超长章节（>wc_max*1.15）按段落边界硬性截断
    if wc > int(wc_max * 1.15):
        output = _truncate_to_word_count(output, wc_max)
        wc = _cn_count(output)

    # 跨事件一致性校验（轻量，零成本）
    checker = ConsistencyChecker(state)
    ok, reason = checker.check_chapter_draft(output, state.current_chapter)
    if not ok:
        # 校验失败：带原因重试一次
        retry_prompt = user_prompt + (
            f"\n\n# 一致性校验失败\n\n上次输出存在以下问题：\n{reason}\n\n"
            f"请严格沿用条目池中已有人物的名字和身份，重新输出完整正文。"
        )
        result = call_llm(system_prompt, retry_prompt, temperature=0.7, max_tokens=8192)
        new_output = _strip_title(result["output"])
        new_wc = _cn_count(new_output)
        # 采用新结果（即使字数稍差，一致性优先）
        if new_wc >= wc_min * 0.8:
            output = new_output
            wc = new_wc

    ch.content = output
    ch.word_count = wc

    return {}


def _truncate_to_word_count(text: str, max_cn: int) -> str:
    """按段落边界截断，保留完整段落，使汉字数不超过 max_cn。"""
    if not text:
        return text
    paragraphs = text.split("\n")
    result = []
    cn_count = 0
    for p in paragraphs:
        p_cn = len([c for c in p if '\u4e00' <= c <= '\u9fff'])
        if cn_count + p_cn > max_cn and result:
            break
        result.append(p)
        cn_count += p_cn
    return "\n".join(result)


def _strip_title(text: str) -> str:
    """去掉开头的 markdown 标题行（# 第X章 / # 章节名 等）"""
    if not text:
        return text
    lines = text.split("\n")
    # 跳过开头的空行和标题行
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        # 标题行：# 开头，或者"第X章"开头
        if line.startswith("#"):
            idx += 1
            continue
        if line.startswith("第") and ("章" in line or "节" in line) and len(line) < 30:
            idx += 1
            continue
        break
    return "\n".join(lines[idx:])
