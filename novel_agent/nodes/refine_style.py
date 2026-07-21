"""
节点：文风润色
把初稿按照作者文风进行润色优化。
展示：Multi-Agent / Dynamic State Updates
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from novel_agent.state import NovelState
from novel_agent.utils.llm import call_llm
from novel_agent.utils.state_adapter import node_wrapper
from novel_agent.utils.prompt_context import PromptContextBuilder


@node_wrapper
def refine_style(state: NovelState) -> dict:
    """润色章节正文"""
    if not state.pending_refinement:
        return {}

    evt = state.events.get(state.current_event)
    if not evt:
        return {"last_error": f"事件{state.current_event} 不存在"}

    ch_num = state.pending_refinement[0]
    ch = evt.chapters.get(ch_num)
    if not ch:
        state.pending_refinement.pop(0)
        return {"last_error": f"第{ch_num}章不存在"}

    # 人工确认过的章节，永远不重润色
    if ch.status == "human_confirmed":
        state.pending_refinement.pop(0)
        return {}

    if ch.refined_content:
        state.pending_refinement.pop(0)
        # 已经人工确认的章节不覆盖状态
        if ch.status != "human_confirmed":
            ch.status = "refined"
        return {}

    system_prompt = open(
        os.path.join(os.path.dirname(__file__), "..", "prompts", "system_refiner.md"),
        encoding="utf-8",
    ).read()

    style_features = state.writing_style_features or "（首次润色，请严格遵循文风规则）"
    system_prompt = system_prompt.replace("{style_features}", style_features)

    # 使用统一的上下文构建器
    builder = PromptContextBuilder(state)
    hint = (state.modification_hint or "").strip()
    user_prompt = builder.for_chapter_refine(ch_num, hint=hint)

    # 字数硬性区间：目标 ±10%
    target_wc = state.words_per_chapter
    wc_min = int(target_wc * 0.9)
    wc_max = int(target_wc * 1.1)
    orig_wc = len([c for c in (ch.content or "") if '\u4e00' <= c <= '\u9fff'])

    result = call_llm(system_prompt, user_prompt, temperature=0.6, max_tokens=8192)

    def _cn_count(s: str) -> int:
        return len([c for c in s if '\u4e00' <= c <= '\u9fff'])

    refined = _strip_title(result["output"])
    refined_wc = _cn_count(refined)

    # 润色后字数超区间：最多重试 2 次，按"更接近目标"策略选用
    for attempt in range(2):
        if wc_min <= refined_wc <= wc_max:
            break
        direction = "少于最低" if refined_wc < wc_min else "多于最高"
        fix_prompt = user_prompt + f"\n\n# 字数警告（第{attempt+1}次重试）\n\n你上次的润色输出有 {refined_wc} 字（{direction}要求 {wc_min}-{wc_max}）。请严格控制字数，重新输出润色后的正文（保持原稿情节和篇幅）。"
        fix_result = call_llm(system_prompt, fix_prompt, temperature=0.6, max_tokens=8192)
        fix_text = _strip_title(fix_result["output"])
        fix_wc = _cn_count(fix_text)
        # 采用"更接近目标"的结果
        if abs(fix_wc - target_wc) < abs(refined_wc - target_wc):
            refined = fix_text
            refined_wc = fix_wc

    # 仍超区间：回退原稿或硬性截断
    if refined_wc > wc_max or refined_wc < wc_min:
        # 比较回退原稿和当前 refined 哪个更接近目标
        if abs(orig_wc - target_wc) < abs(refined_wc - target_wc):
            refined = ch.content
            refined_wc = orig_wc
        # 仍超区间：按段落边界截断到 wc_max（不留容差，硬性保证不超上限）
        if refined_wc > wc_max:
            refined = _truncate_to_word_count(refined, wc_max)
            refined_wc = _cn_count(refined)
        # 字数不足下限时无法硬性补字，保留当前结果（字数不足比超长影响小）

    ch.refined_content = refined
    ch.word_count = refined_wc
    ch.status = "refined"

    # 移到待确认队列（防御性初始化，避免 None.append 崩溃）
    state.pending_human_review = state.pending_human_review or []
    state.pending_human_review.append(ch_num)
    state.pending_refinement.pop(0)

    # 同步写回文件
    from novel_agent.utils.file_io import save_chapter_to_file
    save_chapter_to_file(state, state.current_event, ch_num)

    return {}


def _strip_title(text: str) -> str:
    """去掉开头的 markdown 标题行（LLM 经常无视"不要写标题"指令）"""
    if not text:
        return text
    lines = text.split("\n")
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        if line.startswith("#"):
            idx += 1
            continue
        if line.startswith("第") and ("章" in line or "节" in line) and len(line) < 30:
            idx += 1
            continue
        break
    return "\n".join(lines[idx:])


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
