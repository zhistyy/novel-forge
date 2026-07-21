"""
统一的提示词上下文构建器

所有节点（事件纲/章节规划/正文/润色/质检）通过本模块获取 LLM 上下文，
避免分散构建导致跨事件设定丢失。

6 层组织：
  L1 全局：plot + tone + writing_style_features
  L2 累积设定：完整条目池（按节点裁剪）
  L3 主线进度：已完成事件概述链
  L4 事件纲：前事件纲 + 当前事件纲
  L5 章节层：章节规划 + 上一章末尾
  L6 任务指令：字数要求 + 用户修改意见
"""

from __future__ import annotations
import re
from typing import Optional

from novel_agent.state import NovelState, EventState, ChapterState, EntryPool


# ── 工具函数 ─────────────────────────────────────────

def _extract_plan_overview(plan: str) -> str:
    """从事件纲中抽取概述段落（避免传全文撑爆上下文）"""
    if not plan:
        return ""
    m = re.search(r"## 概述\s*\n(.*?)(?=\n## |\Z)", plan, re.DOTALL)
    if m:
        return m.group(1).strip()
    return plan.strip()[:300]


def _cn_count(s: str) -> int:
    return len([c for c in (s or "") if '\u4e00' <= c <= '\u9fff'])


# ── 上下文构建器 ─────────────────────────────────────

class PromptContextBuilder:
    """统一的提示词上下文构建器。

    所有节点通过本类获取 user_prompt，确保跨事件状态完整传递。
    """

    def __init__(self, state: NovelState):
        self.state = state

    # ── L1 全局块 ──
    def _global_block(self, include_plot: bool = True, plot_max: int = 3000) -> str:
        """plot + tone + writing_style_features"""
        parts = []
        if include_plot:
            plot = self.state.plot or ""
            if len(plot) > plot_max:
                plot = plot[:plot_max] + "\n…（截断）"
            parts.append(f"# 全书剧情（大主线）\n\n{plot or '（未提供）'}")
        if self.state.tone:
            parts.append(f"# 基调\n\n{self.state.tone}")
        if self.state.writing_style_features:
            parts.append(f"# 作者风格特征\n\n{self.state.writing_style_features}")
        return "\n\n".join(parts)

    # ── L2 条目池块 ──
    def _entries_block(self, mode: str = "all") -> str:
        """条目池块。

        mode:
          "all"        — 全部条目（一句话摘要），用于事件纲/章节规划
          "current"    — 当前事件相关的条目（完整正文），用于正文/润色
          "summary"    — 全部条目摘要（同 all），用于质检
          "full_all"   — 全部条目完整正文（仅在条目较少时使用）
        """
        s = self.state
        if mode == "current":
            evt = s.events.get(s.current_event)
            evt_name = evt.event_name if evt else ""
            entries = s.entries.get_entries_for_event(evt_name)
            if not entries:
                # 没有标记 appears_in 的条目时，回退到全部条目摘要
                summary = s.entries.all_entries_summary().strip()
                return f"# 相关条目（未标记，回退全摘要）\n\n{summary or '（尚无条目）'}"
            blocks = [e.to_prompt_block() for e in entries]
            return "# 相关条目（完整设定，必须严格沿用）\n\n" + "\n\n".join(blocks)
        if mode == "full_all":
            entries = s.entries.all_entries()
            if not entries:
                return "# 已建立的条目池\n\n（尚无条目）"
            blocks = [e.to_prompt_block() for e in entries]
            return "# 已建立的条目池（完整设定，必须严格沿用）\n\n" + "\n\n".join(blocks)
        # all / summary
        summary = s.entries.all_entries_summary().strip()
        return f"# 已建立的条目池（必须严格沿用，不得改名、不得改身份）\n\n{summary or '（尚无已建立条目）'}"

    # ── L3 进度块 ──
    def _progress_block(self, until_event: int) -> str:
        """已完成事件的概述摘要链（事件1 ~ until_event-1）"""
        if until_event <= 1:
            return "# 已完成事件进度摘要\n\n（本事件是第一个事件）"
        lines = []
        for n in range(1, until_event):
            evt = self.state.events.get(n)
            if evt and evt.plan:
                ov = _extract_plan_overview(evt.plan)
                if ov:
                    lines.append(f"### 事件{n}\n{ov}")
        if not lines:
            return "# 已完成事件进度摘要\n\n（无）"
        return "# 已完成事件进度摘要\n\n" + "\n\n".join(lines)

    # ── L4 事件纲块 ──
    def _prev_event_plan_block(self, evt: EventState) -> str:
        """上一事件的事件纲全文"""
        if evt.event_num <= 1:
            return "# 上一事件的事件纲（设定与人物必须延续）\n\n（本事件是第一个事件）"
        prev = self.state.events.get(evt.event_num - 1)
        if not prev or not prev.plan:
            return "# 上一事件的事件纲（设定与人物必须延续）\n\n（无）"
        return f"# 上一事件的事件纲（设定与人物必须延续）\n\n{prev.plan.strip()}"

    def _current_event_plan_block(self, evt: EventState, max_chars: int = 2500) -> str:
        """当前事件纲"""
        plan = (evt.plan or "").strip()
        if not plan:
            return "# 当前事件纲\n\n（尚未生成）"
        if len(plan) > max_chars:
            plan = plan[:max_chars] + "\n…（截断）"
        return f"# 当前事件纲\n\n{plan}"

    # ── L5 章节层块 ──
    def _prev_chapter_tail_block(self, ch_num: int, length: int = 500) -> str:
        """上一章末尾正文（跨事件也能取到）"""
        prev_num = ch_num - 1
        if prev_num < 1:
            return "# 上一章末尾\n\n（本章是第一章）"
        # 找到 prev_num 所属的事件
        for evt in self.state.events.values():
            ch = evt.chapters.get(prev_num)
            if ch:
                tail = (ch.refined_content or ch.content or "")[-length:]
                if tail:
                    return f"# 上一章末尾\n\n{tail}"
                return "# 上一章末尾\n\n（上一章无正文）"
        return "# 上一章末尾\n\n（无）"

    def _chapter_plan_block(self, ch: ChapterState) -> str:
        """章节规划"""
        return f"# 本章规划\n\n{ch.plan or '（无）'}"

    # ── L6 任务指令块 ──
    def _task_block(self, task_desc: str, hint: str = "") -> str:
        parts = [task_desc]
        if hint:
            hint = hint.strip()
            if hint:
                parts.append(f"# 用户修改意见（请优先满足）\n\n{hint}\n\n请基于以上意见调整。")
        return "\n\n".join(parts)

    # ── 节点级上下文构建 ──────────────────────────────

    def for_event_plan(self, evt: EventState, hint: str = "") -> str:
        """事件纲生成的 user_prompt"""
        start_ch, end_ch = evt.chapter_range
        chapters_count = end_ch - start_ch + 1 if end_ch >= start_ch else 10
        parts = [
            self._global_block(include_plot=True),
            self._entries_block(mode="all"),
            self._progress_block(until_event=evt.event_num),
            self._prev_event_plan_block(evt),
            self._prev_chapter_tail_block(start_ch, length=400),
            f"# 要规划的事件\n\n事件{evt.event_num}，共 {chapters_count} 章（第{start_ch}-{end_ch}章）\n每章字数要求：约{self.state.words_per_chapter}字",
            self._task_block(
                "请按格式输出本事件的事件纲。\n\n**硬性要求**：\n"
                "1. 必须沿用\"已建立的条目池\"中的人物名字、身份、关系，不得创造新名字替代已有角色\n"
                "2. 必须延续\"上一事件的事件纲\"中的人物设定和主线推进方向\n"
                "3. 不得让已有人物突然换身份（例如前妻不能变成合作者、老太太不能变成老婆）\n"
                "4. 主线必须沿着\"已完成事件进度摘要\"自然推进，不得跳跃到无关题材",
                hint=hint,
            ),
        ]
        return "\n\n".join(parts)

    def for_chapter_plan(self, ch_num: int, hint: str = "") -> str:
        """章节规划的 user_prompt"""
        s = self.state
        evt = s.events.get(s.current_event)
        if not evt:
            return ""
        ch = evt.chapters.get(ch_num)
        parts = [
            self._global_block(include_plot=True, plot_max=2000),
            self._entries_block(mode="all"),
            self._progress_block(until_event=evt.event_num),
            self._current_event_plan_block(evt),
            self._prev_chapter_tail_block(ch_num, length=500),
            f"# 要规划的章节\n\n事件{evt.event_num}，第{ch_num}章\n字数要求：约{s.words_per_chapter}字",
            self._task_block(
                "请按格式输出本章规划：核心场景 + 关键情节 + 章末钩子。\n"
                "**硬性要求**：必须沿用条目池中的人物名字和身份，不得改名或改身份。",
                hint=hint,
            ),
        ]
        return "\n\n".join(parts)

    def for_chapter_draft(self, ch_num: int, hint: str = "") -> str:
        """章节正文的 user_prompt"""
        s = self.state
        evt = s.events.get(s.current_event)
        if not evt:
            return ""
        ch = evt.chapters.get(ch_num)
        if not ch:
            return ""

        target_wc = s.words_per_chapter
        wc_min = int(target_wc * 0.9)
        wc_max = int(target_wc * 1.1)

        # 抽取本章在事件纲中的规划
        plan_for_chapter = ""
        if evt.plan:
            import re as _re
            m = _re.search(rf"[-*]?\s*第\s*0*{ch_num}\s*章[：:][^\n]*", evt.plan)
            if m:
                plan_for_chapter = m.group(0).strip()

        parts = [
            self._chapter_plan_block(ch),
            self._global_block(include_plot=True, plot_max=2000),
            self._entries_block(mode="current"),
            self._progress_block(until_event=evt.event_num),
            self._current_event_plan_block(evt, max_chars=2000),
            self._prev_chapter_tail_block(ch_num, length=500),
        ]
        if plan_for_chapter:
            parts.append(
                f"# 本章事件纲规划（必须严格遵循）\n\n{plan_for_chapter}\n\n"
                "**硬性要求**：\n"
                "- 必须实现上述规划中的核心情节\n"
                "- 章末钩子必须与规划一致（允许措辞调整，不允许主题更换）\n"
                "- 不得引入事件纲关键人物段落之外的新主要角色\n"
                "- 不得跳过规划中的关键场景"
            )
        parts.append(
            self._task_block(
                f"# 写作要求\n\n"
                f"- **字数：必须在 {wc_min}-{wc_max} 字之间（目标 {target_wc} 字，按汉字计数）**\n"
                f"- 事件：{evt.event_name}\n"
                f"- 章节：第{ch_num}章\n"
                f"- **必须沿用条目池中已有人物的名字和身份，不得改名或改身份**\n"
                f"- **必须严格按本章事件纲规划写作，不得偏离主线**\n\n"
                f"直接输出第{ch_num}章正文，不要解释，不要写章节标题。",
                hint=hint,
            ),
        )
        return "\n\n".join(parts)

    def for_chapter_refine(self, ch_num: int, hint: str = "") -> str:
        """章节润色的 user_prompt"""
        s = self.state
        evt = s.events.get(s.current_event)
        if not evt:
            return ""
        ch = evt.chapters.get(ch_num)
        if not ch:
            return ""

        target_wc = s.words_per_chapter
        wc_min = int(target_wc * 0.9)
        wc_max = int(target_wc * 1.1)
        orig_wc = _cn_count(ch.content or "")

        parts = [
            f"# 需要润色的正文\n\n{ch.content or ''}",
            self._entries_block(mode="current"),
            self._chapter_plan_block(ch),
            f"# 字数要求（重要）\n\n- 原稿字数：{orig_wc} 字\n- 润色后字数必须保持在 {wc_min}-{wc_max} 字之间（目标 {target_wc} 字）\n- 不得添加新情节，不得扩写或缩写，只优化表达\n- 字数统计按汉字计算（标点、英文、数字不计）\n- **必须沿用条目池中已有人物的名字和身份**",
        ]
        if hint:
            hint = hint.strip()
            if hint:
                parts.append(f"# 用户修改意见（请优先满足）\n\n{hint}\n\n请基于以上意见调整润色方向。")
        return "\n\n".join(parts)

    def for_chapter_review(self, ch_num: int) -> str:
        """章节质检的 user_prompt"""
        s = self.state
        evt = s.events.get(s.current_event)
        if not evt:
            return ""
        ch = evt.chapters.get(ch_num)
        if not ch:
            return ""

        # 抽取本章在事件纲中的规划（按章号匹配章节规划段落）
        plan_for_chapter = ""
        if evt.plan:
            # 匹配 "- 第N章：xxx" 或 "第N章：xxx" 行
            import re as _re
            m = _re.search(rf"[-*]?\s*第\s*0*{ch_num}\s*章[：:][^\n]*", evt.plan)
            if m:
                plan_for_chapter = m.group(0).strip()

        parts = [
            f"# 章节正文\n\n{ch.content or ''}",
            self._entries_block(mode="summary"),
            self._current_event_plan_block(evt, max_chars=1500),
            self._progress_block(until_event=evt.event_num),
        ]
        if plan_for_chapter:
            parts.append(
                f"# 本章事件纲规划（必须遵循）\n\n{plan_for_chapter}\n\n"
                "**事件纲偏离检查**：本章正文是否实现了上述规划的核心情节和章末钩子？\n"
                "- 如果正文明显偏离规划（例如：规划是A场景，正文写成B场景；规划章末钩子是X，正文写成了Y），必须 FAIL\n"
                "- 如果只是细节调整（例如：对话措辞不同、次要动作不同），不构成偏离，可以 PASS"
            )
        parts.append(
            "# 检查要求\n\n请按质检清单逐项检查。第一行必须输出 PASS 或 FAIL 标记。\n"
            "**额外检查**：\n"
            "- 章节中的人物名字是否与条目池一致（不得出现条目池之外的新人名替代已有角色）\n"
            "- 已有人物的身份/职业是否与条目池一致\n"
            "- 章节是否引入了事件纲关键人物段落之外的新主要角色（如果引入且无铺垫，必须 FAIL）\n"
            "- 如发现跨事件设定漂移，必须 FAIL"
        )
        return "\n\n".join(parts)
