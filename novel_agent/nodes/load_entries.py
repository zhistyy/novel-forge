"""
节点：按需加载条目
当事件开始时，检查该事件需要哪些条目，自动加载缺失的。
同时如果事件纲是空或模板，先调用 LLM 生成事件纲。
展示：State Schema / Dynamic Updates / Conditional Edge
"""

from __future__ import annotations
import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from novel_agent.utils.state_adapter import node_wrapper
from novel_agent.utils.llm import call_llm
from novel_agent.utils.prompt_context import PromptContextBuilder
from novel_agent.utils.consistency_checker import ConsistencyChecker
from novel_agent.state import NovelState, EntryState, EventState


# 模板事件纲的特征文本（来自 orchestrator._tool_create_project）
_TEMPLATE_MARKERS = ["（待填写）", "待规划"]


def _update_event_chapter_range(state: NovelState, evt: EventState):
    """从事件纲正文解析 chapter_range；解析失败时用 total_chapters/total_events 兜底。

    避免 chapter_range=(0,0) 导致 _get_next_step_type 无法推进事件。
    """
    import re as _re
    plan_content = evt.plan or ""
    # 优先匹配"## 章节规划（第X-Y章）"标题内的范围，避免误匹配概述里"承接第1-2章"等
    m = _re.search(r"章节规划\s*[（(]\s*第\s*(\d+)\s*-\s*(\d+)\s*章\s*[）)]", plan_content)
    if m:
        evt.chapter_range = (int(m.group(1)), int(m.group(2)))
        return
    # 回退：匹配任意"第X-Y章"，取最后一个（章节规划通常在文档后部）
    matches = _re.findall(r"第\s*(\d+)\s*-\s*(\d+)\s*章", plan_content)
    if matches:
        evt.chapter_range = (int(matches[-1][0]), int(matches[-1][1]))
        return
    # 兜底：按 total_chapters / total_events 均匀分配（最后一个事件吸收余数）
    total_e = state.total_events or 1
    total_c = state.total_chapters or 0
    cpe = max(1, total_c // total_e) if total_e > 0 else total_c
    i = evt.event_num
    start = (i - 1) * cpe + 1
    end = total_c if i == total_e else i * cpe
    evt.chapter_range = (start, end)


def _is_template_plan(plan: str) -> bool:
    """检测事件纲是否是初始模板"""
    if not plan or not plan.strip():
        return True
    # 短到只有标题级
    if len(plan.strip()) < 80:
        return True
    # 含模板标记且没有实质概述
    if any(marker in plan for marker in _TEMPLATE_MARKERS):
        # 简单判定：如果模板标记出现且没有"概述"段落实质内容
        if "## 概述" in plan:
            after = plan.split("## 概述", 1)[-1]
            # 取概述段落（到下一个 ## 或末尾）
            next_section = after.find("## ")
            overview = after[:next_section] if next_section > 0 else after
            overview = overview.strip()
            # 概述字数少于 50 视为模板
            if len(overview) < 50 or any(m in overview for m in _TEMPLATE_MARKERS):
                return True
        else:
            return True
    return False


def _generate_event_plan(state: NovelState, evt: EventState) -> str:
    """调用 LLM 生成事件纲

    使用 PromptContextBuilder 构建完整上下文（含条目池+前事件纲+进度摘要），
    生成后用 ConsistencyChecker 校验，发现冲突则带原因重试，最多 2 次。
    """
    system_prompt = open(
        os.path.join(os.path.dirname(__file__), "..", "prompts", "system_event_planner.md"),
        encoding="utf-8",
    ).read()

    builder = PromptContextBuilder(state)
    checker = ConsistencyChecker(state)
    hint = (state.modification_hint or "").strip()

    user_prompt = builder.for_event_plan(evt, hint=hint)

    # 第一次生成
    result = call_llm(system_prompt, user_prompt, temperature=0.7, max_tokens=4096)
    plan_text = result["output"]

    # 一致性校验 + 重试（最多 2 次）
    for attempt in range(2):
        ok_name, name_reason = checker.check_event_plan(plan_text)
        ok_identity, identity_reason = checker.check_identity_drift(plan_text)
        if ok_name and ok_identity:
            break
        # 校验失败：把拒绝原因加入 prompt，重新生成
        reasons = []
        if not ok_name:
            reasons.append(name_reason)
        if not ok_identity:
            reasons.append(identity_reason)
        retry_prompt = user_prompt + (
            f"\n\n# 一致性校验失败（第{attempt+1}次重试）\n\n"
            f"上次输出存在以下问题：\n- " + "\n- ".join(reasons) +
            "\n\n请严格沿用条目池中的人物名字和身份，重新输出完整事件纲。"
        )
        result = call_llm(system_prompt, retry_prompt, temperature=0.6, max_tokens=4096)
        # 只在新结果通过校验时采用，否则保留旧结果（避免越改越糟）
        new_plan = result["output"]
        new_ok_name, _ = checker.check_event_plan(new_plan)
        new_ok_identity, _ = checker.check_identity_drift(new_plan)
        if (new_ok_name and new_ok_identity) or attempt == 1:
            plan_text = new_plan

    return plan_text


@node_wrapper
def load_entries_for_event(state: NovelState) -> dict:
    """
    按需加载：检查当前事件的 appears_in，加载所有关联条目。
    如果事件纲是空或模板，先调用 LLM 生成事件纲。
    如果有条目不存在，标记待创建。
    """
    evt = state.events.get(state.current_event)
    if not evt:
        return {"last_error": f"事件{state.current_event} 不存在"}

    event_name = f"事件{state.current_event}"

    # ── 先检测事件纲是否需要生成 ──
    if _is_template_plan(evt.plan):
        new_plan = _generate_event_plan(state, evt)
        if new_plan and not new_plan.startswith("（"):
            # 加上标题行
            if not new_plan.lstrip().startswith("#"):
                new_plan = f"# 事件{state.current_event}\n\n{new_plan}"
            evt.plan = new_plan
            evt.status = "planned"
            # 解析 chapter_range（LLM 生成的事件纲可能不含"第X-Y章"格式，
            # 不设置会导致 _get_next_step_type 无法推进事件）
            _update_event_chapter_range(state, evt)
        else:
            # 生成失败：返回错误，让 step_engine 决定是否重试
            return {"last_error": f"事件{state.current_event}纲生成失败，LLM 返回为空或异常"}
    else:
        # 事件纲已就绪：确保 chapter_range 已正确解析（磁盘加载时已解析，
        # 但如果是从 DB 恢复或早期创建的事件，可能还是 (0,0)）
        if evt.chapter_range == (0, 0):
            _update_event_chapter_range(state, evt)

    # ── 加载条目 ──
    loaded = state.entries.get_entries_for_event(event_name)

    # 扫描事件纲的所有"关键XX"段落，提取条目池中没有的新条目
    evt_plan = evt.plan
    if evt_plan:
        new_entries_needed = _detect_new_entries(state, evt_plan, event_name)
        if new_entries_needed:
            from novel_agent.state import EntryState
            from novel_agent.utils.file_io import save_entry_to_file, get_project_dir
            for item in new_entries_needed:
                entry = EntryState(
                    name=item["name"],
                    category=item["category"],
                    one_line=item["one_line"],
                    content=item["content"],
                    version=1,
                    appears_in=[event_name],
                )
                state.entries.upsert(item["category"], entry)
                # 落盘到文件
                try:
                    save_entry_to_file(state, item["category"], entry)
                except Exception:
                    pass
            state.pending_entry_updates.extend(new_entries_needed)

    # 标记现有条目出现在本事件（用于按需加载）
    for entry in state.entries.get_entries_for_event(event_name):
        entry.touch_appears_in(event=state.current_event)

    state.events[state.current_event].status = "entries_loaded"
    return {"last_error": ""}


# 段落标题 → 分类映射（事件纲扫描用）
_SECTION_HEADING_MAP = [
    (r"##\s*关键人物", "人物设定"),
    (r"##\s*关键势力", "势力设定"),
    (r"##\s*关键概念", "概念设定"),
    (r"##\s*关键道具", "道具设定"),
    (r"##\s*关键地点", "地点设定"),
]


def _detect_new_entries(state: NovelState, evt_plan: str, event_name: str) -> list[dict]:
    """检测事件纲中是否有条目池中没有的新条目（多分类扫描）。

    扫描 "## 关键人物" / "## 关键势力" / "## 关键概念" / "## 关键道具" / "## 关键地点" 段落，
    按 "- 名字（注释）：描述" 格式提取，描述作为 one_line。
    """
    # 收集所有已有条目名
    existing_names = set()
    from novel_agent.state import ENTRY_CATEGORIES
    for cat in ENTRY_CATEGORIES:
        pool = getattr(state.entries, cat, {})
        existing_names.update(pool.keys())

    # (name, category, one_line)
    mentions: list[tuple[str, str, str]] = []

    # 按分类扫描各个 "## 关键XX" 段落
    for pat, cat in _SECTION_HEADING_MAP:
        m = re.search(pat + r"\s*\n(.*?)(?=\n## |\Z)", evt_plan, re.DOTALL)
        if not m:
            continue
        section = m.group(1)
        for line in section.split("\n"):
            line = line.strip()
            if not line.startswith("-"):
                continue
            body = line[1:].strip()
            # 匹配：**名字**（注释）：描述 / 名字（注释）：描述 / 名字：描述
            # 兼容 "主角陈默：xxx"（前缀"主角"会被去掉）
            pm = re.match(
                r"\*{0,2}\s*([^\s*（(：:]{1,8})\s*[\*]{0,2}\s*(?:[（(]([^）)]*)[）)])?\s*[\*]{0,2}\s*[：:]\s*(.+)",
                body,
            )
            if not pm:
                continue
            raw_name = pm.group(1).strip().strip("*")
            note = (pm.group(2) or "").strip()
            desc = pm.group(3).strip()

            # 去掉"主角"前缀
            name = raw_name
            if name.startswith("主角"):
                name = name[2:]

            if not name or name in {"主角", "新登场", "沿用", "无"}:
                continue
            if len(name) > 10:
                continue
            # 过滤明显不是名字的关键词
            if name in {"主线核心", "主线", "事件", "大主线", "事件分布", "硬约束",
                        "爽点", "爽点布局", "注意事项", "概述", "章节规划"}:
                continue

            # 注释里如果含"沿用"，说明是沿用条目，不需要新建
            if "沿用" in note:
                continue

            mentions.append((name, cat, desc[:120]))

    new_items = []
    seen_names = set()
    for name, cat, one_line in mentions:
        if not name or name in existing_names or name in seen_names:
            continue
        seen_names.add(name)
        new_items.append({
            "name": name,
            "category": cat,
            "one_line": one_line or f"（待设定管理员补充）",
            "content": f"{name}：{one_line}" if one_line else f"{name}（待补充）",
            "action": "create",
            "event": event_name,
        })

    return new_items
