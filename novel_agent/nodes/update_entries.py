"""
节点：条目池更新
在 review PASS 后扫描本章正文，提取新条目 + 更新已有条目。
展示：State Update（基于内容增量更新条目池）
"""

from __future__ import annotations
import sys, os, re, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from novel_agent.state import NovelState, EntryState, ENTRY_CATEGORIES
from novel_agent.utils.llm import call_llm
from novel_agent.utils.state_adapter import node_wrapper
from novel_agent.utils.file_io import save_entry_to_file


_SYSTEM_PROMPT = """角色：条目池维护员

你负责在每章正文写完后扫描内容，识别需要新增或更新的设定条目，确保条目池与小说内容保持同步。

## 工作原则

1. 保守原则：只提取"有名字且回收多次"的实体。一次性路人/工具人不建条目
2. 分类原则：
   - 人物设定：有名字的角色（主角、配角、反派）
   - 势力设定：组织、帮派、公司、团队
   - 概念设定：核心术语、研究项目、特殊规则
   - 道具设定：关键物品（钥匙、U盘、保险柜、武器等）
   - 地点设定：主要场景（仓库、研究所、市场等）
3. 更新原则：已有条目如果在本章获得了新信息（性格补充、身份变化、状态改变），才更新
4. 不重复原则：如果条目池已有该名字，不要在 new_entries 中重复创建

## 输出格式（严格 JSON）

只输出一个 JSON 对象，不要任何解释、markdown 代码块、前后缀文字：

{
  "new_entries": [
    {
      "name": "黑夹克头目",
      "category": "人物设定",
      "one_line": "不明势力首领，四十多岁戴墨镜，冷酷暴力",
      "content": "黑夹克头目：不明势力的首领，四十多岁，戴墨镜，说话带冷笑，持枪威胁。"
    }
  ],
  "updates": [
    {
      "name": "陈默",
      "category": "人物设定",
      "reason": "学会使用枪支",
      "append_content": "在第12章中从打手手中抢到枪，开始具备持枪作战能力。"
    }
  ]
}

## 判断标准

创建新条目：
- 人物：有名字 + 至少出现在2个场景中 + 有性格/身份描写
- 势力：有名字 + 多次出现 + 有组织特征
- 概念：被多次提及 + 有定义
- 道具：关键物品 + 推动情节
- 地点：主要场景 + 多次出现

更新已有条目：
- 身份/状态变化（如"假死"→"现身"）
- 新能力/新信息（如"学会用枪"）
- 关系变化（如"敌对"→"合作"）

不创建/不更新：
- 一次性路人（如"出租车司机"只出现一次）
- 已有条目无新信息
- 没有名字的群体（如"打手们"）

## 输出约束

- 严格 JSON，不要 markdown 代码块标记
- 如果没有任何新条目和更新，输出 {"new_entries": [], "updates": []}
- 不要包含注释、说明文字、思考过程"""


def _build_user_prompt(state: NovelState, ch_num: int) -> str:
    """构建 user_prompt：本章正文 + 现有条目池摘要 + 当前事件名"""
    evt = state.events.get(state.current_event)
    if not evt:
        return ""
    event_name = f"事件{state.current_event}"
    ch = evt.chapters.get(ch_num)
    content = (ch.content or "").strip() if ch else ""

    # 现有条目池摘要（按分类）
    pool_lines = []
    for cat in ENTRY_CATEGORIES:
        pool = getattr(state.entries, cat, {})
        if not pool:
            continue
        pool_lines.append(f"## {cat}")
        for name, e in pool.items():
            pool_lines.append(f"- {name} (v{e.version})：{e.one_line}")
    pool_summary = "\n".join(pool_lines) if pool_lines else "（条目池为空）"

    return f"""## 当前章节
事件：{event_name}
章节号：第{ch_num}章

## 本章正文
{content}

## 现有条目池（已存在的条目，更新时按 name 匹配）
{pool_summary}

## 任务
扫描本章正文，按需创建新条目或更新已有条目。严格按 JSON 格式输出，不要任何额外文字。"""


def _parse_json_response(text: str) -> dict:
    """容错解析 LLM 返回的 JSON"""
    if not text:
        return {"new_entries": [], "updates": []}
    text = text.strip()
    # 去掉可能的 markdown 代码块标记
    if text.startswith("```"):
        # 去掉 ```json 或 ``` 开头
        text = re.sub(r"^```(?:json)?\s*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    # 找到第一个 { 和最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return {"new_entries": [], "updates": []}
    json_str = text[start:end + 1]
    try:
        return json.loads(json_str)
    except Exception:
        return {"new_entries": [], "updates": []}


def _apply_updates(state: NovelState, parsed: dict, ch_num: int) -> dict:
    """把解析结果应用到 state.entries，并落盘"""
    event_num = state.current_event
    event_name = f"事件{event_num}"
    new_count = 0
    update_count = 0
    skipped = []

    # 处理新条目
    for item in parsed.get("new_entries", []):
        name = (item.get("name") or "").strip()
        cat = (item.get("category") or "").strip()
        one_line = (item.get("one_line") or "").strip()
        content = (item.get("content") or "").strip()
        if not name or not cat or cat not in ENTRY_CATEGORIES:
            skipped.append({"name": name, "reason": "字段缺失或分类非法"})
            continue
        # 已存在则跳过
        existing = state.entries.get(cat, name)
        if existing:
            skipped.append({"name": name, "reason": "条目已存在"})
            continue
        if not one_line:
            one_line = content[:80] if content else "（待补充）"
        if not content:
            content = f"{name}：{one_line}"
        entry = EntryState(
            name=name,
            category=cat,
            one_line=one_line[:120],
            content=content,
            version=1,
            appears_in=[event_name],
            change_history=[{
                "v": 1, "event": event_num, "chapter": ch_num,
                "reason": "本章首次登场，自动创建",
                "ts": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }],
        )
        state.entries.upsert(cat, entry)
        try:
            save_entry_to_file(state, cat, entry)
        except Exception:
            pass
        new_count += 1

    # 处理已有条目更新
    for item in parsed.get("updates", []):
        name = (item.get("name") or "").strip()
        cat = (item.get("category") or "").strip()
        reason = (item.get("reason") or "本章新信息").strip()
        append_content = (item.get("append_content") or "").strip()
        if not name or not cat or not append_content:
            skipped.append({"name": name, "reason": "更新字段缺失"})
            continue
        existing = state.entries.get(cat, name)
        if not existing:
            skipped.append({"name": name, "reason": "条目池中未找到"})
            continue
        # 追加 content + 调用 update 写历史
        new_content = existing.content.rstrip() + "\n" + append_content
        existing.update(
            reason=reason,
            event=event_num,
            chapter=ch_num,
            new_content=new_content,
            extra_appears_in=event_name,
        )
        try:
            save_entry_to_file(state, cat, existing)
        except Exception:
            pass
        update_count += 1

    return {
        "new_entries_created": new_count,
        "existing_entries_updated": update_count,
        "skipped": skipped,
    }


@node_wrapper
def update_entries_after_chapter(state_dict: dict) -> dict:
    """扫描本章正文，更新条目池。"""
    state: NovelState = state_dict  # type: ignore
    ch_num = state.current_chapter
    evt = state.events.get(state.current_event)
    if not evt:
        return {"last_error": f"事件{state.current_event} 不存在", "entry_updates": {}}
    ch = evt.chapters.get(ch_num)
    if not ch or not ch.content:
        return {"last_error": "本章无正文，跳过条目更新", "entry_updates": {}}

    # review FAIL 的章节不更新条目（避免污染）
    if ch.review_feedback and not _passed_review(ch.review_feedback):
        return {"last_error": "本章 review 未通过，跳过条目更新", "entry_updates": {}}

    user_prompt = _build_user_prompt(state, ch_num)
    result = call_llm(
        _SYSTEM_PROMPT,
        user_prompt,
        temperature=0.3,
        max_tokens=2048,
    )
    parsed = _parse_json_response(result.get("output", ""))
    stats = _apply_updates(state, parsed, ch_num)

    # 写入 pending_entry_updates 便于 UI 展示
    state.pending_entry_updates.append({
        "chapter": ch_num,
        "event": state.current_event,
        "stats": stats,
    })

    return {
        "last_error": "",
        "entry_updates": stats,
        "llm_output": result.get("output", "")[:500],
    }


def _passed_review(review_feedback: str) -> bool:
    """review_feedback 含 PASS 标记视为通过"""
    if not review_feedback:
        return True
    head = review_feedback.lstrip()[:200].upper()
    first_line = review_feedback.lstrip().split("\n", 1)[0].strip().strip("*#-").upper()
    if first_line.startswith("PASS"):
        return True
    if first_line.startswith("FAIL"):
        return False
    if re.search(r"不通过|FAIL", head):
        return False
    if re.search(r"通过[：:]\s*\**\s*是\b|PASS", head):
        return True
    return True  # 无法判定时默认通过
