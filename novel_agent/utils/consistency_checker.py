"""
跨事件一致性校验器

在 LLM 输出事件纲/章节正文后调用，检查：
1. 是否引入新人物名替代已有角色
2. 已有人物的身份/职业是否被改变
3. 是否引入新势力/概念/道具名替代已有设定
4. 主线方向是否偏离

规则匹配，不调 LLM，零成本。
发现冲突时返回 (False, 原因)，由调用方决定是否重试。
"""

from __future__ import annotations
import re
from typing import Optional

from novel_agent.state import NovelState, EntryState, ENTRY_CATEGORIES


# ── 名字提取 ───────────────────────────────────────

# 标记符号内的名字：【张三】、《张三》、[张三]
_MARK_NAME_RE = re.compile(r"[【《\[]([^】》\]]{1,10})[】》\]]")
# "关键XX" 段落里 "- 张三：" 或 "- **张三**：" 模式
_KEY_ENTRY_RE = re.compile(r"^[\-\*]\s*\**([^：*]{1,10})\**\s*[：:]", re.MULTILINE)
# 常见身份关键词（用于检测身份变更）
_IDENTITY_KEYWORDS = {
    "前妻": ["前妻", "前夫人", "离异妻子"],
    "老婆": ["老婆", "妻子", "夫人", "媳妇"],
    "老太太": ["老太太", "老妇人", "老奶奶", "阿婆"],
    "修理工": ["修理工", "维修工", "电器工", "修电器的"],
    "同事": ["同事", "工友"],
    "合作者": ["合作者", "合伙人", "搭档"],
    "警察": ["警察", "刑警", "警官", "公安"],
    "学生": ["学生", "徒弟", "弟子"],
}


def _extract_names_from_text(text: str) -> set[str]:
    """从文本中提取标记符号内的名字（【】《》[]）"""
    if not text:
        return set()
    names = set()
    for m in _MARK_NAME_RE.finditer(text):
        name = m.group(1).strip()
        if 2 <= len(name) <= 8 and not _is_common_word(name):
            names.add(name)
    return names


def _extract_key_entries_from_section(plan: str, section_name: str) -> set[str]:
    """从事件纲的指定段落（如「关键人物」「关键势力」）提取条目名"""
    if not plan:
        return set()
    m = re.search(rf"##\s*{re.escape(section_name)}\s*\n(.*?)(?=\n## |\Z)", plan, re.DOTALL)
    if not m:
        return set()
    section = m.group(1)
    names = set()
    for nm in _KEY_ENTRY_RE.finditer(section):
        name = nm.group(1).strip().strip("*")
        # 去掉"主角"前缀
        if name.startswith("主角"):
            name = name[2:]
        if 2 <= len(name) <= 10 and not _is_common_word(name):
            names.add(name)
    return names


def _extract_key_persons_from_plan(plan: str) -> set[str]:
    """[兼容] 从事件纲「关键人物」段落提取人名"""
    return _extract_key_entries_from_section(plan, "关键人物")


# ── 常见非人名词过滤 ─────────────────────────────────

_NON_NAME_WORDS = {
    "主角", "陈默", "本事件", "本章", "概述", "规划", "钩子", "爽点", "注意事项",
    "憋屈", "铺垫", "主线", "支线", "冲突", "决策", "反应", "心理", "变化",
    "身份", "职业", "关系", "性格", "行动", "场景", "情节", "信息", "线索",
    "通过", "新登场", "沿用", "本事件", "前事件", "上事件", "无", "如有",
    "老钱", "老烟枪",  # 这些是绰号，单独处理
    "势力", "概念", "道具", "地点", "人物", "条目", "清单",
}


def _is_common_word(name: str) -> bool:
    """判断是否是常见非人名词"""
    return name in _NON_NAME_WORDS


# ── 校验器 ───────────────────────────────────────────

# 段落标题 → 分类 映射（用于多分类漂移检测）
_SECTION_TO_CATEGORY = [
    ("关键人物", "人物设定"),
    ("关键势力", "势力设定"),
    ("关键概念", "概念设定"),
    ("关键道具", "道具设定"),
    ("关键地点", "地点设定"),
]


class ConsistencyChecker:
    """跨事件一致性校验器"""

    def __init__(self, state: NovelState):
        self.state = state

    def _existing_entries_by_category(self, category: str) -> set[str]:
        """获取指定分类下所有已建立的条目名"""
        pool = getattr(self.state.entries, category, {})
        return set(pool.keys())

    def _existing_person_names(self) -> set[str]:
        """[兼容] 获取条目池中所有已建立的人物名"""
        return self._existing_entries_by_category("人物设定")

    def _existing_person_identities(self) -> dict[str, str]:
        """获取条目池中每个人物的身份关键词（用于检测身份变更）。

        只从 one_line 提取，因为 content 可能提到其他人的身份
        （例如陈默的 content 里有"前妻是林薇"，不能据此判定陈默是前妻）。
        """
        result = {}
        for name, entry in self.state.entries.人物设定.items():
            # 只用 one_line（一句话简介，描述的就是这个人自己）
            text = entry.one_line or ""
            for identity, keywords in _IDENTITY_KEYWORDS.items():
                if any(kw in text for kw in keywords):
                    result[name] = identity
                    break
        return result

    def check_event_plan(self, plan: str) -> tuple[bool, str]:
        """检查事件纲是否与已有设定冲突（多分类）。

        返回: (是否通过, 拒绝原因)
        通过 = True 表示无冲突；通过 = False 表示有冲突，应拒绝并重试。
        """
        if not plan:
            return True, ""

        # 按分类检测条目名漂移
        all_drifts = []
        for section_name, category in _SECTION_TO_CATEGORY:
            existing = self._existing_entries_by_category(category)
            if not existing:
                continue  # 该分类还没条目，跳过
            plan_names = _extract_key_entries_from_section(plan, section_name)
            if not plan_names:
                continue
            new_names = plan_names - existing - _NON_NAME_WORDS
            # 过滤身份词
            suspicious = []
            for nm in new_names:
                if nm in _IDENTITY_KEYWORDS or any(kw in nm for kws in _IDENTITY_KEYWORDS.values() for kw in kws):
                    continue
                if len(nm) < 2:
                    continue
                suspicious.append(nm)
            # 至少2个已有 + 2个新名 才判冲突（单人增减是正常剧情推进）
            if len(existing) >= 2 and len(suspicious) >= 2:
                all_drifts.append(
                    f"{category}：新名（{', '.join(suspicious)}）疑似替代已有（{', '.join(list(existing)[:5])}）"
                )

        if all_drifts:
            return False, "；".join(all_drifts) + "。请检查是否沿用原有条目名。"
        return True, ""

    def check_chapter_draft(self, content: str, ch_num: int) -> tuple[bool, str]:
        """检查章节正文是否引入新名字替代已有角色。

        注意：章节正文允许引入新配角（比如"老刘"、"老王"），所以判定更宽松。
        只有当正文出现明显是「替代已有角色」的新名字时才拒绝。
        """
        if not content:
            return True, ""

        existing_names = self._existing_person_names()
        if len(existing_names) < 2:
            return True, ""

        # 提取正文中的人名（仅【】标记的）
        content_names = _extract_names_from_text(content)
        new_names = content_names - existing_names - _NON_NAME_WORDS

        # 章节正文比较宽松：只有当新名字数 >= 3 且已有角色数 >= 3 时才判定冲突
        if len(new_names) >= 3 and len(existing_names) >= 3:
            return False, (
                f"正文引入了多个新人物名（{', '.join(list(new_names)[:5])}），"
                f"条目池已有角色（{', '.join(list(existing_names)[:5])}）。"
                f"疑似替换已有角色，请检查。"
            )

        return True, ""

    def check_identity_drift(self, plan: str) -> tuple[bool, str]:
        """检查事件纲中已有人物的身份是否被改变。

        只在「## 关键人物」段落里按 "- 名字：身份描述" 格式检测，
        避免概述段落里多人描述互相干扰（例如"前妻林薇...周玉兰老太太"
        会被误判为林薇的身份是老太太）。
        """
        if not plan:
            return True, ""

        existing_identities = self._existing_person_identities()
        if not existing_identities:
            return True, ""

        # 只取「## 关键人物」段落
        m = re.search(r"##\s*关键人物\s*\n(.*?)(?=\n## |\Z)", plan, re.DOTALL)
        if not m:
            return True, ""
        section = m.group(1)

        # 按行解析 "- 名字：身份描述" 格式
        drifts = []
        for line in section.split("\n"):
            line = line.strip()
            if not line or not line.startswith("-"):
                continue
            # 去掉前导 -
            line = line.lstrip("-* ").strip()
            # 找冒号分割
            if "：" not in line and ":" not in line:
                continue
            # 取冒号前的名字
            for sep in ["：", ":"]:
                if sep in line:
                    name_part = line.split(sep, 1)[0].strip().strip("*")
                    desc_part = line.split(sep, 1)[1].strip()
                    break
            else:
                continue

            # 检查这个名字是否在条目池里
            if name_part not in existing_identities:
                continue

            expected_identity = existing_identities[name_part]
            # 在 desc_part 里查找冲突的身份关键词
            for other_identity, kws in _IDENTITY_KEYWORDS.items():
                if other_identity == expected_identity:
                    continue
                if _is_identity_conflict(expected_identity, other_identity):
                    for kw in kws:
                        if kw in desc_part:
                            drifts.append(
                                f"{name_part}（条目池：{expected_identity}）"
                                f"在事件纲关键人物段落中被描述为「{other_identity}」"
                            )
                            break

        if drifts:
            return False, "身份漂移：" + "；".join(drifts)
        return True, ""


def _is_identity_conflict(a: str, b: str) -> bool:
    """判断两个身份是否互斥（不能同时成立）"""
    conflict_pairs = {
        frozenset({"前妻", "老婆"}),
        frozenset({"老太太", "老婆"}),
        frozenset({"老太太", "前妻"}),
        frozenset({"同事", "老婆"}),
        frozenset({"同事", "前妻"}),
        frozenset({"合作者", "前妻"}),
        frozenset({"学生", "老婆"}),
        frozenset({"修理工", "警察"}),
    }
    return frozenset({a, b}) in conflict_pairs
