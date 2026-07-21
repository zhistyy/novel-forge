"""
StepEngine — 分步执行引擎

LangGraph 的每一步 Node 执行被拆成离散的 Step，
每步执行完 pause，前端展示结果，用户确认后继续。

架构:
  StepEngine
    ├─ WORKFLOW: 预定义的步骤顺序 [load, plan, draft, review, refine, confirm]
    ├─ state: NovelState (持久化在内存)
    ├─ steps: list[Step] (已完成 + 当前)
    └─ status: idle → running → paused → running → paused → ... → completed

  每步状态:
    pending: 还没执行
    running: 正在执行中
    paused: ✅ 执行完，等用户确认
    completed: 用户已确认通过
    error: ❌ 出错
    skipped: - 不需要执行（如已有条目无需加载）
"""

from __future__ import annotations

import sys, os, time, re, threading, json
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from novel_agent.state import NovelState
from novel_agent.utils.file_io import load_project_to_state, save_chapter_to_file


# ── LangGraph 节点函数（已带 @node_wrapper，接受 dict 返回 dict）──

import novel_agent.nodes.load_entries as _load_mod
import novel_agent.nodes.plan_chapter as _plan_mod
import novel_agent.nodes.draft_chapter as _draft_mod
import novel_agent.nodes.review_chapter as _review_mod
import novel_agent.nodes.refine_style as _refine_mod
import novel_agent.nodes.update_entries as _update_mod
from novel_agent.utils.state_adapter import _ensure_dataclass_state
from novel_agent.nodes.review_chapter import _parse_review_verdict


# ── Step 数据类 ──

@dataclass
class Step:
    """单步执行记录"""
    id: str                          # step_1, step_2, ...
    type: str                        # load_entries|plan|draft|review|refine|confirm
    status: str                      # pending|running|completed|paused|error|skipped
    title: str                       # 用户可见的简短名称
    description: str                 # 简短描述
    summary: str = ""                # 执行结果摘要（如"2个条目已加载"）
    preview: str = ""                # 结果预览（300字）
    detail: str = ""                 # 完整结果（详情面板展示）
    meta: dict = field(default_factory=dict)   # 字数、条目数等
    actions: list = field(default_factory=list)  # 当前可用操作
    error: str = ""
    duration_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id, "type": self.type, "status": self.status,
            "title": self.title, "description": self.description,
            "summary": self.summary, "preview": self.preview, "detail": self.detail,
            "meta": self.meta, "actions": self.actions, "error": self.error,
            "duration_ms": self.duration_ms,
        }


# ── 步骤定义 ──

STEP_DEFS = {
    "load_entries": ("加载条目", "加载当前事件所需的设定条目"),
    "plan": ("生成规划", "为本章生成详细写作规划"),
    "draft": ("起草正文", "根据规划写出完整章节正文"),
    "review": ("质检", "检查章节质量和设定一致性"),
    "update_entries": ("更新条目池", "扫描本章正文，新增/更新设定条目"),
    "refine": ("文风润色", "按作者风格优化正文"),
    "confirm": ("等待确认", "人工审阅并确认本章"),
}


# ── 执行引擎 ──

class StepEngine:
    """分步执行引擎"""

    def __init__(self):
        self.project: str = ""
        self.state: Optional[NovelState] = None
        self.steps: list[Step] = []
        self.current_index: int = 0
        self.status: str = "idle"  # idle|running|paused|completed|error
        self.error: str = ""
        self._modification: str = ""  # 用户修改指令（旧字段，保留兼容）
        self.auto_run: bool = False  # 自动连跑模式（不 pause，跑到结束/失败才停）
        self._stop_requested: bool = False  # 用户请求停止（用于自动连跑模式中断）
        self._engine_thread: Optional[threading.Thread] = None  # 后台执行线程
        self._thread_lock = threading.Lock()  # 保护线程启停
        self._review_counts: dict[int, int] = {}  # 每章 review 次数（防止死循环）

    # ── 公开方法 ──

    def start(self, project: str, auto_run: bool = True) -> dict:
        """开始一个完整的写作流程。auto_run=True 自动连跑，False 每步 pause 等确认"""
        self._wait_thread_and_clear()
        self.project = project
        self.state = load_project_to_state(project)
        self.steps = []
        self.current_index = 0
        self._modification = ""
        self.auto_run = auto_run
        self._stop_requested = False
        self.status = "running"
        self.error = ""
        self._review_counts = {}  # 重置 review 计数
        # 重置 modification_hint，避免上一轮的修改意见污染新一轮
        if self.state:
            self.state.modification_hint = ""

        # 执行第一步（同步或异步取决于 auto_run）
        if auto_run:
            self._launch_thread()
            return {"status": "running", "project": project}
        return self._run_next()

    def continue_(self) -> dict:
        """继续执行下一步"""
        if self.status != "paused":
            return {"error": "引擎不在暂停状态"}
        self.status = "running"
        self._modification = ""
        # continue_ 不清 modification_hint：用户在 confirm 暂停时可能
        # 想带着意见继续，只有 refine 后或新一轮 start 才清空
        if self.auto_run:
            self._launch_thread()
            return {"status": "running"}
        return self._run_next()

    def modify(self, instruction: str, target_chapter: int = 0) -> dict:
        """
        针对性重跑：基于用户意见重新执行相关步骤。
        - instruction: 用户修改意见
        - target_chapter: 可选，指定回退到哪一章重跑。0 表示从当前章重跑。

        内部自动处理 stop：如果引擎正在 running，先 stop 等线程退出。
        """
        if not self.state:
            return {"error": "引擎尚未启动"}
        if not self.project:
            return {"error": "未指定项目"}

        # 如果正在 running，先 stop
        if self.status == "running":
            self._stop_requested = True
            try:
                self._wait_thread_and_clear()
            except RuntimeError as e:
                return {"error": str(e)}
            # 此时线程已退出，status 应为 paused
            if self.status == "running":
                return {"error": "等待引擎停止超时，请稍后再试"}

        # 现在 status 应该是 paused/completed/error/idle，都可以接管
        try:
            self._wait_thread_and_clear()
        except RuntimeError as e:
            return {"error": str(e)}

        # 把意见注入到 state，节点会读取
        self.state.modification_hint = instruction.strip()
        self._modification = instruction.strip()

        # 回退到目标章节：清空该章的产出，让 _get_next_step_type 重新走 plan→draft→review→refine
        # 未指定 target_chapter 时，默认回退到 current_chapter（避免章节已是 refined 状态被跳过，
        # 导致用户意见不生效）
        target = target_chapter if target_chapter > 0 else (self.state.current_chapter if self.state else 0)
        if target > 0:
            # 校验 target 是有效章节；无效则不重置（避免 completed 后误调用）
            target_exists = any(target in evt.chapters for evt in self.state.events.values())
            if target_exists:
                self._reset_chapter(target)
                self.state.current_chapter = target
                # 立即持久化重置后的章节状态到 DB，避免线程启动前 Flask 崩溃
                # 导致 load_from_db 恢复旧 refined 状态、modification_hint 失效
                self._persist_chapter(self.state.current_event, target)
                self._persist_workflow()
            elif self.steps:
                # target 无效（如已完成后的兜底）：仅回退一步历史
                self.steps = self.steps[:-1]
                self.current_index = len(self.steps)
        elif self.steps:
            self.steps = self.steps[:-1]
            self.current_index = len(self.steps)

        self._stop_requested = False
        self.status = "running"
        self.error = ""

        # 后台线程跑（避免 HTTP 阻塞）
        self._launch_thread()
        return {"status": "running", "instruction": instruction, "target_chapter": target_chapter}

    def stop(self) -> dict:
        """请求停止执行（auto_run 模式下会等当前 LLM 调用返回后才真正停）"""
        self._stop_requested = True
        self.status = "paused"
        return {"success": True, "message": "已请求停止"}

    def set_auto_run(self, auto: bool) -> dict:
        """运行中切换自动连跑模式"""
        self.auto_run = auto
        return {"success": True, "auto_run": auto}

    # ── 线程管理 ──

    def _launch_thread(self):
        """启动后台线程跑 _run_next"""
        with self._thread_lock:
            if self._engine_thread and self._engine_thread.is_alive():
                # 旧线程还在跑，先等它结束（不应该发生，但保险）
                return
            t = threading.Thread(target=self._thread_main, daemon=True)
            self._engine_thread = t
            t.start()

    def _thread_main(self):
        """后台线程主函数"""
        try:
            self._run_next()
        except Exception as e:
            self.status = "error"
            self.error = f"线程异常: {e}"

    def _wait_thread_and_clear(self):
        """等待已有线程结束（用于 start/modify 前清理）。
        超时后抛异常，避免双线程竞态。
        """
        with self._thread_lock:
            t = self._engine_thread
        if t and t.is_alive():
            # 最多等 120 秒（LLM 调用最长 ~60s，加余量）
            t.join(timeout=120)
            if t.is_alive():
                raise RuntimeError("引擎线程未在 120s 内停止，可能有 LLM 调用卡住")
        with self._thread_lock:
            self._engine_thread = None

    def _reset_chapter(self, ch_num: int) -> bool:
        """清空指定章节的产出，让工作流重新生成（含 plan，让用户意见能影响规划阶段）。

        同时把 current_event 切到该章所属事件，避免跨事件重跑时找错事件。
        返回 True 表示找到并重置了章节，False 表示未找到。
        """
        if not self.state:
            return False
        for evt_num, evt in self.state.events.items():
            if ch_num in evt.chapters:
                ch = evt.chapters[ch_num]
                ch.plan = ""  # 清空规划，让 plan_chapter 重新生成（注入 modification_hint）
                ch.content = ""
                ch.refined_content = ""
                ch.review_feedback = ""
                ch.word_count = 0
                ch.status = "draft"
                ch.entries_updated = False  # 重置条目更新标记，让 update_entries 重新跑
                # 清掉该章的 review 计数，让重试次数重新开始
                self._review_counts.pop(ch_num, None)
                # 从待确认/待润色队列里移除
                if ch_num in (self.state.pending_human_review or []):
                    self.state.pending_human_review.remove(ch_num)
                if ch_num in (self.state.pending_refinement or []):
                    self.state.pending_refinement.remove(ch_num)
                # 切到该章所属事件（跨事件重跑时必须更新，否则 _get_next_step_type
                # 会去 current_event 里找章节，找不到就误判为完成）
                self.state.current_event = evt_num
                return True
        return False

    def get_state(self) -> dict:
        """获取完整状态（前端轮询用）"""
        # 计算章节进度
        written_chapters = 0
        total_chapters = 0
        if self.state:
            for evt in self.state.events.values():
                for ch in evt.chapters.values():
                    total_chapters += 1
                    if ch.status in ("refined", "human_confirmed"):
                        written_chapters += 1
        # 条目池统计（按分类）
        entries_summary = {}
        total_entries = 0
        if self.state:
            from novel_agent.state import ENTRY_CATEGORIES
            for cat in ENTRY_CATEGORIES:
                pool = getattr(self.state.entries, cat, {}) or {}
                if pool:
                    entries_summary[cat] = len(pool)
                    total_entries += len(pool)
        return {
            "status": self.status,
            "project": self.project,
            "current_index": self.current_index,
            "total_steps": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
            "error": self.error,
            "auto_run": self.auto_run,
            "chapter_counter": self.state.chapter_counter if self.state else 0,
            "current_event": self.state.current_event if self.state else 0,
            "total_events": self.state.total_events if self.state else 0,
            "total_words": self._get_total_words(),
            "pending_confirmed": self._count_confirmed_chapters(),
            "pending_human_review": self.state.pending_human_review if self.state else [],
            "pending_refinement": self.state.pending_refinement if self.state else [],
            "written_chapters": written_chapters,
            "total_chapters": total_chapters,
            "entries_summary": entries_summary,
            "total_entries": total_entries,
        }

    # ── 内部执行 ──

    def _run_next(self) -> dict:
        """执行下一步。auto_run 模式下会循环跑到结束/失败/用户停止"""
        while True:
            # 用户请求停止
            if self._stop_requested:
                self._stop_requested = False
                self.status = "paused"
                self._persist_workflow()
                return self.get_state()

            step_type = self._get_next_step_type()
            if step_type is None:
                self.status = "completed"
                self._persist_workflow()
                return {"status": "completed", "message": "所有步骤已完成"}

            step_id = f"step_{len(self.steps) + 1}"
            title, desc = STEP_DEFS.get(step_type, (step_type, ""))
            step = Step(
                id=step_id, type=step_type, status="running",
                title=title, description=desc,
            )
            self.steps.append(step)
            self.current_index = len(self.steps)

            try:
                t0 = time.time()
                result = self._execute(step_type)
                dt = int((time.time() - t0) * 1000)

                step.duration_ms = dt
                step.summary = result.get("summary", "")
                step.preview = result.get("preview", "")
                step.detail = result.get("detail", "")
                step.meta = result.get("meta", {})
                step.actions = result.get("actions", ["continue"])

                if result.get("error"):
                    step.status = "error"
                    step.error = result["error"]
                    self.status = "error"
                    self.error = result["error"]
                    return self.get_state()

                # 步骤执行成功
                if self.auto_run and not self._stop_requested:
                    # 自动连跑模式：标记完成，立即继续下一步
                    # 但 confirm 步骤必须 pause（需要人工确认章节）
                    if step_type == "confirm":
                        step.status = "paused"
                        self.status = "paused"
                        # 意见已用完，清空
                        if self.state:
                            self.state.modification_hint = ""
                        # 持久化
                        if self.state:
                            self._persist_chapter(self.state.current_event, self.state.current_chapter)
                            self._persist_workflow()
                        return self.get_state()
                    step.status = "completed"
                    # 意见仅在 refine 后清空（plan/draft/refine 都需要读取用户意见）
                    # 在 draft 后清空会导致 refine 拿不到意见
                    if step_type == "refine" and self.state:
                        self.state.modification_hint = ""
                    # 持久化到数据库
                    if self.state and step_type == "load_entries":
                        # load_entries 可能生成了事件纲，单独持久化事件
                        self._persist_event(self.state.current_event)
                    if self.state and step_type in ("plan", "draft", "review", "refine", "update_entries"):
                        self._persist_chapter(self.state.current_event, self.state.current_chapter)
                    # update_entries 修改了条目池，需要持久化条目
                    if self.state and step_type == "update_entries":
                        self._persist_entries()
                    # 每步都持久化 workflow 状态，保证 Flask 重启后 current_chapter/pending 列表
                    # 不会落后于实际进度（否则 load_from_db 恢复旧指针，靠递归跳过已完成章节恢复，
                    # 章节多了递归深度会爆）
                    self._persist_workflow()
                    # 继续循环跑下一步
                    continue
                else:
                    # 手动模式：每步 pause 等确认
                    step.status = "paused"
                    self.status = "paused"
                    if self.state and step_type == "load_entries":
                        self._persist_event(self.state.current_event)
                    if self.state and step_type in ("plan", "draft", "review", "refine", "update_entries"):
                        self._persist_chapter(self.state.current_event, self.state.current_chapter)
                    if self.state and step_type == "update_entries":
                        self._persist_entries()
                    self._persist_workflow()
                    return self.get_state()

            except Exception as e:
                step.status = "error"
                step.error = str(e)
                self.status = "error"
                self.error = str(e)
                return self.get_state()

    def _get_next_step_type(self) -> str | None:
        """
        根据当前状态判断下一步类型。
        以 state.current_chapter 为唯一权威来源，自动跳过已有章节、推进事件。

        用 while 循环代替递归，避免跳过大量已完成章节时栈溢出
        （如 Flask 重启后从 DB 恢复，current_chapter 落后于实际进度）。
        """
        state = self.state
        if not state:
            return None

        while True:
            # 清理 pending_refinement 中已过时的条目（current_chapter 之前的章节）。
            # 这些通常是错误恢复后残留的：refine 失败 → 用户 modify 跳到后续章节 →
            # 旧条目留在队列里。不清掉会导致 step 7 条件误判、润色错章或跨事件找不到章节。
            if state.pending_refinement:
                cleaned = [ch for ch in state.pending_refinement if ch >= state.current_chapter]
                if len(cleaned) != len(state.pending_refinement):
                    state.pending_refinement = cleaned

            evt = state.events.get(state.current_event)
            if not evt:
                return None

            # 1. 需要先加载条目？
            if evt.status != "entries_loaded":
                return "load_entries"

            start_ch, end_ch = evt.chapter_range
            current_ch = state.current_chapter

            # 2. 当前事件全部写完 → 推进下一事件
            if end_ch > 0 and current_ch > end_ch:
                if state.current_event < state.total_events:
                    state.current_event += 1
                    next_evt = state.events.get(state.current_event)
                    if next_evt:
                        ns, ne = next_evt.chapter_range
                        state.current_chapter = ns if ns > 0 else current_ch
                    continue  # 重新检查新事件
                else:
                    return None  # 所有事件完成

            # 3. 检查当前章节的写作状态
            ch = evt.chapters.get(current_ch)

            # 4. 需要规划？
            # 已有正文且状态为 refined/human_confirmed → 视为已完成的章节，不需要重新规划
            # （human_confirmed 章节从磁盘加载时 plan 为空，但已完成无需重跑）
            if ch and ch.content and ch.status in ("refined", "human_confirmed"):
                pass  # 跳过规划、起草、质检，走到润色/确认或直接完成
            elif not ch or not ch.plan:
                return "plan"

            # 5. 需要起草？
            if not ch.content:
                return "draft"

            # 6. 需要质检？
            # 触发条件：status=draft（刚起草）且无 review_feedback
            # 加 review 次数限制：每章最多重试 2 次，超过就强制通过（避免死循环）
            # 注意：refined/human_confirmed 状态的章节不再返工（避免已完成的章节被重写）
            review_count = self._review_counts.get(current_ch, 0)
            if ch.status == "draft" and not ch.review_feedback:
                return "review"
            if (ch.status == "draft" and ch.review_feedback
                    and not _parse_review_verdict(ch.review_feedback)
                    and review_count < 2):
                # 上次没通过，且未达上限 → 返工起草后重新质检
                # 清空旧内容让 draft 重写
                ch.content = ""
                ch.word_count = 0
                ch.review_feedback = ""
                ch.status = "draft"
                return "draft"

            # 6.5 需要更新条目池？（review PASS 后、refine 前跑一次）
            # 触发条件：review 已通过（status=reviewed）+ 还没跑过 update_entries
            if ch.status == "reviewed" and _parse_review_verdict(ch.review_feedback or ""):
                if not ch.entries_updated:
                    return "update_entries"

            # 7. 需要润色？
            if state.pending_refinement and current_ch in state.pending_refinement:
                return "refine"

            # 8. 需要确认？
            # auto_run 模式下：refined 状态自动确认（写文件 + 标记 human_confirmed），直接推进
            # 手动模式：refined 状态返回 confirm 让用户确认
            if ch.status == "refined":
                if self.auto_run:
                    # 自动确认：写文件 + 改状态
                    ch.status = "human_confirmed"
                    if current_ch in (state.pending_human_review or []):
                        state.pending_human_review.remove(current_ch)
                    try:
                        save_chapter_to_file(state, state.current_event, current_ch)
                    except Exception:
                        pass
                    # 推进到下一章
                    state.current_chapter = current_ch + 1
                    self._review_counts.pop(current_ch, None)
                    continue  # 重新检查下一章
                else:
                    return "confirm"

            # 当前章节已完成 → 推进到下一章
            state.current_chapter = current_ch + 1
            # 清掉旧章节的 review 计数避免内存累积
            self._review_counts.pop(current_ch, None)
            continue  # 重新检查下一章

    def _execute(self, step_type: str) -> dict:
        """实际执行一个步骤"""
        state = self.state

        if step_type == "load_entries":
            return self._do_load_entries(state)
        elif step_type == "plan":
            return self._do_plan(state)
        elif step_type == "draft":
            return self._do_draft(state)
        elif step_type == "review":
            return self._do_review(state)
        elif step_type == "update_entries":
            return self._do_update_entries(state)
        elif step_type == "refine":
            return self._do_refine(state)
        elif step_type == "confirm":
            return self._do_confirm(state)
        return {"error": f"未知步骤类型: {step_type}"}

    # ── 各步骤实现 ──

    def _do_load_entries(self, state: NovelState) -> dict:
        """加载条目（如果事件纲是模板/空，会先调用 LLM 生成）"""
        # 记录调用前的事件 plan，用于判断是否新生成
        evt_before = state.events.get(state.current_event)
        plan_before = evt_before.plan if evt_before else ""
        plan_before_len = len(plan_before)

        state_dict = state.to_checkpoint_dict()
        result_dict = _load_mod.load_entries_for_event(state_dict)
        new_state = NovelState.from_checkpoint_dict(result_dict)
        new_state = _ensure_dataclass_state(new_state)
        state.__dict__.update(new_state.__dict__)

        # 节点返回错误（如事件纲生成失败）→ 整个步骤失败
        last_error = result_dict.get("last_error", "") if isinstance(result_dict, dict) else ""
        if last_error:
            return {
                "error": last_error,
                "summary": f"❌ {last_error}",
                "preview": last_error,
                "detail": last_error,
                "meta": {"event": state.current_event},
                "actions": ["modify"],
            }

        evt = state.events.get(state.current_event)
        entries = state.entries.get_entries_for_event(f"事件{state.current_event}") if evt else []

        # 对比前后 plan 判断是否新生成
        plan_after = evt.plan if evt else ""
        plan_after_len = len(plan_after)
        # 判断：新 plan 比旧 plan 长得多（实质内容），且新 plan 不是模板
        plan_generated = (
            plan_after_len > 100
            and plan_after_len > plan_before_len + 100
            and "（待填写）" not in plan_after
        )
        # plan 已就绪：新生成 OR 之前已经是实质内容
        plan_ready = (
            plan_after_len > 200
            and "（待填写）" not in plan_after
        )

        # 调试输出（持久化失败时排查用，仅记录到 step.summary 不影响引擎状态）
        plan_gen_warning = ""
        if plan_after_len < 100:
            plan_gen_warning = f"⚠ 事件纲生成可能失败（长度={plan_after_len}），请检查 LLM 输出"

        parts = []
        if plan_generated:
            parts.append("✓ 事件纲已生成")
        elif plan_ready:
            parts.append("✓ 事件纲已就绪")
        if plan_gen_warning:
            parts.append(plan_gen_warning)
        parts.append(f"{len(entries)}个条目已加载" if entries else "无关联条目")
        summary = " · ".join(parts)

        if plan_ready and evt and evt.plan:
            preview = evt.plan[:600]
            detail = evt.plan
        else:
            preview = "\n".join(f"• {e.name} (v{e.version}) — {e.one_line}" for e in entries[:5]) if entries else "当前事件未关联特定条目"
            detail = "\n".join(f"**{e.name}** (v{e.version})\n类别: {e.category}\n简介: {e.one_line}\n{e.content[:300]}\n---" for e in entries) if entries else "无"

        meta = {"entries_loaded": len(entries), "event": state.current_event}
        if plan_generated:
            meta["event_plan_generated"] = True
        if plan_ready:
            meta["event_plan_ready"] = True

        return {
            "summary": summary,
            "preview": preview[:500],
            "detail": detail,
            "meta": meta,
            "actions": ["continue"],
        }

    def _do_plan(self, state: NovelState) -> dict:
        """生成规划"""
        state_dict = state.to_checkpoint_dict()
        result_dict = _plan_mod.plan_chapter(state_dict)
        new_state = NovelState.from_checkpoint_dict(result_dict)
        new_state = _ensure_dataclass_state(new_state)
        state.__dict__.update(new_state.__dict__)

        # 节点可能返回 last_error（如事件不存在）
        last_error = result_dict.get("last_error", "") if isinstance(result_dict, dict) else ""
        if last_error:
            return {
                "error": last_error,
                "summary": f"❌ {last_error}",
                "preview": last_error,
                "detail": last_error,
                "meta": {"chapter_num": state.current_chapter, "event": state.current_event},
                "actions": ["modify"],
            }

        evt = state.events.get(state.current_event)
        ch = evt.chapters.get(state.current_chapter) if evt else None
        plan_text = ch.plan if ch else ""

        summary = f"第{state.current_chapter}章规划完成"
        preview = plan_text[:400] if plan_text else "(无)"
        detail = plan_text if plan_text else "(无规划内容)"

        return {
            "summary": summary,
            "preview": preview,
            "detail": detail,
            "meta": {"chapter_num": state.current_chapter, "event": state.current_event},
            "actions": ["continue", "modify"],
        }

    def _do_draft(self, state: NovelState) -> dict:
        """起草正文"""
        state_dict = state.to_checkpoint_dict()
        result_dict = _draft_mod.draft_chapter(state_dict)
        new_state = NovelState.from_checkpoint_dict(result_dict)
        new_state = _ensure_dataclass_state(new_state)
        state.__dict__.update(new_state.__dict__)

        # 节点可能返回 last_error（如事件不存在、章节未规划）
        last_error = result_dict.get("last_error", "") if isinstance(result_dict, dict) else ""
        if last_error:
            return {
                "error": last_error,
                "summary": f"❌ {last_error}",
                "preview": last_error,
                "detail": last_error,
                "meta": {"chapter_num": state.current_chapter, "event": state.current_event},
                "actions": ["modify"],
            }

        evt = state.events.get(state.current_event)
        ch = evt.chapters.get(state.current_chapter) if evt else None
        content = ch.content if ch else ""
        wc = len(re.findall(r"[\u4e00-\u9fff]", content))

        preview = content[:500] if content else "(无)"
        detail = content if content else "(无)"

        return {
            "summary": f"第{state.current_chapter}章 · {wc}字",
            "preview": preview,
            "detail": detail,
            "meta": {"chapter_num": state.current_chapter, "word_count": wc, "event": state.current_event},
            "actions": ["continue", "modify"],
        }

    def _do_review(self, state: NovelState) -> dict:
        """质检"""
        # 计数 +1
        ch_num = state.current_chapter
        self._review_counts[ch_num] = self._review_counts.get(ch_num, 0) + 1

        state_dict = state.to_checkpoint_dict()
        result_dict = _review_mod.review_chapter(state_dict)
        new_state = NovelState.from_checkpoint_dict(result_dict)
        new_state = _ensure_dataclass_state(new_state)
        state.__dict__.update(new_state.__dict__)

        # 节点可能返回 last_error（如事件不存在、没有可质检的章节）
        last_error = result_dict.get("last_error", "") if isinstance(result_dict, dict) else ""
        if last_error:
            return {
                "error": last_error,
                "summary": f"❌ {last_error}",
                "preview": last_error,
                "detail": last_error,
                "meta": {"chapter_num": state.current_chapter, "review_count": self._review_counts.get(ch_num, 0)},
                "actions": ["modify"],
            }

        evt = state.events.get(state.current_event)
        ch = evt.chapters.get(state.current_chapter) if evt else None
        feedback = ch.review_feedback if ch else ""
        passed = _parse_review_verdict(feedback) if feedback else False
        review_count = self._review_counts.get(ch_num, 0)

        # 强制通过条件：达到重试上限
        forced_pass = False
        if not passed and review_count >= 2:
            forced_pass = True
            passed = True
            # 改写 review_feedback 开头为 PASS，让 _get_next_step_type 的 update_entries 条件能通过
            if ch:
                if feedback:
                    ch.review_feedback = (
                        f"PASS（强制：已达重试上限{review_count}次）\n\n"
                        f"--- 原始 FAIL 反馈 ---\n{feedback}"
                    )
                else:
                    # feedback 为空（review 节点未写入反馈）：写入默认 PASS 反馈
                    ch.review_feedback = (
                        f"PASS（强制：已达重试上限{review_count}次，原始反馈为空）"
                    )

        summary = "✅ 质检通过" if passed else f"❌ 质检未通过（第{review_count}次，上限2次）"
        if forced_pass:
            summary = f"⚠ 质检未通过但已达重试上限({review_count}次)，强制进入润色"
        # preview 和 detail 都用最终的 review_feedback（forced_pass 时已改写为 PASS 开头）
        final_feedback = ch.review_feedback if ch else (feedback or "")
        preview = final_feedback[:500] if final_feedback else "(无反馈)"
        detail = final_feedback if final_feedback else "(无)"

        if passed:
            # 标记为已审查，准备润色
            if ch:
                ch.status = "reviewed"
                state.pending_refinement = state.pending_refinement or []
                if state.current_chapter not in state.pending_refinement:
                    state.pending_refinement.append(state.current_chapter)

        actions = ["continue"] if passed else ["modify"]
        return {
            "summary": summary,
            "preview": preview,
            "detail": detail,
            "meta": {"chapter_num": state.current_chapter, "passed": passed, "review_count": review_count},
            "actions": actions,
        }

    def _do_update_entries(self, state: NovelState) -> dict:
        """条目池更新：扫描本章正文，新增/更新设定条目"""
        ch_num = state.current_chapter
        state_dict = state.to_checkpoint_dict()
        result_dict = _update_mod.update_entries_after_chapter(state_dict)
        new_state = NovelState.from_checkpoint_dict(result_dict)
        new_state = _ensure_dataclass_state(new_state)
        state.__dict__.update(new_state.__dict__)

        # 节点可能返回 last_error（如"本章无正文"或"review 未通过"）
        last_error = result_dict.get("last_error", "") if isinstance(result_dict, dict) else ""

        # 标记本章已跑过 update_entries。
        # 无论成功还是跳过（last_error）都标记 True，避免 _get_next_step_type 死循环重试。
        # "review 未通过"的情况 status 还是 draft，根本不会进入 update_entries 分支，所以不影响。
        evt = state.events.get(state.current_event)
        ch = evt.chapters.get(ch_num) if evt else None
        if ch:
            ch.entries_updated = True

        stats = result_dict.get("entry_updates", {}) if isinstance(result_dict, dict) else {}
        new_n = stats.get("new_entries_created", 0) if isinstance(stats, dict) else 0
        upd_n = stats.get("existing_entries_updated", 0) if isinstance(stats, dict) else 0
        skipped_n = len(stats.get("skipped", [])) if isinstance(stats, dict) else 0

        if last_error:
            summary = f"⚠ 跳过条目更新：{last_error}"
            preview = last_error
            detail = last_error
        else:
            summary = f"新增 {new_n} 条 / 更新 {upd_n} 条 / 跳过 {skipped_n} 条"
            preview = (result_dict.get("llm_output", "") if isinstance(result_dict, dict) else "")[:500]
            detail = json.dumps(stats, ensure_ascii=False, indent=2) if stats else "(无变更)"

        return {
            "summary": summary,
            "preview": preview,
            "detail": detail,
            "meta": {
                "chapter_num": ch_num,
                "new_entries": new_n,
                "updated_entries": upd_n,
                "skipped": skipped_n,
            },
            "actions": ["continue"],
        }

    def _do_refine(self, state: NovelState) -> dict:
        """文风润色"""
        # 润色目标：current_chapter（与 _get_next_step_type step 7 条件
        # `current_ch in pending_refinement` 一致）。不再用 pending_refinement[0]，
        # 避免错误恢复后队首残留旧条目导致润色错章。
        refine_target = state.current_chapter

        state_dict = state.to_checkpoint_dict()
        result_dict = _refine_mod.refine_style(state_dict)
        new_state = NovelState.from_checkpoint_dict(result_dict)
        new_state = _ensure_dataclass_state(new_state)
        state.__dict__.update(new_state.__dict__)

        # 节点可能返回 last_error（如事件不存在、章节不存在）
        last_error = result_dict.get("last_error", "") if isinstance(result_dict, dict) else ""
        if last_error:
            return {
                "error": last_error,
                "summary": f"❌ {last_error}",
                "preview": last_error,
                "detail": last_error,
                "meta": {"chapter_num": refine_target},
                "actions": ["modify"],
            }

        evt = state.events.get(state.current_event)
        refined_ch = evt.chapters.get(refine_target) if evt else None
        refined_text = refined_ch.refined_content if refined_ch else ""
        wc = len(re.findall(r"[\u4e00-\u9fff]", refined_text)) if refined_text else 0

        summary = f"润色完成 · {wc}字" if refined_text else "无需润色"
        preview = refined_text[:500] if refined_text else "(无变化)"
        detail = refined_text if refined_text else "(无)"

        # 注：refine_style 节点内部已处理 status="refined" 和 pending_human_review append
        return {
            "summary": summary,
            "preview": preview,
            "detail": detail,
            "meta": {"word_count": wc, "chapter_num": refine_target},
            "actions": ["continue"],
        }

    def _do_confirm(self, state: NovelState) -> dict:
        """等待确认"""
        evt = state.events.get(state.current_event)
        ch_num = state.current_chapter
        ch = evt.chapters.get(ch_num) if evt else None

        content = ch.refined_content or ch.content if ch else ""
        wc = len(re.findall(r"[\u4e00-\u9fff]", content)) if content else 0

        preview = content[:500] if content else "(无)"
        detail = content if content else "(无)"

        return {
            "summary": f"第{ch_num}章 · {wc}字，请确认",
            "preview": preview,
            "detail": detail,
            "meta": {"chapter_num": ch_num, "word_count": wc},
            "actions": ["confirm", "modify"],
        }

    def confirm_chapter(self, ch_num: int) -> dict:
        """确认章节（由前端 step/confirm API 调用）"""
        if not self.state:
            return {"error": "状态未初始化"}
        # 统一转 int（前端可能传字符串，pending 列表里存的是 int）
        try:
            ch_num = int(ch_num)
        except (TypeError, ValueError):
            return {"error": f"非法章号: {ch_num}"}

        for evt_num, evt in self.state.events.items():
            ch = evt.chapters.get(ch_num)
            if ch:
                ch.status = "human_confirmed"
                if ch_num in (self.state.pending_human_review or []):
                    self.state.pending_human_review.remove(ch_num)
                save_chapter_to_file(self.state, evt_num, ch_num)
                # 持久化到 DB：章节状态 + pending_human_review 列表。
                # 若不持久化，Flask 重启后 load_from_db 会用 DB 里的旧状态
                # （"refined"）覆盖磁盘的 "human_confirmed"，导致用户需要重新确认。
                self._persist_chapter(evt_num, ch_num)
                self._persist_workflow()

                # 更新当前步骤
                if self.steps:
                    self.steps[-1].status = "completed"
                    self.steps[-1].summary = f"✅ 第{ch_num}章已确认"

                return {"success": True, "chapter": ch_num}

        return {"error": f"第{ch_num}章不存在"}

    # ── 辅助 ──

    def _persist_chapter(self, evt_num: int, ch_num: int):
        """把当前 state 里的章节状态同步到数据库（plan/content/refined/review/status/word_count）"""
        if not self.state:
            return
        evt = self.state.events.get(evt_num)
        if not evt:
            return
        ch = evt.chapters.get(ch_num)
        if not ch:
            return
        try:
            from dashboard.core import crud
            from dashboard.core.db import get_conn
            # 找到项目 id 和事件 id
            proj = crud.get_project_by_name(self.project)
            if not proj:
                return
            db_evt = crud.get_event(proj.id, evt_num)
            if not db_evt:
                return
            # 章节可能还没建，先确保存在
            db_ch = crud.get_chapter(db_evt.id, ch_num)
            if not db_ch:
                crud.create_chapter(db_evt.id, ch_num, plan=ch.plan or "", status=ch.status or "pending")
            # 更新所有字段
            # word_count 优先用节点已设置的 ch.word_count（refine_style 会设润色稿字数）；
            # 没有时才重新计算。避免用初稿字数覆盖润色稿字数。
            if ch.word_count > 0:
                word_count = ch.word_count
            elif ch.refined_content:
                word_count = len(re.findall(r"[\u4e00-\u9fff]", ch.refined_content))
            elif ch.content:
                word_count = len(re.findall(r"[\u4e00-\u9fff]", ch.content))
            else:
                word_count = 0
            crud.update_chapter_by_event(
                db_evt.id, ch_num,
                plan=ch.plan or "",
                content=ch.content or "",
                refined_content=ch.refined_content or "",
                review_feedback=ch.review_feedback or "",
                status=ch.status or "pending",
                word_count=word_count,
            )
            # 同步事件 plan 到数据库
            if evt.plan and evt.plan != db_evt.plan:
                crud.update_event(db_evt.id, plan=evt.plan, status=evt.status or "planned")
                # 同时写到文件
                self._save_event_plan_to_file(evt_num, evt.plan)
        except Exception as e:
            # 持久化失败不阻塞流程，只记录
            self.error = f"持久化失败(ch{ch_num}): {e}"

    def _persist_event(self, evt_num: int):
        """持久化事件 plan 到数据库 + 文件（load_entries 生成事件纲后调用）。"""
        if not self.state:
            return
        evt = self.state.events.get(evt_num)
        if not evt or not evt.plan:
            return
        # 模板 plan 不持久化
        if "（待填写）" in evt.plan and len(evt.plan) < 200:
            return
        try:
            from dashboard.core import crud
            proj = crud.get_project_by_name(self.project)
            if not proj:
                return
            db_evt = crud.get_event(proj.id, evt_num)
            if not db_evt:
                return
            # 1. 同步到数据库（如果不同）
            if evt.plan != db_evt.plan:
                crud.update_event(db_evt.id, plan=evt.plan, status=evt.status or "planned")
            # 2. 文件落盘（复用辅助函数）
            self._save_event_plan_to_file(evt_num, evt.plan)
        except Exception as e:
            self.error = f"持久化事件{evt_num}失败: {e}"

    def _save_event_plan_to_file(self, evt_num: int, plan: str):
        """把事件 plan 写到 projects/<项目名>/事件N/事件纲.md"""
        try:
            from pathlib import Path
            base = Path(__file__).resolve().parent.parent / "projects" / self.project
            evt_dir = base / f"事件{evt_num}"
            evt_dir.mkdir(parents=True, exist_ok=True)
            (evt_dir / "事件纲.md").write_text(plan, encoding="utf-8")
        except Exception:
            pass

    def _persist_entries(self):
        """持久化所有条目到数据库 + 文件（update_entries 修改条目后调用）。"""
        if not self.state:
            return
        try:
            from dashboard.core import crud
            from novel_agent.utils.file_io import save_entry_to_file
            from novel_agent.state import ENTRY_CATEGORIES
            proj = crud.get_project_by_name(self.project)
            if not proj:
                return
            # 全量同步条目到数据库 + 文件
            for cat in ENTRY_CATEGORIES:
                pool = getattr(self.state.entries, cat, {})
                for name, entry in pool.items():
                    # 1. 文件落盘（含 change_history）
                    try:
                        save_entry_to_file(self.state, cat, entry)
                    except Exception:
                        pass
                    # 2. 数据库 upsert
                    # appears_in 必须存 JSON 字符串（app.py 读取时用 json.loads 解析）。
                    # 早期版本用逗号分隔，会与 json.loads 不兼容，导致条目详情接口 500。
                    try:
                        appears_in_json = json.dumps(
                            entry.appears_in if isinstance(entry.appears_in, list) else list(entry.appears_in or []),
                            ensure_ascii=False,
                        )
                        crud.upsert_entry(
                            project_id=proj.id,
                            name=entry.name,
                            category=entry.category,
                            one_line=entry.one_line,
                            content=entry.content,
                            version=entry.version,
                            appears_in=appears_in_json,
                        )
                    except Exception:
                        pass
        except Exception as e:
            self.error = f"持久化条目失败: {e}"

    def _persist_workflow(self):
        """把当前工作流状态同步到数据库"""
        if not self.state:
            return
        try:
            from dashboard.core import crud
            import json as _json
            proj = crud.get_project_by_name(self.project)
            if not proj:
                return
            # extra 字段保存易失状态：review_counts、modification_hint
            extra = {
                "review_counts": self._review_counts,
                "modification_hint": self.state.modification_hint or "",
                "auto_run": self.auto_run,
            }
            crud.upsert_workflow(
                proj.id,
                current_event=self.state.current_event,
                current_chapter=self.state.current_chapter,
                pending_review=self.state.pending_human_review or [],
                pending_refine=self.state.pending_refinement or [],
                status=self.status,
                extra=_json.dumps(extra, ensure_ascii=False),
            )
        except Exception:
            pass

    def load_from_db(self, project: str) -> dict:
        """从数据库恢复工作流状态（Flask 重启后调用）

        返回 {"success": bool, "status": str, "message": str}
        """
        from dashboard.core import crud
        from novel_agent.state import ChapterState
        import json as _json
        proj = crud.get_project_by_name(project)
        if not proj:
            return {"success": False, "message": f"项目不存在: {project}"}
        wf = crud.get_workflow(proj.id)
        if not wf or wf.status == "idle":
            return {"success": False, "message": "工作流未启动过"}

        # 加载项目到 state（从磁盘读取，已 refine/confirm 的章节会被加载）
        self.project = project
        self.state = load_project_to_state(project)
        if not self.state:
            return {"success": False, "message": "加载项目失败"}

        # 从 DB 覆盖章节状态（磁盘只在 refine/confirm 时写盘，draft/review 阶段
        # 的内容只在 DB 里。不覆盖会丢失进行中的章节）
        try:
            db_events = crud.get_events_for_project(proj.id)
            for db_evt in db_events:
                evt_state = self.state.events.get(db_evt.num)
                if not evt_state:
                    continue
                # 从 DB 恢复 chapter_range（磁盘加载可能因事件纲缺失而为 (0,0)，
                # 导致 _get_next_step_type 无法推进事件）
                if db_evt.ch_range_start and db_evt.ch_range_end:
                    evt_state.chapter_range = (db_evt.ch_range_start, db_evt.ch_range_end)
                # 同步事件 plan（DB 可能比磁盘更新，如 load_entries 刚生成事件纲）
                if db_evt.plan and not evt_state.plan:
                    evt_state.plan = db_evt.plan
                for db_ch in crud.get_chapters_for_event(db_evt.id):
                    ch_state = evt_state.chapters.get(db_ch.num)
                    if ch_state:
                        # 磁盘已加载：用 DB 字段覆盖（DB 是最新的）
                        ch_state.plan = db_ch.plan or ch_state.plan
                        ch_state.content = db_ch.content or ch_state.content
                        ch_state.refined_content = db_ch.refined_content or ch_state.refined_content
                        ch_state.review_feedback = db_ch.review_feedback or ch_state.review_feedback
                        ch_state.word_count = db_ch.word_count or ch_state.word_count
                        # DB status 比 disk 的 "human_confirmed" 默认值更准确
                        ch_state.status = db_ch.status or ch_state.status
                        # entries_updated 不在 DB 里：按 status 推断。
                        # refined/human_confirmed 必然已跑过 update_entries（它是 refine 的前置步骤）；
                        # reviewed 可能跑过也可能没跑过，保守设 False（最多幂等重跑一次）。
                        if ch_state.status in ("refined", "human_confirmed"):
                            ch_state.entries_updated = True
                    else:
                        # 磁盘没有（draft/review 阶段）：从 DB 创建
                        _status = db_ch.status or "draft"
                        ch_state = ChapterState(
                            chapter_num=db_ch.num,
                            content=db_ch.content or "",
                            refined_content=db_ch.refined_content or "",
                            review_feedback=db_ch.review_feedback or "",
                            plan=db_ch.plan or "",
                            word_count=db_ch.word_count or 0,
                            status=_status,
                            entries_updated=_status in ("refined", "human_confirmed"),
                        )
                        evt_state.chapters[db_ch.num] = ch_state
        except Exception:
            pass

        # 从 workflow 表恢复指针
        self.state.current_event = wf.current_event or 1
        self.state.current_chapter = wf.current_chapter or 1
        # pending_review/pending_refine 可能是 JSON、空串或旧格式；容错解析
        def _safe_load_list(val):
            if not val:
                return []
            if isinstance(val, list):
                return val
            try:
                parsed = _json.loads(val)
                return parsed if isinstance(parsed, list) else []
            except (ValueError, TypeError):
                return []
        self.state.pending_human_review = _safe_load_list(wf.pending_review)
        self.state.pending_refinement = _safe_load_list(wf.pending_refine)

        # 从 extra 恢复易失状态
        try:
            extra = _json.loads(wf.extra or "{}")
            self._review_counts = {int(k): int(v) for k, v in (extra.get("review_counts") or {}).items()}
            self.state.modification_hint = extra.get("modification_hint", "")
            self.auto_run = extra.get("auto_run", True)
        except Exception:
            self._review_counts = {}
            self.auto_run = True

        # 状态映射：DB 的 running 视为 paused（重启后等待用户继续）
        self.status = "paused" if wf.status == "running" else wf.status
        self.steps = []  # 步骤历史不恢复（用户可重新查看章节内容）
        self.current_index = 0
        self._stop_requested = False
        self.error = ""

        return {
            "success": True,
            "status": self.status,
            "message": f"已恢复到 evt{self.state.current_event}/ch{self.state.current_chapter}",
        }

    def _get_total_words(self) -> int:
        if not self.state:
            return 0
        return sum(
            ch.word_count
            for evt in self.state.events.values()
            for ch in evt.chapters.values()
        )

    def _count_confirmed_chapters(self) -> int:
        if not self.state:
            return 0
        return sum(
            1 for evt in self.state.events.values()
            for ch in evt.chapters.values()
            if ch.status == "human_confirmed"
        )
