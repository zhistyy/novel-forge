"""
Novel Agent 工作台 — Flask 后端

集成了:
  - 数据库 (SQLite CRUD)
  - StepEngine (分步写作引擎)
  - DeepSeek AI
"""

from __future__ import annotations

import sys, os, json, threading
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, jsonify, request, render_template

from novel_agent.utils.file_io import load_project_to_state
from novel_agent.utils.llm import call_llm, API_KEY as LLM_API_KEY

from dashboard.step_engine import StepEngine
from dashboard.core.db import init_db
from dashboard.core.crud import (
    get_project_by_name, list_projects, get_events_for_project,
    get_chapter, get_chapters_for_event,
    list_entries, get_entry, create_entry, update_entry, delete_entry,
    import_project_from_fs, get_workflow,
)
from dashboard.core.prompt_manager import ensure_default_prompts

app = Flask(__name__)

PROJECTS_DIR = Path(__file__).resolve().parent.parent / "projects"

# ── 全局实例 ──
_engine = StepEngine()

# ── 日志缓冲区 ──
_log_buf: list[dict] = []
_log_lock = threading.Lock()

def _add_log(kind: str, msg: str, detail: str = ""):
    with _log_lock:
        _log_buf.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "kind": kind, "msg": msg, "detail": detail,
        })
        if len(_log_buf) > 200:
            _log_buf[:] = _log_buf[-100:]


def _safe_json_loads(val, default=None):
    """容错解析 JSON 字符串。用于解析 DB 中可能为 None / 空串 / 旧格式（逗号分隔）的字段。
    - None / 空串 → default
    - list / dict → 原样返回
    - 合法 JSON 字符串 → 解析结果
    - 旧格式逗号分隔字符串 → 拆分成 list
    - 解析失败 → default
    """
    if default is None:
        default = []
    if val is None:
        return default
    if isinstance(val, (list, dict)):
        return val
    if not isinstance(val, str):
        return default
    s = val.strip()
    if not s:
        return default
    # JSON 格式优先
    if s.startswith("[") or s.startswith("{"):
        try:
            return json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return default
    # 旧格式：逗号分隔字符串 → list
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [s]


def _safe_int(val, default: int = 0) -> int:
    """容错转 int：前端可能传字符串、null 或非数字值。"""
    if val is None:
        return default
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        return default


# ── 启动初始化 ──

def _startup():
    init_db()
    ensure_default_prompts()
    # 自动导入已有项目到数据库
    if PROJECTS_DIR.exists():
        for p in sorted(PROJECTS_DIR.iterdir()):
            if not p.is_dir() or p.name.startswith("."):
                continue
            existing = get_project_by_name(p.name)
            if not existing:
                try:
                    pid = import_project_from_fs(p.name)
                    if pid:
                        _add_log("system", f"导入项目: {p.name}")
                except Exception as e:
                    _add_log("system", f"导入失败: {p.name} - {e}")
    _add_log("system", f"启动完成 | DeepSeek {'已连接' if LLM_API_KEY else '未配置'}")

_startup()


# ── 页面 ──

@app.route("/")
def index():
    resp = render_template("index.html")
    from flask import make_response
    r = make_response(resp)
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r


# ── API: 项目列表 ──

@app.route("/api/projects")
def api_projects():
    db_projects = list_projects()
    result = []
    for p in db_projects:
        events = get_events_for_project(p.id)
        total_ch = written_ch = total_words = 0
        for evt in events:
            chs = get_chapters_for_event(evt.id)
            total_ch += len(chs)
            written_ch += sum(1 for c in chs if c.status in ("refined", "human_confirmed"))
            total_words += sum(c.word_count for c in chs)
        result.append({
            "name": p.name,
            "genre": p.genre,
            "written_chapters": written_ch,
            "total_words": total_words,
        })
    return jsonify(result)


# ── API: 项目详情 ──

@app.route("/api/project/<name>")
def api_project_detail(name: str):
    # 先从数据库获取，失败则从文件加载
    proj = get_project_by_name(name)
    if not proj:
        state = _load_state_fs(name)
        if not state:
            return jsonify({"error": "项目未找到"}), 404
        # 快速响应文件模式的数据
        return jsonify(_project_state_to_json(name, state))

    events = get_events_for_project(proj.id)
    events_data = []
    # 一次遍历收集所有统计，避免每个事件的章节被查询三次
    total_words = 0
    written_chapters = 0
    total_chapters = 0
    for evt in events:
        chapters = get_chapters_for_event(evt.id)
        ch_data = []
        for ch in chapters:
            ch_data.append({
                "num": ch.num, "status": ch.status, "word_count": ch.word_count,
                "has_plan": bool(ch.plan), "has_content": bool(ch.content),
                "has_refined": bool(ch.refined_content),
            })
            total_words += ch.word_count
            if ch.status in ("refined", "human_confirmed"):
                written_chapters += 1
        total = evt.ch_range_end - evt.ch_range_start + 1
        total_chapters += total
        written = sum(1 for c in chapters if c.status in ("refined", "human_confirmed"))
        events_data.append({
            "num": evt.num, "name": f"事件{evt.num}", "status": evt.status,
            "plan": (evt.plan or "")[:300],
            "chapter_range": f"{evt.ch_range_start}-{evt.ch_range_end}",
            "written": written, "total": total,
            "chapters": ch_data,
        })

    entries = list_entries(proj.id)
    entries_data = {}
    for e in entries:
        cat = e.category
        if cat not in entries_data:
            entries_data[cat] = []
        entries_data[cat].append({
            "name": e.name, "version": e.version, "one_line": e.one_line,
        })

    wf = get_workflow(proj.id)

    return jsonify({
        "name": proj.name, "genre": proj.genre,
        "status": wf.status if wf else "idle",
        "current_event": wf.current_event if wf else 1,
        "chapter_counter": wf.current_chapter if wf else 1,
        "written_chapters": written_chapters,
        "total_chapters": total_chapters,
        "total_events": len(events),
        "total_words": total_words,
        "words_per_chapter": proj.words_per_chapter or 1000,
        "tone": proj.tone or "",
        "updated_at": proj.updated_at or "",
        "events": events_data, "entries": entries_data,
        "pending_human_review": _safe_json_loads(wf.pending_review) if wf else [],
        "pending_refinement": _safe_json_loads(wf.pending_refine) if wf else [],
        "has_api_key": bool(LLM_API_KEY),
    })


def _load_state_fs(name: str):
    try:
        return load_project_to_state(name)
    except Exception:
        return None


def _project_state_to_json(name: str, state):
    """快速将文件系统的 NovelState 转为 JSON"""
    events_data = []
    for evt_num in sorted(state.events.keys()):
        evt = state.events[evt_num]
        chapters = []
        for ch_num in sorted(evt.chapters.keys()):
            ch = evt.chapters[ch_num]
            chapters.append({
                "num": ch_num, "status": ch.status, "word_count": ch.word_count,
                "has_plan": bool(ch.plan), "has_content": bool(ch.content),
                "has_refined": bool(ch.refined_content),
            })
        start, end = evt.chapter_range
        if end == 0: end = start + 17
        written = sum(1 for c in evt.chapters.values() if c.status in ("refined", "human_confirmed"))
        events_data.append({
            "num": evt_num, "name": f"事件{evt_num}", "status": evt.status,
            "plan": (evt.plan or "")[:300],
            "chapter_range": f"{start}-{end}", "written": written, "total": end - start + 1,
            "chapters": chapters,
        })
    entries_data = {}
    from novel_agent.state import ENTRY_CATEGORIES
    for cat in ENTRY_CATEGORIES:
        pool = getattr(state.entries, cat, {})
        if pool:
            entries_data[cat] = [
                {"name": e.name, "version": e.version, "one_line": e.one_line}
                for e in pool.values()
            ]
    total_words = sum(ch.word_count for evt in state.events.values() for ch in evt.chapters.values())
    return {
        "name": state.project_name, "genre": state.genre,
        "status": state.status, "current_event": state.current_event,
        "chapter_counter": state.chapter_counter, "total_events": state.total_events,
        "total_words": total_words, "events": events_data, "entries": entries_data,
        "pending_human_review": state.pending_human_review,
        "pending_refinement": state.pending_refinement,
        "has_api_key": bool(LLM_API_KEY),
    }


# ── API: 单章内容 ──

@app.route("/api/project/<name>/chapter/<int:ch_num>")
def api_chapter_by_num(name: str, ch_num: int):
    proj = get_project_by_name(name)
    if not proj:
        # Fallback to FS
        state = _load_state_fs(name)
        if not state:
            return jsonify({"error": "项目未找到"}), 404
        for evt_num, evt in state.events.items():
            ch = evt.chapters.get(ch_num)
            if ch:
                return jsonify({
                    "event": evt_num, "event_num": evt_num, "chapter": ch_num,
                    "plan": ch.plan, "content": ch.content,
                    "refined_content": ch.refined_content,
                    "review_feedback": getattr(ch, "review_feedback", "") or "",
                    "status": ch.status, "word_count": ch.word_count,
                })
        return jsonify({"error": "章节不存在"}), 404

    events = get_events_for_project(proj.id)
    for evt in events:
        ch = get_chapter(evt.id, ch_num)
        if ch:
            return jsonify({
                "event": evt.num, "event_num": evt.num, "chapter": ch_num,
                "plan": ch.plan, "content": ch.content,
                "refined_content": ch.refined_content,
                "review_feedback": getattr(ch, "review_feedback", "") or "",
                "status": ch.status, "word_count": ch.word_count,
            })
    return jsonify({"error": "章节不存在"}), 404


# ── API: Step 引擎 ──

@app.route("/api/step/state")
def api_step_state():
    return jsonify(_engine.get_state())

@app.route("/api/step/start", methods=["POST"])
def api_step_start():
    """启动引擎。auto_run=true 自动连跑（默认，后台线程），false 每步 pause（同步）"""
    data = request.get_json() or {}
    project = data.get("project", "")
    auto_run = data.get("auto_run", True)
    if not project: return jsonify({"error": "缺少 project"}), 400
    _add_log("engine", f"Step 引擎启动 (auto_run={auto_run})", project)
    result = _engine.start(project, auto_run=auto_run)
    return jsonify(result)

@app.route("/api/step/continue", methods=["POST"])
def api_step_continue():
    """继续执行。auto_run 模式下由引擎内部启动后台线程"""
    result = _engine.continue_()
    _add_log("engine", "继续下一步" if result.get("status") != "completed" else "所有步骤已完成")
    return jsonify(result)

@app.route("/api/step/modify", methods=["POST"])
def api_step_modify():
    """针对性重跑：基于用户意见 + 可选目标章节"""
    data = request.get_json() or {}
    instruction = data.get("instruction", "")
    target_chapter = _safe_int(data.get("target_chapter"), 0)
    if not instruction.strip():
        return jsonify({"error": "需要填写修改意见"}), 400
    _add_log("engine", f"针对性重跑: {instruction[:50]} (target_ch={target_chapter})")
    result = _engine.modify(instruction, target_chapter=target_chapter)
    return jsonify(result)

@app.route("/api/step/stop", methods=["POST"])
def api_step_stop():
    result = _engine.stop()
    _add_log("engine", "Step 引擎已停止")
    return jsonify(result)

@app.route("/api/step/confirm", methods=["POST"])
def api_step_confirm():
    data = request.get_json() or {}
    ch_num = data.get("ch_num", 0)
    if not ch_num: return jsonify({"error": "缺少 ch_num"}), 400
    result = _engine.confirm_chapter(ch_num)
    _add_log("engine", f"第{ch_num}章已确认")
    return jsonify(result)

@app.route("/api/step/set_auto_run", methods=["POST"])
def api_step_set_auto_run():
    """运行中切换自动连跑模式"""
    data = request.get_json() or {}
    auto = data.get("auto_run", True)
    return jsonify(_engine.set_auto_run(auto))


@app.route("/api/step/resume", methods=["POST"])
def api_step_resume():
    """Flask 重启后从 DB 恢复工作流状态。

    请求体: {"project": "项目名"}
    返回: {"success": bool, "status": str, "message": str}
    """
    data = request.get_json() or {}
    project = data.get("project", "")
    if not project:
        return jsonify({"error": "缺少 project"}), 400
    result = _engine.load_from_db(project)
    _add_log("engine", f"恢复工作流: {result.get('message', '')}", project)
    return jsonify(result)


# ── API: 一键创建项目并启动 ──

@app.route("/api/project/create_and_start", methods=["POST"])
def api_project_create_and_start():
    """表单式创建项目 + 自动启动引擎。
    接收 name/genre/tone/规模，建项目后立即跑全流程。
    引擎在后台线程跑，HTTP 立即返回。前端通过 /api/step/state 轮询进度。
    """
    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    genre = (data.get("genre") or "").strip()
    tone = (data.get("tone") or "").strip()
    plot = (data.get("plot") or "").strip()
    total_events = _safe_int(data.get("total_events"), 3)
    total_chapters = _safe_int(data.get("total_chapters"), total_events * 3)
    words_per_chapter = _safe_int(data.get("words_per_chapter"), 1000)
    # 保护下限，避免传 0 或负数
    if total_events < 1: total_events = 1
    if total_chapters < 1: total_chapters = total_events * 3
    if words_per_chapter < 100: words_per_chapter = 1000
    auto_run = data.get("auto_run", True)

    if not name:
        return jsonify({"error": "需要项目名称"}), 400
    if not genre:
        return jsonify({"error": "需要小说类型"}), 400

    # 创建项目（脱耦 Brain 系统）
    from dashboard.core.project_creator import create_project
    create_result = create_project(
        name=name,
        genre=genre,
        tone=tone,
        plot=plot,
        total_events=total_events,
        total_chapters=total_chapters,
        words_per_chapter=words_per_chapter,
    )
    if not create_result.get("success"):
        return jsonify({"error": create_result.get("error", "创建项目失败")}), 400

    # 引擎内部会启动后台线程跑（auto_run=True 时）
    _engine.start(name, auto_run=auto_run)

    _add_log("engine", f"项目 {name} 已创建并启动引擎", name)
    return jsonify({
        "success": True,
        "project": name,
        "message": create_result.get("text", ""),
        "engine_status": "running",
        "hint": "通过 GET /api/step/state 轮询进度",
    })


# ── API: 修改已完成章节 ──

@app.route("/api/project/<name>/tone", methods=["POST"])
def api_project_update_tone(name: str):
    """更新项目基调"""
    data = request.get_json() or {}
    tone = data.get("tone", "")
    proj = get_project_by_name(name)
    if not proj:
        return jsonify({"error": "项目不存在"}), 404
    # 更新数据库
    from dashboard.core import crud
    crud.update_project(proj.id, tone=tone)
    # 同步到 engine 内存
    if _engine.state and _engine.project == name:
        _engine.state.tone = tone
    _add_log("project", f"更新基调", name)
    return jsonify({"success": True, "tone": tone})


@app.route("/api/project/<name>", methods=["DELETE"])
def api_project_delete(name: str):
    """删除项目（数据库 + 文件系统）"""
    from dashboard.core import crud
    import shutil
    from pathlib import Path
    proj = get_project_by_name(name)
    if not proj:
        return jsonify({"error": "项目不存在"}), 404
    try:
        crud.delete_project(proj.id)
    except Exception as e:
        return jsonify({"error": f"删除数据库失败: {e}"}), 500
    # 删除文件
    root = Path(__file__).resolve().parent.parent / "projects" / name
    if root.exists():
        try:
            shutil.rmtree(root)
        except Exception as e:
            return jsonify({"error": f"删除文件失败: {e}"}), 500
    # 如果引擎在跑这个项目，停止并彻底清理
    if _engine.project == name:
        _engine.stop()
        # 等线程真正退出，避免线程在退出前把 status 改回 "paused" 覆盖 "idle"
        try:
            _engine._wait_thread_and_clear()
        except RuntimeError:
            pass  # 线程卡在 LLM 调用，120s 内未退出，强制清理
        _engine.project = ""
        _engine.steps = []
        _engine.status = "idle"
        _engine.state = None  # 清空 state，避免后续 modify 等调用误用已删除项目的内存状态
        _engine._review_counts = {}
    _add_log("project", f"删除项目 {name}")
    return jsonify({"success": True})


@app.route("/api/project/<name>/chapter/<int:ch_num>/unlock", methods=["POST"])
def api_chapter_unlock(name: str, ch_num: int):
    """解锁章节：把 human_confirmed 状态改回 refined，让工作流可以重新生成"""
    proj = get_project_by_name(name)
    if not proj:
        return jsonify({"error": "项目不存在"}), 404
    events = get_events_for_project(proj.id)
    for evt in events:
        ch = get_chapter(evt.id, ch_num)
        if ch:
            from dashboard.core import crud
            crud.update_chapter_by_event(evt.id, ch_num, status="refined")
            if _engine.state and _engine.project == name:
                eng_evt = _engine.state.events.get(evt.num)
                if eng_evt and eng_evt.chapters.get(ch_num):
                    eng_evt.chapters[ch_num].status = "refined"
            _add_log("chapter", f"解锁 {name} 第{ch_num}章", name)
            return jsonify({"success": True, "chapter": ch_num, "status": "refined"})
    return jsonify({"error": f"第{ch_num}章不存在"}), 404


@app.route("/api/project/<name>/chapter/<int:ch_num>/edit", methods=["POST"])
def api_chapter_edit(name: str, ch_num: int):
    """直接编辑章节内容（暂停后修改已完成工作的入口）。
    body: {"field": "content|refined_content|plan", "text": "新内容", "confirm": true}
    confirm=true 时把章节状态改为 human_confirmed，防止后续工作流覆盖。
    """
    data = request.get_json() or {}
    field = data.get("field", "refined_content")
    text = data.get("text", "")
    confirm = bool(data.get("confirm", False))
    if field not in ("content", "refined_content", "plan", "review_feedback"):
        return jsonify({"error": f"非法字段：{field}"}), 400

    proj = get_project_by_name(name)
    if not proj:
        return jsonify({"error": "项目不存在"}), 404

    events = get_events_for_project(proj.id)
    for evt in events:
        ch = get_chapter(evt.id, ch_num)
        if ch:
            from dashboard.core import crud
            # 仅在编辑正文类字段时才更新 word_count，避免编辑 plan/review_feedback
            # 时把字数错误地设为该字段字数
            is_content_field = field in ("refined_content", "content")
            if is_content_field:
                word_count = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
            else:
                word_count = ch.word_count
            update_kwargs = {field: text}
            if is_content_field:
                update_kwargs["word_count"] = word_count
            # 编辑润色稿或原始稿时，如果用户选择 confirm，标记为已人工确认
            is_confirm_field = field in ("refined_content", "content")
            new_status = "human_confirmed" if (confirm and is_confirm_field) else ch.status
            if confirm and is_confirm_field:
                update_kwargs["status"] = new_status
            crud.update_chapter_by_event(evt.id, ch_num, **update_kwargs)
            # 同步到 engine 内存（如果引擎在跑这个项目）
            if _engine.state and _engine.project == name:
                eng_evt = _engine.state.events.get(evt.num)
                if eng_evt and eng_evt.chapters.get(ch_num):
                    eng_ch = eng_evt.chapters[ch_num]
                    setattr(eng_ch, field, text)
                    if is_content_field:
                        eng_ch.word_count = word_count
                    if confirm and is_confirm_field:
                        eng_ch.status = "human_confirmed"
                        # 从待确认队列移除
                        pending = _engine.state.pending_human_review or []
                        if ch_num in pending:
                            pending.remove(ch_num)
            _add_log("chapter", f"编辑 {name} 第{ch_num}章 {field}" +
                     ("（已确认）" if confirm else ""), name)
            return jsonify({
                "success": True, "chapter": ch_num, "field": field,
                "word_count": word_count,
                "status": new_status,
            })
    return jsonify({"error": f"第{ch_num}章不存在"}), 404


# ── API: 条目 CRUD ──

@app.route("/api/entries/<name>", methods=["GET", "POST", "PUT", "DELETE"])
def api_entries(name: str):
    project = request.args.get("project", "")
    if not project:
        return jsonify({"error": "缺少 project 参数"}), 400
    proj = get_project_by_name(project)
    if not proj:
        return jsonify({"error": f"项目不存在: {project}"}), 404

    if request.method == "GET":
        entry = get_entry(proj.id, name)
        if not entry:
            return jsonify({"error": "条目不存在"}), 404
        return jsonify({
            "name": entry.name, "category": entry.category,
            "one_line": entry.one_line, "content": entry.content,
            "version": entry.version,
            "appears_in": _safe_json_loads(entry.appears_in),
        })

    elif request.method == "POST":
        data = request.get_json() or {}
        entry = create_entry(proj.id, name, data.get("category", "人物设定"),
                           one_line=data.get("one_line", ""),
                           content=data.get("content", ""),
                           appears_in=data.get("appears_in", []),
                           version=1)
        if not entry:
            return jsonify({"error": "创建条目失败"}), 500
        _add_log("db", f"创建条目: {name}")
        return jsonify({"success": True, "name": entry.name, "version": entry.version})

    elif request.method == "PUT":
        data = request.get_json() or {}
        entry = update_entry(proj.id, name, **data)
        if not entry:
            return jsonify({"error": "条目不存在或更新失败"}), 404
        _add_log("db", f"更新条目: {name}")
        return jsonify({"success": True, "name": entry.name, "version": entry.version})

    elif request.method == "DELETE":
        ok = delete_entry(proj.id, name)
        _add_log("db", f"删除条目: {name}")
        return jsonify({"success": ok})


@app.route("/api/entries", methods=["GET"])
def api_list_entries():
    project = request.args.get("project", "")
    category = request.args.get("category", "")
    if not project:
        return jsonify([])
    proj = get_project_by_name(project)
    if not proj:
        return jsonify([])
    entries = list_entries(proj.id, category)
    return jsonify([
        {
            "name": e.name, "category": e.category,
            "one_line": e.one_line, "version": e.version,
            "appears_in": _safe_json_loads(e.appears_in),
        }
        for e in entries
    ])


# ── API: DeepSeek AI ──

@app.route("/api/ai/refine", methods=["POST"])
def ai_refine():
    data = request.get_json() or {}
    text = data.get("text", "")
    style = data.get("style", "")
    if not text: return jsonify({"error": "缺少 text"}), 400
    if not LLM_API_KEY:
        return jsonify({"error": "未配置 API Key", "mock": True, "output": text}), 200
    system = "你是一个专业的小说润色师。请优化以下文本的文学表现力，保持情节和人物不变。"
    if style: system += f"\n\n作者风格参考：\n{style[:500]}"
    try:
        result = call_llm(system, f"请润色以下正文：\n\n{text}", temperature=0.6, max_tokens=8192)
        return jsonify({"output": result["output"], "tokens_in": result.get("tokens_in", 0),
                        "tokens_out": result.get("tokens_out", 0), "duration_ms": result.get("duration_ms", 0)})
    except Exception as e:
        return jsonify({"error": f"LLM 调用失败: {e}"}), 500


# ── API: 日志 ──

@app.route("/api/logs")
def api_logs():
    kind = request.args.get("kind", "")
    with _log_lock:
        logs = [l for l in _log_buf if l["kind"] == kind] if kind else list(_log_buf)
    return jsonify(logs[-100:])


# ── 启动 ──

if __name__ == "__main__":
    _add_log("system", f"Forge 工作台启动 | 数据库: 已初始化")
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
