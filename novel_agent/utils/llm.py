"""
LLM 调用工具 — 独立实现，不依赖旧 scripts/
"""

from __future__ import annotations

import os
import time
import json
from typing import Optional, Any

from novel_agent.utils.file_io import read_env

# 首次加载时读取环境变量
read_env()

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com").rstrip("/")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "deepseek-chat")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    stream: bool = False,
) -> dict:
    """
    调用 LLM，返回 {"output": str, "tokens_in": int, "tokens_out": int, "duration_ms": int}
    """
    if not API_KEY:
        return {
            "output": f"（模拟输出 — 未配置 API_KEY，请在 .env 中设置 DEEPSEEK_API_KEY）\n\n【System】\n{system_prompt[:200]}...\n\n【User】\n{user_prompt[:200]}...",
            "tokens_in": 0, "tokens_out": 0, "duration_ms": 0,
        }

    import requests

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    t0 = time.time()
    try:
        resp = requests.post(
            f"{API_BASE}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        output = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return {
            "output": output,
            "tokens_in": usage.get("prompt_tokens", 0),
            "tokens_out": usage.get("completion_tokens", 0),
            "duration_ms": int((time.time() - t0) * 1000),
        }
    except Exception as e:
        return {
            "output": f"（LLM 调用失败: {e}）",
            "tokens_in": 0, "tokens_out": 0, "duration_ms": int((time.time() - t0) * 1000),
        }


def call_llm_with_messages(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    model: Optional[str] = None,
    temperature: float = 0.4,
    max_tokens: int = 2048,
    tool_choice: str = "auto",
) -> dict:
    """
    支持 function calling 的 LLM 调用（DeepSeek 兼容 OpenAI tools 格式）。

    参数：
        messages: OpenAI 消息列表，可包含 system/user/assistant/tool 角色消息
        tools:    OpenAI function schema 列表，形如
                  [{"type":"function","function":{"name":..., "description":..., "parameters":{...}}}]
        tool_choice: "auto" | "none" | {"type":"function","function":{"name":"xxx"}}

    返回：
        {
            "content": str,                  # 助理文本回复（可能为空字符串，纯工具调用时）
            "tool_calls": list[dict] | None, # 工具调用列表，每项 {id, name, arguments(dict)}
            "finish_reason": str,
            "tokens_in": int, "tokens_out": int, "duration_ms": int,
        }
    """
    if not API_KEY:
        return {
            "content": "（未配置 API_KEY，无法启动助理型 Brain。请在 .env 设置 DEEPSEEK_API_KEY）",
            "tool_calls": None,
            "finish_reason": "no_api_key",
            "tokens_in": 0, "tokens_out": 0, "duration_ms": 0,
        }

    import requests

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice

    t0 = time.time()
    try:
        resp = requests.post(
            f"{API_BASE}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        msg = choice.get("message", {})

        # 解析 tool_calls（OpenAI 格式）
        raw_calls = msg.get("tool_calls")
        parsed_calls = None
        if raw_calls:
            parsed_calls = []
            for tc in raw_calls:
                fn = tc.get("function", {})
                args_str = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except Exception:
                    args = {"_raw": args_str}
                parsed_calls.append({
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "arguments": args,
                })

        usage = data.get("usage", {})
        return {
            "content": msg.get("content") or "",
            "tool_calls": parsed_calls,
            "finish_reason": choice.get("finish_reason", ""),
            "tokens_in": usage.get("prompt_tokens", 0),
            "tokens_out": usage.get("completion_tokens", 0),
            "duration_ms": int((time.time() - t0) * 1000),
        }
    except Exception as e:
        return {
            "content": f"（LLM 调用失败: {e}）",
            "tool_calls": None,
            "finish_reason": "error",
            "tokens_in": 0, "tokens_out": 0,
            "duration_ms": int((time.time() - t0) * 1000),
        }
