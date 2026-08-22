
# Use for Stanford AI Hub and replace for a different LLM API if necessary 

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

# AI Hub SDK — once `pip install -e <som-guideai/aihub>` has been run.
from aihub import get_openai_client, get_anthropic_client, get_gemini_client

try:
    from aihub_cost import log_call as _log_call
except ImportError:
    _log_call = None

_KEY_FILE = Path.home() / ".aihub_key"


def _resolve_api_key() -> str:
    key = os.environ.get("AIHUB_API_KEY")
    if key:
        return key.strip()
    if _KEY_FILE.exists():
        return _KEY_FILE.read_text().strip()
    raise RuntimeError(
        "AI Hub API key not found. Set $AIHUB_API_KEY or write the key to "
        f"{_KEY_FILE}"
    )


DEFAULT_MODELS = {
    "gpt":    "gpt-5",
    "claude": "us.anthropic.claude-sonnet-4-6",
    "gemini": "gemini-2.5-pro",
}

# Output cap. info_categories needs ~1,000–1,500; judge calls need ~500.
# 4096 gives headroom for either.
MAX_OUTPUT_TOKENS = 16000


_clients: dict[str, Any] = {}


def _get_client(provider: str):
    if provider in _clients:
        return _clients[provider]
    api_key = _resolve_api_key()
    if provider == "gpt":
        c = get_openai_client(api_key=api_key)
    elif provider == "claude":
        c = get_anthropic_client(api_key=api_key)
    elif provider == "gemini":
        c = get_gemini_client(api_key=api_key)
    else:
        raise ValueError(f"Unknown AI Hub provider: {provider}")
    _clients[provider] = c
    return c


def _call_gpt_sync(user_prompt: str, system_instructions: Optional[str]) -> str:
    client = _get_client("gpt")
    messages = []
    if system_instructions:
        messages.append({"role": "system", "content": system_instructions})
    messages.append({"role": "user", "content": user_prompt})
    resp = client.chat.completions.create(
        model                 = DEFAULT_MODELS["gpt"],
        messages              = messages,
        max_completion_tokens = MAX_OUTPUT_TOKENS,
        response_format       = {"type": "json_object"},
    )
    if _log_call is not None and getattr(resp, "usage", None) is not None:
        _log_call(
            model         = DEFAULT_MODELS["gpt"],
            input_tokens  = resp.usage.prompt_tokens,
            output_tokens = resp.usage.completion_tokens,
            prompt_chars  = (len(system_instructions or "") + len(user_prompt)),
        )
    return resp.choices[0].message.content


def _call_claude_sync(user_prompt: str, system_instructions: Optional[str]) -> str:
    client = _get_client("claude")
    kwargs: dict[str, Any] = {
        "model":      DEFAULT_MODELS["claude"],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "messages":   [{"role": "user", "content": user_prompt}],
    }
    if system_instructions:
        kwargs["system"] = system_instructions
    resp = client.messages.create(**kwargs)
    if _log_call is not None and getattr(resp, "usage", None) is not None:
        _log_call(
            model         = DEFAULT_MODELS["claude"],
            input_tokens  = resp.usage.input_tokens,
            output_tokens = resp.usage.output_tokens,
            prompt_chars  = (len(system_instructions or "") + len(user_prompt)),
        )
    return resp.content[0].text


def _call_gemini_sync(user_prompt: str, system_instructions: Optional[str]) -> str:
    client = _get_client("gemini")
    config: dict[str, Any] = {
        "max_output_tokens":  MAX_OUTPUT_TOKENS,
        "response_mime_type": "application/json",
    }
    if system_instructions:
        config["system_instruction"] = system_instructions
    resp = client.models.generate_content(
        model    = DEFAULT_MODELS["gemini"],
        contents = user_prompt,
        config   = config,
    )
    if _log_call is not None:
        meta = getattr(resp, "usage_metadata", None)
        if meta is not None:
            _log_call(
                model         = DEFAULT_MODELS["gemini"],
                input_tokens  = meta.prompt_token_count,
                output_tokens = meta.candidates_token_count,
                prompt_chars  = (len(system_instructions or "") + len(user_prompt)),
            )
    return resp.text


async def send_single_message(
    user_prompt:         str,
    system_instructions: Optional[str] = None,
    model_id:            str           = "gpt",
) -> str:
    """
    Async wrapper around AI Hub provider clients. The underlying SDK calls
    are synchronous, so we run them on a worker thread to avoid blocking
    the asyncio event loop.

    model_id values: "gpt", "claude", "gemini".
    """
    if model_id == "gpt":
        return await asyncio.to_thread(_call_gpt_sync, user_prompt, system_instructions)
    if model_id == "claude":
        return await asyncio.to_thread(_call_claude_sync, user_prompt, system_instructions)
    if model_id == "gemini":
        return await asyncio.to_thread(_call_gemini_sync, user_prompt, system_instructions)
    raise ValueError(
        f"Unknown model_id {model_id!r}. Expected one of: gpt, claude, gemini."
    )


def parse_llm_json(
    response:          str,
    expected_fields:   list,
    validate_booleans: bool = False,
) -> dict[str, Any]:
    # Strategy 1: direct parse
    try:
        result = json.loads(response)
        if expected_fields:
            _validate_schema(result, expected_fields, validate_booleans)
        return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: extract from markdown code blocks
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(1))
            if expected_fields:
                _validate_schema(result, expected_fields, validate_booleans)
            return result
        except (json.JSONDecodeError, ValueError):
            response = m.group(1)

    # Strategy 3: find a complete JSON object inside the response
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(0))
            if expected_fields:
                _validate_schema(result, expected_fields, validate_booleans)
            return result
        except (json.JSONDecodeError, ValueError):
            response = m.group(0)

    # Strategy 4: clean common formatting issues
    cleaned = _clean_json_string(response)
    try:
        result = json.loads(cleaned)
        if expected_fields:
            _validate_schema(result, expected_fields, validate_booleans)
        return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 5: fix common LLM mistakes
    fixed = _fix_common_llm_errors(cleaned)
    try:
        result = json.loads(fixed)
        if expected_fields:
            _validate_schema(result, expected_fields, validate_booleans)
        return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 6: line-by-line field extraction (last resort, booleans only)
    if expected_fields:
        result = _extract_fields_line_by_line(response, expected_fields)
        if result:
            _validate_schema(result, expected_fields, validate_booleans)
            return result

    raise ValueError(
        f"Could not parse JSON after all recovery attempts. Response: {response[:500]}"
    )


def _validate_schema(
    data:              dict[str, Any],
    expected_fields:   list,
    validate_booleans: bool = False,
) -> None:
    missing = set(expected_fields) - set(data.keys())
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")
    extra = set(data.keys()) - set(expected_fields)
    if extra:
        raise ValueError(f"Unexpected fields found: {sorted(extra)}")
    if validate_booleans:
        bad = {k: type(v).__name__ for k, v in data.items() if not isinstance(v, bool)}
        if bad:
            raise ValueError(f"All fields must be boolean. Found: {bad}")


def _clean_json_string(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start: end + 1]
    return text


def _fix_common_llm_errors(text: str) -> str:
    text = re.sub(r'[a-zA-Z_]+("[\w_]+"\s*:)', r"\1", text)
    text = re.sub(r'("[^"]+")[\w\s]+(:)', r"\1\2", text)
    text = re.sub(r'([,{]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
    text = re.sub(r'("[^"]+"\s*:\s*)"\1', r"\1", text)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    text = re.sub(r':\s*[a-zA-Z_]+\s*"([^"]+)"', r': "\1"', text)
    text = re.sub(r'(,\s*)[a-zA-Z_]+\s*("[\w_]+"\s*:)', r"\1\2", text)
    text = re.sub(r'("[^"]+"\s*:\s*)$', r"\1false", text)
    lines = text.split("\n")
    fixed_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if (
            fixed_lines and stripped
            and ":" not in stripped
            and not stripped.endswith(("{", "}", ","))
        ):
            fixed_lines[-1] += " " + stripped
        else:
            fixed_lines.append(line)
    return "\n".join(fixed_lines)


def _extract_fields_line_by_line(
    text:            str,
    expected_fields: list,
) -> Optional[dict[str, Any]]:
    result: dict[str, Any] = {}
    for field in expected_fields:
        for pattern in (
            rf'"{field}"\s*:\s*(true|false)',
            rf"{field}\s*:\s*(true|false)",
        ):
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                result[field] = m.group(1).strip().lower() == "true"
                break
    return result if len(result) == len(expected_fields) else None
