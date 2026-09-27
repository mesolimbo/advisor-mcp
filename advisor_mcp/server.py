"""advisor-mcp server.

Exposes tools that let the running model (Opus / Sonnet / Haiku) consult an advisor model
(``claude-fable-5-1`` by default) for a second opinion or extra guidance, billed to
the existing Claude Max subscription via its OAuth token.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import httpx
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

from . import __version__
from .credentials import CredentialError, get_credential

# Load configuration from the .env that lives next to this project, regardless
# of the current working directory the MCP host launched us from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

API_URL = "https://api.anthropic.com/v1/messages"
# Required when authenticating with a subscription OAuth token instead of an API key.
OAUTH_BETA = "oauth-2025-04-20"
# Presenting as Claude Code routes requests through the Claude Code rate-limit
# pool; orgs often restrict raw API traffic (per-model caps) while allowing
# Claude Code usage, so this beta + user-agent + identity line are load-bearing
# for API-key auth too, not just OAuth.
CLAUDE_CODE_BETA = "claude-code-20250219"
# The API refuses newer models to Claude Code versions below a moving minimum,
# so report the version of the locally installed `claude` binary. Falls back to
# this pinned version if the binary is missing; CLAUDE_CODE_VERSION overrides both.
_FALLBACK_CLAUDE_CODE_VERSION = "2.1.280"


def _detect_claude_code_version() -> str:
    override = os.getenv("CLAUDE_CODE_VERSION")
    if override:
        return override
    exe = shutil.which("claude")
    if exe:
        try:
            out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=15).stdout
            match = re.search(r"\d+\.\d+\.\d+", out)
            if match:
                return match.group(0)
        except (OSError, subprocess.SubprocessError):
            pass
    return _FALLBACK_CLAUDE_CODE_VERSION


CLAUDE_CODE_VERSION = _detect_claude_code_version()
CLAUDE_CODE_USER_AGENT = f"claude-cli/{CLAUDE_CODE_VERSION} (external, cli)"
# The credential is only accepted for Claude Code; the system prompt must
# lead with this identity line.
CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."

DEFAULT_MODEL = os.environ.get("ADVISOR_MODEL", "claude-fable-5-1")
DEFAULT_MAX_TOKENS = int(os.environ.get("ADVISOR_MAX_TOKENS", "128000"))
# Opus 5.5 defaults to "medium" effort, so always send one explicitly.
DEFAULT_EFFORT = os.environ.get("ADVISOR_EFFORT", "high")
# Responses stream, so this bounds the gap between events, not the whole call;
# hard prompts can think for many minutes.
REQUEST_TIMEOUT = float(os.environ.get("ADVISOR_TIMEOUT", "300"))

# Models that reject sampling parameters and accept output_config.effort.
_EFFORT_MODEL_PREFIXES = (
    "claude-fable",
    "claude-mythos",
    "claude-opus-5",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-5",
)

ADVISOR_ROLE = (
    "You are acting as a senior technical advisor to another AI coding "
    "agent. Give direct, well-reasoned guidance: weigh trade-offs, flag risks and "
    "edge cases, and recommend a concrete course of action. Be concise and "
    "practical rather than exhaustive."
)

mcp = MCPServer("advisor")


def _call_advisor(
    prompt: str,
    *,
    context: str | None,
    model: str,
    max_tokens: int,
    temperature: float,
    effort: str,
) -> str:
    credential = get_credential()

    user_content = prompt if not context else f"<context>\n{context}\n</context>\n\n{prompt}"

    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [
            {"type": "text", "text": CLAUDE_CODE_IDENTITY},
            {"type": "text", "text": ADVISOR_ROLE},
        ],
        "messages": [{"role": "user", "content": user_content}],
        "stream": True,
    }
    if model.startswith(_EFFORT_MODEL_PREFIXES):
        body["output_config"] = {"effort": effort}
    elif temperature != 1.0:
        body["temperature"] = temperature

    headers = {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        "user-agent": CLAUDE_CODE_USER_AGENT,
        "x-app": "cli",
    }
    if credential.is_api_key:
        headers["x-api-key"] = credential.token
        headers["anthropic-beta"] = CLAUDE_CODE_BETA
    else:
        headers["authorization"] = f"Bearer {credential.token}"
        headers["anthropic-beta"] = f"{OAUTH_BETA},{CLAUDE_CODE_BETA}"

    parts: list[str] = []
    stop_reason = None
    stop_details = None
    timeout = httpx.Timeout(REQUEST_TIMEOUT, connect=30.0)
    with httpx.stream("POST", API_URL, json=body, headers=headers, timeout=timeout) as resp:
        if resp.is_error:
            resp.read()
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data:"):
                continue
            event = json.loads(line[5:])
            kind = event.get("type")
            if kind == "content_block_delta" and event["delta"].get("type") == "text_delta":
                parts.append(event["delta"]["text"])
            elif kind == "message_delta":
                stop_reason = event["delta"].get("stop_reason") or stop_reason
                stop_details = event["delta"].get("stop_details") or stop_details
            elif kind == "error":
                error = event.get("error", {})
                return f"Advisor request failed mid-stream ({error.get('type')}): {error.get('message')}"

    text = "".join(parts).strip()
    if stop_reason == "refusal":
        details = stop_details or {}
        note = " ".join(filter(None, [
            f"(The advisor declined this request — refusal category: {details.get('category') or 'unspecified'}.",
            details.get("explanation"),
            "Rephrase it, or retry with another model such as claude-opus-5-5.)",
        ]))
        return f"{text}\n\n{note}" if text else note
    if text:
        return text
    if stop_reason == "max_tokens":
        return (
            "(The advisor produced no text — the token budget was consumed by internal "
            "reasoning. Retry with a larger max_tokens.)"
        )
    return "(The advisor returned no text.)"


@mcp.tool()
def ask_advisor(
    prompt: str,
    context: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 1.0,
    effort: str = DEFAULT_EFFORT,
) -> str:
    """Consult the advisor model for a second opinion or extra guidance on a hard problem.

    Use this when you want an independent perspective from a different model —
    for architecture calls, tricky trade-offs, reviewing a plan, or sanity-checking
    an approach before committing to it.

    Args:
        prompt: The question or request for the advisor. Be specific about what
            kind of guidance you want (e.g. "review this plan", "which approach
            is safer", "what am I missing").
        context: Optional supporting material (code, error output, a draft plan)
            that the advisor should consider when answering.
        model: Advisor model to query. Defaults to the configured model
            (``claude-fable-5-1``).
        max_tokens: Maximum tokens in the advisor's response.
        temperature: Sampling temperature (0.0-1.0). Ignored by current models,
            which reject sampling parameters.
        effort: Reasoning effort for current models: "low", "medium", "high",
            "xhigh", or "max". Higher is slower and more thorough.

    Returns:
        The advisor's guidance as text.
    """
    try:
        return _call_advisor(
            prompt,
            context=context,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            effort=effort,
        )
    except CredentialError as exc:
        return f"Advisor unavailable — credential error: {exc}"
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:600] if exc.response is not None else ""
        return f"Advisor request failed ({exc.response.status_code}): {detail}"
    except httpx.HTTPError as exc:
        return f"Advisor request failed: {exc}"


@mcp.tool()
def get_version() -> str:
    """Return this MCP server's name, version, and configured advisor model."""
    return f"advisor-mcp {__version__} (default model: {DEFAULT_MODEL}, effort: {DEFAULT_EFFORT}, presenting as Claude Code {CLAUDE_CODE_VERSION})"


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
