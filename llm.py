"""
llm.py — shared Claude helper for the protection engine.

Handles the Claude-specific quirks in ONE place:
  - newer models don't accept `temperature`
  - responses may include a ThinkingBlock before the text block
  - JSON responses may be wrapped in ```json ... ``` fences

Embeddings stay on OpenAI (see rag.py) — this is only for chat/generation.
"""

import os
import json as _json
import logging
from anthropic import Anthropic

log = logging.getLogger(__name__)

_key = os.getenv("ANTHROPIC_API_KEY")

from anthropic import Anthropic
...
anthropic_client = Anthropic(
    api_key=_key,
    timeout=90.0,
    max_retries=1,
)

# model roles - matched to how often and how urgently each route fires
# /analyze fires every few seconds during the live call and returns small JSON,
# so it runs on the fastest model. If its questions/prompts ever read noticeably
# weaker, set this back to "claude-sonnet-5" - it's a one-line change.
MODEL_ANALYZE = "claude-haiku-4-5-20251001"  # live, fires often - must land in ~1-2s
MODEL_SCRIPT  = "claude-sonnet-5"   # read-aloud script, streamed - quality matters
MODEL_PICTURE = "claude-sonnet-5"   # Your Picture links + guidance - the emotional pivot, precomputed
MODEL_PREPARE = "claude-sonnet-5"   # pre-call prep / email recap
MODEL_SUMMARY = "claude-opus-5"     # once per call, post-call, high-stakes - max quality
MODEL_QUALIFY = "claude-sonnet-5"
MODEL_SMS     = "claude-sonnet-5"


def _extract_text(resp):
    """Grab the text block, skipping any ThinkingBlock newer models return."""
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            return block.text
    for block in resp.content:
        if hasattr(block, "text"):
            return block.text
    return ""


def _strip_fences(text):
    t = (text or "").strip()
    if t.startswith("```"):
        # remove the opening ```json / ``` and closing ```
        t = t.split("```", 2)
        # t like ['', 'json\n{...}\n', ''] or ['', '{...}', '']
        if len(t) >= 2:
            body = t[1]
            if body.lstrip().lower().startswith("json"):
                body = body.lstrip()[4:]
            return body.strip()
    return t


def claude_text(system, user, model, max_tokens=1500):
    """Return Claude's plain text response."""
    resp = anthropic_client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
    )
    _log_cache_usage(resp)
    return _extract_text(resp).strip()


def _repair_truncated_json(text):
    """Best-effort recovery for a response that got cut off mid-value when it hit
    max_tokens. Walks the text tracking bracket/string depth, remembers the last
    point it was at a clean "key": value boundary (a comma at the top of the
    current nesting), cuts everything after that, and closes whatever objects/
    arrays were still open. Returns None if nothing was ever safely complete.

    e.g. '{"a":"1","b":{"c":"2","d":"unterm' -> '{"a":"1","b":{"c":"2"}}'
    - loses "d", keeps everything that finished before it."""
    stack = []
    last_safe_stack = None
    last_safe_end = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
        elif ch == "," and stack:
            last_safe_end = i
            last_safe_stack = list(stack)
    if last_safe_stack is None:
        return None
    closers = {"{": "}", "[": "]"}
    return text[:last_safe_end].rstrip() + "".join(closers[c] for c in reversed(last_safe_stack))


def claude_json(system, user, model, max_tokens=1500, effort=None):
    """Return Claude's response parsed as JSON (handles fences).
    effort: optional "low"/"medium"/"high" - limits how long the model reasons
    before answering (reasoning counts against max_tokens)."""
    full_system = system + "\n\nReturn ONLY valid JSON. No preamble, no markdown, no code fences."
    extra = {"output_config": {"effort": effort}} if effort else {}
    resp = anthropic_client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": full_system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
        **extra,
    )
    _log_cache_usage(resp)
    raw = _extract_text(resp)
    cleaned = _strip_fences(raw)
    try:
        return _json.loads(cleaned)
    except ValueError:
        # Tolerate stray text around the JSON: parse the outermost {...} block.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return _json.loads(cleaned[start:end + 1])
            except ValueError:
                pass
        # Hit the token cap mid-value: recover whatever fields finished before
        # the cut, rather than losing everything the adviser just said.
        if getattr(resp, "stop_reason", None) == "max_tokens":
            repaired = _repair_truncated_json(cleaned)
            if repaired:
                try:
                    result = _json.loads(repaired)
                    log.warning("claude_json: hit max_tokens and was truncated - recovered a "
                                "partial result (%d of %d chars); some fields will be missing",
                                len(repaired), len(cleaned))
                    return result
                except ValueError:
                    pass
        log.error("claude_json could not parse response (stop_reason=%s, output_tokens=%s): %r",
                  getattr(resp, "stop_reason", "?"),
                  getattr(getattr(resp, "usage", None), "output_tokens", "?"),
                  raw[:500])
        raise


def _log_cache_usage(resp):
    """Log whether the prompt cache was hit, so we can verify caching works."""
    try:
        u = resp.usage
        created = getattr(u, "cache_creation_input_tokens", 0) or 0
        read = getattr(u, "cache_read_input_tokens", 0) or 0
        if read:
            log.info("prompt cache HIT: %d tokens read from cache", read)
        elif created:
            log.info("prompt cache MISS (creating): %d tokens cached for next time", created)
    except Exception:
        pass
