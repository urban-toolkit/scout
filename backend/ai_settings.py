"""Generic LLM chat: settings persistence + provider dispatch.

SCOUT has no user accounts, so this is one global config (unlike Curio's
per-user ``user.llm_*`` fields) stored in ``data/ai_settings.json`` next to
the other on-disk state ``server.py`` keeps under ``DATA_DIR``. The provider
dispatch below mirrors the shape of Curio's ``agents/providers.py``, trimmed
to a single blocking chat call - no streaming, no tool use, no agents.
"""

from __future__ import annotations

import json
from pathlib import Path

SETTINGS_PATH = Path("data/ai_settings.json")

_DEFAULT_SETTINGS = {
    "api_type": "openai_compatible",  # "openai_compatible" | "anthropic" | "gemini"
    "base_url": "",
    "api_key": "",
    "model": "",
}


def _read() -> dict:
    if not SETTINGS_PATH.exists():
        return dict(_DEFAULT_SETTINGS)
    try:
        data = json.loads(SETTINGS_PATH.read_text())
    except json.JSONDecodeError:
        return dict(_DEFAULT_SETTINGS)
    return {**_DEFAULT_SETTINGS, **data}


def _write(settings: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2))


def is_configured(s: dict | None = None) -> bool:
    s = s or _read()
    if not s["model"]:
        return False
    if s["api_type"] == "openai_compatible" and s["base_url"]:
        # A local/self-hosted endpoint (Ollama, LM Studio, vLLM, ...) may need
        # no key at all.
        return True
    return bool(s["api_key"])


def get_public_settings() -> dict:
    """Settings safe to send to the frontend: never the raw key."""
    s = _read()
    return {
        "apiType": s["api_type"],
        "baseUrl": s["base_url"],
        "model": s["model"],
        "hasApiKey": bool(s["api_key"]),
        "configured": is_configured(s),
    }


def update_settings(patch: dict) -> dict:
    """Apply a partial update and return the new public settings.

    A blank/absent ``apiKey`` means "keep the saved one" (same convention as
    Curio's AI Settings panel); pass ``clearApiKey: true`` to remove it.
    """
    s = _read()
    if patch.get("apiType"):
        s["api_type"] = patch["apiType"]
    if "baseUrl" in patch:
        s["base_url"] = patch["baseUrl"] or ""
    if "model" in patch:
        s["model"] = patch["model"] or ""
    if patch.get("clearApiKey"):
        s["api_key"] = ""
    elif patch.get("apiKey"):
        s["api_key"] = patch["apiKey"]
    _write(s)
    return get_public_settings()


class ChatError(ValueError):
    """A user-facing chat failure (not configured, provider error, ...)."""


def list_provider_models(
    api_type: str | None, base_url: str | None, api_key: str | None
) -> dict:
    """The models an OpenAI-compatible endpoint says it serves.

    Mirrors Curio's ``/api/agents/provider-models``: called mid-edit, before
    the user saves, so each argument that is blank falls back to what is
    already on disk - an already-configured user can refresh the list
    without retyping their key.

    Only ``openai_compatible`` is listable: Anthropic and Gemini have no
    equivalent ``/models`` in the shape the OpenAI SDK speaks, so they return
    an empty, non-listable result and the caller keeps a free-text field.
    """
    saved = _read()
    api_type = (api_type or "").strip() or saved["api_type"]
    base_url = (base_url or "").strip() or saved["base_url"]
    api_key = (api_key or "").strip() or saved["api_key"]

    if api_type != "openai_compatible":
        return {"models": [], "listable": False}

    from openai import OpenAI

    kwargs = {"api_key": api_key or "no-key", "timeout": 20.0}
    if base_url:
        kwargs["base_url"] = base_url
    try:
        listing = OpenAI(**kwargs).models.list()
    except Exception as exc:
        # A rejected key, an unreachable host, or an endpoint without /models
        # all mean "cannot offer a choice" - reported as a ChatError (400) the
        # panel can show verbatim, since the user is mid-edit.
        raise ChatError(f"Could not list models: {exc}") from exc

    models = sorted({m.id for m in listing.data if getattr(m, "id", None)})
    return {"models": models, "listable": True}


def run_chat(messages: list[dict], max_output_tokens: int | None = None) -> str:
    """Send ``messages`` (OpenAI-style ``[{"role", "content"}, ...]``) to the
    configured provider and return the assistant's reply text."""
    s = _read()
    if not is_configured(s):
        raise ChatError(
            "AI is not configured yet. Open AI Settings and add a provider, "
            "API key and model."
        )

    api_type = s["api_type"]
    api_key = s["api_key"]
    base_url = s["base_url"]
    model = s["model"]

    try:
        if api_type == "anthropic":
            import anthropic

            system_parts = [m["content"] for m in messages if m["role"] == "system"]
            chat_messages = [m for m in messages if m["role"] != "system"]
            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model=model,
                system="\n".join(system_parts) if system_parts else anthropic.NOT_GIVEN,
                messages=chat_messages,
                max_tokens=max_output_tokens or 1024,
            )
            return resp.content[0].text

        if api_type == "gemini":
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            system_parts = [m["content"] for m in messages if m["role"] == "system"]
            chat_messages = [m for m in messages if m["role"] != "system"]
            history = []
            for m in chat_messages[:-1]:
                role = "user" if m["role"] == "user" else "model"
                history.append({"role": role, "parts": [m["content"]]})
            last_user_msg = chat_messages[-1]["content"] if chat_messages else ""
            system_instruction = "\n".join(system_parts) if system_parts else None
            gen_model = genai.GenerativeModel(model, system_instruction=system_instruction)
            chat = gen_model.start_chat(history=history)
            response = chat.send_message(last_user_msg)
            return response.text

        # openai_compatible (default) - also covers Ollama, LM Studio, vLLM, etc.
        from openai import OpenAI

        kwargs = {"api_key": api_key or "no-key"}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        completion = client.chat.completions.create(model=model, messages=messages)
        return completion.choices[0].message.content
    except ChatError:
        raise
    except Exception as e:
        raise ChatError(f"AI request failed: {e}") from e
