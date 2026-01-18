#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemini API compatibility layer for OpenAI-style code.

This module provides a drop-in replacement for OpenAI's chat completions API
that actually calls Google's Gemini API. It handles authentication, message
formatting, and response parsing to maintain compatibility with existing code.
"""

import os
import json
import base64
import requests
from typing import List, Dict, Any, Optional
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv

# Google auth (service account) optional imports
from google.oauth2 import service_account  # type: ignore
import google.auth.transport.requests as tr  # type: ignore

# Import debug logging utility
# from debug_utils import print
# Token usage tracker
try:
    from token_usage import add_call_usage  # type: ignore
except Exception:
    def add_call_usage(_usage):
        return

load_dotenv()

# Configuration
API_VERSION = os.getenv("GEMINI_API_VERSION", "v1beta")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
API_KEY = os.getenv("GOOGLE_API_KEY")  # optional; service account can be used instead
SCOPES = ["https://www.googleapis.com/auth/generative-language"]

# Token caching for service account auth
_TOKEN_CACHE = {"token": None, "expiry": None}

# Last usage metadata for external inspection (non-threadsafe best-effort)
_LAST_USAGE: Optional[Dict[str, int]] = None

def get_last_usage() -> Dict[str, int]:
    """Return last seen usage metadata from a Gemini call in this process.
    Keys: prompt_tokens, candidates_tokens, total_tokens.
    Returns empty dict if none recorded.
    """
    return dict(_LAST_USAGE or {})


def _load_creds():
    """Load service account credentials from env or file, if available."""
    if os.getenv("GCP_SA_KEY_JSON"):
        info = json.loads(os.environ["GCP_SA_KEY_JSON"])
        return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)

    if os.getenv("GCP_SA_KEY_B64"):
        raw = base64.b64decode(os.environ["GCP_SA_KEY_B64"])
        info = json.loads(raw.decode("utf-8"))
        return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)

    path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if path:
        p = Path(path)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        return service_account.Credentials.from_service_account_file(str(p), scopes=SCOPES)

    return None


def _access_token() -> Optional[str]:
    """Return a valid OAuth access token using service account, if configured.

    Returns None if service account creds are not available.
    """
    now = datetime.now(timezone.utc)
    if _TOKEN_CACHE["token"] and _TOKEN_CACHE["expiry"]:
        expiry = _TOKEN_CACHE["expiry"]
        if getattr(expiry, "tzinfo", None) is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        if expiry - now > timedelta(minutes=2):
            return _TOKEN_CACHE["token"]

    creds = _load_creds()
    if not creds:
        return None
    creds.refresh(tr.Request())
    _TOKEN_CACHE["token"] = creds.token
    _TOKEN_CACHE["expiry"] = creds.expiry if creds.expiry else now + timedelta(minutes=30)
    return _TOKEN_CACHE["token"]


def _merge_messages_for_gemini(messages: List[Dict[str, Any]]):
    """Convert OpenAI-style messages to Gemini contents.
       Gemini accepts roles 'user' and 'model'. We fold all system prompts
       into the first user turn as plain text.
    """
    system_texts = [m["content"] for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    contents = []
    first_user_idx = next((i for i, m in enumerate(non_system) if m["role"] == "user"), None)

    for i, m in enumerate(non_system):
        role = "user" if m["role"] != "assistant" else "model"
        text = str(m.get("content", ""))
        if i == first_user_idx and system_texts:
            text = "System instructions:\n" + "\n".join(system_texts) + "\n\n" + text
        contents.append({"role": role, "parts": [{"text": text}]})
    return contents


def _normalize_model_name(model_name: Optional[str]) -> str:
    """Normalize common aliases and strip unsupported suffixes like '-latest'."""
    name = (model_name or MODEL).strip()
    # Allow users to pass 'models/xxx' too
    if name.startswith("models/"):
        name = name[len("models/"):]
    # Strip '-latest' suffix which is not resolvable on some API versions
    if name.endswith("-latest"):
        name = name[:-len("-latest")]
    # Map a few common aliases
    aliases = {
        "gemini-flash": "gemini-2.5-flash",
        "gemini-pro": "gemini-2.5-pro",
    }
    return aliases.get(name, name)


def _fallback_models_for(name: str) -> List[str]:
    """Return a sequence of fallback model names for robustness."""
    base = _normalize_model_name(name)
    if "pro" in base:
        return [base, "gemini-2.5-pro", "gemini-1.5-pro", "gemini-2.5-flash"]
    # default to flash family
    return [base, "gemini-2.5-flash", "gemini-1.5-flash"]


class OpenAICompat:
    """Shim so existing code like:
         resp = openai.chat.completions.create(model=..., messages=...)
         resp.choices[0].message.content
       keeps working, but hits Gemini instead (non-streaming).
    """
    def __init__(self, default_model: Optional[str] = None):
        self.default_model = default_model or MODEL
        self.chat = SimpleNamespace(completions=self)

    def create(self,
               model: Optional[str] = None,
               messages: List[Dict[str, Any]] = None,
               temperature: float = 0.0,
               max_tokens: int = 1024,
               n: int = 1,
               **kwargs):
        requested_model = model or self.default_model
        models_to_try = _fallback_models_for(requested_model)

        last_error: Optional[Any] = None
        data = None

        # Choose auth mechanism once
        headers: Dict[str, str] = {}
        if API_KEY:
            headers["x-goog-api-key"] = API_KEY
        else:
            token = _access_token()
            if not token:
                raise RuntimeError("No GOOGLE_API_KEY or service account credentials found for Gemini API auth")
            headers["Authorization"] = f"Bearer {token}"

        payload = {
            "contents": _merge_messages_for_gemini(messages or []),
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_tokens),
                "candidateCount": int(n),
            }
        }

        # Try the requested model and fallbacks if 404 or unsupported
        for model_name in models_to_try:
            normalized = _normalize_model_name(model_name)
            url = f"https://generativelanguage.googleapis.com/{API_VERSION}/models/{normalized}:generateContent"
            r = requests.post(url, headers=headers, json=payload, timeout=90)
            try:
                r.raise_for_status()
                data = r.json()
                break
            except requests.HTTPError as e:
                try:
                    detail = r.json()
                except Exception:
                    detail = r.text
                last_error = (r.status_code, detail)
                # If not a 404/unsupported error, don't continue fallbacks
                if r.status_code != 404:
                    raise RuntimeError(f"Gemini API error {r.status_code}: {detail}") from e
                # else continue with next fallback

        if data is None:
            status, detail = last_error if last_error else ("unknown", "unknown")
            raise RuntimeError(f"Gemini API error {status}: {detail}")
        
        # Debug: Log the raw response structure
        print(f"Gemini API raw response: {json.dumps(data, ensure_ascii=False, indent=2)[:500]}...")
        
        # Extract usage metadata if present
        usage_md = data.get("usageMetadata") or {}
        usage = {
            "prompt_tokens": int(usage_md.get("promptTokenCount") or 0),
            "candidates_tokens": int(usage_md.get("candidatesTokenCount") or 0),
            "total_tokens": int(usage_md.get("totalTokenCount") or 0),
        }
        global _LAST_USAGE
        _LAST_USAGE = usage
        print(f"Gemini usage: {usage}")
        try:
            add_call_usage(usage)
        except Exception as e:
            print(f"Token tracker add_call_usage failed: {e}")
        
        # Build OpenAI-like response object
        choices = []
        for c in data.get("candidates", []):
            txt = ""
            try:
                # Debug: Log the candidate structure
                print(f"Processing candidate: {json.dumps(c, ensure_ascii=False, indent=2)[:300]}...")
                
                # Check if this is a malformed response (missing parts)
                if "content" in c and "parts" not in c["content"]:
                    print(f"Malformed response detected - missing 'parts' in content")
                    # Check if it's a MAX_TOKENS error
                    if c.get("finishReason") == "MAX_TOKENS":
                        txt = ""
                    else:
                        txt = ""
                else:
                    txt = c["content"]["parts"][0]["text"]
                    print(f"Extracted text: {txt[:100]}...")
            except Exception as e:
                print(f"Error extracting text from candidate: {e}")
                print(f"Candidate structure: {c}")
                txt = ""
            choices.append(SimpleNamespace(message=SimpleNamespace(content=txt)))
        
        # Handle empty/blocked responses gracefully
        if not choices:
            print("No candidates found, creating empty choice")
            choices = [SimpleNamespace(message=SimpleNamespace(content=""))]
        
        result = SimpleNamespace(choices=choices, usage=usage)
        print(f"Final result structure: {result.choices[0].message.content[:100]}...")
        return result
