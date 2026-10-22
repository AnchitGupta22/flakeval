"""Talking to a model: per-call random seeds, and a caller for any OpenAI-compatible chat endpoint.

A system can instead supply its own Python function (key "call"), so any other SDK or local model works too.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import zlib


def call_seed(base_seed: int, item_id: str, run: int) -> int:
    """A different seed for every (question, run) so repeated runs genuinely differ, yet the study is reproducible."""
    return zlib.crc32(f"{base_seed}|{item_id}|{run}".encode()) & 0x7FFFFFFF


def make_caller(system: dict, max_tokens: int):
    """Returns f(prompt, system_prompt, temperature, seed) -> answer text."""
    if system.get("call"):
        user_fn = system["call"]
        return lambda prompt, system_prompt, temperature, seed: user_fn(
            prompt=prompt, system_prompt=system_prompt, temperature=temperature, seed=seed, max_tokens=max_tokens)

    for key in ("model", "base_url"):
        if not system.get(key):
            raise SystemExit(f"System {system.get('name')!r} needs {key!r} (or a 'call' function).")
    api_key = None
    if system.get("api_key_env"):
        api_key = os.environ.get(system["api_key_env"])
        if not api_key:
            raise SystemExit(f"Environment variable {system['api_key_env']} is not set (needed for system "
                             f"{system['name']!r}). Set it in your terminal before running.")
    url = system["base_url"].rstrip("/") + "/chat/completions"
    limit_param = system.get("token_limit_param", "max_tokens")

    def caller(prompt, system_prompt, temperature, seed):
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + \
                   [{"role": "user", "content": prompt}]
        body = {"model": system["model"], "messages": messages, limit_param: max_tokens}
        if temperature is not None:
            body["temperature"] = temperature
        if system.get("send_seed", True):
            body["seed"] = seed
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
        last_error = None
        for attempt in range(4):  # retry busy/overloaded servers; fail loudly on real errors
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    return json.loads(resp.read())["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504):
                    last_error = f"HTTP {e.code}"
                else:
                    detail = e.read().decode("utf-8", "replace")[:300]
                    raise RuntimeError(f"{system['name']}: server answered HTTP {e.code} ({detail}). Check model name, "
                                       f"base_url and API key.") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last_error = str(e)
            time.sleep(2 ** attempt)
        raise RuntimeError(f"{system['name']}: gave up after 4 attempts ({last_error}). Is the server running?")

    return caller
