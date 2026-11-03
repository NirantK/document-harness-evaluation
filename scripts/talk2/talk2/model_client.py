"""Chat backends for the answer model, behind one function: chat(messages, tools) -> ChatTurn.

The default backend is local Ollama through its native /api/chat endpoint, which accepts `think`, `seed` and
`num_ctx` (the OpenAI-compatible endpoint ignores them). OpenRouter is the fallback; its key comes from
OPENROUTER_API_KEY or the macOS keychain item 'openrouter.ai-talk2'.

Messages use one neutral shape: {"role", "content", "tool_calls": [{"id", "name", "arguments": dict}],
"tool_call_id", "name"}; each backend converts it.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

import httpx

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
# Hosted endpoints of the same weights that accept both tools and a seed, with the quantization each one lists on OpenRouter.
HOSTED_QUANTIZATION = {"Darkbloom": "fp4", "DeepInfra": "bf16"}
HOSTED_PROVIDER = os.environ.get("TALK2_HOSTED_PROVIDER", "Darkbloom")
LOCAL_QUANTIZATION = "nvfp4"
HOSTED_RETRY_ATTEMPTS = 8
HOSTED_TIMEOUT_SECONDS = 300
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "hf.co/unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS"
SEED = int(os.environ.get("TALK2_SEED", "42"))  # a repeat under another seed measures run-to-run noise
CONTEXT_LENGTH = 49152
# A generation that does not stop held a request open for 30 minutes on 1 Nov while the runner's memory grew.
MAX_OUTPUT_TOKENS = 4096
# Reasoning counts against the output limit: at 4096 the model spent the whole limit reasoning and wrote no reply.
THINKING_OUTPUT_TOKENS = 16384
REQUEST_TIMEOUT_SECONDS = 900
PAUSE_BELOW_FREE_MEMORY_PERCENT = 15
RESUME_AT_FREE_MEMORY_PERCENT = 40
PAUSE_BELOW_FREE_DISK_GIB = 8
RESOURCE_POLL_SECONDS = 30
SERVER_RETRY_ATTEMPTS = 20
# Unloading discards the runner's prompt prefix cache, which turns a 0.1 s prompt into 8.6 s at 26k tokens, so the model stays
# loaded and is released only when memory is short.
LOADED_MODEL_KEEP_ALIVE = "30m"
UNLOAD_BELOW_FREE_MEMORY_PERCENT = 25


@dataclass
class ChatTurn:
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    provider: str = ""
    thinking: str = ""


def free_memory_percent() -> int:
    return int(subprocess.run(["sysctl", "-n", "kern.memorystatus_level"], capture_output=True, text=True).stdout)


def free_disk_gib() -> float:
    return shutil.disk_usage("/").free / 2**30


def unload_model(model: str) -> None:
    """Release the model's memory. The MLX runner's retained prompt cache grows while a model stays loaded."""
    httpx.post(OLLAMA_GENERATE_URL, json={"model": model, "keep_alive": 0}, timeout=120)


def wait_for_resources(model: str) -> None:
    """Between model calls, when no request is in flight: unload and wait while memory or disk is short."""
    if free_memory_percent() >= PAUSE_BELOW_FREE_MEMORY_PERCENT and free_disk_gib() >= PAUSE_BELOW_FREE_DISK_GIB:
        return
    print(f"RESOURCE PAUSE free_memory={free_memory_percent()}% free_disk={free_disk_gib():.0f}GiB", file=sys.stderr, flush=True)
    unload_model(model)
    while free_memory_percent() < RESUME_AT_FREE_MEMORY_PERCENT or free_disk_gib() < PAUSE_BELOW_FREE_DISK_GIB:
        time.sleep(RESOURCE_POLL_SECONDS)
    print(f"RESOURCE RESUME free_memory={free_memory_percent()}%", file=sys.stderr, flush=True)


def post_to_openrouter(body: dict) -> dict:
    """Retry rate limits, server errors and error bodies, so a provider outage is not recorded as a wrong answer."""
    problem = ""
    for attempt in range(HOSTED_RETRY_ATTEMPTS):
        try:
            response = httpx.post(OPENROUTER_URL, json=body, timeout=HOSTED_TIMEOUT_SECONDS,
                                  headers={"Authorization": f"Bearer {read_openrouter_key()}"})
            if response.status_code == 200:
                data = response.json()
                if data.get("choices"):
                    return data
                problem = f"no choices: {str(data)[:200]}"
            else:
                problem = f"HTTP {response.status_code}: {response.text[:200]}"
        except (httpx.TransportError, json.JSONDecodeError) as error:
            problem = f"{type(error).__name__}: {error}"
        print(f"OPENROUTER RETRY {attempt + 1}/{HOSTED_RETRY_ATTEMPTS} {problem}", file=sys.stderr, flush=True)
        time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"OpenRouter unavailable after {HOSTED_RETRY_ATTEMPTS} attempts: {problem}")


def release_memory_if_short(model: str) -> None:
    if free_memory_percent() < UNLOAD_BELOW_FREE_MEMORY_PERCENT:
        unload_model(model)


class MalformedToolCall(RuntimeError):
    """Ollama could not parse the tool call the model wrote, and rejected the whole reply."""


def post_to_ollama(body: dict) -> httpx.Response:
    """Retry when the server is down or its runner died, so an outage is not recorded as a wrong answer."""
    for attempt in range(SERVER_RETRY_ATTEMPTS):
        try:
            response = httpx.post(OLLAMA_URL, json=body, timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code < 500:
                return response
            problem = f"HTTP {response.status_code}: {response.text[:200]}"
            if "XML syntax error" in response.text:  # the same request fails the same way, so waiting cannot help
                raise MalformedToolCall(problem)
        except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError) as error:
            problem = f"{type(error).__name__}: {error}"
        print(f"OLLAMA RETRY {attempt + 1}/{SERVER_RETRY_ATTEMPTS} {problem}", file=sys.stderr, flush=True)
        time.sleep(RESOURCE_POLL_SECONDS)
        wait_for_resources(body["model"])
    raise RuntimeError(f"Ollama unavailable after {SERVER_RETRY_ATTEMPTS} attempts: {problem}")


def read_openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    return subprocess.run(
        ["security", "find-generic-password", "-s", "openrouter.ai-talk2", "-w"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def ollama_messages(messages: list[dict]) -> list[dict]:
    converted = []
    for message in messages:
        if message["role"] == "tool":
            converted.append({"role": "tool", "content": message["content"], "tool_name": message.get("name", "")})
        elif message.get("tool_calls"):
            converted.append({"role": "assistant", "content": message.get("content", ""),
                              **({"thinking": message["thinking"]} if message.get("thinking") else {}),
                              "tool_calls": [{"function": {"name": c["name"], "arguments": c["arguments"]}}
                                             for c in message["tool_calls"]]})
        else:
            converted.append({"role": message["role"], "content": message["content"]})
    return converted


def openai_messages(messages: list[dict]) -> list[dict]:
    converted = []
    for message in messages:
        if message["role"] == "tool":
            converted.append({"role": "tool", "content": message["content"], "tool_call_id": message["tool_call_id"]})
        elif message.get("tool_calls"):
            converted.append({"role": "assistant", "content": message.get("content", ""),
                              "tool_calls": [{"id": c["id"], "type": "function",
                                              "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                             for c in message["tool_calls"]]})
        else:
            converted.append({"role": message["role"], "content": message["content"]})
    return converted


def chat(messages: list[dict], tools: list[dict] | None = None, model: str = DEFAULT_MODEL,
         backend: str = "ollama", think: bool = False) -> ChatTurn:
    if backend == "ollama":
        wait_for_resources(model)
        body = {"model": model, "messages": ollama_messages(messages), "stream": False, "think": think,
                "keep_alive": LOADED_MODEL_KEEP_ALIVE, "options": {"temperature": 0, "seed": SEED, "num_ctx": CONTEXT_LENGTH,
                            "num_predict": MAX_OUTPUT_TOKENS}}
        if tools:
            body["tools"] = tools
        response = post_to_ollama(body)
        response.raise_for_status()
        data = response.json()
        message = data["message"]
        calls = [{"id": f"call_{index}", "name": c["function"]["name"], "arguments": c["function"]["arguments"] or {}}
                 for index, c in enumerate(message.get("tool_calls") or [])]
        return ChatTurn(message.get("content") or "", calls, data.get("prompt_eval_count", 0),
                        data.get("eval_count", 0), "ollama", message.get("thinking") or "")
    if backend == "openrouter":
        body = {"model": model, "messages": openai_messages(messages), "temperature": 0, "seed": SEED,
                "max_tokens": THINKING_OUTPUT_TOKENS if think else MAX_OUTPUT_TOKENS, "reasoning": {"enabled": think},
                "provider": {"order": [HOSTED_PROVIDER], "allow_fallbacks": False}}
        if tools:
            body["tools"] = tools
        data = post_to_openrouter(body)
        message = data["choices"][0]["message"]
        calls = []
        for c in message.get("tool_calls") or []:
            try:
                arguments = json.loads(c["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {"_unparsed": c["function"]["arguments"]}
            calls.append({"id": c["id"], "name": c["function"]["name"], "arguments": arguments})
        usage = data.get("usage") or {}
        return ChatTurn(message.get("content") or "", calls, usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0), data.get("provider") or HOSTED_PROVIDER,
                        message.get("reasoning") or "")
    raise ValueError(f"unknown backend: {backend}")
