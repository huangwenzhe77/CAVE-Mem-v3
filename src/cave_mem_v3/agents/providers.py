"""Metered, stateless actor calls; providers make exactly one physical attempt.

Design references: legacy CAVE-Mem's per-decision prompt rebuilding and MemRL's
full request tracing. Neither package is imported; retrying or losing usage is
not part of this contract.
"""
from __future__ import annotations

import copy
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from cave_mem_v3.common.contracts import PublicContext, canonical, digest, finite, plain


class ProviderError(RuntimeError):
    """A transport/provider failure, retaining its possibly charged attempt."""

    def __init__(self, message: str, attempt: dict[str, Any], *, retryable: bool = False) -> None:
        super().__init__(message)
        self.attempt = copy.deepcopy(attempt)
        self.retryable = retryable


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    usage: dict[str, float | None]
    finish_reason: str | None
    model: str
    fingerprint: str | None
    attempt: dict[str, Any]
    response_id: str | None = None
    requested_sampling: dict[str, Any] = field(default_factory=dict)
    applied_sampling: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActorResponse:
    action: dict[str, Any]
    text: str
    usage: dict[str, float | None]
    prompt_hash: str
    messages: tuple[dict[str, str], ...]
    attempts: tuple[dict[str, Any], ...]
    model: str
    finish_reason: str | None = None
    fingerprint: str | None = None
    requested_sampling: dict[str, Any] = field(default_factory=dict)
    applied_sampling: dict[str, Any] = field(default_factory=dict)
    parse_error: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("action", "usage", "messages", "attempts", "requested_sampling", "applied_sampling"):
            object.__setattr__(self, field_name, copy.deepcopy(getattr(self, field_name)))
        if not isinstance(self.action, dict) or not self.action:
            raise ValueError("actor action must be a nonempty object")
        canonical(self)


class Actor(Protocol):
    def propose(self, context: PublicContext, *, stage: str, seed: int,
                reminder: str | None = None, call_id: str = "") -> ActorResponse: ...
    def configuration(self) -> dict: ...
    def capabilities(self) -> dict: ...


class Provider(Protocol):
    def generate(self, messages: list[dict[str, str]], *, stage: str, seed: int,
                 call_id: str = "") -> ProviderResponse: ...
    def configuration(self) -> dict: ...
    def capabilities(self) -> dict: ...


@dataclass(frozen=True)
class OpenAIConfig:
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    base_url: str = ""
    api_key_env: str = "CAVE_MODEL_API_KEY"
    revision: str = ""
    local_model_path: str = ""
    response_model: str = ""
    expected_fingerprint: str = ""
    temperature: float = 0.0
    max_output_tokens: int = 512
    timeout_seconds: float = 60.0
    send_seed: bool = False
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    cached_input_price_per_million: float | None = None
    token_limit_parameter: str = "max_tokens"
    send_temperature: bool = True

    def __post_init__(self) -> None:
        if not self.model or self.base_url and not self.base_url.startswith(("http://", "https://")):
            raise ValueError("model and an empty or HTTP(S) base URL are required")
        if not isinstance(self.revision, str) or not isinstance(self.local_model_path, str):
            raise ValueError("revision/local_model_path must be strings; leave empty until server setup")
        if not isinstance(self.api_key_env, str):
            raise ValueError("api_key_env must be a string; empty permits an unauthenticated local server")
        for name in ("response_model", "expected_fingerprint"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string; empty means unspecified")
        if self.token_limit_parameter not in ("max_tokens", "max_completion_tokens"):
            raise ValueError("unsupported token limit parameter")
        if type(self.send_temperature) is not bool:
            raise ValueError("send_temperature must be boolean")
        if type(self.max_output_tokens) is not int or self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be a positive integer")
        finite(self.temperature, "temperature", minimum=0)
        if finite(self.timeout_seconds, "timeout_seconds", minimum=0) == 0:
            raise ValueError("timeout_seconds must be positive")
        for name in ("input_price_per_million", "output_price_per_million", "cached_input_price_per_million"):
            if getattr(self, name) is not None:
                finite(getattr(self, name), name, minimum=0)


class OpenAICompatibleProvider:
    """One chat-completions HTTP request. No SDK or hidden retry layer.

    The key is read only when a request is explicitly made. It is absent from
    configuration(), attempt metadata and errors. Tests may inject a transport
    returning a decoded response object; that remains one physical attempt.
    """

    def __init__(self, config: OpenAIConfig, *, transport: Callable[[dict], dict] | None = None) -> None:
        self.config = config
        self.transport = transport

    def configuration(self) -> dict:
        return {"provider": "openai-compatible-single-attempt.v3", **plain(self.config)}

    def capabilities(self) -> dict:
        return {"stateless": True, "physical_attempts_per_generate": 1,
                "seed_requested": self.config.send_seed, "deterministic_sampling_certified": False,
                "sampling_coupling": "provider_noise_not_certified", "usage": "reported_or_unknown",
                "response_model_checked": True, "weight_revision_certified": False,
                "revision_assurance": "operator_declared_not_server_attested"}

    @staticmethod
    def _count(value: Any, name: str) -> int | None:
        if value is None:
            return None
        if type(value) is not int or value < 0:
            raise ValueError(f"invalid provider {name}")
        return value

    def _usage(self, raw: dict) -> dict:
        usage = raw.get("usage") or {}
        input_tokens = self._count(usage.get("prompt_tokens", usage.get("input_tokens")), "input tokens")
        output_tokens = self._count(usage.get("completion_tokens", usage.get("output_tokens")), "output tokens")
        details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
        cached = self._count(details.get("cached_tokens"), "cached tokens")
        api_cost = None
        if input_tokens is not None and output_tokens is not None and self.config.input_price_per_million is not None and self.config.output_price_per_million is not None:
            if cached and self.config.cached_input_price_per_million is None:
                api_cost = None
            else:
                cached = cached or 0
                if cached > input_tokens:
                    raise ValueError("cached tokens exceed total input tokens")
                api_cost = ((input_tokens - cached) * self.config.input_price_per_million
                            + cached * (self.config.cached_input_price_per_million or 0.0)
                            + output_tokens * self.config.output_price_per_million) / 1_000_000
        return {"model_calls": 1.0, "physical_attempts": 1.0,
                "input_tokens": input_tokens, "output_tokens": output_tokens,
                "cached_input_tokens": cached, "api_cost": api_cost}

    def generate(self, messages: list[dict[str, str]], *, stage: str, seed: int,
                 call_id: str = "") -> ProviderResponse:
        if self.transport is None and (not self.config.base_url or not self.config.revision):
            raise ValueError("actor is not configured: set base_url and a pinned model revision on the server; no weights are downloaded automatically")
        if not messages or any(set(m) != {"role", "content"} or not isinstance(m["content"], str) for m in messages):
            raise ValueError("messages must contain role/content strings")
        payload = {"model": self.config.model, "messages": copy.deepcopy(messages),
                   self.config.token_limit_parameter: self.config.max_output_tokens}
        if self.config.send_temperature:
            payload["temperature"] = self.config.temperature
        if self.config.send_seed:
            payload["seed"] = seed
        requested = {k: v for k, v in payload.items() if k != "messages"}
        attempt = {"attempt_id": call_id or digest([stage, seed, payload])[:24],
                   "stage": stage, "status": "started", "request_hash": digest(payload),
                   "requested_sampling": requested, "started_at_unix": time.time(),
                   "usage": {"model_calls": 1.0, "physical_attempts": 1.0,
                             "input_tokens": None, "output_tokens": None, "api_cost": None}}
        started = time.perf_counter()
        try:
            if self.transport is not None:
                raw = self.transport(copy.deepcopy(payload))
            else:
                key = os.environ.get(self.config.api_key_env) if self.config.api_key_env else None
                if self.config.api_key_env and not key:
                    attempt["usage"]["physical_attempts"] = 0.0
                    raise ValueError(f"required credential environment variable is unset: {self.config.api_key_env}")
                headers = {"Content-Type": "application/json"}
                if key:
                    headers["Authorization"] = "Bearer " + key
                request = urllib.request.Request(self.config.base_url.rstrip("/") + "/chat/completions",
                                                data=canonical(payload).encode("utf-8"),
                                                headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    raw = json.loads(response.read().decode("utf-8"))
            usage = self._usage(raw)
            attempt["usage"] = usage
            response_model = raw.get("model")
            fingerprint = raw.get("system_fingerprint")
            if fingerprint == "":
                fingerprint = None
            attempt.update(response_hash=digest(raw), provider_model=response_model,
                           provider_fingerprint=fingerprint)
            if not isinstance(response_model, str) or not response_model:
                attempt["identity_error"] = "response_model_missing"
                raise ValueError("provider must report its response model")
            if response_model != (self.config.response_model or self.config.model):
                attempt["identity_error"] = "response_model_mismatch"
                raise ValueError("provider response model differs from the registered identity")
            if fingerprint is not None and not isinstance(fingerprint, str):
                attempt["identity_error"] = "response_fingerprint_invalid"
                raise ValueError("provider fingerprint must be a string or unknown")
            if self.config.expected_fingerprint and fingerprint != self.config.expected_fingerprint:
                attempt["identity_error"] = "response_fingerprint_mismatch"
                raise ValueError("provider fingerprint differs from the registered identity")
            attempt["response_identity"] = {"model": response_model, "fingerprint": fingerprint,
                "fingerprint_status": "reported" if fingerprint is not None else "unknown",
                "weight_revision_certified": False}
            choice = raw["choices"][0]
            text = choice["message"].get("content")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("provider returned no textual action")
            attempt.update(status="completed", response_hash=digest(raw), usage=usage,
                           finish_reason=choice.get("finish_reason"))
            return ProviderResponse(text, usage, choice.get("finish_reason"), response_model,
                                    fingerprint, attempt, raw.get("id"), requested,
                                    {"seed": "requested_unconfirmed" if self.config.send_seed else "not_requested"})
        except Exception as exc:
            status = exc.code if isinstance(exc, urllib.error.HTTPError) else None
            # Never include response bodies/headers: some gateways echo credentials.
            attempt.update(status="failed", error_class=type(exc).__name__, http_status=status)
            attempt["wall_seconds"] = time.perf_counter() - started
            attempt["ended_at_unix"] = time.time()
            retryable = status in (408, 409, 425, 429) or status is not None and status >= 500 or isinstance(exc, (TimeoutError, urllib.error.URLError))
            raise ProviderError(f"provider attempt failed: {type(exc).__name__}"
                + (f" (HTTP {status})" if status else "")
                + (" (" + attempt["identity_error"] + ")" if attempt.get("identity_error") else ""),
                attempt, retryable=retryable) from exc
        finally:
            attempt["wall_seconds"] = time.perf_counter() - started
            attempt["ended_at_unix"] = time.time()


def render_messages(context: PublicContext, *, reminder: str | None = None,
                    system_prompt: str = "Return exactly one executable action, without explanation.",
                    demonstrations: tuple[str, ...] = ()) -> tuple[dict[str, str], ...]:
    """Only actual executed history persists. A reminder is local to this call."""
    sections = []
    if demonstrations:
        sections.append("Fixed demonstrations:\n" + "\n\n".join(demonstrations))
    sections += ["Task:\n" + context.instruction, "Executed history:\n" + canonical(context.history),
                 "Current observation:\n" + context.observation, "Visible facts:\n" + canonical(context.facts),
                 f"Remaining execution steps: {context.remaining_steps}"]
    if context.admissible_actions:
        sections.append("Admissible actions:\n" + "\n".join(context.admissible_actions))
    if reminder is not None:
        raise ValueError("phase-one historical actor cannot receive reminders")
    return ({"role": "system", "content": system_prompt}, {"role": "user", "content": "\n\n".join(sections)})


def parse_json_action(text: str) -> dict:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean, flags=re.IGNORECASE)
    action = json.loads(clean)
    if not isinstance(action, dict) or not isinstance(action.get("type"), str):
        raise ValueError("expected JSON action object containing type")
    return action


class ProviderActor:
    """Explicit prompt rendering around a stateless, metered provider."""

    def __init__(self, provider: Provider, *, parser: Callable[[str], dict] = parse_json_action,
                 system_prompt: str = "Return one JSON executable action containing type; no explanation.",
                 demonstrations: tuple[str, ...] = (), prompt_version: str = "actor-prompt.v3") -> None:
        self.provider = provider
        self.parser = parser
        self.system_prompt = system_prompt
        self.demonstrations = tuple(demonstrations)
        self.prompt_version = prompt_version

    def configuration(self) -> dict:
        return {"actor": "stateless-provider-actor.v3", "provider": self.provider.configuration(),
                "prompt_version": self.prompt_version, "system_prompt_hash": digest(self.system_prompt),
                "demonstrations_hash": digest(self.demonstrations), "parser": self.parser.__name__}

    def capabilities(self) -> dict:
        return {**self.provider.capabilities(), "history_owner": "collector", "reminder_persistence": "call_only"}

    def propose(self, context: PublicContext, *, stage: str, seed: int,
                reminder: str | None = None, call_id: str = "") -> ActorResponse:
        messages = render_messages(context, reminder=reminder, system_prompt=self.system_prompt, demonstrations=self.demonstrations)
        response = self.provider.generate(list(messages), stage=stage, seed=seed, call_id=call_id)
        try:
            action = self.parser(response.text)
        except (ValueError, TypeError, KeyError) as exc:
            # A completed model response with malformed syntax is behavior, not
            # missing infrastructure. Retain usage and let action validation
            # produce the preregistered invalid-action failure outcome.
            attempt = copy.deepcopy(response.attempt)
            attempt.update(action_parse_status="invalid", action_parse_error=type(exc).__name__)
            return ActorResponse({"type": "__unparsed_action__"}, response.text, response.usage,
                                 digest(messages), messages, (attempt,), response.model,
                                 response.finish_reason, response.fingerprint,
                                 response.requested_sampling, response.applied_sampling,
                                 parse_error=type(exc).__name__)
        return ActorResponse(action, response.text, response.usage, digest(messages), messages,
                             (response.attempt,), response.model, response.finish_reason,
                             response.fingerprint, response.requested_sampling, response.applied_sampling)


class ScriptedActor:
    """Fixture actor: follow a visible preparation plan, otherwise operate."""

    def __init__(self, *, error_rate: float = 0.0) -> None:
        if not 0 <= finite(error_rate, "error_rate") <= 1:
            raise ValueError("error_rate must be in [0,1]")
        self.error_rate = error_rate

    def configuration(self) -> dict:
        return {"actor": "scripted-preparation.v3",
                "error_rate": self.error_rate, "research_role": "fixture_only"}

    def capabilities(self) -> dict:
        return {"stateless": True, "deterministic_sampling_certified": True,
                "sampling_coupling": "stage_step_keyed", "history_owner": "collector",
                "physical_attempts_per_generate": 0, "usage": "exact_fixture_zero"}

    def propose(self, context: PublicContext, *, stage: str, seed: int,
                reminder: str | None = None, call_id: str = "") -> ActorResponse:
        import random
        rng = random.Random(seed)
        phase = context.facts.get("phase", "ready")
        if phase == "ready":
            prepared = context.facts.get("prepared", False)
            operation = "prepare" if not prepared and context.facts.get("intent") == "prepare" else "operate"
        elif phase == "operated":
            operation = "finish"
        else:
            operation = "finish"
        if rng.random() < self.error_rate:
            operation = "invalid"
        action = {"type": operation}
        messages = render_messages(context, reminder=reminder)
        usage = {"model_calls": 1.0, "physical_attempts": 0.0, "input_tokens": 0.0,
                 "output_tokens": 0.0, "cached_input_tokens": 0.0, "api_cost": 0.0}
        attempt = {"attempt_id": call_id or digest([stage, seed, messages])[:24], "stage": stage,
                   "status": "completed", "provider": "scripted-no-network", "request_hash": digest(messages),
                   "usage": usage, "wall_seconds": 0.0, "physical": False}
        return ActorResponse(action, canonical(action), usage, digest(messages), messages, (attempt,), "scripted-preparation.v3", "stop", "scripted-fixed-v3", {"seed": seed}, {"seed": seed})
