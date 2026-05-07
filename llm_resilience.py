"""LLM call resilience — timeout, retry, and error handling for all agent invocations.

Every LLM call site should use:
  - ``init_chat_model_with_timeout()`` for model initialization
  - ``resilient_invoke()`` for agent/graph invocation with retry
  - ``is_harvestable_error()`` for deciding whether to salvage partial results
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.chat_models import init_chat_model
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LLM_TIMEOUT = 300  # 5 minutes — matches hunt_agent / hunt_planner
MAX_RETRIES = 3
RETRY_MIN_WAIT = 1   # seconds
RETRY_MAX_WAIT = 30  # seconds

# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

_TRANSIENT_PATTERNS = (
    "connection error",
    "server disconnected",
    "timeout",
    "timed out",
    "rate limit",
    "429",
    "500",
    "502",
    "503",
    "504",
    "overloaded",
    "capacity",
)

_HARVESTABLE_PATTERNS = (
    *_TRANSIENT_PATTERNS,
    "recursion limit",
)


def _is_transient_error(exc: BaseException) -> bool:
    """Return True if *exc* is a transient LLM API error worth retrying."""
    err_str = str(exc).lower()
    return any(p in err_str for p in _TRANSIENT_PATTERNS)


def is_harvestable_error(exc: BaseException) -> bool:
    """Return True if *exc* allows harvesting partial results from the toolkit.

    Superset of transient errors — also includes recursion-limit errors where
    the agent ran out of steps but may have accumulated findings.
    """
    err_str = str(exc).lower()
    return any(p in err_str for p in _HARVESTABLE_PATTERNS)


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------

def init_chat_model_with_timeout(
    model_spec: str,
    timeout: int = DEFAULT_LLM_TIMEOUT,
) -> Any:
    """Initialize a chat model with standard timeout configuration.

    Consolidates the ``_init_hunt_model`` / ``_init_planner_model`` pattern
    into a single reusable function.
    """
    if model_spec.startswith(("azure:", "azure_openai:")):
        return _init_azure_chat_model(model_spec, timeout=timeout)

    kwargs: dict[str, Any] = {"request_timeout": timeout}
    if model_spec.startswith("openai:"):
        kwargs["use_responses_api"] = True
    return init_chat_model(model_spec, **kwargs)


def _init_azure_chat_model(model_spec: str, timeout: int = DEFAULT_LLM_TIMEOUT) -> Any:
    """Initialize Azure OpenAI using the active Predator profile."""
    import os

    _, _, deployment = model_spec.partition(":")
    if not deployment:
        raise ValueError(f"Invalid Azure model spec: {model_spec}")

    try:
        from predator.config.llm_profiles import get_profile
    except Exception:
        get_profile = None  # type: ignore[assignment]

    resolved = None
    if get_profile is not None:
        profile = get_profile()
        if profile is not None:
            resolved = profile.resolve(deployment)

    endpoint = (
        (resolved.endpoint if resolved else None)
        or os.environ.get("AZURE_OPENAI_ENDPOINT")
    )
    api_version = (
        (resolved.api_version if resolved else None)
        or os.environ.get("AZURE_OPENAI_API_VERSION")
        or os.environ.get("OPENAI_API_VERSION")
    )
    deployment_name = (resolved.deployment if resolved else None) or deployment
    if not endpoint:
        raise ValueError("Azure profile is missing endpoint. Set AZURE_OPENAI_ENDPOINT or llm-profiles.local.yaml.")
    if not api_version:
        raise ValueError("Azure profile is missing api_version. Set AZURE_OPENAI_API_VERSION or llm-profiles.local.yaml.")

    kwargs: dict[str, Any] = {
        "azure_endpoint": endpoint,
        "azure_deployment": deployment_name,
        "api_version": api_version,
        "request_timeout": timeout,
    }
    credential_type = getattr(resolved, "credential_type", None)
    if credential_type:
        kwargs["azure_ad_token_provider"] = _build_azure_token_provider(resolved)
    else:
        key_env = (resolved.api_key_env if resolved else None) or "AZURE_OPENAI_API_KEY"
        api_key = os.environ.get(key_env) or os.environ.get("AZURE_OPENAI_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key
        else:
            kwargs["azure_ad_token_provider"] = _build_azure_token_provider(resolved)

    if not deployment_name.startswith(("o1", "o3", "gpt-5")):
        kwargs["temperature"] = 0

    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(**kwargs)


def _build_azure_token_provider(resolved: Any) -> Any:
    """Build an Azure bearer token provider from profile credential settings."""
    from azure.identity import get_bearer_token_provider

    credential_type = getattr(resolved, "credential_type", None)
    credential_args = getattr(resolved, "credential_args", {}) or {}
    if credential_type == "AzureCliCredential":
        from azure.identity import AzureCliCredential

        credential = AzureCliCredential(**credential_args)
    else:
        from azure.identity import DefaultAzureCredential

        credential = DefaultAzureCredential()
    return get_bearer_token_provider(
        credential,
        "https://cognitiveservices.azure.com/.default",
    )


# ---------------------------------------------------------------------------
# Resilient invocation
# ---------------------------------------------------------------------------

def resilient_invoke(
    agent: Any,
    messages: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    context: str = "",
    max_retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    """Invoke an agent with automatic retry on transient LLM API failures.

    Parameters
    ----------
    agent:
        Any object with an ``.invoke()`` method (CompiledStateGraph, etc.).
    messages:
        The input dict, typically ``{"messages": [HumanMessage(...)]}``.
    config:
        Optional LangGraph config dict (run_name, metadata, callbacks, etc.).
    context:
        Human-readable label for log messages (e.g. ``"hunt_planner_HUNT-06"``).
    max_retries:
        Maximum number of attempts (including the first call).

    Returns
    -------
    dict
        The agent's response dict.

    Raises
    ------
    Exception
        Re-raised after exhausting retries for transient errors, or
        immediately for non-transient errors.
    """
    @retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
        retry=retry_if_exception(_is_transient_error),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _invoke() -> dict[str, Any]:
        if config is not None:
            return agent.invoke(messages, config=config)
        return agent.invoke(messages)

    ctx = f" [{context}]" if context else ""
    logger.debug("resilient_invoke%s starting (max_retries=%d)", ctx, max_retries)
    return _invoke()
