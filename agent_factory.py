"""DeepAgents factory wiring for Wolfpack extraction."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

from predator.config.llm_profiles import ResolvedModel, normalize_profile_name


RunMode = str
ProfileResolver = Callable[[str], dict[str, str | None]]
CreateAgentFn = Callable[..., Any]


@dataclass(frozen=True)
class BuiltAgent:
    """Agent instance bundled with resolved LLM metadata."""

    agent: Any
    provider: str
    model: str
    deployment: str | None
    langsmith_enabled: bool
    # Extended metadata for status display
    agent_name: str = ""
    model_spec: str = ""
    system_prompt: str = ""
    subagent_specs: tuple[dict[str, Any], ...] = ()
    run_mode: str = ""


ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the Wolfpack Extraction Orchestrator.

Your task is to extract atomic threat behaviors from threat report sections.
Each context window contains a section of a threat report with an anchor ID,
line range, snippet text, and associated MITRE ATT&CK TTP IDs.

For each context window, extract ALL observable threat behaviors as structured
AtomicBehaviorCandidate objects. Follow recall-first extraction: capture every
discernible behavior, even low-confidence ones. Downstream validation will filter.

Output a JSON array of AtomicBehaviorCandidate objects with these fields:
- behavior_id: "B1", "B2", etc. (sequential)
- claim: concise description of the behavior
- evidence_refs: list of evidence references (anchor, lines, snippet, ttp_ids)
- observables: IOCs, file names, domains, IPs found in the text
- telemetry_requirements: log_sources and required_fields for validation
- confidence: 0.0-1.0 based on evidence strength
- source_agent: "behavior_extractor"
"""

BEHAVIOR_EXTRACTOR_SYSTEM_PROMPT = """\
You are the Wolfpack Behavior Extractor subagent.

Given threat report sections, extract individual atomic threat behaviors.
Each behavior must reference the specific anchor and line range from the source.
Include observables (IOCs, filenames, domains, IPs) found in each section.
Assign confidence based on how explicitly the behavior is described.
Extract multiple behaviors per section when the text describes multiple actions.
"""


def _default_subagents() -> list[dict[str, Any]]:
    return [
        {
            "name": "behavior_extractor",
            "description": "Extracts atomic threat behaviors from report sections with anchor provenance and observable identification.",
            "system_prompt": BEHAVIOR_EXTRACTOR_SYSTEM_PROMPT,
        },
    ]


def _resolve_langsmith_enabled(llm_profile_name: str, env: dict[str, str] | None = None) -> bool:
    """Enable tracing only for the non-corp profile and only when env keys exist."""

    if normalize_profile_name(llm_profile_name) != "non-corp":
        return False

    env_data = env or dict(os.environ)
    api_key = (env_data.get("LANGSMITH_API_KEY") or "").strip()
    tracing_flag = (env_data.get("LANGSMITH_TRACING") or "").strip().lower()

    return bool(api_key and tracing_flag in {"1", "true", "yes", "on"})


def gate_langsmith(langsmith_enabled: bool) -> None:
    """Enable or disable LangSmith tracing based on profile resolution.

    Call this after loading .env and resolving the profile.

    When *enabled*, actively sets ``LANGSMITH_TRACING=true`` and clears the
    ``langsmith.utils.get_env_var`` LRU cache so that ``tracing_is_enabled()``
    picks up the current state (the cache can hold stale pre-load_dotenv values).

    When *disabled*, forces ``LANGSMITH_TRACING=false`` (only if it was
    previously set — avoids injecting the var into environments without
    LangSmith at all).
    """
    if langsmith_enabled:
        os.environ["LANGSMITH_TRACING"] = "true"
    else:
        current = os.environ.get("LANGSMITH_TRACING")
        if current is not None:
            os.environ["LANGSMITH_TRACING"] = "false"

    # langsmith.utils.get_env_var uses @functools.lru_cache — if anything
    # called tracing_is_enabled() before load_dotenv populated the env,
    # the stale "not found" result is cached forever.  Clear it.
    try:
        from langsmith.utils import get_env_var as _ls_get_env_var

        _ls_get_env_var.cache_clear()
    except (ImportError, AttributeError):
        pass


@contextmanager
def _temporary_profile(llm_profile_name: str):
    """Temporarily switch active Predator profile for one resolution call."""

    original = os.environ.get("PREDATOR_PROFILE")
    os.environ["PREDATOR_PROFILE"] = llm_profile_name
    try:
        from predator.config.llm_profiles import reset_cache

        reset_cache()
        yield
    finally:
        if original is None:
            os.environ.pop("PREDATOR_PROFILE", None)
        else:
            os.environ["PREDATOR_PROFILE"] = original
        from predator.config.llm_profiles import reset_cache

        reset_cache()


def _resolve_profile_via_predator(llm_profile_name: str) -> dict[str, str | None]:
    """Resolve provider/model/deployment via Predator's profile router."""

    from predator.config.llm_profiles import get_profile

    with _temporary_profile(llm_profile_name):
        profile = get_profile()

    if profile is None:
        raise ValueError(
            "Unable to resolve LLM profile via Predator config. "
            f"Missing or invalid profile: {llm_profile_name}"
        )

    resolved = profile.resolve(None)
    _apply_resolved_environment(resolved)
    provider = _provider_for_langchain(resolved.provider)
    return {
        "provider": provider,
        "model": resolved.deployment,
        "deployment": resolved.deployment,
    }


def _provider_for_langchain(provider: str | None) -> str | None:
    """Translate Predator provider names to LangChain model provider names."""
    if provider == "azure":
        return "azure_openai"
    return provider


def _apply_resolved_environment(resolved: ResolvedModel) -> None:
    """Populate env vars expected by LangChain model factories."""
    if resolved.provider != "azure":
        return
    if resolved.endpoint and not os.environ.get("AZURE_OPENAI_ENDPOINT"):
        os.environ["AZURE_OPENAI_ENDPOINT"] = resolved.endpoint
    if resolved.api_version:
        os.environ.setdefault("AZURE_OPENAI_API_VERSION", resolved.api_version)
        os.environ.setdefault("OPENAI_API_VERSION", resolved.api_version)
    if resolved.deployment:
        os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT", resolved.deployment)
    if resolved.api_key_env and not resolved.credential_type:
        key_value = os.environ.get(resolved.api_key_env)
        if key_value and not os.environ.get("AZURE_OPENAI_API_KEY"):
            os.environ["AZURE_OPENAI_API_KEY"] = key_value


def _default_create_agent(**kwargs: Any) -> Any:
    """Construct DeepAgent using installed deepagents package."""

    from deepagents import create_deep_agent

    return create_deep_agent(**kwargs)


def resolve_llm_profile(
    llm_profile_name: str,
    *,
    resolve_profile: ProfileResolver | None = None,
) -> dict[str, str | None]:
    """Resolve LLM profile metadata for manifest population."""
    resolver = resolve_profile or _resolve_profile_via_predator
    return resolver(llm_profile_name)


def build_extraction_agent(
    *,
    run_mode: RunMode,
    llm_profile_name: str,
    resolve_profile: ProfileResolver | None = None,
    create_agent: CreateAgentFn | None = None,
) -> BuiltAgent:
    """Build a DeepAgents extraction agent according to Wolfpack mode contracts."""

    if run_mode not in {"multi_agent_primary", "single_agent_control"}:
        raise ValueError(f"Unsupported run_mode for factory: {run_mode}")

    resolved = resolve_llm_profile(llm_profile_name, resolve_profile=resolve_profile)
    provider = resolved.get("provider")
    model = resolved.get("model")
    deployment = resolved.get("deployment")
    if not provider:
        raise ValueError("Resolved profile is missing provider")

    model_spec = f"{provider}:{model}"
    langsmith_enabled = _resolve_langsmith_enabled(llm_profile_name)

    agent_name = "wolfpack_extraction_orchestrator"
    subagents_list = _default_subagents() if run_mode == "multi_agent_primary" else []

    model_for_agent: Any = model_spec
    if create_agent is None:
        from wolfpack.llm_resilience import init_chat_model_with_timeout

        model_for_agent = init_chat_model_with_timeout(model_spec)

    kwargs: dict[str, Any] = {
        "model": model_for_agent,
        "system_prompt": ORCHESTRATOR_SYSTEM_PROMPT,
        "name": agent_name,
    }

    if subagents_list:
        kwargs["subagents"] = subagents_list

    factory = create_agent or _default_create_agent
    agent = factory(**kwargs)

    return BuiltAgent(
        agent=agent,
        provider=str(provider),
        model=str(model),
        deployment=str(deployment) if deployment else None,
        langsmith_enabled=langsmith_enabled,
        agent_name=agent_name,
        model_spec=model_spec,
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        subagent_specs=tuple(subagents_list),
        run_mode=run_mode,
    )
