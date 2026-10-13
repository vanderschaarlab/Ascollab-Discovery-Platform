import logging
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings
from langchain_openai.chat_models.base import BaseChatOpenAI
from omegaconf import DictConfig, OmegaConf
from openai import APIError, AzureOpenAI, OpenAI, RateLimitError

log = logging.getLogger(__name__)


class FallbackEmbeddings:
    """Wrapper that provides failover between a primary and optional backup embedder.

    This class attempts an embedding call against the primary
    `AzureOpenAIEmbeddings` client and, if a transient error occurs (e.g.,
    `RateLimitError` or `APIError`), optionally retries the same call against a
    backup client.

    Attributes:
        primary: The primary embeddings client.
        backup: An optional backup embeddings client used on primary failure.
    """

    def __init__(self, primary: AzureOpenAIEmbeddings, backup: AzureOpenAIEmbeddings | None = None):
        """Initialize the failover embeddings wrapper.

        Args:
            primary: Primary `AzureOpenAIEmbeddings` instance.
            backup: Optional backup `AzureOpenAIEmbeddings` instance for failover.
        """
        self.primary = primary
        self.backup = backup

    def _try(self, primary_fn, backup_fn=None):
        """Execute `primary_fn`, falling back to `backup_fn` on transient OpenAI errors.

        Catches `RateLimitError` and `APIError`. If a backup function is provided,
        it will be invoked on those failures; otherwise, the exception is re-raised.

        Args:
            primary_fn: Zero-argument callable that performs the primary action.
            backup_fn: Optional zero-argument callable for the backup action.

        Returns:
            The return value of whichever function succeeds.

        Raises:
            RateLimitError: If the primary call fails with a rate limit error and no backup.
            APIError: If the primary call fails with an API error and no backup.
        """
        try:
            return primary_fn()
        except (RateLimitError, APIError):
            if not backup_fn:
                raise
            return backup_fn()

    def embed_query(self, text: str):
        """Embed a single query string with failover.

        Args:
            text: The query text to embed.

        Returns:
            List[float] or provider-specific embedding vector format.
        """
        return self._try(
            lambda: self.primary.embed_query(text),
            (lambda: self.backup.embed_query(text)) if self.backup else None,
        )

    def embed_documents(self, texts: list[str]):
        """Embed multiple documents with failover.

        Args:
            texts: List of document strings to embed.

        Returns:
            List of embedding vectors, one per input document.
        """
        return self._try(
            lambda: self.primary.embed_documents(texts),
            (lambda: self.backup.embed_documents(texts)) if self.backup else None,
        )


# Endpoint files (sys_config/llm_api/*.yaml, sys_config/embedder_api/*.yaml) choose a provider:
#   azure  (default): azure_deployment, azure_endpoint, api_key, api_version
#   openai          : model, api_key, optional base_url (any OpenAI-compatible server)
PROVIDERS = ("azure", "openai")


def endpoint_provider(cfg: Mapping[str, Any]) -> str:
    """Return the endpoint's provider (``azure`` when unset)."""
    provider = str(cfg.get("provider") or "azure").strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown endpoint provider {provider!r}; expected one of {PROVIDERS}.")
    return provider


ENDPOINT_SETUP_HINT = "see docs/models.md"

# Endpoint key -> its file, for messages when Hydra's choices are not available.
_ENDPOINT_FILES = {
    "llm_primary": "sys_config/llm_api/default.yaml",
    "embedder_primary": "sys_config/embedder_api/default.yaml",
    "gpt4o": "sys_config/llm_api/gpt-4o.yaml",
    "gpt5": "sys_config/llm_api/gpt-5.yaml",
    "gpt5-high": "sys_config/llm_api/gpt-5-high.yaml",
    "deepseek": "sys_config/llm_api/deepseek.yaml",
    "llama": "sys_config/llm_api/llama.yaml",
}


class EndpointConfigError(ValueError):
    """An endpoint file is missing or still the unfilled template."""


def endpoint_file(key: str) -> str:
    """The endpoint file behind ``endpoints.<key>`` (from Hydra's choices when running)."""
    try:
        from hydra.core.hydra_config import HydraConfig

        for group, choice in HydraConfig.get().runtime.choices.items():
            if choice and str(group).endswith(f"@endpoints.{key}"):
                return f"sys_config/{str(group).split('@', 1)[0]}/{choice}.yaml"
    except Exception:
        pass
    return _ENDPOINT_FILES.get(key, f"the endpoint file for endpoints.{key}")


def require_endpoints(endpoints_cfg: Mapping[str, Any] | None, keys: Iterable[str]) -> None:
    """Fail early, naming the file, when a required endpoint file was not created."""
    missing = [k for k in keys if not endpoints_cfg or endpoints_cfg.get(k) is None]
    if missing:
        files = ", ".join(endpoint_file(k) for k in missing)
        raise EndpointConfigError(
            f"Missing endpoint file(s): {files}. Copy the default_to_complete.yaml template in "
            f"the same folder and fill it in ({ENDPOINT_SETUP_HINT})."
        )


def _require_filled(cfg_dict: Mapping[str, Any], key: str | None) -> None:
    """Reject an endpoint that is still the template (no deployment or model name)."""
    field = "model" if endpoint_provider(cfg_dict) == "openai" else "azure_deployment"
    if not cfg_dict.get(field):
        where = endpoint_file(key) if key else "An endpoint file"
        raise EndpointConfigError(
            f"{where} is not filled in ({field} is empty); {ENDPOINT_SETUP_HINT}."
        )


def _plain_dict(cfg: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        out = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(out, dict)
        return {str(k): v for k, v in out.items()}
    return dict(cfg)


# Always needed for Azure
_ALWAYS_KEYS: set[str] = {
    "azure_deployment",
    "azure_endpoint",
    "api_key",
    "api_version",
    # these are client-level, safe for all backends
    "timeout",
    "max_retries",
    "seed",
}

# Always passed for OpenAI-compatible endpoints
_OPENAI_ALWAYS_KEYS: set[str] = {
    "model",
    "api_key",
    "base_url",
    "organization",
    "timeout",
    "max_retries",
    "seed",
}

# Only GPT-4o is allowed to receive sampling knobs explicitly.
# GPT-5 is allowed to receive the reasoning_effort knob.
# Everything else will rely on server defaults and we explicitly
# null out the sampling attributes on the client object.
_OPTIONAL_SAMPLING_BY_BACKEND: dict[str, set[str]] = {
    "gpt4o": {"temperature", "top_p", "max_tokens"},
    "gpt5": {"reasoning_effort"},
    "llama": set(),
    "deepseek": set(),
    "generic": set(),
}


def _infer_backend(cfg: dict[str, Any]) -> str:
    """Infer a backend name from the Azure deployment (or OpenAI model) name.

    Args:
        cfg: Mapping containing an `azure_deployment` or `model` key.

    Returns:
        Canonical backend label such as 'gpt4o', 'gpt5', 'gpt5-high', 'llama', 'deepseek', or 'generic'.
    """
    dep = str(cfg.get("azure_deployment") or cfg.get("model") or "").lower()
    if "gpt-5" in dep or "gpt5" in dep:
        return "gpt5"
    if "gpt-4" in dep or "gpt4o" in dep:
        return "gpt4o"
    if "llama" in dep:
        return "llama"
    if "deepseek" in dep:
        return "deepseek"
    return "generic"


def _mk_llm(cfg: Mapping[str, Any] | DictConfig, key: str | None = None) -> BaseChatOpenAI:
    """Instantiate a single chat client (Azure or OpenAI-compatible) from Hydra kwargs.

    Args:
        cfg: Dict-like endpoint configuration (see ``PROVIDERS``).
        key: The ``endpoints.<key>`` it came from, used to name the file in errors.

    Returns:
        Configured chat client with sampling knobs filtered appropriately.
    """
    # Drop explicit Nones from config
    cfg_dict = {k: v for k, v in _plain_dict(cfg).items() if v is not None}
    provider = endpoint_provider(cfg_dict)
    _require_filled(cfg_dict, key)

    backend = _infer_backend(cfg_dict)
    optional_sampling = _OPTIONAL_SAMPLING_BY_BACKEND.get(
        backend, _OPTIONAL_SAMPLING_BY_BACKEND["generic"]
    )

    # For gpt-4o we allow sampling knobs; for others we don't.
    always = _OPENAI_ALWAYS_KEYS if provider == "openai" else _ALWAYS_KEYS
    allowed_keys = always | optional_sampling

    filtered = {k: v for k, v in cfg_dict.items() if k in allowed_keys}
    if provider != "openai" and backend in ("deepseek", "llama"):
        # Non-OpenAI models served through Azure can sit behind servers that require a model
        # name in the request body (a DeepSeek deployment answered 400 "model: Input should be a
        # valid string" without it). Azure picks the deployment from the URL either way. GPT
        # deployments are left as they were: a model name would switch on LangChain's
        # model-specific request handling.
        filtered.setdefault("model", cfg_dict["azure_deployment"])
    llm: BaseChatOpenAI = (
        ChatOpenAI(**filtered) if provider == "openai" else AzureChatOpenAI(**filtered)
    )

    # ── Belt & braces: ensure non-gpt4o backends do NOT send sampling params ──
    if backend != "gpt4o":
        # The OpenAI clients typically only send non-None values,
        # so setting these to None prevents them from hitting the wire.
        # (And also makes downstream debugging clearer.)
        try:
            llm.temperature = None
        except AttributeError:
            pass
        try:
            llm.top_p = None
        except AttributeError:
            pass
        try:
            llm.max_tokens = None
        except AttributeError:
            pass

    return llm


def _mk_embedder(cfg: Mapping[str, Any] | DictConfig, key: str | None = None) -> Embeddings:
    """Instantiate an embeddings client (Azure or OpenAI-compatible) from Hydra kwargs."""
    cfg_dict = {k: v for k, v in _plain_dict(cfg).items() if v is not None}
    provider = endpoint_provider(cfg_dict)
    _require_filled(cfg_dict, key)
    cfg_dict.pop("provider", None)
    if provider == "openai":
        return OpenAIEmbeddings(**cfg_dict)
    return AzureOpenAIEmbeddings(**cfg_dict)


def load_endpoint_config(path: str | Path) -> dict[str, Any] | None:
    """Load one endpoint file outside Hydra; ``None`` if it does not exist.

    Values that need the Hydra run config (such as ``seed: ${SEED}``) resolve to ``None``;
    environment interpolations such as ``${oc.env:OPENAI_API_KEY}`` resolve normally.
    """
    path = Path(path)
    if not path.exists():
        return None
    raw = OmegaConf.load(path)
    return {
        str(key): OmegaConf.select(raw, str(key), throw_on_resolution_failure=False) for key in raw
    }


def make_sdk_client(
    cfg: Mapping[str, Any] | None, timeout: float | None = None
) -> tuple[OpenAI, str] | None:
    """Raw OpenAI SDK client and model name, for direct ``chat.completions`` calls.

    Args:
        cfg: One endpoint file's contents (``provider``, key, and deployment or model).
        timeout: Request timeout in seconds; the SDK default when ``None``.

    Returns:
        ``(client, model)``, or ``None`` when the endpoint is not filled in, so callers can
        fall back.
    """
    if not cfg:
        return None
    provider = endpoint_provider(cfg)
    api_key = cfg.get("api_key")
    kwargs: dict[str, Any] = {"api_key": api_key}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if provider == "openai":
        model = cfg.get("model")
        if not (api_key and model):
            return None
        if cfg.get("base_url"):
            kwargs["base_url"] = cfg["base_url"]
        return OpenAI(**kwargs), str(model)
    model = cfg.get("azure_deployment")
    if not (api_key and model and cfg.get("azure_endpoint")):
        return None
    kwargs.update(azure_endpoint=cfg["azure_endpoint"], api_version=cfg.get("api_version"))
    return AzureOpenAI(**kwargs), str(model)


def build_llm_and_embedder(
    llm_primary: OmegaConf,
    embedder_primary: OmegaConf,
) -> tuple[BaseChatOpenAI, Embeddings]:
    """Instantiate the primary LLM and embedder.

    Args:
        llm_primary: Hydra config for the primary chat model.
        embedder_primary: Hydra config for the primary embeddings client.

    Returns:
        Tuple of (chat model, embeddings client).
    """
    llm = _mk_llm(llm_primary, key="llm_primary")
    embedder = _mk_embedder(embedder_primary, key="embedder_primary")
    return llm, embedder


# ------------  multi-backend support ------------
_DEFAULT_POOL_KEYS = ("gpt4o", "gpt5", "gpt5-high", "deepseek", "llama")


def mix_backends(
    counts: Mapping[str, int] | None = None, fractions: Mapping[str, float] | None = None
) -> list[str]:
    """Backends that a model mix gives a non-zero share, in first-seen order."""
    keys: list[str] = []
    for share in (counts or {}, fractions or {}):
        for key, value in share.items():
            if value and float(value) > 0 and key not in keys:
                keys.append(str(key))
    return keys


def build_llm_pool(
    endpoints_cfg: Mapping[str, Any] | None,
    keys: Iterable[str] | None = None,
    *,
    primary_alias: bool = True,
) -> dict[str, BaseChatOpenAI]:
    """Build a dictionary of named LLM backends from Hydra endpoints.* namespaces.

    Args:
        endpoints_cfg: Hydra config with one namespace per backend (e.g. gpt4o, gpt5, gpt5-high,
            deepseek, llama) plus llm_primary.
        keys: Backends to build. Pass the backends a run uses, so endpoint files for unused
            backends can stay unfilled templates. ``None`` builds every known backend that is
            configured.
        primary_alias: Also build "primary" from llm_primary when it is configured (the run's
            fallback for mix backends without a file). The ranker passes False unless its judge
            is "primary", so it builds only its judge.

    Returns:
        Mapping of backend key to instantiated chat client, plus "primary" for llm_primary.
    """
    pool: dict[str, BaseChatOpenAI] = {}
    endpoints_cfg = endpoints_cfg or {}
    for key in _DEFAULT_POOL_KEYS if keys is None else keys:
        if key in endpoints_cfg and endpoints_cfg[key] is not None:
            pool[key] = _mk_llm(endpoints_cfg[key], key=key)
    # Include llm_primary as a fallback alias if present, unless the caller does not use it
    if (
        primary_alias
        and "llm_primary" in endpoints_cfg
        and endpoints_cfg["llm_primary"] is not None
    ):
        pool.setdefault("primary", _mk_llm(endpoints_cfg["llm_primary"], key="llm_primary"))
    return pool


def scale_counts(counts: Mapping[str, int], total: int) -> dict[str, int]:
    """Scale per-backend counts to ``total`` agents, keeping proportions (largest remainder).

    Ties go to the backend listed first, as in ``scripts/experiments/lib/common.sh``.
    """
    weight = sum(counts.values())
    if weight <= 0 or total <= 0:
        return dict.fromkeys(counts, 0)
    exact = {k: total * c / weight for k, c in counts.items()}
    scaled = {k: int(v) for k, v in exact.items()}
    order = sorted(counts, key=lambda k: -(exact[k] - scaled[k]))  # stable: first listed wins ties
    for k in order[: total - sum(scaled.values())]:
        scaled[k] += 1
    return scaled


def realize_mix(
    pool: dict[str, BaseChatOpenAI],
    num_agents: int,
    counts: dict[str, int] | None = None,
    fractions: dict[str, float] | None = None,
    shuffle: bool = True,
    rng: random.Random | None = None,
) -> list[tuple[str, BaseChatOpenAI]]:
    """Allocate LLMs for agents according to counts/fractions, with fallback.

    Args:
        pool: Mapping of backend key to LLM client.
        num_agents: Total number of agents to assign.
        counts: Optional explicit counts per backend.
        fractions: Optional fractional allocation per backend when counts are absent/short.
        shuffle: Whether to shuffle the assignment order.
        rng: Optional random generator for reproducible shuffling.

    Returns:
        List of tuples `(backend_key, llm_client)` of length `num_agents`.

    Raises:
        RuntimeError: If no LLMs are available to assign.
    """
    rng = rng or random.Random()
    plan: list[str] = []

    missing = [k for k in mix_backends(counts, fractions) if k not in pool]
    if missing:
        log.warning("!" * 72)
        log.warning(
            "[llm_backends] requested backends %s have no endpoint file (%s), so their agents "
            "are filled from llm_backends.fractions if set, otherwise from llm_primary; %s.",
            missing,
            ", ".join(endpoint_file(k) for k in missing),
            ENDPOINT_SETUP_HINT,
        )
        log.warning("!" * 72)

    # Prefer counts if provided
    if counts:
        usable = {k: int(c) for k, c in counts.items() if k in pool and int(c) > 0}
        if sum(usable.values()) > num_agents:
            scaled = scale_counts(usable, num_agents)
            log.warning(
                "[llm_backends] counts %s add up to more than network.num_agents=%d; "
                "scaled to %s (largest remainder).",
                usable,
                num_agents,
                scaled,
            )
            usable = scaled
        for k, c in usable.items():
            plan.extend([k] * c)

    # If still short, fill via fractions
    if len(plan) < num_agents and fractions:
        remaining = num_agents - len(plan)
        items = [(k, p) for k, p in fractions.items() if k in pool and p > 0]
        s = sum(p for _, p in items) or 1.0
        # allocate by floor, then fill remainder by descending fractional parts
        raw = [(k, remaining * (p / s)) for k, p in items]
        base = {k: int(v) for k, v in raw}
        remainder = remaining - sum(base.values())
        # sort by fractional part desc
        frac_order = sorted(((k, v - int(v)) for (k, v) in raw), key=lambda x: x[1], reverse=True)
        for i in range(remainder):
            base[frac_order[i % len(frac_order)][0]] += 1
        for k, c in base.items():
            plan.extend([k] * c)

    # If still short, fill with 'primary' or any available
    if len(plan) < num_agents:
        fallback_key = (
            "primary" if "primary" in pool else (next(iter(pool.keys())) if pool else None)
        )
        if fallback_key is None:
            raise RuntimeError("No LLMs available to assign.")
        plan.extend([fallback_key] * (num_agents - len(plan)))

    # If too long, trim
    plan = plan[:num_agents]

    if shuffle:
        rng.shuffle(plan)

    return [(k, pool[k]) for k in plan]
