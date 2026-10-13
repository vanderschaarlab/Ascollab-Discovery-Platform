from env_setup.llama_emulation import is_llama_backend, make_llama_auto_tools_prompt
from env_setup.sandbox_tools import (
    _emit_arxiv_code,
    _emit_search_agent_registry_code,
)
from env_setup.tools import _extract_query, _with_defaults


def test_emit_arxiv_code_bakes_configuration_and_signature():
    code, sig = _emit_arxiv_code({"top_k_results": 2})

    assert "_ARXIV_CFG = {'top_k_results': 2}" in code
    assert "def search_arxiv(query: str) -> str:" in sig


def test_emit_registry_signatures():
    registry_code, registry_sig = _emit_search_agent_registry_code({})

    assert "registry_query_tool" in registry_code
    assert "establish_collaboration" in registry_sig


def test_tool_defaults_and_query_extraction():
    defaults = _with_defaults("arxiv", {"top_k_results": 1})

    assert defaults["top_k_results"] == 1
    assert "load_max_docs" in defaults
    assert _extract_query({"query": "rna", "pmcid": "PMC1"}) == "rna"
    assert "PMC1" in _extract_query({"pmcid": "PMC1"})


def test_llama_backend_detection_and_prompt_constraints():
    class Llm:
        azure_deployment = "llama-70b"

    assert is_llama_backend(Llm()) is True
    prompt = make_llama_auto_tools_prompt("Base prompt", valid_agent_ids=["agent_01"])

    assert "Base prompt" in prompt
    assert "agent_01" in prompt
    assert 'Action: {"tool": "<tool_name>"' in prompt


def test_llama_detection_covers_both_providers():
    from langchain_openai import AzureChatOpenAI, ChatOpenAI

    def azure(deployment):
        # Built offline; no request is made.
        return AzureChatOpenAI(
            azure_deployment=deployment,
            azure_endpoint="https://example.invalid",
            api_key="not-a-key",
            api_version="2024-12-01-preview",
        )

    assert is_llama_backend(azure("Llama-3.3-70B-Instruct")) is True
    assert is_llama_backend(azure("gpt-5")) is False
    openai_llama = ChatOpenAI(model="meta-llama-3.1-70b-instruct", api_key="not-a-key")
    assert is_llama_backend(openai_llama) is True
    assert is_llama_backend(ChatOpenAI(model="gpt-4o", api_key="not-a-key")) is False
