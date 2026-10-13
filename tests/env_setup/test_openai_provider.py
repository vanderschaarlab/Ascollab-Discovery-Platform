"""OpenAI-compatible endpoints, exercised over real HTTP against a local fake server.

No key or network access is needed: the fake server answers the two calls the pipeline makes
(chat completions and embeddings) in the OpenAI wire format. Azure stays the default provider.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings

from env_setup.build_llms import (
    _mk_embedder,
    _mk_llm,
    build_llm_and_embedder,
    build_llm_pool,
    load_endpoint_config,
    make_sdk_client,
)

FAKE_KEY = "sk-test-not-a-real-key"


class _FakeOpenAI(BaseHTTPRequestHandler):
    calls: list[dict] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).calls.append(
            {"path": self.path, "auth": self.headers.get("Authorization"), "body": body}
        )
        if self.path.endswith("/chat/completions"):
            payload = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "pong",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        elif self.path.endswith("/embeddings"):
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            payload = {
                "object": "list",
                "model": body["model"],
                "data": [
                    {"object": "embedding", "index": i, "embedding": [0.1, 0.2, 0.3]}
                    for i in range(len(inputs))
                ],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
        else:
            self.send_error(404)
            return
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_openai():
    _FakeOpenAI.calls = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAI)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


def _chat_cfg(base_url, model="gpt-4o-mini"):
    return {
        "provider": "openai",
        "model": model,
        "api_key": FAKE_KEY,
        "base_url": base_url,
        "max_retries": 0,
        "timeout": 10,
    }


def _embed_cfg(base_url):
    # check_embedding_ctx_length=false sends raw strings, as non-OpenAI servers expect.
    return {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "api_key": FAKE_KEY,
        "base_url": base_url,
        "check_embedding_ctx_length": False,
        "max_retries": 0,
    }


def test_chat_and_embedding_calls_go_through_the_openai_path(fake_openai):
    llm, embedder = build_llm_and_embedder(_chat_cfg(fake_openai), _embed_cfg(fake_openai))
    assert isinstance(llm, ChatOpenAI) and not isinstance(llm, AzureChatOpenAI)
    assert isinstance(embedder, OpenAIEmbeddings)

    assert llm.invoke("ping").content == "pong"
    assert embedder.embed_query("kidney cancer") == pytest.approx([0.1, 0.2, 0.3])

    chat, embed = _FakeOpenAI.calls
    assert chat["path"] == "/v1/chat/completions"
    assert chat["auth"] == f"Bearer {FAKE_KEY}"
    assert chat["body"]["model"] == "gpt-4o-mini"
    assert embed["path"] == "/v1/embeddings"
    assert embed["body"]["model"] == "text-embedding-3-small"


def test_pool_backends_can_mix_providers(fake_openai):
    from omegaconf import OmegaConf

    endpoints = OmegaConf.create(
        {
            "llm_primary": _chat_cfg(fake_openai),
            "qwen": _chat_cfg(fake_openai, model="qwen2.5-72b-instruct"),
        }
    )
    pool = build_llm_pool(endpoints, keys=["qwen"])
    assert pool["qwen"].invoke("ping").content == "pong"
    assert _FakeOpenAI.calls[-1]["body"]["model"] == "qwen2.5-72b-instruct"


def test_direct_sdk_client_uses_the_same_endpoint_file(fake_openai, tmp_path):
    # The prior-paper summariser calls the SDK directly with this client.
    path = tmp_path / "default.yaml"
    path.write_text(
        "provider: openai\n"
        "model: gpt-4o-mini\n"
        "api_key: ${oc.env:FAKE_OPENAI_KEY}\n"
        f"base_url: {fake_openai}\n"
        "seed: ${SEED}\n",
        encoding="utf-8",
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("FAKE_OPENAI_KEY", FAKE_KEY)
        cfg = load_endpoint_config(path)
    assert cfg is not None and cfg["seed"] is None  # needs the Hydra run config
    client, model = make_sdk_client(cfg)
    resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "x"}])
    assert resp.choices[0].message.content == "pong"
    assert _FakeOpenAI.calls[-1]["auth"] == f"Bearer {FAKE_KEY}"


def test_azure_stays_the_default_provider():
    azure = {
        "azure_deployment": "gpt-5",
        "azure_endpoint": "https://example.invalid",
        "api_key": FAKE_KEY,
        "api_version": "2024-12-01-preview",
    }
    llm = _mk_llm(azure)
    assert isinstance(llm, AzureChatOpenAI)
    assert llm.deployment_name == "gpt-5"
    assert isinstance(_mk_embedder({**azure, "azure_deployment": "emb"}), AzureOpenAIEmbeddings)
    client, model = make_sdk_client(azure)
    assert type(client).__name__ == "AzureOpenAI" and model == "gpt-5"


def test_unfilled_endpoints_have_no_sdk_client_and_unknown_providers_fail():
    assert make_sdk_client(None) is None
    assert make_sdk_client({"provider": "openai", "api_key": None, "model": "m"}) is None
    assert make_sdk_client({"azure_deployment": None, "api_key": None}) is None
    with pytest.raises(ValueError, match="Unknown endpoint provider"):
        _mk_llm({"provider": "bedrock", "model": "m"})


def test_azure_non_openai_models_send_their_deployment_as_the_model():
    # Some servers behind Azure deployments of non-OpenAI models reject a request whose body has
    # no model name; GPT deployments keep sending none.
    azure = {
        "azure_endpoint": "https://example.invalid",
        "api_key": FAKE_KEY,
        "api_version": "2024-12-01-preview",
    }
    assert _mk_llm({**azure, "azure_deployment": "DeepSeek-R1"}).model_name == "DeepSeek-R1"
    assert _mk_llm({**azure, "azure_deployment": "Llama-3.3-70B"}).model_name == "Llama-3.3-70B"
    assert _mk_llm({**azure, "azure_deployment": "gpt-4o"}).model_name is None
    assert _mk_llm({**azure, "azure_deployment": "gpt-5"}).model_name is None
