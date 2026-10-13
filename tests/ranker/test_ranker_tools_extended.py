from ranker.ranker_tools import SandboxToolState, build_ranker_tools


class _FakeSticky:
    def __init__(self):
        self.ensure_calls = 0
        self.close_calls = 0
        self.runs = []

    def ensure(self):
        self.ensure_calls += 1

    def run(self, code, **kwargs):
        self.runs.append((code, kwargs))
        return {
            "status": "success",
            "stdout": "hello\n",
            "stderr": "",
            "result": "42",
            "images": [{"data": "abc"}],
        }

    def close(self):
        self.close_calls += 1


class _FakeSandboxClient:
    def __init__(self):
        self.sticky = _FakeSticky()
        self.created = []

    def sticky_session(self, **kwargs):
        self.created.append(kwargs)
        return self.sticky


def test_sandbox_tool_state_creates_session_lazily_and_formats_output():
    client = _FakeSandboxClient()
    state = SandboxToolState(client, {"code_timeout_seconds": 5})

    text = state.run_code("print('hello')")

    assert client.sticky.ensure_calls == 1
    assert client.sticky.runs[0][1]["timeout"] == 5
    assert "Status: success" in text
    assert "stdout:\nhello" in text
    assert "result: 42" in text
    assert "image(s) produced" in text
    state.cleanup()
    assert client.sticky.close_calls == 1


def test_build_ranker_tools_returns_cleanup_that_swallows_errors():
    client = _FakeSandboxClient()
    tools, cleanup = build_ranker_tools(
        {"sandbox": {"enabled": True, "code_timeout_seconds": 3}},
        sandbox_client=client,
    )

    assert list(tools) == ["sandbox"]
    assert '- sandbox: args = { "code": str }' in tools["sandbox"]["tool_signature"]
    tools["sandbox"]["object"].invoke({"code": "1 + 1"})
    cleanup()
    cleanup()
    assert client.sticky.close_calls == 1


def test_build_ranker_tools_disabled_config_has_no_tools():
    tools, cleanup = build_ranker_tools(
        {"arxiv": {"enabled": False}, "sandbox": {"enabled": False}}
    )

    assert tools == {}
    cleanup()
