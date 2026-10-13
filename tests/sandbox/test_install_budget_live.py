"""Live check: a missing import must not read as an execution timeout.

Requires a running sandbox server (``SANDBOX_BASE_URL``) built from the current
``sandbox/`` tree -- the daemon half of the fix ships inside the
``agent-sandbox:py311`` image, so rebuild it before running this.
"""

import os

import pytest


@pytest.fixture
def sandbox_client():
    """Build a SandboxClient, or skip if no server is configured."""
    url = os.environ.get("SANDBOX_BASE_URL")
    if not url:
        pytest.skip("SANDBOX_BASE_URL not set")
    from env_setup.sandbox_utils import SandboxClient

    return SandboxClient(base_url=url)


@pytest.mark.sandbox
def test_missing_import_reports_an_install_result_not_a_timeout(sandbox_client):
    session_id = sandbox_client.create_session()
    try:
        response = sandbox_client.run(
            session_id,
            "import definitely_not_a_real_package_xyz",
            auto_install=True,
            timeout=30,
        )
    finally:
        sandbox_client.shutdown(session_id)

    # The install phase is bounded, so the job returns a normal (failing) cell
    # result rather than blowing the execution deadline and restarting.
    assert response.get("timed_out") is not True
    assert response.get("restarted") is not True
    assert response.get("pre_install") is not None
