import base64

from env_setup.sandbox_utils import (
    extract_sandbox_response,
    get_session_id,
    is_sandbox_execution_successful,
)


def test_extract_sandbox_response_summarizes_text_and_images():
    image_blob = base64.b64encode(b"png").decode("ascii")
    text, images, ok = extract_sandbox_response(
        {
            "status": "success",
            "stdout": "hello",
            "stderr": "",
            "result": "42",
            "images": [{"type": "png", "data": image_blob}],
        }
    )

    assert ok is True
    assert "Status: success" in text
    assert "stdout: hello" in text
    assert "result: 42" in text
    assert "IMAGES =" in text
    assert images == [image_blob]


def test_extract_sandbox_response_trims_long_fields():
    text, _images, ok = extract_sandbox_response(
        {"status": "error", "stderr": "x" * 20},
        trim=True,
        max_chars=5,
    )

    assert ok is False
    assert "trimmed observation" in text


def test_sandbox_response_success_and_session_id_helpers():
    assert is_sandbox_execution_successful({"status": "success"}) is True
    assert is_sandbox_execution_successful({"status": "error"}) is False
    assert is_sandbox_execution_successful("success") is False
    assert get_session_id({"session_metadata": {"session_id": "sid_1"}}) == "sid_1"
    assert get_session_id({}) is None
