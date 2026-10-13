from system.code_evidence import has_code_script_recording_anomaly


def test_code_script_recording_anomaly_detected_when_success_without_script():
    assert has_code_script_recording_anomaly(
        successful_code_executions=1,
        code_script="",
    )


def test_code_script_recording_anomaly_not_detected_when_script_present():
    assert not has_code_script_recording_anomaly(
        successful_code_executions=1,
        code_script="print('ok')",
    )


def test_code_script_recording_anomaly_not_detected_without_success():
    assert not has_code_script_recording_anomaly(
        successful_code_executions=0,
        code_script="",
    )
