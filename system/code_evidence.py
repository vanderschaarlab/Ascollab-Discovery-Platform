def has_code_script_recording_anomaly(
    *,
    successful_code_executions: int,
    code_script: str,
) -> bool:
    return int(successful_code_executions or 0) > 0 and not str(code_script or "").strip()
