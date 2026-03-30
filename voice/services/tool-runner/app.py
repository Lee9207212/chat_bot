from __future__ import annotations

import json
import multiprocessing
import os
import traceback
from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel


def _as_bool(value: str, default: bool = False) -> bool:
    raw = str(value if value is not None else "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


RUNNER_DEBUG = _as_bool(os.getenv("TOOL_RUNNER_DEBUG", "0"))
MAX_EXEC_TIMEOUT_SEC = float(os.getenv("TOOL_RUNNER_MAX_EXEC_TIMEOUT_SEC", "10") or "10")
MAX_MEMORY_MB = int(float(os.getenv("TOOL_RUNNER_MAX_MEMORY_MB", "256") or "256"))
MAX_CPU_SEC = int(float(os.getenv("TOOL_RUNNER_MAX_CPU_SEC", "2") or "2"))
ALLOWED_PATH_PREFIXES = [
    p.strip()
    for p in str(
        os.getenv("TOOL_RUNNER_ALLOWED_PATH_PREFIXES", "/app/workspace/_system/custom_tools")
    ).split(",")
    if p.strip()
]

app = FastAPI(title="tool-runner", version="0.1.0")


class ValidateBundleRequest(BaseModel):
    spec_text: str
    manifest_text: str
    tests_text: str
    tool_code: str


def _validate_local_bundle(spec_text: str, manifest_text: str, tests_text: str, tool_code: str) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    passed = True

    def record(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed
        checks.append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            passed = False

    try:
        spec_payload = json.loads(spec_text or "{}")
        if not isinstance(spec_payload, dict):
            raise ValueError("spec.json must be object")
        record("spec_json_parse", True, "")
    except Exception as exc:
        spec_payload = {}
        record("spec_json_parse", False, str(exc))

    try:
        manifest_payload = json.loads(manifest_text or "{}")
        if not isinstance(manifest_payload, dict):
            raise ValueError("manifest.json must be object")
        record("manifest_json_parse", True, "")
    except Exception as exc:
        manifest_payload = {}
        record("manifest_json_parse", False, str(exc))

    try:
        tests_payload = json.loads(tests_text or "{}")
        tests_list = tests_payload.get("tests") if isinstance(tests_payload, dict) else None
        if not isinstance(tests_list, list) or not tests_list:
            raise ValueError("contract_tests.json must include non-empty tests[]")
        record("contract_tests_json_parse", True, "")
    except Exception as exc:
        tests_payload = {"tests": []}
        record("contract_tests_json_parse", False, str(exc))

    entrypoint = str(manifest_payload.get("entrypoint") or "").strip()
    record("manifest_entrypoint", bool(entrypoint), entrypoint or "missing manifest.entrypoint")

    try:
        compile(tool_code or "", "tool.py", "exec")
        record("tool_code_compile", True, "")
    except Exception as exc:
        record("tool_code_compile", False, str(exc))

    spec_version = str(spec_payload.get("version") or "").strip()
    manifest_version = str(manifest_payload.get("version") or "").strip()
    record(
        "version_consistency",
        bool(spec_version and manifest_version and spec_version == manifest_version),
        f"spec={spec_version or '-'} manifest={manifest_version or '-'}",
    )
    return {
        "status": "pass" if passed else "fail",
        "checks": checks,
        "tests_payload": tests_payload,
    }


def _subset_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        for key, value in expected.items():
            if key not in actual:
                return False
            if not _subset_match(value, actual.get(key)):
                return False
        return True
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        if len(expected) > len(actual):
            return False
        for idx, value in enumerate(expected):
            if not _subset_match(value, actual[idx]):
                return False
        return True
    return expected == actual


def _run_tool_worker(code: str, payload: Any, queue: multiprocessing.Queue) -> None:
    try:
        try:
            import resource

            if MAX_MEMORY_MB > 0:
                memory_bytes = MAX_MEMORY_MB * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            if MAX_CPU_SEC > 0:
                resource.setrlimit(resource.RLIMIT_CPU, (MAX_CPU_SEC, MAX_CPU_SEC))
        except Exception:
            pass
        namespace: Dict[str, Any] = {}
        exec(compile(code, "tool.py", "exec"), namespace)
        run_fn = namespace.get("run")
        if not callable(run_fn):
            raise RuntimeError("tool.py must expose callable run(payload)")
        result = run_fn(payload if isinstance(payload, dict) else {})
        queue.put({"ok": True, "result": result})
    except Exception as exc:
        queue.put(
            {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=4),
            }
        )


def _execute_tool_with_timeout(code: str, payload: Any, timeout_sec: float) -> Dict[str, Any]:
    queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=1)
    process = multiprocessing.Process(target=_run_tool_worker, args=(code, payload, queue), daemon=True)
    process.start()
    process.join(timeout=max(0.1, min(timeout_sec, MAX_EXEC_TIMEOUT_SEC)))
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        return {"ok": False, "error": f"timeout after {timeout_sec:.1f}s", "timeout": True}
    if queue.empty():
        return {"ok": False, "error": "no_result"}
    try:
        return queue.get_nowait()
    except Exception as exc:
        return {"ok": False, "error": f"result_unavailable: {exc}"}


def _walk_values(value: Any, out: List[str]) -> None:
    if isinstance(value, str):
        out.append(value)
        return
    if isinstance(value, dict):
        for nested in value.values():
            _walk_values(nested, out)
        return
    if isinstance(value, list):
        for nested in value:
            _walk_values(nested, out)


def _is_path_allowed(path_value: str) -> bool:
    text = str(path_value or "").strip()
    if not text:
        return True
    if not text.startswith("/"):
        return True
    return any(text.startswith(prefix) for prefix in ALLOWED_PATH_PREFIXES)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "service": "tool-runner"}


@app.post("/validate-bundle")
async def validate_bundle(payload: ValidateBundleRequest) -> Dict[str, Any]:
    base = _validate_local_bundle(
        spec_text=payload.spec_text,
        manifest_text=payload.manifest_text,
        tests_text=payload.tests_text,
        tool_code=payload.tool_code,
    )
    checks = list(base.get("checks") or [])
    status = str(base.get("status") or "fail")
    tests_payload = base.get("tests_payload") if isinstance(base.get("tests_payload"), dict) else {"tests": []}
    tests = tests_payload.get("tests")
    if not isinstance(tests, list):
        tests = []

    for idx, test_case in enumerate(tests, start=1):
        if not isinstance(test_case, dict):
            checks.append({"name": f"contract_test_{idx}", "ok": False, "detail": "test case must be object"})
            status = "fail"
            continue
        case_name = str(test_case.get("name") or f"case_{idx}").strip()[:120]
        case_input = test_case.get("input") if isinstance(test_case.get("input"), dict) else {}
        flat_values: List[str] = []
        _walk_values(case_input, flat_values)
        disallowed_paths = [item for item in flat_values if item.startswith("/") and not _is_path_allowed(item)]
        if disallowed_paths:
            checks.append(
                {
                    "name": f"contract_test:{case_name}:path_whitelist",
                    "ok": False,
                    "detail": f"disallowed absolute path: {disallowed_paths[0]}",
                }
            )
            status = "fail"
            continue
        timeout_sec = float(test_case.get("timeout_sec") or 6.0)
        output = _execute_tool_with_timeout(payload.tool_code, case_input, timeout_sec)
        if not output.get("ok"):
            checks.append(
                {
                    "name": f"contract_test:{case_name}",
                    "ok": False,
                    "detail": str(output.get("error") or "execution_failed"),
                }
            )
            status = "fail"
            continue
        expect_contains = test_case.get("expect_contains")
        result = output.get("result")
        if expect_contains is not None and not _subset_match(expect_contains, result):
            checks.append(
                {
                    "name": f"contract_test:{case_name}",
                    "ok": False,
                    "detail": "result does not match expect_contains",
                }
            )
            status = "fail"
            continue
        checks.append(
            {
                "name": f"contract_test:{case_name}",
                "ok": True,
                "detail": "pass",
            }
        )

    if RUNNER_DEBUG:
        return {"status": status, "checks": checks, "runner": "sandbox", "debug": {"tests_count": len(tests)}}
    return {"status": status, "checks": checks, "runner": "sandbox"}
