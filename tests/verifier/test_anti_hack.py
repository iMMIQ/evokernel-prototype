from dataclasses import dataclass

from evokernel.benchmarks.cpu_simd_tasks import build_vector_add_task
from evokernel.verifier.anti_hack import (
    check_for_disallowed_patterns,
    inspect_with_model,
)
from evokernel.verifier.core import verify_candidate


def test_anti_hack_rejects_numpy_shortcuts():
    result = check_for_disallowed_patterns("import numpy as np\nnp.add(a, b)")

    assert result.passed is False
    assert result.error_category == "anti_hack"


def test_anti_hack_rejects_unaliased_numpy_shortcuts():
    result = check_for_disallowed_patterns("import numpy\nnumpy.add(a, b)")

    assert result.passed is False
    assert result.error_category == "anti_hack"


def test_anti_hack_rejects_numpy_submodule_imports():
    result = check_for_disallowed_patterns("from numpy.linalg import norm")

    assert result.passed is False
    assert result.error_category == "anti_hack"


def test_anti_hack_rejects_tensor_library_submodule_imports():
    result = check_for_disallowed_patterns(
        "from torch.nn import functional as F"
    )

    assert result.passed is False
    assert result.error_category == "anti_hack"


def test_anti_hack_rejects_dynamic_import():
    result = check_for_disallowed_patterns(
        '__import__("numpy").add(a, b)'
    )

    assert result.passed is False
    assert result.error_category == "anti_hack"


def test_anti_hack_rejects_importlib():
    result = check_for_disallowed_patterns(
        'import importlib\nimportlib.import_module("numpy")'
    )

    assert result.passed is False
    assert result.error_category == "anti_hack"


def test_anti_hack_rejects_subprocess():
    result = check_for_disallowed_patterns(
        'subprocess.run(["python", "-c", "import numpy"])'
    )

    assert result.passed is False
    assert result.error_category == "anti_hack"


def test_anti_hack_rejects_eval():
    result = check_for_disallowed_patterns(
        'eval("__import__(\\"numpy\\")")'
    )

    assert result.passed is False
    assert result.error_category == "anti_hack"


def test_anti_hack_rejects_os_system():
    result = check_for_disallowed_patterns('os.system("python -c import numpy")')

    assert result.passed is False
    assert result.error_category == "anti_hack"


def test_anti_hack_allows_plain_candidate_code():
    result = check_for_disallowed_patterns(
        'extern "C" void evokernel_entry() {}'
    )

    assert result.passed is True
    assert result.error_category is None


def test_anti_hack_regex_failure_short_circuits_inspector():
    """When regex catches a violation, the inspector should NOT be called."""
    inspector_calls = []

    def fake_inspector(**kwargs):
        inspector_calls.append(1)
        from evokernel.verifier.anti_hack import AntiHackResult
        return AntiHackResult(passed=True, error_category=None, feedback_summary=None)

    result = check_for_disallowed_patterns(
        "import numpy as np\nnp.add(a, b)",
        inspector={"base_url": "http://fake", "api_key": "k", "model": "m"},
    )

    assert result.passed is False
    assert inspector_calls == []


def test_anti_hack_inspector_not_called_when_none():
    """No inspector = no model call, just regex."""
    result = check_for_disallowed_patterns(
        'extern "C" void evokernel_entry() { int x = 1; }',
        inspector=None,
    )

    assert result.passed is True
    assert result.error_category is None


def test_inspect_with_model_flags_cheating(httpx_mock):
    httpx_mock.add_response(
        json={
            "choices": [
                {
                    "message": {
                        "content": '{"cheating": true, "reason": "uses dlopen to load numpy"}'
                    }
                }
            ]
        }
    )

    result = inspect_with_model(
        candidate_code='void* h = dlopen("libnumpy.so", 1);',
        base_url="https://example.invalid/v1",
        api_key="test",
        model="test-model",
    )

    assert result.passed is False
    assert result.error_category == "anti_hack_inspector"
    assert "dlopen" in result.feedback_summary


def test_inspect_with_model_passes_clean_code(httpx_mock):
    httpx_mock.add_response(
        json={
            "choices": [
                {
                    "message": {
                        "content": '{"cheating": false, "reason": "legitimate SIMD code"}'
                    }
                }
            ]
        }
    )

    result = inspect_with_model(
        candidate_code='#include <immintrin.h>\nvoid foo() { __m256 v = _mm256_setzero_ps(); }',
        base_url="https://example.invalid/v1",
        api_key="test",
        model="test-model",
    )

    assert result.passed is True
    assert result.error_category is None


def test_inspect_with_model_handles_http_failure(httpx_mock):
    httpx_mock.add_exception(Exception("connection refused"))

    result = inspect_with_model(
        candidate_code="void foo() {}",
        base_url="https://example.invalid/v1",
        api_key="test",
        model="test-model",
    )

    assert result.passed is True
    assert "inspector call failed" in result.feedback_summary


def test_inspect_with_model_handles_malformed_response(httpx_mock):
    httpx_mock.add_response(json={"choices": [{"message": {"content": "true cheating detected"}}]})

    result = inspect_with_model(
        candidate_code="void foo() {}",
        base_url="https://example.invalid/v1",
        api_key="test",
        model="test-model",
    )

    assert result.passed is False
    assert result.error_category == "anti_hack_inspector"


@dataclass(slots=True)
class _UnusedBackend:
    called: bool = False

    def materialize_candidate(self, task, candidate_code, attempt_id):
        self.called = True
        raise AssertionError("anti-hack failures must short-circuit verification")


def test_verify_candidate_short_circuits_on_anti_hack_failure():
    backend = _UnusedBackend()

    outcome = verify_candidate(
        backend=backend,
        task=build_vector_add_task(),
        candidate_code="import numpy as np\nnp.add(a, b)",
        attempt_id="attempt-anti-hack",
    )

    assert outcome.anti_hack_passed is False
    assert outcome.compile_passed is False
    assert outcome.correctness_passed is False
    assert outcome.error_category == "anti_hack"
    assert backend.called is False


def test_verify_candidate_passes_inspector_through(httpx_mock):
    """verify_candidate should forward inspector to check_for_disallowed_patterns."""
    httpx_mock.add_response(
        json={
            "choices": [
                {
                    "message": {
                        "content": '{"cheating": true, "reason": "hardcoded output"}'
                    }
                }
            ]
        }
    )

    backend = _UnusedBackend()
    outcome = verify_candidate(
        backend=backend,
        task=build_vector_add_task(),
        candidate_code='void foo() {}',
        attempt_id="attempt-inspector",
        inspector={
            "base_url": "https://example.invalid/v1",
            "api_key": "test",
            "model": "test-model",
        },
    )

    assert outcome.anti_hack_passed is False
    assert outcome.error_category == "anti_hack_inspector"
    assert backend.called is False
