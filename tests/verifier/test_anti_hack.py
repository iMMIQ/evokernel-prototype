from dataclasses import dataclass

from evokernel.benchmarks.cpu_simd_tasks import build_vector_add_task
from evokernel.verifier.anti_hack import check_for_disallowed_patterns
from evokernel.verifier.core import verify_candidate


def test_anti_hack_rejects_numpy_shortcuts():
    result = check_for_disallowed_patterns("import numpy as np\nnp.add(a, b)")

    assert result.passed is False
    assert result.error_category == "anti_hack"


def test_anti_hack_allows_plain_candidate_code():
    result = check_for_disallowed_patterns(
        'extern "C" void evokernel_entry() {}'
    )

    assert result.passed is True
    assert result.error_category is None


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
