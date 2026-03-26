import pytest

from evokernel.config import load_runtime_config
from evokernel.domain.errors import ConfigLoadError
from evokernel.domain.models import EpisodeState, VerificationOutcome


def test_load_runtime_config_reads_lambda_multiplier(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[retrieval]\nfinal_context_count=4\nover_retrieval_lambda=3\n",
        encoding="utf-8",
    )

    config = load_runtime_config(config_path)

    assert config.retrieval.over_retrieval_lambda == 3


def test_episode_state_starts_in_drafting():
    state = EpisodeState.initial(task_id="vector_add", budget=30)

    assert state.stage.value == "drafting"
    assert state.remaining_budget == 30
    assert state.start_points == []


def test_verification_outcome_requires_all_gates_for_feasibility():
    outcome = VerificationOutcome(
        anti_hack_passed=True,
        compile_passed=True,
        correctness_passed=False,
        latency_ms=None,
        error_category="wrong_answer",
        feedback_summary="mismatch at output[3]",
    )

    assert outcome.is_feasible is False


def test_load_runtime_config_reads_generator_provider_fields(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[generator]\nprovider=\"openai_compatible\"\nmodel=\"gpt-5.4\"\nbase_url=\"https://api.example.test/v1\"\napi_key_env=\"OPENAI_API_KEY\"\n",
        encoding="utf-8",
    )

    config = load_runtime_config(config_path)

    assert config.generator.provider == "openai_compatible"
    assert config.generator.model == "gpt-5.4"


def test_load_runtime_config_reads_backend_benchmark_and_artifact_paths(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[runtime]\nbackend=\"cpu_simd\"\nartifact_dir=\"artifacts\"\nlog_dir=\"logs\"\n[benchmark]\ntasks=[\"vector_add\", \"reduce_sum\"]\n",
        encoding="utf-8",
    )

    config = load_runtime_config(config_path)

    assert config.runtime.backend == "cpu_simd"
    assert config.benchmark.tasks == ["vector_add", "reduce_sum"]


def test_load_runtime_config_wraps_invalid_typed_values(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[retrieval]\nover_retrieval_lambda=\"three\"\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigLoadError):
        load_runtime_config(config_path)


def test_load_runtime_config_rejects_unknown_keys(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[runtime]\nbackend=\"cpu_simd\"\nunknown_option=true\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigLoadError):
        load_runtime_config(config_path)
