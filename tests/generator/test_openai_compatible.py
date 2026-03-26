import pytest

from evokernel.config import GeneratorConfig
from evokernel.generator.openai_compatible import OpenAICompatibleGenerator


def test_openai_compatible_generator_builds_responses_payload():
    generator = OpenAICompatibleGenerator(
        model="gpt-5.4",
        base_url="https://example.invalid/v1",
        api_key="test",
    )

    payload = generator.build_payload(system_prompt="sys", user_prompt="usr")

    assert payload["model"] == "gpt-5.4"
    assert payload["input"][0]["role"] == "system"


def test_openai_compatible_generator_generate_uses_http_client(httpx_mock):
    httpx_mock.add_response(
        json={
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": "void evokernel_entry() {}",
                        }
                    ]
                }
            ]
        }
    )
    generator = OpenAICompatibleGenerator(
        model="gpt-5.4",
        base_url="https://example.invalid/v1",
        api_key="test",
    )

    result = generator.generate_from_prompts(system_prompt="sys", user_prompt="usr")

    assert "evokernel_entry" in result.code
    request = httpx_mock.get_requests()[0]
    assert str(request.url) == "https://example.invalid/v1/responses"
    assert request.headers["Authorization"] == "Bearer test"
    assert request.headers["Content-Type"] == "application/json"
    assert request.read().decode("utf-8") == (
        '{"model":"gpt-5.4","input":['
        '{"role":"system","content":[{"type":"input_text","text":"sys"}]},'
        '{"role":"user","content":[{"type":"input_text","text":"usr"}]}'
        "]}"
    )


def test_openai_compatible_generator_from_config_requires_api_key_env(monkeypatch):
    monkeypatch.delenv("MISSING_TEST_API_KEY", raising=False)

    with pytest.raises(ValueError, match="MISSING_TEST_API_KEY"):
        OpenAICompatibleGenerator.from_config(
            GeneratorConfig(
                model="gpt-5.4",
                base_url="https://example.invalid/v1",
                api_key_env="MISSING_TEST_API_KEY",
            )
        )


def test_openai_compatible_generator_generate_fails_when_response_has_no_output_text(
    httpx_mock,
):
    httpx_mock.add_response(
        json={"output": [{"content": [{"type": "refusal", "text": "no"}]}]}
    )
    generator = OpenAICompatibleGenerator(
        model="gpt-5.4",
        base_url="https://example.invalid/v1",
        api_key="test",
    )

    with pytest.raises(ValueError, match="No usable output_text"):
        generator.generate_from_prompts(system_prompt="sys", user_prompt="usr")
