import pytest

from evokernel.config import GeneratorConfig
from evokernel.generator.openai_compatible import (
    OpenAICompatibleGenerator,
    _strip_code_fences,
)


def test_openai_compatible_generator_builds_chat_completions_payload():
    generator = OpenAICompatibleGenerator(
        model="gpt-5.4",
        base_url="https://example.invalid/v1",
        api_key="test",
    )

    payload = generator.build_payload(system_prompt="sys", user_prompt="usr")

    assert payload["model"] == "gpt-5.4"
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    assert payload["messages"][1] == {"role": "user", "content": "usr"}


def test_openai_compatible_generator_generate_uses_http_client(httpx_mock):
    httpx_mock.add_response(
        json={
            "choices": [
                {
                    "message": {
                        "content": "void evokernel_entry() {}",
                    }
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
    assert str(request.url) == "https://example.invalid/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer test"
    assert request.headers["Content-Type"] == "application/json"


def test_openai_compatible_generator_from_config_prefers_direct_key():
    generator = OpenAICompatibleGenerator.from_config(
        GeneratorConfig(
            model="gpt-5.4",
            base_url="https://example.invalid/v1",
            api_key="direct-key",
            api_key_env="MISSING_TEST_API_KEY",
        )
    )
    assert generator.api_key == "direct-key"


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


def test_openai_compatible_generator_generate_fails_when_no_content(
    httpx_mock,
):
    httpx_mock.add_response(
        json={"choices": [{"message": {"content": None}}]}
    )
    generator = OpenAICompatibleGenerator(
        model="gpt-5.4",
        base_url="https://example.invalid/v1",
        api_key="test",
    )

    with pytest.raises(ValueError, match="No content found"):
        generator.generate_from_prompts(system_prompt="sys", user_prompt="usr")


def test_strip_code_fences_removes_wrapping_fences():
    assert _strip_code_fences("```c\nvoid foo() {}\n```") == "void foo() {}"


def test_strip_code_fences_removes_cpp_fence():
    assert _strip_code_fences("```cpp\nint x;\n```") == "int x;"


def test_strip_code_fences_no_fence():
    assert _strip_code_fences("void foo() {}") == "void foo() {}"


def test_generator_strips_markdown_code_fences(httpx_mock):
    httpx_mock.add_response(
        json={
            "choices": [
                {
                    "message": {
                        "content": "```cpp\nvoid evokernel_entry() {}\n```",
                    }
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

    assert result.code == "void evokernel_entry() {}"
