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
