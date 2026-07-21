from jang_tools.mistral3.runtime import encode_with_image


class _Tokenizer:
    def encode(self, prompt):
        assert prompt == "The capital of France is"
        return [1, 2, 3]


def test_text_only_encode_does_not_require_optional_pixtral_runtime():
    ids, image = encode_with_image(
        _Tokenizer(),
        "The capital of France is",
        image_path=None,
        image_token_id=10,
    )

    assert ids == [1, 2, 3]
    assert image is None
