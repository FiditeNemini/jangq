from pathlib import Path


def test_nemotron_omni_mlx_llm_loader_uses_non_strict_model_load(monkeypatch, tmp_path):
    """The LLM-only load must ignore RADIO/Parakeet tensors in omni bundles."""

    from jang_tools.nemotron_omni_chat import _load_mlx_lm_ignoring_omni_extras
    import mlx_lm.utils as mlx_utils

    calls = {}
    fake_model = object()
    fake_tokenizer = object()

    def fake_load_model(model_path, lazy=False, strict=True, model_config=None):
        calls["load_model"] = {
            "model_path": model_path,
            "lazy": lazy,
            "strict": strict,
            "model_config": model_config,
        }
        return fake_model, {"eos_token_id": [11, 2]}

    def fake_load_tokenizer(model_path, tokenizer_config=None, eos_token_ids=None):
        calls["load_tokenizer"] = {
            "model_path": model_path,
            "tokenizer_config": tokenizer_config,
            "eos_token_ids": eos_token_ids,
        }
        return fake_tokenizer

    monkeypatch.setattr(mlx_utils, "load_model", fake_load_model)
    monkeypatch.setattr(mlx_utils, "load_tokenizer", fake_load_tokenizer)

    model, tokenizer = _load_mlx_lm_ignoring_omni_extras(tmp_path)

    assert model is fake_model
    assert tokenizer is fake_tokenizer
    assert calls["load_model"]["model_path"] == Path(tmp_path)
    assert calls["load_model"]["lazy"] is False
    assert calls["load_model"]["strict"] is False
    assert calls["load_tokenizer"]["eos_token_ids"] == [11, 2]


def test_nemotron_omni_encoder_view_stubs_duplicate_torch_llm(tmp_path):
    """The PyTorch encoder view must not instantiate a second full LLM."""

    from jang_tools.nemotron_omni_chat import _populate_omni_encoder_view

    bundle = tmp_path / "bundle"
    view = tmp_path / "view"
    bundle.mkdir()
    view.mkdir()
    (bundle / "config.json").write_text('{"model_type":"nemotron_h"}')
    (bundle / "config_omni.json").write_text('{"model_type":"omni"}')
    (bundle / "modeling_nemotron_h.py").write_text("REAL_LLM = True\n")
    (bundle / "modeling.py").write_text("MODEL = True\n")

    _populate_omni_encoder_view(bundle, view)

    assert (view / "config.json").is_symlink()
    assert (view / "config.json").resolve() == bundle / "config_omni.json"
    assert not (view / "modeling_nemotron_h.py").is_symlink()
    stub = (view / "modeling_nemotron_h.py").read_text()
    assert "class NemotronHForCausalLM" in stub
    assert "stubbed in vMLX Omni" in stub
    assert (view / "modeling.py").is_symlink()


def test_nemotron_omni_video_prompt_uses_image_placeholders():
    """Nemotron-Omni video embeds occupy <image> slots, not literal <video>."""

    from jang_tools.nemotron_omni_chat import OmniChat

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

    chat = OmniChat.__new__(OmniChat)
    chat.tokenizer = FakeTokenizer()

    prompt = chat._build_prompt("Describe.", n_video_tokens=3)

    assert "<img><image><image><image></img>" in prompt
    assert "<video>" not in prompt
