import importlib
import json
import subprocess
import sys
from pathlib import Path


CONVERTER = (
    Path(__file__).resolve().parents[1]
    / "jang_tools"
    / "dsv4"
    / "convert_dsv4_jangtq.py"
)
AFFINE_CONVERTER = (
    Path(__file__).resolve().parents[1]
    / "jang_tools"
    / "dsv4"
    / "convert_dsv4_jang.py"
)


def test_dsv4_converter_preserves_f32_control_tensors():
    """DSV4 mHC/Sinkhorn/router/sink tensors must not be downcast to fp16.

    The previous converter path read every passthrough tensor with
    out_dtype=torch.float16, which is exactly how the local bad bundle ended
    up with 344/344 critical control tensors stored as F16.
    """
    src = CONVERTER.read_text()

    assert "CRITICAL_F32_RE" in src
    assert "def read_passthrough" in src
    assert "idx.dtype_of(name) == torch.float32" in src
    assert "CRITICAL_F32_RE.match(name)" in src
    assert "read_passthrough(idx, name)" in src
    assert 'idx.read_tensor(name, out_dtype=torch.float16)' not in src


def test_dsv4_converter_finalizes_jangtq_shipping_layout_by_default():
    """A fixed DSV4 JANGTQ artifact must not stop at raw per-expert shards.

    The shipped bundle needs prestacked switch_mlp tensors and the
    jangtq_runtime sidecar so Python/Swift loaders do not silently diverge on
    codebook/sign generation or restacking behavior.
    """
    src = CONVERTER.read_text()

    assert "def finalize_jangtq_bundle" in src
    assert "rebundle_jangtq_stacked" in src
    assert "build_jangtq_sidecar" in src
    assert "--no-prestack" in src
    assert "--no-sidecar" in src
    assert "prestack=not args.no_prestack" in src
    assert "build_sidecar=not args.no_sidecar" in src
    assert "if FORMAT == \"jangtq\":" in src


def test_dsv4_converter_v3_profile_still_lifts_hash_layers():
    """The size target is the V3/K family, not uniform 2-bit everywhere."""
    src = CONVERTER.read_text()

    assert 'VARIANT == "V3"' in src
    assert "int(m.group(1)) < 3" in src
    assert "return 4, \"mxtq\", 0" in src
    assert "return profile_bits, \"mxtq\", 0" in src


def test_dsv4_converter_metadata_declares_f32_controls_and_mtp_policy():
    """Runtime metadata must tell loaders what was preserved or dropped."""
    src = CONVERTER.read_text()

    assert '"critical_f32_preserved": True' in src
    assert '"critical_control_tensors": "source-f32"' in src
    assert '"dsv4_runtime_requirements"' in src
    assert '"limited_swiglu_tq_patch": FORMAT == "jangtq"' in src
    assert "if drop_mtp:" in src
    assert 'src_cfg["num_nextn_predict_layers"] = 0' in src


def test_dsv4_converter_config_has_top_level_jangtq_metadata():
    """config.json and jang_config.json must agree on JANGTQ routing metadata."""
    src = CONVERTER.read_text()

    assert 'src_cfg["weight_format"] = "mxtq"' in src
    assert 'src_cfg["mxtq_bits"] = mxtq_bits_meta' in src
    assert 'src_cfg["routed_expert_bit_plan"] = routed_expert_bit_plan' in src
    assert 'quant_cfg["routed_expert_bit_plan"] = routed_expert_bit_plan' in src


def test_dsv4_converter_v3_metadata_declares_hash_layer_bit_plan():
    """V3 is mixed routed bits; metadata must not imply uniform 2-bit routed experts."""
    src = CONVERTER.read_text()

    assert "def build_routed_expert_bit_plan" in src
    assert '"hash_layer_bits": 4' in src
    assert '"hash_layer_indices": [0, 1, 2]' in src
    assert '"smooth_layer_bits": profile_bits' in src


def test_dsv4_converter_classify_uses_v3_safe_plan(monkeypatch, tmp_path):
    """DSV4_V3_PLAN_PATH must be a real converter input, not stale docs."""
    conv = importlib.import_module("jang_tools.dsv4.convert_dsv4_jangtq")
    monkeypatch.setattr(conv, "FORMAT", "jangtq")
    monkeypatch.setattr(conv, "VARIANT", "V3")

    plan_path = tmp_path / "bit_plan_v3_safe.json"
    plan_path.write_text(json.dumps({
        "plan": {
            "model.layers.7.mlp.switch_mlp.SWITCH_MLP_LAYER": 4,
            "model.layers.7.self_attn.wq_b": 4,
            "model.layers.8.mlp.shared_experts.down_proj": 6,
            "model.embed_tokens": 4,
        },
        "config": {"routed_gs": 64, "default_gs": 32},
    }))
    monkeypatch.setenv("DSV4_V3_PLAN_PATH", str(plan_path))

    assert conv.classify("layers.7.ffn.experts.0.w1.weight", 2) == (4, "mxtq", 0)
    assert conv.classify("layers.7.attn.wq_b.weight", 2) == (4, "affine", 32)
    assert conv.classify("layers.8.ffn.shared_experts.w2.weight", 2) == (6, "affine", 32)
    assert conv.classify("embed.weight", 2) == (4, "affine", 32)


def test_dsv4_converter_rejects_unproven_affine_routed_plan(monkeypatch, tmp_path):
    """Do not ship mixed affine-routed/TQ-routed DSV4 until runtime support is proven."""
    conv = importlib.import_module("jang_tools.dsv4.convert_dsv4_jangtq")
    monkeypatch.setattr(conv, "FORMAT", "jangtq")
    monkeypatch.setattr(conv, "VARIANT", "V3")

    plan_path = tmp_path / "bit_plan_v3_affine_routed.json"
    plan_path.write_text(json.dumps({
        "plan": {
            "model.layers.9.mlp.switch_mlp.SWITCH_MLP_LAYER": 8,
        },
        "config": {"routed_gs": 64, "default_gs": 32},
    }))
    monkeypatch.setenv("DSV4_V3_PLAN_PATH", str(plan_path))

    try:
        conv.classify("layers.9.ffn.experts.0.w1.weight", 2)
    except ValueError as exc:
        assert "affine routed" in str(exc)
    else:
        raise AssertionError("expected unproven affine-routed DSV4 plan to fail")


def test_dsv4_converter_v3_metadata_summarizes_external_plan(monkeypatch, tmp_path):
    """Bundle metadata must describe the actual external V3 routed layer plan."""
    conv = importlib.import_module("jang_tools.dsv4.convert_dsv4_jangtq")
    monkeypatch.setattr(conv, "FORMAT", "jangtq")
    monkeypatch.setattr(conv, "VARIANT", "V3")

    plan_path = tmp_path / "bit_plan_v3_late_layers.json"
    plan_path.write_text(json.dumps({
        "plan": {
            "model.layers.0.mlp.switch_mlp.SWITCH_MLP_LAYER": 4,
            "model.layers.1.mlp.switch_mlp.SWITCH_MLP_LAYER": 4,
            "model.layers.2.mlp.switch_mlp.SWITCH_MLP_LAYER": 4,
            "model.layers.37.mlp.switch_mlp.SWITCH_MLP_LAYER": 4,
            "model.layers.39.mlp.switch_mlp.SWITCH_MLP_LAYER": 4,
        },
        "config": {"routed_gs": 64, "default_gs": 32},
    }))
    monkeypatch.setenv("DSV4_V3_PLAN_PATH", str(plan_path))

    meta = conv.build_routed_expert_bit_plan(2)

    assert meta["plan_source"] == "DSV4_V3_PLAN_PATH"
    assert meta["plan_file"] == plan_path.name
    assert meta["routed_layer_bits"] == {"0": 4, "1": 4, "2": 4, "37": 4, "39": 4}
    assert len(meta["plan_sha256"]) == 64


def test_dsv4_converter_v3_metadata_allows_explicit_hash_layer_override(monkeypatch, tmp_path):
    """Explicit plans may spend the five-layer sub-80 budget away from hash layers."""
    conv = importlib.import_module("jang_tools.dsv4.convert_dsv4_jangtq")
    monkeypatch.setattr(conv, "FORMAT", "jangtq")
    monkeypatch.setattr(conv, "VARIANT", "V3")

    plan_path = tmp_path / "bit_plan_v3_worst5_sub80.json"
    plan_path.write_text(json.dumps({
        "plan": {
            "model.layers.0.mlp.switch_mlp.SWITCH_MLP_LAYER": 2,
            "model.layers.1.mlp.switch_mlp.SWITCH_MLP_LAYER": 2,
            "model.layers.2.mlp.switch_mlp.SWITCH_MLP_LAYER": 2,
            "model.layers.23.mlp.switch_mlp.SWITCH_MLP_LAYER": 4,
            "model.layers.25.mlp.switch_mlp.SWITCH_MLP_LAYER": 4,
            "model.layers.28.mlp.switch_mlp.SWITCH_MLP_LAYER": 4,
            "model.layers.34.mlp.switch_mlp.SWITCH_MLP_LAYER": 4,
            "model.layers.36.mlp.switch_mlp.SWITCH_MLP_LAYER": 4,
        },
        "config": {"routed_only": True},
    }))
    monkeypatch.setenv("DSV4_V3_PLAN_PATH", str(plan_path))

    meta = conv.build_routed_expert_bit_plan(2)

    assert meta["routed_layer_bits"] == {
        "0": 2,
        "1": 2,
        "2": 2,
        "23": 4,
        "25": 4,
        "28": 4,
        "34": 4,
        "36": 4,
    }
    assert "hash_layer_bits" not in meta
    assert meta["hash_layer_default_bits"] == 4
    assert meta["hash_layer_bits_source"] == "DSV4_V3_PLAN_PATH"


def test_dsv4_converter_help_renders_percent_literals():
    """argparse help strings must escape percent signs."""
    result = subprocess.run(
        [sys.executable, str(CONVERTER), "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "80% MMLU" in result.stdout


def test_dsv4_converter_default_variant_is_runtime_candidate_v3():
    """Bare JANGTQ2 conversion must not silently build the legacy MTP baseline."""
    src = CONVERTER.read_text()

    assert 'VARIANT = "V3"' in src
    assert 'ap.add_argument("--variant", default="V3"' in src
    assert "std=legacy baseline with MTP shipped" in src
    assert "production candidate needs DSV4_V3_PLAN_PATH" in src


def test_dsv4_docs_separate_std_baseline_from_runtime_candidate():
    """Public DSV4 docs must not present uniform std JANGTQ2 as the proven lane."""
    readme = (
        Path(__file__).resolve().parents[1]
        / "jang_tools"
        / "dsv4"
        / "README.md"
    ).read_text()
    examples = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "dsv4_flash"
        / "README.md"
    ).read_text()

    assert "--variant V3" in readme
    assert "DSV4_V3_PLAN_PATH" in readme
    assert "uniform `std` JANGTQ2" in readme
    assert "not a production-cleared DSV4 runtime candidate" in readme
    assert "DSV4_POOL_QUANT` | `0`" in examples


def test_dsv4_affine_converter_separates_routed_and_bookend_group_sizes():
    """Affine JANG DSV4 must use 128-group routed experts without touching bookends."""
    conv = importlib.import_module("jang_tools.dsv4.convert_dsv4_jang")

    assert conv.classify(
        "layers.2.ffn.experts.0.w1.weight",
        profile_bits=2,
        bookend_bits=8,
        routed_group_size=128,
        bookend_group_size=64,
    ) == (2, "affine", 128)
    assert conv.classify(
        "mtp.0.ffn.experts.255.w3.weight",
        profile_bits=2,
        bookend_bits=8,
        routed_group_size=128,
        bookend_group_size=64,
    ) == (2, "affine", 128)
    assert conv.classify(
        "layers.2.attn.wq_b.weight",
        profile_bits=2,
        bookend_bits=8,
        routed_group_size=128,
        bookend_group_size=64,
    ) == (8, "affine", 64)
    assert conv.classify(
        "layers.2.ffn.gate.weight",
        profile_bits=2,
        bookend_bits=8,
        routed_group_size=128,
        bookend_group_size=64,
    ) == (16, "passthrough", 0)


def test_dsv4_affine_converter_supports_selected_4bit_routed_layers():
    """Pure JANG selected-layer compromise keeps MTP/default routed at 2-bit."""
    conv = importlib.import_module("jang_tools.dsv4.convert_dsv4_jang")
    routed_layer_bits = {23: 4, 25: 4, 28: 4, 34: 4, 36: 4}

    for proj in ("w1", "w2", "w3"):
        assert conv.classify(
            f"layers.23.ffn.experts.7.{proj}.weight",
            profile_bits=2,
            bookend_bits=8,
            routed_group_size=128,
            bookend_group_size=64,
            routed_layer_bits=routed_layer_bits,
        ) == (4, "affine", 128)

    assert conv.classify(
        "layers.24.ffn.experts.7.w2.weight",
        profile_bits=2,
        bookend_bits=8,
        routed_group_size=128,
        bookend_group_size=64,
        routed_layer_bits=routed_layer_bits,
    ) == (2, "affine", 128)
    assert conv.classify(
        "mtp.0.ffn.experts.7.w2.weight",
        profile_bits=2,
        bookend_bits=8,
        routed_group_size=128,
        bookend_group_size=64,
        routed_layer_bits=routed_layer_bits,
    ) == (2, "affine", 128)


def test_dsv4_affine_converter_supports_jang_k_down_projection_bits():
    """Pure JANG_K can lift only DSV4 w2/down routed projections."""
    conv = importlib.import_module("jang_tools.dsv4.convert_dsv4_jang")
    routed_projection_bits = conv.parse_routed_projection_bits("down=4")

    assert routed_projection_bits == {"w2": 4}
    assert conv.classify(
        "layers.7.ffn.experts.3.w1.weight",
        profile_bits=2,
        bookend_bits=8,
        routed_group_size=128,
        bookend_group_size=64,
        routed_projection_bits=routed_projection_bits,
    ) == (2, "affine", 128)
    assert conv.classify(
        "layers.7.ffn.experts.3.w2.weight",
        profile_bits=2,
        bookend_bits=8,
        routed_group_size=128,
        bookend_group_size=64,
        routed_projection_bits=routed_projection_bits,
    ) == (4, "affine", 128)
    assert conv.classify(
        "layers.7.ffn.experts.3.w3.weight",
        profile_bits=2,
        bookend_bits=8,
        routed_group_size=128,
        bookend_group_size=64,
        routed_projection_bits=routed_projection_bits,
    ) == (2, "affine", 128)
    assert conv.classify(
        "mtp.0.ffn.experts.3.w2.weight",
        profile_bits=2,
        bookend_bits=8,
        routed_group_size=128,
        bookend_group_size=64,
        routed_projection_bits=routed_projection_bits,
    ) == (4, "affine", 128)
    assert conv.classify(
        "layers.23.ffn.experts.3.w1.weight",
        profile_bits=2,
        bookend_bits=8,
        routed_group_size=128,
        bookend_group_size=64,
        routed_layer_bits={23: 4},
        routed_projection_bits=routed_projection_bits,
    ) == (4, "affine", 128)


def test_dsv4_affine_converter_records_mtp_chat_cache_and_group_contracts():
    """The affine bundle must be self-describing for later MTP/cache activation."""
    src = AFFINE_CONVERTER.read_text()

    assert "routed_group_size" in src
    assert "bookend_group_size" in src
    assert "quant_overrides" in src
    assert "group_size=gsz" in src
    assert '"native_cache_schema": "deepseek_v4_v7"' in src
    assert '"generic_turboquant_kv": False' in src
    assert '"sliding_window": src_cfg.get("sliding_window")' in src
    assert '"compress_ratios": src_cfg.get("compress_ratios")' in src
    assert '"model_family": "deepseek_v4"' in src
    assert '"chat_template_source": "tokenizer_config"' in src
    assert "DSV4_CHAT_TEMPLATE_JINJA" in src
    assert '"runtime_self_spec_enabled": False' in src
    assert '"mtp_mode": "preserved_disabled"' in src
    assert '"rope_parameters"' in src
    assert "routed_layer_bits" in src
    assert "--routed-4bit-layers" in src
