import json
import subprocess
import sys
from pathlib import Path


def test_mtp_inspector_reports_qwen_visual_tensors(tmp_path):
    model_dir = tmp_path / "qwen36"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
        },
    }))
    (model_dir / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "model.visual.patch_embed.proj.weight": "model-00001-of-00001.safetensors",
            "model.visual.blocks.0.attn.qkv.weight": "model-00001-of-00001.safetensors",
            "mtp.fc.weight": "model-00001-of-00001.safetensors",
            "model.language_model.layers.0.mlp.up_proj.weight": "model-00001-of-00001.safetensors",
        }
    }))

    script = Path(__file__).parents[1] / "examples" / "mtp" / "inspect_mtp_bundle.py"
    proc = subprocess.run(
        [sys.executable, str(script), str(model_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["artifact_has_vision_weights"] is True
    assert data["visual_tensor_count"] == 2
    assert data["visual_tensor_samples"] == [
        "model.visual.blocks.0.attn.qkv.weight",
        "model.visual.patch_embed.proj.weight",
    ]


def test_qwen36_mtp_runtime_probe_strict_passes_with_mtp_and_visual(tmp_path):
    model_dir = tmp_path / "qwen36"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
        },
    }))
    (model_dir / "preprocessor_config.json").write_text("{}")
    (model_dir / "video_preprocessor_config.json").write_text("{}")
    (model_dir / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "model.visual.patch_embed.proj.weight": "model-00001-of-00001.safetensors",
            "mtp.fc.weight": "model-00001-of-00001.safetensors",
            "mtp.layers.0.self_attn.q_proj.weight": "model-00001-of-00001.safetensors",
            "model.language_model.layers.0.mlp.up_proj.weight": "model-00001-of-00001.safetensors",
        }
    }))

    script = Path(__file__).parents[1] / "examples" / "mtp" / "qwen36_mtp_runtime_probe.py"
    proc = subprocess.run(
        [sys.executable, str(script), str(model_dir), "--strict", "--no-headers"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["ok"] is True
    assert data["mtp_tensor_count"] == 2
    assert data["visual_tensor_count"] == 1
    assert data["has_preprocessor_config"] is True
    assert data["has_video_preprocessor_config"] is True


def test_qwen36_mtp_runtime_probe_strict_fails_without_visual(tmp_path):
    model_dir = tmp_path / "qwen36"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {"mtp_num_hidden_layers": 1},
    }))
    (model_dir / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "mtp.fc.weight": "model-00001-of-00001.safetensors",
        }
    }))

    script = Path(__file__).parents[1] / "examples" / "mtp" / "qwen36_mtp_runtime_probe.py"
    proc = subprocess.run(
        [sys.executable, str(script), str(model_dir), "--strict", "--no-headers"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 2
    data = json.loads(proc.stdout)
    assert data["ok"] is False
    assert "missing visual tensor weights" in data["errors"]
