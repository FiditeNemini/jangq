from __future__ import annotations

import json

import pytest

from jang_tools.jangtq_decode_profiler import (
    CoherencySummary,
    SwitchCallStats,
    classify_switchglu_call,
    loader_kind_for_model_info,
    summarize_coherency,
    write_profile_report,
)
from jang_tools.jangrt.inference_mode import ensure_inference_mode


class _FakeProj:
    def __init__(self, in_features=3072, out_features=1536, bits=2):
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits


class _FakeSwitch:
    def __init__(self, training=False):
        self.gate_proj = _FakeProj()
        self.up_proj = _FakeProj()
        self.down_proj = _FakeProj(bits=4)
        self.training = training


class _FakeArray:
    def __init__(self, shape, size=None, ndim=None):
        self.shape = shape
        self.ndim = len(shape) if ndim is None else ndim
        self.size = size

    def reshape(self, *shape):
        return _FakeArray(shape)

    def squeeze(self, _axis):
        return self


def test_classify_switchglu_call_marks_single_token_topk_as_fast_path():
    switch = _FakeSwitch()
    x = _FakeArray((1, 1, 3072))
    indices = _FakeArray((1, 8), size=8)

    result = classify_switchglu_call(switch, x, indices, tq_linear_type=_FakeProj)

    assert result.fast_path_eligible is True
    assert result.reason == "decode-fast-path"
    assert result.batch == 1
    assert result.top_k == 8
    assert result.down_bits == 4


def test_classify_switchglu_call_explains_slow_prefill_and_training_paths():
    prefill = classify_switchglu_call(
        _FakeSwitch(),
        _FakeArray((1, 32, 3072)),
        _FakeArray((32, 8), size=256),
        tq_linear_type=_FakeProj,
    )
    training = classify_switchglu_call(
        _FakeSwitch(training=True),
        _FakeArray((1, 1, 3072)),
        _FakeArray((1, 8), size=8),
        tq_linear_type=_FakeProj,
    )

    assert prefill.fast_path_eligible is False
    assert prefill.reason == "prefill-or-large-routing"
    assert training.fast_path_eligible is False
    assert training.reason == "training"


def test_switch_call_stats_tracks_fast_slow_and_shapes():
    stats = SwitchCallStats()

    stats.record(
        True,
        0.001,
        "decode-fast-path",
        batch=1,
        top_k=8,
        bits=2,
        down_bits=4,
    )
    stats.record(
        False,
        0.002,
        "prefill-or-large-routing",
        batch=32,
        top_k=8,
        bits=2,
        down_bits=4,
    )

    payload = stats.to_dict()
    assert payload["timing_mode"] == "python_call_no_mx_sync"
    assert "--sync-switchglu" in payload["timing_note"]
    assert payload["total_calls"] == 2
    assert payload["fast_calls"] == 1
    assert payload["slow_calls"] == 1
    assert payload["slow_reasons"] == {"prefill-or-large-routing": 1}
    assert payload["shape_counts"]["batch=1,k=8,bits=2,down_bits=4"] == 1


def test_summarize_coherency_flags_empty_reasoning_only_and_repetition():
    empty = summarize_coherency("")
    reasoning_only = summarize_coherency("<think>private chain")
    looping = summarize_coherency("alpha beta gamma " * 80)
    ok = summarize_coherency("The answer is 45.")

    assert empty.status == "empty"
    assert reasoning_only.status == "reasoning_only_or_unclosed"
    assert looping.status == "looping_risk"
    assert looping.repeat_4gram_ratio > 0.9
    assert ok.status == "ok"


def test_summarize_coherency_strips_closed_reasoning_before_visible_checks():
    raw = (
        "The model reasoned about the arithmetic first. 17 + 28 = 45.\n"
        "</think>\n\n"
        "17 plus 28 equals 45."
    )

    summary = summarize_coherency(raw)

    assert summary.status == "ok"
    assert summary.visible_text == "17 plus 28 equals 45."
    assert summary.visible_chars == len("17 plus 28 equals 45.")
    assert summary.full_chars == len(raw.strip())
    assert summary.reasoning_chars > summary.visible_chars


def test_write_profile_report_round_trips_json(tmp_path):
    report_path = tmp_path / "profile.json"
    coherency = summarize_coherency("The answer is 4.")
    stats = SwitchCallStats()
    stats.record(
        True,
        0.001,
        "decode-fast-path",
        batch=1,
        top_k=8,
        bits=2,
        down_bits=2,
    )

    write_profile_report(
        report_path,
        model_path="/models/example",
        prompt="2+2?",
        output_text="The answer is 4.",
        prompt_tokens=4,
        generated_tokens=5,
        total_seconds=0.25,
        decode_seconds=0.2,
        switch_stats=stats,
        coherency=coherency,
        step_records=[{"index": 1, "text": "The", "finish_reason": "stop"}],
        model_info={"model_type": "minimax_m2"},
    )

    data = json.loads(report_path.read_text())
    assert data["model"]["model_type"] == "minimax_m2"
    assert data["speed"]["decode_tokens_per_second"] == pytest.approx(25.0)
    assert data["switchglu"]["fast_calls"] == 1
    assert data["switchglu"]["timing_mode"] == "python_call_no_mx_sync"
    assert data["coherency"]["status"] == "ok"
    assert data["coherency"]["visible_text"] == "The answer is 4."
    assert data["acceptance"] == {
        "notes": [],
        "requires_review": False,
        "status": "pass",
    }
    assert data["finish_reason"] == "stop"
    assert data["output_text"] == "The answer is 4."


def test_write_profile_report_marks_length_runs_as_incomplete(tmp_path):
    report_path = tmp_path / "profile.json"

    write_profile_report(
        report_path,
        model_path="/models/example",
        prompt="2+2?",
        output_text="The answer",
        prompt_tokens=4,
        generated_tokens=2,
        total_seconds=0.25,
        decode_seconds=0.2,
        switch_stats=SwitchCallStats(),
        coherency=summarize_coherency("The answer"),
        step_records=[{"index": 2, "text": "answer", "finish_reason": "length"}],
        model_info={"model_type": "minimax_m2"},
    )

    data = json.loads(report_path.read_text())
    assert data["coherency"]["status"] == "ok"
    assert data["acceptance"]["status"] == "incomplete_length"
    assert data["acceptance"]["requires_review"] is True
    assert "max_tokens" in data["acceptance"]["notes"][0]


def test_loader_kind_routes_laguna_to_dedicated_runtime():
    assert loader_kind_for_model_info({"model_type": "laguna"}) == "laguna-runtime"
    assert loader_kind_for_model_info({"model_type": "minimax_m2"}) == "generic-jangtq"


class _FakeModel:
    def __init__(self, children=()):
        self.training = True
        self.children = list(children)
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        self.training = False
        for child in self.children:
            child.training = False
        return self

    def named_modules(self):
        yield "", self
        for idx, child in enumerate(self.children):
            yield f"child_{idx}", child


def test_ensure_inference_mode_calls_eval_and_reports_remaining_training_modules():
    model = _FakeModel(children=[_FakeModel()])

    report = ensure_inference_mode(model, label="unit")

    assert model.eval_called is True
    assert report["eval_called"] is True
    assert report["training_modules_remaining"] == 0
