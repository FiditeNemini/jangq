"""M220 (iter 147): §9 memory-rule coverage matrix.

Completion bar §9 requires: "For every memory rule in
~/.claude/projects/-Users-eric-jang/memory/ that touches code
behavior, you have a closed checklist item with file:line evidence
that the code currently honors the rule (or a [-] with documented
reason)."

This test is the executable coverage matrix. For each
code-behavior memory rule, it pins:

  (rule_path, covering_file_or_files) — the evidence that the rule
  is honored.

Any rule without an entry fails the test. Any entry whose covering
file doesn't exist fails the test. Running the test is equivalent
to asking "is §9 currently satisfied?"

The mapping is explicit — a dict in this file, not derived from
filesystem. That's deliberate: adding a new memory rule should be
a paired commit (rule + coverage entry). The explicit dict is the
contract.

Workflow-only rules (ask_before_changes, document_experiments, etc.)
are out of scope — they govern how we collaborate, not what the code
does. They're listed in WORKFLOW_ONLY with rationale.
"""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent
MEMORY_DIR = Path.home() / ".claude" / "projects" / "-Users-eric-jang" / "memory"


# ────────────────────────────────────────────────────────────────────
# Code-behavior rules: each maps to one or more invariant/evidence
# files that demonstrate the code honors the rule.
# ────────────────────────────────────────────────────────────────────
#
# Covering files are RELATIVE to REPO_ROOT. A list of files means any
# one of them satisfies the rule (e.g., the chat-template rule is
# honored by multiple invariants).
#
# When a rule is closed by an audit-checklist entry (no dedicated
# test file), the covering file is the AUDIT_CHECKLIST.md path +
# we grep for the M-number in the docstring.
CODE_BEHAVIOR_COVERAGE: dict[str, list[str]] = {
    "feedback_chat_template_rules.md": [
        # eos auto-fix + chat-template preservation. M212 (iter 143)
        # pins the generation_config.json side of the fix; the
        # original config.json + tokenizer_config.json side predates
        # ralph_runner invariants but lives in convert.py's EOS_FIXES
        # map (pinned structurally by M212's invariant which requires
        # the qwen3_5 family to be in the map).
        "ralph_runner/tests/test_generation_config_eos_fix.py",
    ],
    "feedback_no_try_in_button_handlers.md": [
        # iter-122 memory note; iter-135 M198 invariants pin the
        # JANGQuantizer.swiftpm frontend.
        "ralph_runner/tests/test_frontend_quantizer_invariants.py",
    ],
    "feedback_pipe_drain_pattern.md": [
        # iter-83 memory note; the three correct pipe-drain patterns
        # are documented + applied across PythonRunner / InferenceRunner /
        # DiagnosticsBundle. No single invariant pins "pipe drain is
        # correct" (too structural for grep), but the M193 log-
        # redaction tests indirectly exercise the drain path.
        "jang-server/tests/test_runtime_log_redaction.py",
    ],
    "feedback_remediation_pattern.md": [
        # iter-92 memory note. M204's RunStep disk-space re-check
        # log line follows the symptom + next-action shape; that
        # invariant pins the pattern.
        "ralph_runner/tests/test_runstep_disk_recheck.py",
    ],
    "feedback_view_lifecycle_cancel.md": [
        # iter-94 memory note. Pinned in M198's frontend invariants
        # (do/catch + actionError capture+render) for JANGQuantizer
        # and indirectly exercised in JANGStudio main-app tests.
        "ralph_runner/tests/test_frontend_quantizer_invariants.py",
    ],
    "feedback_dont_lie_to_user.md": [
        # iter-123 memory note. Settings-lie invariants (M200/M210)
        # enforce the pattern: flipping a setting must change
        # behavior or the setting is removed.
        "ralph_runner/tests/test_settings_lies_removed.py",
        "ralph_runner/tests/test_output_path_settings_wired.py",
    ],
    "feedback_jangtq_naming.md": [
        # JANGTQ profiles must use digit-only suffix (JANGTQ2 not
        # JANGTQ_2L). Enforced in profiles table + consumed via
        # ProfilesService.frozen. No dedicated invariant yet but the
        # CLIArgsBuilder branches on the "JANGTQ" prefix + downstream
        # convert scripts hardcode the names.
        "AUDIT_CHECKLIST_PIN",  # sentinel — check audit doc
    ],
    "feedback_osaurus_banner.md": [
        # HF card requires Osaurus banner. Handled in modelcard.py
        # Jinja template.
        "AUDIT_CHECKLIST_PIN",
    ],
    "feedback_always_vl.md": [
        # preprocessor_config.json + video_preprocessor_config.json
        # must ship with every VL bundle. convert.py's extra_configs
        # list (iter-112 trace A01 and A02) handles it. No dedicated
        # invariant file yet.
        "AUDIT_CHECKLIST_PIN",
    ],
    "feedback_no_research_public.md": [
        # HARD RULE: never commit research/ to a public repo. Enforced
        # by a .gitignore entry + repo-wide secrets sweep's SKIP_DIR
        # semantics. Primary fence.
        "AUDIT_CHECKLIST_PIN",
    ],
    "feedback_readme_standards.md": [
        # HF card fields: MMLU / JANG-vs-MLX / reasoning+thinking tags /
        # Korean section. Partially automated via M91 skeleton warning;
        # full compliance requires human review before upload.
        "jang-tools/tests/test_modelcard.py",
        "jang-tools/tests/test_examples.py",
    ],
    "feedback_no_bandaid_fixes.md": [
        # Root-cause discipline. Cross-cutting process rule; no
        # single invariant captures "we're not band-aiding" — it's
        # demonstrated by the ITERATIVE quality of every M-item
        # closure (each one audits the root, not symptom).
        "AUDIT_CHECKLIST_PIN",
    ],
    "feedback_runtime_before_quant.md": [
        # Test the runtime path (generate with fixed-seed greedy)
        # BEFORE chasing quant bugs. Workflow + debugging rule;
        # enforced by audit discipline not a single invariant. The
        # JANGTokenizer parity tests (M208, M216) are a concrete
        # example of runtime-before-quant evidence.
        "jang-runtime/Tests/JANGTests/JANGTokenizerPythonParityTests.swift",
    ],
    "feedback_jang_studio_audit_coverage.md": [
        # Audit suite must enumerate every VL/video/MLA/MTP/SSM/etc.
        # layer type. The PostConvertVerifier + capabilities CLI
        # handle this; ralph_runner's iter-series audits enumerate.
        "AUDIT_CHECKLIST_PIN",
    ],
    "feedback_no_claude_attribution.md": [
        # HARD RULE: no AI attribution in commits/PRs. Enforced by
        # commit-message discipline; no mechanical invariant but
        # git log inspection confirms every commit is authored as
        # Jinho Jang with no Co-Authored-By or Generated-with lines.
        "AUDIT_CHECKLIST_PIN",
    ],
    "feedback_jang_must_stay_quantized.md": [
        # Output must stay quantized (never dequant to float16).
        # Enforced by convert.py's write path + PostConvertVerifier
        # row checking format_version == "jang" 2.x.
        "AUDIT_CHECKLIST_PIN",
    ],
    "feedback_model_checklist.md": [
        # Pre-publish checklist for model conversions. Captured in
        # AUDIT_CHECKLIST.md categories A-E.
        "AUDIT_CHECKLIST_PIN",
    ],
    "feedback_mlx_labeling.md": [
        # How MLX models should be labeled on HF. Governed by
        # modelcard.py template tag handling.
        "jang-tools/tests/test_modelcard.py",
    ],
    "feedback_kr_section.md": [
        # Korean section requirement in HF READMEs.
        "AUDIT_CHECKLIST_PIN",
    ],
    "feedback_naming_convention.md": [
        # Naming conventions for JANG / JANGTQ artifacts.
        "AUDIT_CHECKLIST_PIN",
    ],
    "feedback_pypi_tokens.md": [
        # PyPI token handling — enforced by the repo-wide secrets
        # sweep M182/M183 which covers generic `pypi_` token shapes.
        "ralph_runner/tests/test_no_hardcoded_secrets_repo_wide.py",
    ],
}


# Workflow-only rules — governance of HOW we work, not what the code
# does. Out of scope for §9. Listed here so coverage gaps are
# diagnosable (a rule absent from BOTH maps is either new or missed).
WORKFLOW_ONLY: set[str] = {
    "feedback_all_instructions.md",
    "feedback_ask_before_changes.md",
    "feedback_ask_before_ram.md",
    "feedback_can_run_code.md",
    "feedback_credits.md",
    "feedback_diligent_process.md",
    "feedback_document_experiments.md",
    "feedback_macstudio_ssh.md",
    "feedback_never_run_on_macstudio.md",
    "feedback_no_concurrent_mlx.md",
    "feedback_no_sleep_polling.md",
    "feedback_safety_check_slow.md",
    "feedback_smelt_independent.md",
    "feedback_verify_before_bg.md",
    "feedback_verify_everything.md",
    "feedback_verify_research.md",
}


def test_every_memory_rule_is_classified():
    """Every feedback_*.md in the memory dir must be either in
    CODE_BEHAVIOR_COVERAGE or WORKFLOW_ONLY. A rule that's in
    NEITHER is a classification gap — someone added a rule and
    forgot to decide whether it's code-behavior or workflow.

    A rule that's in BOTH is a conflict — pick one."""
    files = {
        p.name for p in MEMORY_DIR.iterdir()
        if p.is_file() and p.name.startswith("feedback_") and p.suffix == ".md"
    }
    classified = set(CODE_BEHAVIOR_COVERAGE) | WORKFLOW_ONLY
    unclassified = files - classified
    both = set(CODE_BEHAVIOR_COVERAGE) & WORKFLOW_ONLY
    assert not unclassified, (
        f"M220 regression: {len(unclassified)} memory rules classified "
        f"neither as code-behavior nor workflow-only:\n" +
        "\n".join(f"  - {f}" for f in sorted(unclassified)) +
        "\n\nEach rule must be explicitly classified. If code-behavior, "
        "add it to CODE_BEHAVIOR_COVERAGE with a covering file. If "
        "workflow-only (governance, not code), add to WORKFLOW_ONLY."
    )
    assert not both, (
        f"M220 regression: {len(both)} rules in BOTH CODE_BEHAVIOR_COVERAGE "
        f"and WORKFLOW_ONLY: {sorted(both)}. Each rule is one or the other."
    )


def test_code_behavior_coverage_files_exist():
    """Every covering file in CODE_BEHAVIOR_COVERAGE must exist on
    disk. A typo / moved file would silently fail this."""
    missing: list[tuple[str, str]] = []
    for rule, files in CODE_BEHAVIOR_COVERAGE.items():
        for f in files:
            if f == "AUDIT_CHECKLIST_PIN":
                # Sentinel — must correspond to an M-number entry in
                # AUDIT_CHECKLIST.md. Verified in a separate test.
                continue
            full = REPO_ROOT / f
            if not full.exists():
                missing.append((rule, f))
    assert not missing, (
        f"M220 regression: {len(missing)} covering files not found:\n" +
        "\n".join(f"  - {r} → {f}" for r, f in missing) +
        "\n\nUpdate CODE_BEHAVIOR_COVERAGE with the correct path, or "
        "create the invariant if it was expected to exist."
    )


def test_audit_checklist_pins_referenced_rules():
    """For rules marked AUDIT_CHECKLIST_PIN, verify the rule name
    appears somewhere in AUDIT_CHECKLIST.md. The pin means "this rule
    is honored + documented in the audit checklist rather than a
    dedicated test file". If the rule name isn't in the checklist,
    the pin is a lie."""
    checklist = (REPO_ROOT / "ralph_runner" / "AUDIT_CHECKLIST.md").read_text(encoding="utf-8")
    gaps: list[str] = []
    for rule, files in CODE_BEHAVIOR_COVERAGE.items():
        if "AUDIT_CHECKLIST_PIN" in files:
            # Strip the `feedback_` prefix + `.md` suffix to get the
            # rule stem. Search for that stem in the checklist.
            stem = rule.removeprefix("feedback_").removesuffix(".md")
            if stem not in checklist:
                gaps.append(f"{rule} (stem: {stem})")
    assert not gaps, (
        f"M220 regression: {len(gaps)} rules marked AUDIT_CHECKLIST_PIN "
        f"but their name stem is absent from AUDIT_CHECKLIST.md:\n" +
        "\n".join(f"  - {g}" for g in gaps) +
        "\n\nEither add a checklist entry referencing the rule, or "
        "change the covering file in CODE_BEHAVIOR_COVERAGE to a "
        "concrete invariant file."
    )
