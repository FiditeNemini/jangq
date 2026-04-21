"""M221 (iter 148): README-USER.md factual-claim invariants.

Completion bar §10 requires README-USER.md's spot-check to be backed
by command-output evidence, not just a visual read. Iter-148 ran the
spot-check and found 2 factual drifts vs. actual code:
  - README said "All 15 subcommands" → actual 16 (recommend was
    added without updating the README).
  - README said "12-row post-convert audit" → actual 14 rows in
    VerifyID (diskSizeSanity + tokenizerClassConcrete added post-
    initial-README).

M221 fixes the two counts AND pins each against the code so any
future code-side addition fires this test, reminding the author to
update the README alongside.

Other README-USER.md claims verified iter-148 with command-output
evidence (all currently accurate — no fix needed):
  - `pip install 'jang[mlx]'` — PyPI jang 2.3.2 + mlx extra exists.
  - `from jang_tools.loader import load_jang_model` — importable.
  - `from jang_tools.load_jangtq_vlm import load_jangtq_vlm_model` — importable.
  - Settings has 5 tabs (General, Advanced, Performance, Diagnostics, Updates).
  - Settings has 27 persisted preferences (post-M200 count).
  - 15 JANG profiles + JANGTQ2/3/4 listed match VALID_PROFILES + ProfilesService.frozen.
  - docs/adoption/README.md, PORTING.md, EXAMPLES/, FORMAT.md — all present.
  - "Pre-flight panel runs 10 checks" — PreflightRunner has 10 `out.append` calls.

These invariants verify ONLY the numeric claims that are susceptible
to code-side drift. Qualitative claims ("lets you chat with the
converted model", "generates snippets") aren't counted — they'd need
end-to-end stranger-walkthrough evidence (§8 work).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent
README_USER = REPO_ROOT / "README-USER.md"


def test_readme_subcommand_count_matches_cli():
    """Pin: README's "All N subcommands" claim must match the actual
    number of jang_tools subparsers. If a future PR adds a subcommand
    without updating the README, this test fires."""
    # Ask the CLI directly for its subcommands.
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "--help"],
        capture_output=True, text=True,
        cwd=str(REPO_ROOT / "jang-tools"),
        check=True,
    )
    # Parse the `{a,b,c,...}` list from the subcommand positional —
    # skip `{json,off}` and similar flag-choice lists. The subcommand
    # block follows `positional arguments:` in argparse default output.
    positional_idx = r.stdout.find("positional arguments:")
    assert positional_idx != -1, "argparse help output lacks `positional arguments:` — format changed"
    tail = r.stdout[positional_idx:]
    m = re.search(r"\{([a-z,\-]+)\}", tail)
    assert m is not None, (
        "Couldn't parse subcommand list from `python -m jang_tools --help`. "
        "Help output format changed?"
    )
    actual_count = len(m.group(1).split(","))
    # README's claim.
    readme = README_USER.read_text(encoding="utf-8")
    rm = re.search(r"All\s+(\d+)\s+subcommands are documented", readme)
    assert rm is not None, (
        "M221 regression: README-USER.md's 'All N subcommands are "
        "documented' phrase removed or rephrased. This invariant pins "
        "the specific claim shape; adjust the regex if rephrasing is "
        "deliberate."
    )
    claimed = int(rm.group(1))
    assert claimed == actual_count, (
        f"M221 regression: README claims 'All {claimed} subcommands' "
        f"but `python -m jang_tools --help` reports {actual_count}. "
        f"A subcommand was added (or removed) without updating the "
        f"README-USER.md line. Fix: update the README count to "
        f"{actual_count}, then re-run this test."
    )


def test_readme_verify_row_count_matches_swift_enum():
    """Pin: README's "N-row post-convert audit" must match the
    number of cases in VerifyID. If a future row is added to VerifyID
    without updating the README, this test fires."""
    verify_check_path = REPO_ROOT / "JANGStudio" / "JANGStudio" / "Verify" / "VerifyCheck.swift"
    content = verify_check_path.read_text(encoding="utf-8")
    # Extract the enum body and count comma-separated cases.
    m = re.search(
        r"enum\s+VerifyID\s*:[^{]*\{\s*case\s+([^}]+)\}",
        content, re.DOTALL
    )
    assert m is not None, (
        "M221 regression: VerifyID enum not found in VerifyCheck.swift — "
        "layout changed. Adjust the regex if the refactor was deliberate."
    )
    body = m.group(1)
    # Strip `//` comments inline and by line.
    body_clean = re.sub(r"//[^\n]*", "", body)
    # Each case is a comma-separated identifier. Count them.
    # Cases can span multiple lines + be grouped.
    identifiers = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", body_clean)
    actual_count = len(identifiers)
    # README's claim.
    readme = README_USER.read_text(encoding="utf-8")
    rm = re.search(r"(\d+)-row post-convert audit", readme)
    assert rm is not None, (
        "M221 regression: README-USER.md's 'N-row post-convert audit' "
        "phrase removed or rephrased. Adjust the regex if deliberate."
    )
    claimed = int(rm.group(1))
    assert claimed == actual_count, (
        f"M221 regression: README claims '{claimed}-row post-convert "
        f"audit' but VerifyID has {actual_count} cases. A verify row "
        f"was added (or removed) without updating README-USER.md. Fix: "
        f"update the README count to {actual_count}."
    )


def test_readme_profile_cheatsheet_covers_all_valid_profiles():
    """Pin: every profile in the server's VALID_PROFILES list must
    appear in the README's profile cheat sheet table. If a new
    profile (e.g. JANG_7K) is added without updating the README,
    this test fires."""
    server_py = REPO_ROOT / "jang-server" / "server.py"
    content = server_py.read_text(encoding="utf-8")
    # VALID_PROFILES = [ ... ] — find the list + extract JANG_* entries.
    m = re.search(r"VALID_PROFILES\s*=\s*\[([^\]]+)\]", content, re.DOTALL)
    assert m is not None, "VALID_PROFILES list not found in server.py"
    body = m.group(1)
    profiles = re.findall(r'"(JANG_[A-Z0-9]+)"', body)
    assert profiles, "VALID_PROFILES parsed but no JANG_* entries found"
    readme = README_USER.read_text(encoding="utf-8")
    missing = [p for p in profiles if p not in readme]
    assert not missing, (
        f"M221 regression: {len(missing)} profiles in VALID_PROFILES "
        f"are missing from README-USER.md's cheat sheet table: "
        f"{missing}. A stranger reading the README would not know "
        f"these profiles exist. Add them to the `Profile cheat sheet` "
        f"section."
    )


def test_readme_linked_docs_exist():
    """Pin: every relative-path doc link in README-USER.md resolves.
    A broken link shipped in a release is a stranger-friction bug."""
    readme = README_USER.read_text(encoding="utf-8")
    # Match [text](path) style links where path is a relative path
    # (doesn't start with http:// or https:// or #).
    for m in re.finditer(r"\[[^\]]+\]\(([^)#]+)\)", readme):
        link = m.group(1).strip()
        if link.startswith("http://") or link.startswith("https://"):
            continue
        # Strip trailing slash for directory links.
        rel = link.rstrip("/")
        # Strip backticks that sometimes wrap the link.
        rel = rel.strip("`")
        target = REPO_ROOT / rel
        assert target.exists(), (
            f"M221 regression: README-USER.md links to `{rel}` but "
            f"that path does not exist under {REPO_ROOT}. Either the "
            f"linked file was moved/renamed, or the link had a typo. "
            f"Strangers clicking this link would hit a 404."
        )
