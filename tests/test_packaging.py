"""The public repository holds the package only (RESEARCH.md standing rules 9 and 16).

Checks the directory itself, not `git ls-files`, so a gitignored private file sitting in the folder
still fails the test.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ["CLAUDE.md", "FINDINGS.md", "CORRECTIONS.md", "PLAN.md", "RESULTS.md", "MANUSCRIPT.md",
             "MANUSCRIPT.docx", "references.bib", "references.ris", "references-missing.ris",
             "zotero_keys.json", "data", "checkpoints", "src/make_docx.py", "src/make_refs.py",
             "tests/test_manuscript.py", "tests/test_headline.py"]
FORBIDDEN_SUFFIX = {".npz", ".pt", ".jpg", ".png.cache"}


def test_no_private_or_derived_data_files():
    present = [f for f in FORBIDDEN if (ROOT / f).exists()]
    assert not present, present
    bad = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.suffix in FORBIDDEN_SUFFIX and ".git" not in p.parts]
    assert not bad, bad
