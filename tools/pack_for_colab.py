"""Pack the training code into a zip you can upload straight to Colab.

`notebooks/train_all_colab.ipynb` normally clones this repo. When that is not
possible — the branch is not pushed, the repo is private, or GitHub is simply
unreachable from where you are — upload the zip this produces into Colab's
Files pane instead and the notebook picks it up automatically.

    python -m tools.pack_for_colab

Only `pipeline/` and `tools/` go in: ~150 KB of source. **No datasets, no
weights, no footage** — those are personal data under Egypt's PDPL 151/2020
(CLAUDE.md §7) and the notebook downloads the public datasets itself anyway.
The zip is gitignored: it is a build artefact, and a copy of the source
committed alongside the source goes stale the first time either changes.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "its_code_for_colab.zip"

# Everything the training and eval tools need to run, and nothing else.
INCLUDE_DIRS = ("pipeline", "tools")
INCLUDE_FILES = ("requirements.txt", "pipeline/bytetrack.yaml")


def pack(out: Path) -> int:
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for d in INCLUDE_DIRS:
            for p in sorted((ROOT / d).rglob("*.py")):
                if "__pycache__" in p.parts:
                    continue
                z.write(p, p.relative_to(ROOT).as_posix())
                n += 1
        for f in INCLUDE_FILES:
            p = ROOT / f
            if p.exists():
                z.write(p, p.relative_to(ROOT).as_posix())
                n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = Path(args.out)
    n = pack(out)
    print(f"{n} files -> {out}  ({out.stat().st_size / 1024:.0f} KB)")
    print("Upload it with Colab's Files pane (folder icon, left), then run the "
          "notebook's 'Get the code' cell.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
