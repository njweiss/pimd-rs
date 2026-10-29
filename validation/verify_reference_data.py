"""Regenerate every committed reference data file and compare it with the committed JSON (Q5).

Run from the repository root (CHECK_LONG=1 scripts/check.sh does this):

    pixi run -e ref python validation/verify_reference_data.py

Each crates/*/tests/data/gen_*.py is run with `--out <tmpdir>`. Every JSON file it writes must exist
next to the generator and match it: same keys and list lengths, numbers within rtol 1e-9 / atol 1e-12,
other values exactly. The top-level "versions" block is reported, not compared.
"""

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

RTOL, ATOL = 1e-9, 1e-12


def compare(new, old, path, errors):
    if isinstance(old, dict) and isinstance(new, dict):
        if new.keys() != old.keys():
            errors.append(f"{path}: keys differ: {sorted(new.keys() ^ old.keys())}")
        for k in new.keys() & old.keys():
            compare(new[k], old[k], f"{path}.{k}", errors)
    elif isinstance(old, list) and isinstance(new, list):
        if len(new) != len(old):
            errors.append(f"{path}: length {len(new)} != committed {len(old)}")
        for i, (n, o) in enumerate(zip(new, old)):
            compare(n, o, f"{path}[{i}]", errors)
    elif isinstance(old, (int, float)) and isinstance(new, (int, float)) and not isinstance(old, bool):
        if not math.isclose(new, old, rel_tol=RTOL, abs_tol=ATOL):
            errors.append(f"{path}: {new!r} != committed {old!r}")
    elif new != old:
        errors.append(f"{path}: {new!r} != committed {old!r}")


def main():
    root = Path(__file__).resolve().parent.parent
    generators = sorted(root.glob("crates/*/tests/data/gen_*.py"))
    failed = False
    for gen in generators:
        rel = gen.relative_to(root)
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, str(gen), "--out", tmp], check=True, cwd=root)
            outputs = sorted(Path(tmp).glob("*.json"))
            if not outputs:
                print(f"FAIL {rel}: wrote no JSON")
                failed = True
            for out in outputs:
                committed = gen.parent / out.name
                if not committed.exists():
                    print(f"FAIL {rel}: {out.name} is not committed")
                    failed = True
                    continue
                new, old = json.loads(out.read_text()), json.loads(committed.read_text())
                if new.get("versions") != old.get("versions"):
                    print(f"note {out.name}: versions {new.get('versions')} (committed {old.get('versions')})")
                new.pop("versions", None)
                old.pop("versions", None)
                errors = []
                compare(new, old, out.name, errors)
                status = "FAIL" if errors else "ok  "
                print(f"{status} {rel.parent / out.name}")
                for e in errors:
                    print(f"     {e}")
                failed |= bool(errors)
    print(f"{len(generators)} generator(s) checked")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
