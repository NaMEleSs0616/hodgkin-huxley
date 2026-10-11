"""Regenerate every figure and numeric result.

Runs each script in turn, mirrors its output to the terminal and saves it to
results/<script>.txt so every number quoted in the README can be traced to a log.

    python scripts/run_all.py            # everything (~15 min on a laptop)
    python scripts/run_all.py exp2 exp3  # only scripts whose name contains one of these
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = [
    "model_overview.py",
    "solver_validation.py",
    "exp1_spiking.py",
    "exp2_fi_curve.py",
    "exp3_refractory.py",
    "exp4_blockers.py",
    "exp5_phase_plane.py",
    "exp6_temperature.py",
]


def main(filters: list[str]) -> int:
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    chosen = [s for s in SCRIPTS if not filters or any(f in s for f in filters)]
    failed = []
    for name in chosen:
        print(f"\n=== {name} ===", flush=True)
        start = time.perf_counter()
        proc = subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / name)], cwd=ROOT,
                              capture_output=True, text=True)
        output = proc.stdout + proc.stderr
        print(output, end="")
        (results / name.replace(".py", ".txt")).write_text(output)
        print(f"--- {name}: {'ok' if proc.returncode == 0 else 'FAILED'} in "
              f"{time.perf_counter() - start:.0f} s", flush=True)
        if proc.returncode != 0:
            failed.append(name)
    if failed:
        print("\nFAILED:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
