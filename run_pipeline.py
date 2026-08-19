"""Run the technological-momentum pipeline end to end.

Full run (downloads ~10 GB, embeds ~1M abstracts; hours):
    python run_pipeline.py

Smoke test on synthetic data (minutes, offline except the embedding model):
    python run_pipeline.py --sample

Individual stages:
    python run_pipeline.py --stages 4 5 8 [--sample]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent / "pipeline"

STAGES = {
    1: ("stage01_download.py", False),      # (script, supports --sample)
    2: ("stage02_link_firms.py", False),
    3: ("stage03_returns.py", False),
    4: ("stage04_embed.py", True),
    5: ("stage05_cluster.py", True),
    6: ("stage06_firm_vectors.py", True),
    7: ("stage07_techmom.py", True),
    8: ("stage08_tests.py", True),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true",
                    help="run on synthetic sample data (stages 4-8)")
    ap.add_argument("--stages", type=int, nargs="*",
                    help="subset of stages to run, e.g. --stages 4 5")
    args = ap.parse_args()

    if args.sample:
        stages = args.stages or [4, 5, 6, 7, 8]
        print(">> generating synthetic sample data")
        subprocess.run([sys.executable, str(PIPELINE / "make_sample_data.py")],
                       check=True)
    else:
        stages = args.stages or list(STAGES)

    for s in stages:
        script, samples = STAGES[s]
        cmd = [sys.executable, str(PIPELINE / script)]
        if args.sample and samples:
            cmd.append("--sample")
        print(f"\n{'=' * 60}\n>> stage {s}: {script}\n{'=' * 60}")
        t0 = time.time()
        subprocess.run(cmd, check=True)
        print(f">> stage {s} finished in {time.time() - t0:,.0f}s")


if __name__ == "__main__":
    main()
