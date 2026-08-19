"""Run the technological-momentum pipeline.

USAGE EXAMPLES
    python run_pipeline.py --sample          5-min smoke test on synthetic data
                                             (ALWAYS run this first on a new machine)
    python run_pipeline.py                   full run, every stage in order
    python run_pipeline.py --only tests      just the final tests (stage 8)
    python run_pipeline.py --only techmom tests      signal + tests only
    python run_pipeline.py --from cluster    cluster and everything after it
    python run_pipeline.py --skip download link      all except those two
    python run_pipeline.py --list            show stages and what they need

Stages can be named (download, link, returns, fundamentals, embed, corpus,
cluster, vectors, techmom, tests) or numbered (1, 2, 3, 3b, 4, 4b, 5, 6, 7, 8).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent / "pipeline"

# id, name, script, supports --sample, one-line description
STAGES = [
    ("1",  "download",     "stage01_download.py",       False, "raw data: patents (Zenodo) + SEC tickers"),
    ("2",  "link",         "stage02_link_firms.py",     False, "match patent owners to stock tickers"),
    ("3",  "returns",      "stage03_returns.py",        False, "monthly returns + price controls (Yahoo)"),
    ("3b", "fundamentals", "stage03b_fundamentals.py",  False, "point-in-time B/M, ROE, industry (EDGAR) + risk-free"),
    ("4",  "embed",        "stage04_embed.py",          True,  "abstracts -> vectors (GPU if available; resumable)"),
    ("4b", "corpus",       "stage04b_corpus_sample.py", False, "100k full-corpus sample for cluster fitting"),
    ("5",  "cluster",      "stage05_cluster.py",        True,  "K-means into 500 technology themes"),
    ("6",  "vectors",      "stage06_firm_vectors.py",   True,  "firm-month patent count vectors (12m rolling)"),
    ("7",  "techmom",      "stage07_techmom.py",        True,  "LINK similarity + TECHMOM signal"),
    ("8",  "tests",        "stage08_tests.py",          True,  "Fama-MacBeth + portfolio sorts -> results/"),
]
IDS = {s[0]: s for s in STAGES}
NAMES = {s[1]: s for s in STAGES}


def resolve(token: str):
    s = IDS.get(token) or NAMES.get(token.lower())
    if s is None:
        sys.exit(f"unknown stage '{token}' — use --list to see valid names")
    return s


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true",
                    help="synthetic smoke test (stages embed..tests on fake data)")
    ap.add_argument("--only", nargs="+", metavar="STAGE",
                    help="run ONLY these stages, e.g. --only techmom tests")
    ap.add_argument("--from", dest="from_stage", metavar="STAGE",
                    help="run this stage and everything after it")
    ap.add_argument("--skip", nargs="+", metavar="STAGE", default=[],
                    help="run everything except these stages")
    ap.add_argument("--list", action="store_true", help="show stages and exit")
    # legacy numeric interface, kept for compatibility
    ap.add_argument("--stages", nargs="*", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.list:
        print(f"{'id':>3}  {'name':<13} {'sample?':<8} description")
        for sid, name, _, samp, desc in STAGES:
            print(f"{sid:>3}  {name:<13} {'yes' if samp else 'no':<8} {desc}")
        return

    if args.only:
        selected = [resolve(t) for t in args.only]
    elif args.from_stage:
        start = STAGES.index(resolve(args.from_stage))
        selected = STAGES[start:]
    elif args.stages:
        selected = [resolve(t) for t in args.stages]
    elif args.sample:
        selected = [s for s in STAGES if s[3]]          # sample-capable only
    else:
        selected = list(STAGES)
    skip = {resolve(t)[0] for t in args.skip}
    selected = [s for s in selected if s[0] not in skip]

    if args.sample:
        not_ok = [s[1] for s in selected if not s[3]]
        if not_ok:
            sys.exit(f"--sample does not apply to: {', '.join(not_ok)} "
                     "(these stages only run on real data)")
        print(">> generating synthetic sample data")
        subprocess.run([sys.executable, str(PIPELINE / "make_sample_data.py")],
                       check=True)

    print(">> will run: " + " -> ".join(s[1] for s in selected))
    for sid, name, script, samp, _ in selected:
        cmd = [sys.executable, str(PIPELINE / script)]
        if args.sample and samp:
            cmd.append("--sample")
        print(f"\n{'=' * 60}\n>> stage {sid} ({name}): {script}\n{'=' * 60}")
        t0 = time.time()
        subprocess.run(cmd, check=True)
        print(f">> {name} finished in {time.time() - t0:,.0f}s")
    print("\n>> ALL REQUESTED STAGES FINISHED")


if __name__ == "__main__":
    main()
