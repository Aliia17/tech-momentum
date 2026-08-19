"""Diagnostic — which unmatched assignees hold the most patents?

Ranks assignee organizations that did NOT match an SEC registrant by their
patent count (2009+), to size the subsidiary/renaming problem and build a
manual alias table for the worst offenders.

Output: results/unmatched_top200.csv
Run:    python pipeline/diag_unmatched.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from stage02_link_firms import load_sec_map, normalize


def main() -> None:
    sec = load_sec_map()
    sec_names = set(sec["norm_name"])

    counts = {}
    display = {}
    for chunk in pd.read_csv(
        config.DATA_RAW / "g_assignee_disambiguated.tsv.zip",
        sep="\t", chunksize=2_000_000, dtype=str,
        usecols=lambda c: c in {"patent_id", "disambig_assignee_organization",
                                "assignee_sequence"},
        on_bad_lines="skip",
    ):
        chunk = chunk.dropna(subset=["disambig_assignee_organization"])
        if "assignee_sequence" in chunk.columns:
            chunk = chunk[chunk["assignee_sequence"].astype(float) == 0]
        chunk["norm"] = chunk["disambig_assignee_organization"].map(normalize)
        unmatched = chunk[~chunk["norm"].isin(sec_names)]
        for norm, n in unmatched["norm"].value_counts().items():
            counts[norm] = counts.get(norm, 0) + n
        for norm, name in zip(unmatched["norm"],
                              unmatched["disambig_assignee_organization"]):
            display.setdefault(norm, name)

    top = (pd.Series(counts).sort_values(ascending=False).head(200)
           .rename("patents").reset_index().rename(columns={"index": "norm_name"}))
    top["example_name"] = top["norm_name"].map(display)
    top.to_csv(config.RESULTS / "unmatched_top200.csv", index=False)
    print(top.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
