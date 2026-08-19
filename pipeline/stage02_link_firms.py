"""Stage 02 — link patents to listed US firms.

QuantData gave the paper a subsidiary-consolidated patent->parent mapping.
The free-US equivalent here is simpler (a known limitation to disclose):
disambiguated PatentsView assignee organization names are matched to SEC
registrant names by normalized exact match. This catches the parent entity
when the assignee IS the listed company ("Apple Inc." -> AAPL) but misses
patents held under differently-named subsidiaries.

Outputs
  interim/patents.parquet      patent_id, grant_date, abstract, ticker
  interim/patent_cpc.parquet   patent_id, cpc_subclass (primary, 4-char)
  interim/match_report.txt     match-rate diagnostics

Run:  python pipeline/stage02_link_firms.py
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

# Legal suffixes and noise words stripped before matching.
_SUFFIXES = (
    "incorporated|corporation|company|holdings|holding|group|international|"
    "technologies|technology|inc|corp|llc|ltd|lp|plc|co|sa|ag|nv|se|kk|gmbh"
)
_SUFFIX_RE = re.compile(rf"\b(?:{_SUFFIXES})\b", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^a-z0-9 ]")
_STATE_TAIL = re.compile(r"/[A-Za-z]{1,3}/?\s*$")   # e.g. "QUALCOMM INC/DE"


def normalize(name: str) -> str:
    s = _STATE_TAIL.sub(" ", str(name))
    s = _NON_ALNUM.sub(" ", s.lower().replace("&", " and "))
    s = _SUFFIX_RE.sub(" ", s)
    s = " ".join(s.split())
    if s.startswith("the "):
        s = s[4:]
    return s


# Manual assignee -> ticker aliases for major listed parents whose patents sit
# in differently-named subsidiaries (the QuantData ownership-consolidation
# problem; see results/unmatched_top200.csv for the ranking that built this).
# Keys are normalize()-outputs of assignee names; only unambiguous mappings.
ALIASES = {
    # renamed / IP-holding subsidiaries of US-listed parents
    "google": "GOOGL", "microsoft licensing": "MSFT", "amazon": "AMZN",
    "cisco": "CSCO", "ford global": "F", "gm global operations": "GM",
    "at and t intellectual property i l p": "T",
    "at and t intellectual property ii l p": "T",
    "3m innovative properties": "MMM",
    "minnesota mining and manufacturing": "MMM",
    "e i du pont de nemours and": "DD",
    "hewlett packard development l p": "HPQ", "hewlett packard l p": "HPQ",
    "hewlett packard enterprise development": "HPE",
    "applied materials": "AMAT", "halliburton energy services": "HAL",
    "dell products l p": "DELL", "emc ip": "DELL",
    "united": "RTX", "raytheon": "RTX", "hamilton sundstrand": "RTX",
    "schlumberger": "SLB", "dow chemical": "DOW", "dow global": "DOW",
    "facebook": "META", "corning": "GLW", "goodyear tire and rubber": "GT",
    "verizon patent and licensing": "VZ",
    "boston scientific scimed": "BSX", "cardiac pacemakers": "BSX",
    "whirlpool": "WHR", "uop": "HON", "honeywell limited": "HON",
    "bank of america": "BAC", "kimberly clark worldwide": "KMB",
    "capital one services": "COF", "juniper networks": "JNPR",
    "adobe systems": "ADBE", "harris": "LHX", "ethicon": "JNJ",
    "exxonmobil chemical patents": "XOM",
    "exxonmobil research and engineering": "XOM",
    "semiconductor components industries": "ON",
    "motorola": "MSI", "motorola solutions": "MSI",
    "black and decker": "SWK", "schering": "MRK",
    "monsanto": "MON", "vmware": "VMW", "xilinx": "XLNX",
    "carrier": "CARR", "igt": "IGT", "arm limited": "ARM",
    "qualcomm": "QCOM", "boeing": "BA", "procter and gamble": "PG",
    # US-listed ADR parents (NYSE/Nasdaq) of foreign assignees
    "toyota jidosha kabushiki kaisha": "TM",
    "toyota motor engineering and manufacturing north america": "TM",
    "telefonaktiebolaget lm ericsson publ": "ERIC",
    "u s philips": "PHG", "koninklijke philips n v": "PHG",
    "koninklijke philips electronics": "PHG",
    "philips north america": "PHG",
    "stmicroelectronics s r l": "STM",
    "nokia oy": "NOK", "nxp b v": "NXPI", "nxp usa": "NXPI",
    "asml netherlands b v": "ASML", "shell oil": "SHEL",
    "shell oil compny": "SHEL",  # (sic) typo appears in the raw data
    "research in motion limited": "BB", "blackberry limited": "BB",
}


def load_sec_map() -> pd.DataFrame:
    raw = json.loads((config.DATA_RAW / "company_tickers.json").read_text(encoding="utf-8"))
    df = pd.DataFrame(list(raw.values()))  # fields: cik_str, ticker, title
    df["norm_name"] = df["title"].map(normalize)
    # One normalized name can map to several share classes; keep the shortest
    # ticker (usually the primary listing, e.g. GOOGL/GOOG -> GOOG).
    df = df.sort_values("ticker", key=lambda s: s.str.len())
    return df.drop_duplicates("norm_name")[["norm_name", "ticker", "cik_str"]]


def main() -> None:
    sec = load_sec_map()
    print(f"SEC registrants: {len(sec):,}")

    # ---- assignees: keep organizations, normalize, match to SEC names
    print("Matching assignees (chunked scan of g_assignee_disambiguated) ...")
    matched_chunks = []
    seen_orgs, matched_orgs = set(), set()
    for chunk in pd.read_csv(
        config.DATA_RAW / "g_assignee_disambiguated.tsv.zip",
        sep="\t", chunksize=2_000_000, dtype=str,
        usecols=lambda c: c in {"patent_id", "disambig_assignee_organization",
                                "assignee_sequence"},
        on_bad_lines="skip",
    ):
        chunk = chunk.dropna(subset=["disambig_assignee_organization"])
        # primary assignee only, to give each patent one owner
        if "assignee_sequence" in chunk.columns:
            chunk = chunk[chunk["assignee_sequence"].astype(float) == 0]
        chunk["norm_name"] = chunk["disambig_assignee_organization"].map(normalize)
        seen_orgs.update(chunk["norm_name"].unique())
        # manual aliases take precedence over the SEC name match
        alias_hit = chunk["norm_name"].map(ALIASES)
        aliased = chunk[alias_hit.notna()].copy()
        aliased["ticker"] = alias_hit[alias_hit.notna()]
        m = chunk[alias_hit.isna()].merge(sec, on="norm_name", how="inner")
        matched_orgs.update(m["norm_name"].unique())
        matched_orgs.update(aliased["norm_name"].unique())
        matched_chunks.append(pd.concat([m[["patent_id", "ticker"]],
                                         aliased[["patent_id", "ticker"]]]))
    links = pd.concat(matched_chunks, ignore_index=True).drop_duplicates("patent_id")
    print(f"  assignee orgs seen {len(seen_orgs):,} | matched to a ticker "
          f"{len(matched_orgs):,} | patents linked {len(links):,}")

    # ---- patents: dates for linked patents in the sample window
    print("Scanning g_patent for linked patents ...")
    keep = set(links["patent_id"])
    pat_chunks = []
    for chunk in pd.read_csv(
        config.DATA_RAW / "g_patent.tsv.zip",
        sep="\t", chunksize=500_000, dtype=str,
        usecols=lambda c: c in {"patent_id", "patent_date", "patent_type"},
        on_bad_lines="skip",
    ):
        chunk = chunk[chunk["patent_id"].isin(keep)]
        if "patent_type" in chunk.columns:
            chunk = chunk[chunk["patent_type"] == "utility"]
        chunk = chunk.dropna(subset=["patent_date"])
        chunk = chunk[chunk["patent_date"] >= config.PATENT_START]
        pat_chunks.append(chunk[["patent_id", "patent_date"]])
    patents = pd.concat(pat_chunks, ignore_index=True)
    keep = set(patents["patent_id"])  # narrow further before the abstract scan

    # ---- abstracts live in a separate table in the final release
    print("Scanning g_patent_abstract for abstracts ...")
    abs_chunks = []
    for chunk in pd.read_csv(
        config.DATA_RAW / "g_patent_abstract.tsv.zip",
        sep="\t", chunksize=500_000, dtype=str, on_bad_lines="skip",
    ):
        abs_col = [c for c in chunk.columns if "abstract" in c.lower()][0]
        chunk = chunk[chunk["patent_id"].isin(keep)].dropna(subset=[abs_col])
        abs_chunks.append(chunk[["patent_id", abs_col]]
                          .rename(columns={abs_col: "abstract"}))
    abstracts = pd.concat(abs_chunks, ignore_index=True).drop_duplicates("patent_id")

    patents = (patents.merge(abstracts, on="patent_id")
               .merge(links, on="patent_id"))
    patents = patents.rename(columns={"patent_date": "grant_date"})
    patents["grant_date"] = pd.to_datetime(patents["grant_date"])
    patents.to_parquet(config.DATA_INTERIM / "patents.parquet", index=False)
    print(f"  linked utility patents since {config.PATENT_START}: {len(patents):,} "
          f"across {patents['ticker'].nunique():,} tickers")

    # ---- CPC subclass (classification analogue of the paper's IPC vectors)
    print("Scanning g_cpc_current for primary CPC subclass ...")
    cpc_chunks = []
    for chunk in pd.read_csv(
        config.DATA_RAW / "g_cpc_current.tsv.zip",
        sep="\t", chunksize=5_000_000, dtype=str,
        usecols=lambda c: c in {"patent_id", "cpc_subclass", "cpc_sequence"},
        on_bad_lines="skip",
    ):
        chunk = chunk[chunk["patent_id"].isin(keep)]
        if "cpc_sequence" in chunk.columns:
            chunk = chunk[chunk["cpc_sequence"].astype(float) == 0]
        cpc_chunks.append(chunk[["patent_id", "cpc_subclass"]])
    cpc = pd.concat(cpc_chunks, ignore_index=True).drop_duplicates("patent_id")
    cpc.to_parquet(config.DATA_INTERIM / "patent_cpc.parquet", index=False)
    print(f"  CPC rows: {len(cpc):,}, subclasses: {cpc['cpc_subclass'].nunique()}")

    report = (
        f"assignee orgs seen:    {len(seen_orgs):,}\n"
        f"orgs matched to SEC:   {len(matched_orgs):,}\n"
        f"patents linked:        {len(links):,}\n"
        f"utility patents kept:  {len(patents):,}\n"
        f"tickers covered:       {patents['ticker'].nunique():,}\n"
    )
    (config.DATA_INTERIM / "match_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
