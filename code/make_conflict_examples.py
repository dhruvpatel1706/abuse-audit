#!/usr/bin/env python3
"""Emit the paper's table of shared texts that carry conflicting labels.

Concrete examples: the texts that recur across corpora and
receive opposite binary labels, with a human reading of each. This script builds
that table from the frozen artifacts so no text, label, or Jaccard value is
typed into the manuscript by hand.

Scope: the strict cutoff (Jaccard >= 0.8), which is the "same text" regime. At
that cutoff the audit finds nine conflicting cross-dataset links. They collapse
to a much smaller number of DISTINCT texts, which is itself the point: the
strict-cutoff evidence is narrower than a raw link count suggests, and this
table is what makes that visible.

The `reading` column is authorial interpretation, not a measurement, so it is
kept here in version control and keyed to the normalized text. Everything else
(text, corpora, labels, Jaccard, link and item counts) is read from
data/results/.

Run:  python3 code/make_conflict_examples.py
Out:  paper/camera_ready/table_conflict_examples.tex
      data/results/conflict_examples.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "data", "results")
OUT_TEX = os.path.join(ROOT, "paper", "camera_ready", "table_conflict_examples.tex")

THR = 0.8

PRETTY = {
    "davidson2017": "Davidson",
    "dynahate2021": "DynaHate",
    "hatexplain": "HateXplain",
    "tweeteval_hate": "TweetEval",
    "stormfront": "Stormfront",
}

# Authorial reading, keyed by the normalized text of the node with the LOWER nid
# in each conflicting link. Reviewed by the author; not generated.
READINGS = {  # keys are canon() of the shortest variant

    "karma is a bitch":
        "Generic profanity with no target. Davidson's broad offensive class "
        "marks it positive; the hate-specific corpora do not.",
    "i fucking hate myself":
        "Self-directed, no target group. Not an offensive-versus-hate split at "
        "all: one corpus reads the hate verb as the signal.",
    "user you pussy":
        "Gendered insult aimed at an individual, not a protected group.",
    "u a stupid ho":
        "Gendered insult aimed at an individual. Same split as the line above.",
    "hell yea bitch dis go hard as hell flocka i will suck big dick team slut all day and week month year and century i love big naps":
        "In-group AAVE-inflected banter. Reclaimed usage that the two corpora "
        "score in opposite directions.",
    "she s a hoe she s ugly she s a bitch she s this she s that but you dated her please take a seat":
        "Quoted misogynistic speech that the post itself is criticising.",
}

MAXLEN = 78  # characters of displayed text before elision

# WOAH reporting policy (thewoah.org/policies.html): slurs and swear words are
# obfuscated with * for the middle letters, first and last letter kept, so the
# term stays discernible. Applied to every displayed text and to the JSON copy.
MASK_WORDS = {"bitch", "bitches", "fuck", "fucking", "fucked", "shit", "pussy", "cunt",
              "whore", "slut", "sluts", "dick", "cock", "ho", "hoe", "hoes", "asshole",
              "nigger", "nigga", "niggers", "niggas", "faggot", "fag", "fags", "kike",
              "retard", "retarded", "negro", "dyke", "tranny", "spic", "chink", "coon"}


def mask(s: str) -> str:
    def m(w):
        t = w.group(0)
        if t.lower() in MASK_WORDS and len(t) >= 2:
            return t[0] + "*" * (len(t) - 2) + t[-1] if len(t) > 2 else t[0] + "*"
        return t
    return re.sub(r"[A-Za-z]+", m, s)


def canon(t: str) -> str:
    """Collapse punctuation-only variants of the same text to one key.

    The corpora contain "karma is a bitch", "karma is a bitch." and
    "karma is a bitch..." as separate nodes. They are the same sentence and
    must share one reading and one table row.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", t.lower())).strip()


def tex_escape(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def display(s: str) -> str:
    """Escape first, then append the elision, so the LaTeX is not escaped too."""
    s = mask(s)
    if len(s) <= MAXLEN:
        return tex_escape(s)
    cut = s[:MAXLEN]
    if " " in cut:                     # elide on a word boundary
        cut = cut[:cut.rindex(" ")]
    return tex_escape(cut.rstrip()) + "\\,\\ldots"


def main() -> int:
    nodes = pd.read_csv(os.path.join(RES, "nodes.csv"))
    pairs = pd.read_csv(os.path.join(RES, "near_dup_pairs.csv"))
    pairs["cross_dataset"] = pairs["cross_dataset"].astype(str).str.lower().eq("true")

    norm = dict(zip(nodes["nid"], nodes["norm"]))
    dset = dict(zip(nodes["nid"], nodes["dataset"]))
    lab = dict(zip(nodes["nid"], nodes["label"]))

    links = pairs[(pairs["cross_dataset"])
                  & (pairs["label_conflict"].astype(str).str.lower().eq("true"))
                  & (pairs["jaccard"] >= THR)]
    if links.empty:
        print("no conflicting cross-dataset links at the strict cutoff", file=sys.stderr)
        return 2

    # group links by the text of their lower-nid endpoint
    groups = defaultdict(list)
    variants = defaultdict(set)
    for _, r in links.iterrows():
        a, b = int(r["nid_a"]), int(r["nid_b"])
        key = canon(min((norm[a], norm[b]), key=len))
        groups[key].append((a, b, float(r["jaccard"])))
        variants[key].update((norm[a], norm[b]))

    missing = [k for k in groups if k not in READINGS]
    if missing:
        print("FAIL: no authorial reading for these texts, refusing to emit a "
              "partial table:", file=sys.stderr)
        for k in missing:
            print("   " + repr(k), file=sys.stderr)
        return 3

    rows, payload = [], []
    for key in sorted(groups, key=lambda k: -max(j for _, _, j in groups[k])):
        ls = groups[key]
        # positive corpora and negative corpora across all links for this text
        pos, neg = set(), set()
        for a, b, _ in ls:
            for nid in (a, b):
                (pos if lab[nid] == 1 else neg).add(PRETTY[dset[nid]])
        shown = min(variants[key], key=len)
        rows.append((display(shown), ", ".join(sorted(pos)), ", ".join(sorted(neg)),
                     len(ls), max(j for _, _, j in ls), tex_escape(READINGS[key])))
        payload.append({"text": mask(shown), "canonical_key": mask(key), "variants": sorted(mask(v) for v in variants[key]), "positive_in": sorted(pos), "negative_in": sorted(neg),
                        "n_links": len(ls), "max_jaccard": max(j for _, _, j in ls),
                        "reading": READINGS[key]})

    n_links, n_texts = len(links), len(groups)
    body = "\n".join(
        f"{t} & {p} & {n} & {k} & {j:.2f} \\\\\n\\multicolumn{{5}}{{@{{}}p{{0.97\\textwidth}}@{{}}}}"
        f"{{\\small\\itshape {rd}}} \\\\[2pt]"
        for t, p, n, k, j, rd in rows)

    tex = f"""% GENERATED by code/make_conflict_examples.py -- do not edit by hand.
\\begin{{table*}}[t]
\\centering\\small
\\begin{{tabular}}{{@{{}}p{{0.40\\textwidth}}llcc@{{}}}}
\\toprule
Shared text (normalized) & Abuse in & Not abuse in & Links & $J$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{Every cross-dataset near-duplicate link at the strict cutoff
($J \\geq {THR}$) whose endpoints disagree on the binary label. The {n_links}
conflicting links reduce to {n_texts} distinct texts, and one text
(\\textit{{``karma is a b***h''}}) accounts for {max(len(v) for v in groups.values())} of
them. Italic lines are our reading of each case. Most are the broad
offensive-versus-hate split, but two are not: one is self-directed and one is
reclaimed in-group usage. Slurs and swear words are masked with asterisks
following the workshop's reporting policy. Generated by \\texttt{{code/make\\_conflict\\_examples.py}}.}}
\\label{{tab:conflictex}}
\\end{{table*}}
"""
    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, "w", encoding="utf-8") as fh:
        fh.write(tex)
    with open(os.path.join(RES, "conflict_examples.json"), "w", encoding="utf-8") as fh:
        json.dump({"threshold": THR, "n_links": n_links, "n_distinct_texts": n_texts,
                   "examples": payload}, fh, indent=2, sort_keys=True)
    print(f"{n_links} conflicting links -> {n_texts} distinct texts")
    for p in payload:
        print(f"  J={p['max_jaccard']:.3f} x{p['n_links']}  +{p['positive_in']} "
              f"-{p['negative_in']}  {p['text'][:60]!r}")
    print(f"wrote {OUT_TEX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
