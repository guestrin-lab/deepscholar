#!/usr/bin/env python3
"""Check that dataset/gt_nuggets_outputs is aligned with the dataset CSV.

NuggetCoverageEvaluator reads `<gt_dir>/<file_id>/res.json`, and file_id is the
row index of the dataset CSV (deepscholar_base/main.py writes one output folder
per row index). So folder i must hold the nuggets of CSV row i. This script
verifies, using only the standard library, that:

  * every CSV row i has `<gt_dir>/<i>/res.json`;
  * there are no other folders (out-of-range indices, nested sub-folders);
  * res.json["qid"] equals row i's arxiv_id;
  * the abstract embedded in res.json["query"] equals row i's abstract
    (whitespace-normalised).

Usage:
    python scripts/validate_gt_nuggets.py \
        [--dataset-path dataset/papers_with_related_works.csv] \
        [--gt-dir dataset/gt_nuggets_outputs]
"""
import argparse
import csv
import json
import os
import re
import sys

ABSTRACT_MARKER = "Here is the paper abstract:"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def validate(dataset_path: str, gt_dir: str) -> list[str]:
    """Return a list of human-readable problems (empty if aligned)."""
    csv.field_size_limit(sys.maxsize)
    with open(dataset_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    errors = []

    expected = {str(i) for i in range(len(rows))}
    for name in sorted(os.listdir(gt_dir)):
        path = os.path.join(gt_dir, name)
        if not os.path.isdir(path):
            continue
        if name not in expected:
            errors.append(f"{name}/: unexpected folder (dataset has {len(rows)} rows)")
            continue
        for sub in sorted(os.listdir(path)):
            if os.path.isdir(os.path.join(path, sub)):
                errors.append(f"{name}/{sub}/: unexpected nested folder (never read by the evaluator)")

    for i, row in enumerate(rows):
        res_path = os.path.join(gt_dir, str(i), "res.json")
        if not os.path.exists(res_path):
            errors.append(f"{i}/: missing res.json for row {i} ({row['arxiv_id']})")
            continue
        with open(res_path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("qid") != row["arxiv_id"]:
            errors.append(
                f"{i}/: qid {data.get('qid')} != row {i} arxiv_id {row['arxiv_id']}"
            )
        query = data.get("query", "")
        query_abstract = query.split(ABSTRACT_MARKER, 1)[-1]
        if _norm(query_abstract) != _norm(row["abstract"]):
            errors.append(f"{i}/: query abstract does not match row {i} abstract")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset-path", default="dataset/papers_with_related_works.csv")
    parser.add_argument("--gt-dir", default="dataset/gt_nuggets_outputs")
    args = parser.parse_args()
    errors = validate(args.dataset_path, args.gt_dir)
    for e in errors:
        print(e)
    print(f"{len(errors)} problem(s) found" if errors else "OK: every gt nugget folder matches its CSV row")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
