#!/usr/bin/env python3
"""
Scores a completed manual review against the pipeline's predictions to
estimate PPV and NPV of the templated/revealing classifier.

  PPV = P(human says "templated" | pipeline predicted "templated")
  NPV = P(human says "revealing" | pipeline predicted "revealing")
"""

import argparse
import csv
import json
from collections import Counter, defaultdict


VALID_LABELS = {"templated", "revealing"}


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blind", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    blind_rows = load_csv(args.blind)
    key_rows = load_csv(args.key)
    key_by_id = {r["item_id"]: r for r in key_rows}

    missing_key = []
    missing_label = []
    invalid_label = []
    merged = []

    for row in blind_rows:
        item_id = row["item_id"]
        key = key_by_id.get(item_id)
        if key is None:
            missing_key.append(item_id)
            continue
        human_label = (row.get("human_label") or "").strip().lower()
        if not human_label:
            missing_label.append(item_id)
            continue
        if human_label not in VALID_LABELS:
            invalid_label.append((item_id, row.get("human_label")))
            continue
        merged.append({**key, "human_label": human_label})

    if missing_key:
        print(f"[warn] {len(missing_key)} item_id(s) in blind file have no matching key row: {missing_key[:10]}"
              f"{' ...' if len(missing_key) > 10 else ''}")
    if missing_label:
        print(f"[warn] {len(missing_label)} row(s) have no human_label filled in yet and will be excluded: "
              f"{missing_label[:10]}{' ...' if len(missing_label) > 10 else ''}")
    if invalid_label:
        print(f"[warn] {len(invalid_label)} row(s) have an unrecognized human_label (must be exactly "
              f"'templated' or 'revealing') and will be excluded: {invalid_label[:10]}")

    n_scored = len(merged)
    if n_scored == 0:
        print("[error] no scoreable rows -- did you fill in human_label?")
        return

    tp = fp = tn = fn = 0
    confusion = Counter()
    section_errors = defaultdict(lambda: {"n": 0, "errors": 0})
    rule_errors = defaultdict(lambda: {"n": 0, "errors": 0})

    for row in merged:
        pred = row["predicted_label"]
        human = row["human_label"]
        confusion[(pred, human)] += 1

        if pred == "templated" and human == "templated":
            tp += 1
        elif pred == "templated" and human == "revealing":
            fp += 1
        elif pred == "revealing" and human == "revealing":
            tn += 1
        elif pred == "revealing" and human == "templated":
            fn += 1

        is_error = (pred != human)
        section_key = row.get("section") or "(no_header)"
        section_errors[section_key]["n"] += 1
        section_errors[section_key]["errors"] += int(is_error)
        rule_key = row.get("rule_matched") or "(no_rule / inherited)"
        rule_errors[rule_key]["n"] += 1
        rule_errors[rule_key]["errors"] += int(is_error)

    n_pred_templated = tp + fp
    n_pred_revealing = tn + fn
    ppv = tp / n_pred_templated if n_pred_templated else float("nan")
    npv = tn / n_pred_revealing if n_pred_revealing else float("nan")
    accuracy = (tp + tn) / n_scored

    print()
    print("=" * 72)
    print(f"[score] rows scored: {n_scored}  "
          f"(pipeline-predicted templated: {n_pred_templated}, revealing: {n_pred_revealing})")
    print()
    print("Confusion matrix (predicted x human):")
    print(f"  {'':<22}{'human=templated':>18}{'human=revealing':>18}")
    print(f"  {'predicted=templated':<22}{tp:>18}{fp:>18}")
    print(f"  {'predicted=revealing':<22}{fn:>18}{tn:>18}")
    print()
    print(f"  PPV (templated spans correctly categorized): {ppv*100:.1f}%  ({tp}/{n_pred_templated})")
    print(f"  NPV (revealing spans correctly categorized): {npv*100:.1f}%  ({tn}/{n_pred_revealing})")
    print(f"  Overall accuracy: {accuracy*100:.1f}%  ({tp+tn}/{n_scored})")
    print("=" * 72)

    print()
    print("=== Errors by section (for failure-mode discussion) ===")
    print(f"  {'section':<40}{'n':>6}{'errors':>8}{'error rate':>12}")
    for sec, d in sorted(section_errors.items(), key=lambda kv: -kv[1]["errors"]):
        rate = 100 * d["errors"] / d["n"] if d["n"] else 0.0
        print(f"  {sec[:40]:<40}{d['n']:>6}{d['errors']:>8}{rate:>11.1f}%")

    print()
    print("=== Errors by rule matched ===")
    print(f"  {'rule':<30}{'n':>6}{'errors':>8}{'error rate':>12}")
    for rule, d in sorted(rule_errors.items(), key=lambda kv: -kv[1]["errors"]):
        rate = 100 * d["errors"] / d["n"] if d["n"] else 0.0
        print(f"  {rule[:30]:<30}{d['n']:>6}{d['errors']:>8}{rate:>11.1f}%")

    if args.out_json:
        summary = {
            "n_scored": n_scored,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "ppv": ppv, "npv": npv, "accuracy": accuracy,
            "n_excluded_missing_label": len(missing_label),
            "n_excluded_invalid_label": len(invalid_label),
            "n_excluded_missing_key": len(missing_key),
            "section_errors": {k: v for k, v in section_errors.items()},
            "rule_errors": {k: v for k, v in rule_errors.items()},
        }
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[saved] summary written to {args.out_json}")


if __name__ == "__main__":
    main()
