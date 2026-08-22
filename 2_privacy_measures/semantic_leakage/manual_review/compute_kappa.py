#!/usr/bin/env python3
"""
Compute Cohen's kappa and percent agreement between manual annotations
(annotation_sheet.csv) and the LLM judge (judge_labels_hidden.csv).

Reports overall agreement on `patient_has_diagnosis` (4-way: positive / negative /
ambiguous / not_mentioned), overall and broken out per diagnosis.
"""

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ANNOTATION_SHEET = "annotation_sheet.csv"
JUDGE_HIDDEN = "judge_labels_hidden.csv"


def normalize_label(mentioned, polarity) -> str:
    """Collapse (mentioned, polarity) into a single 4-way category label."""
    if mentioned in (False, "false", "False", "", None):
        return "not_mentioned"
    polarity = (polarity or "").strip().lower()
    if polarity in ("positive", "negative", "ambiguous"):
        return polarity
    return "not_mentioned"


def main():
    ann = pd.read_csv(ANNOTATION_SHEET)
    judge = pd.read_csv(JUDGE_HIDDEN)

    df = ann.merge(judge, on=["generation_id", "diagnosis"], how="inner")

    missing = df["manual_patient_has_diagnosis"].isna() & df["manual_diagnosis_mentioned"].isna()
    if missing.any():
        n_missing = missing.sum()
        print(f"[WARN] {n_missing} rows have no manual annotation yet — excluding from kappa calc.")
        df = df[~missing]

    df["manual_label"] = [
        normalize_label(m, p)
        for m, p in zip(df["manual_diagnosis_mentioned"], df["manual_patient_has_diagnosis"])
    ]
    df["judge_label"] = [
        normalize_label(m, p)
        for m, p in zip(df["judge_diagnosis_mentioned"], df["judge_patient_has_diagnosis"])
    ]

    def report(sub: pd.DataFrame, label: str):
        n = len(sub)
        if n == 0:
            print(f"  {label}: no rows")
            return
        pct_agree_4way = (sub["manual_label"] == sub["judge_label"]).mean()
        kappa_4way = cohen_kappa_score(sub["manual_label"], sub["judge_label"])
        print(f"  {label} (n={n}):")
        print(f"    4-way  — agreement: {pct_agree_4way*100:.1f}%   kappa: {kappa_4way:.3f}")

    print("=== Overall ===")
    report(df, "overall")

    print("\n=== By diagnosis ===")
    for dx, sub in df.groupby("diagnosis"):
        report(sub, dx)

    disagreements = df[df["manual_label"] != df["judge_label"]]
    out_path = "disagreements.csv"
    disagreements.to_csv(out_path, index=False)
    print(f"\n{len(disagreements)} disagreements written to {out_path} for error review.")


if __name__ == "__main__":
    main()
