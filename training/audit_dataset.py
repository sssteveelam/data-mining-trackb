"""Audit a labeled wafer-map pickle before attempting model export.

This command is intentionally model-free.  It is useful after a discovery
notebook has assigned a provisional label such as ``Horizontal_Stripes``:

    python -m training.audit_dataset \
      --dataset /kaggle/working/WM811K_train_v2.pkl \
      --output /kaggle/working/label_audit.json

The audit reports the actual class counts and whether the exporter can make a
stratified train/validation/test split.  It never fabricates metrics and it
does not promote a discovery candidate to a production class.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .export_model import (
    DEFAULT_MIN_SAMPLES_PER_CLASS,
    _unpack_nested_label,
)


def audit_dataset(
    dataset_path: Path,
    *,
    min_samples_per_class: int = DEFAULT_MIN_SAMPLES_PER_CLASS,
) -> dict[str, Any]:
    """Return a JSON-serializable label/data readiness report."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("pandas is required to audit a dataset") from exc

    dataframe = pd.read_pickle(dataset_path).copy()
    if "failureType" not in dataframe.columns:
        raise ValueError("dataset must contain a failureType column")
    dataframe["failureType"] = dataframe["failureType"].map(_unpack_nested_label)
    labeled = dataframe.dropna(subset=["failureType"]).copy()
    labeled["failureType"] = labeled["failureType"].replace({"none": "Normal"})
    counts = labeled["failureType"].value_counts().sort_index()

    insufficient = {
        str(label): int(count)
        for label, count in counts.items()
        if int(count) < min_samples_per_class
    }
    report: dict[str, Any] = {
        "dataset": str(dataset_path),
        "rows_total": int(len(dataframe)),
        "rows_labeled": int(len(labeled)),
        "map_column": (
            "waferMap"
            if "waferMap" in labeled.columns
            else "waferMap_resized"
            if "waferMap_resized" in labeled.columns
            else None
        ),
        "labels": [str(label) for label in counts.index.tolist()],
        "class_counts": {str(label): int(count) for label, count in counts.items()},
        "min_samples_per_class": int(min_samples_per_class),
        "insufficient_classes": insufficient,
        "ready_for_export": not insufficient
        and len(labeled) > 0
        and ("waferMap" in labeled.columns or "waferMap_resized" in labeled.columns),
        "notes": [],
    }

    if "Horizontal_Stripes" in counts.index:
        stripe_count = int(counts["Horizontal_Stripes"])
        if stripe_count < 30:
            report["notes"].append(
                "Horizontal_Stripes has fewer than 30 reviewed samples; "
                "export is technically possible only once the split minimum "
                "is met, but per-class metrics will be statistically fragile."
            )
        if stripe_count < min_samples_per_class:
            report["notes"].append(
                "Horizontal_Stripes is a discovery candidate, not yet "
                "trainable under the configured stratified split."
            )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=DEFAULT_MIN_SAMPLES_PER_CLASS,
    )
    args = parser.parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")
    report = audit_dataset(
        args.dataset,
        min_samples_per_class=args.min_samples_per_class,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["ready_for_export"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
