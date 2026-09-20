"""Train the notebook CNN offline and export a deployable artifact bundle.

Example (Colab/Kaggle):

    python -m training.export_model \
      --dataset /kaggle/input/LSWMD.pkl \
      --output-dir artifacts \
      --model-version cnn-9class-v1

After a reviewed discovery pass, the input may also be a pickle containing
``failureType = "Horizontal_Stripes"`` rows (for example the notebook's
``df_train_version2``).  The exporter infers the number of classes from the
data, so a reviewed ten-class dataset can be exported with:

    python -m training.export_model \
      --dataset /kaggle/working/WM811K_train_v2.pkl \
      --output-dir artifacts-10class \
      --model-version cnn-10class-v1

The exporter deliberately refuses classes with fewer than thirty samples.  A
two-sample CURE candidate is a discovery result, not enough evidence for a
train/validation/test split or a defensible classification metric.

The API never invokes this module.  It only loads the exported files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

from .preprocess import TARGET_SIZE, normalize_wafer_map, validate_wafer_map

SEED = 42
DEFAULT_MODEL_VERSION = "cnn-9class-v1"
DEFAULT_MIN_SAMPLES_PER_CLASS = 30


def _set_seed(seed: int = SEED) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)


def _unpack_nested_label(value: Any) -> str | None:
    """Extract labels stored as ``[['Label']]`` or ``[]`` in WM-811K."""

    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _unpack_nested_label(value[0])
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def _load_dataset(path: Path, *, min_samples_per_class: int = DEFAULT_MIN_SAMPLES_PER_CLASS):
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("pandas is required to export a model") from exc

    dataframe = pd.read_pickle(path)
    dataframe = dataframe.copy()
    dataframe["failureType"] = dataframe["failureType"].map(_unpack_nested_label)
    labeled = dataframe.dropna(subset=["failureType"]).copy()
    labeled["failureType"] = labeled["failureType"].replace({"none": "Normal"})

    if "waferMap" not in labeled.columns and "waferMap_resized" not in labeled.columns:
        raise ValueError(
            "dataset must contain a waferMap or waferMap_resized column"
        )

    class_counts = labeled["failureType"].value_counts()
    too_small = class_counts[class_counts < min_samples_per_class]
    if not too_small.empty:
        details = ", ".join(
            f"{label}={int(count)}" for label, count in too_small.items()
        )
        raise ValueError(
            "every class needs at least "
            f"{min_samples_per_class} samples for the stratified "
            f"train/validation/test split; insufficient classes: {details}. "
            "A discovered label must be reviewed and expanded before it is "
            "used as a production classifier class."
        )

    normal = labeled[labeled["failureType"] == "Normal"]
    defects = labeled[labeled["failureType"] != "Normal"]
    if normal.empty or defects.empty:
        raise ValueError("dataset must contain Normal and defect samples")

    normal_count = min(len(normal), len(defects))
    normal_sampled = normal.sample(n=normal_count, random_state=SEED)
    subset = pd.concat([normal_sampled, defects], ignore_index=True)
    subset = subset.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return subset


def _build_model(num_classes: int):
    try:
        from tensorflow.keras import layers, models
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "TensorFlow is required for training. Run this command in Colab/Kaggle "
            "with tensorflow-cpu installed."
        ) from exc

    model = models.Sequential(
        [
            layers.Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 1)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_model(
    dataset_path: Path,
    output_dir: Path,
    model_version: str = DEFAULT_MODEL_VERSION,
    epochs: int = 10,
    batch_size: int = 128,
    min_samples_per_class: int = DEFAULT_MIN_SAMPLES_PER_CLASS,
) -> dict[str, Any]:
    """Train and export the CNN; return the generated manifest."""

    _set_seed()
    try:
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils.class_weight import compute_class_weight
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("scikit-learn is required for training") from exc

    if min_samples_per_class < 2:
        raise ValueError("min_samples_per_class must be at least 2")
    dataframe = _load_dataset(
        dataset_path,
        min_samples_per_class=min_samples_per_class,
    )
    labels = LabelEncoder()
    y = labels.fit_transform(dataframe["failureType"].to_numpy())
    map_column = "waferMap" if "waferMap" in dataframe.columns else "waferMap_resized"
    # Notebook discovery exports already-resized categorical maps.  The
    # shared preprocessing function is intentionally used for raw maps only;
    # resized maps still need the same /2 normalization and channel dimension.
    if map_column == "waferMap":
        x = np.stack(
            [normalize_wafer_map(value) for value in dataframe[map_column].to_numpy()]
        ).astype(np.float32)
    else:
        resized_maps: list[np.ndarray] = []
        for value in dataframe[map_column].to_numpy():
            validated = validate_wafer_map(value)
            if tuple(validated.shape) != TARGET_SIZE:
                raise ValueError(
                    "waferMap_resized rows must all have shape "
                    f"{TARGET_SIZE}; got {tuple(validated.shape)}"
                )
            resized_maps.append(validated.astype(np.float32) / 2.0)
        x = np.expand_dims(np.stack(resized_maps), axis=-1).astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.15,
        random_state=SEED,
        stratify=y,
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train,
        y_train,
        test_size=(0.15 / 0.85),
        random_state=SEED,
        stratify=y_train,
    )

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {int(class_id): float(weight) for class_id, weight in zip(classes, weights)}

    model = _build_model(len(labels.classes_))
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_valid, y_valid),
        class_weight=class_weight,
        verbose=1,
    )

    probabilities = model.predict(x_test, batch_size=batch_size, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    report = classification_report(
        y_test,
        predictions,
        target_names=labels.classes_.tolist(),
        output_dict=True,
        zero_division=0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.keras"
    labels_path = output_dir / "labels.json"
    preprocess_path = output_dir / "preprocess.json"
    metrics_path = output_dir / "metrics.json"
    manifest_path = output_dir / "manifest.json"

    model.save(model_path)
    labels_path.write_text(
        json.dumps({"labels": labels.classes_.tolist()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    preprocess_path.write_text(
        json.dumps(
            {
                "target_size": list(TARGET_SIZE),
                "allowed_values": [0, 1, 2],
                "normalization_divisor": 2.0,
                "resize_interpolation": "INTER_NEAREST",
                "review_threshold": 0.60,
                "max_input_side": 512,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    class_counts = {
        str(label): int(count)
        for label, count in dataframe["failureType"].value_counts().sort_index().items()
    }
    metrics = {
        "model_version": model_version,
        "label_count": int(len(labels.classes_)),
        "labels": labels.classes_.tolist(),
        "class_counts": class_counts,
        "dataset": {
            "source": str(dataset_path),
            "labeled_samples": int(len(dataframe)),
            "train_samples": int(len(x_train)),
            "validation_samples": int(len(x_valid)),
            "test_samples": int(len(x_test)),
        },
        "classification_report": _json_ready(report),
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": SEED,
            "history": _json_ready(history.history),
        },
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    files = [model_path, labels_path, preprocess_path, metrics_path]
    manifest = {
        "format": (
            "wafer-map-classifier-artifact-v2"
            if len(labels.classes_) > len(
                (
                    "Center",
                    "Donut",
                    "Edge-Loc",
                    "Edge-Ring",
                    "Loc",
                    "Near-full",
                    "Normal",
                    "Random",
                    "Scratch",
                )
            )
            else "wafer-map-classifier-artifact-v1"
        ),
        "model_version": model_version,
        "labels": labels.classes_.tolist(),
        "training_mode": (
            "new-label-retrained"
            if len(labels.classes_) > 9
            else "baseline"
        ),
        "files": {path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size} for path in files},
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=DEFAULT_MIN_SAMPLES_PER_CLASS,
        help=(
            "Minimum rows required for every label before a stratified "
            "train/validation/test split (default: 30)."
        ),
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")
    manifest = export_model(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        model_version=args.model_version,
        epochs=args.epochs,
        batch_size=args.batch_size,
        min_samples_per_class=args.min_samples_per_class,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
