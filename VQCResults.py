import json
from pathlib import Path

import numpy as np
import pandas as pd


def find_result_files(results_dir, file_prefix="VQCS"):
    result_files = []

    for path in Path(results_dir).glob(f"{file_prefix}_*.jsonl"):
        suffix = path.stem.rsplit("_", 1)[-1]

        if suffix.isdigit():
            result_files.append((int(suffix), path))

    result_files.sort()

    if not result_files:
        raise FileNotFoundError(
            f"No numbered {file_prefix}_n.jsonl files found in {results_dir}"
        )

    return result_files


def extract_runs(
    results_dir,
    output_path=None,
    file_prefix="VQCS",
    test_mode=False,
    n_test_results=1000,
    progress_interval=50_000,
):
    result_files = find_result_files(results_dir, file_prefix)

    print(
        f"Found {len(result_files)} result files: "
        + ", ".join(path.name for _, path in result_files)
    )

    rows = []

    for search_repeat, path in result_files:
        print(f"\nLoading {path}...")

        file_results = 0

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                run = record.get("run")
                report = (
                    record
                    .get("metrics", {})
                    .get("classification_report")
                )

                if not isinstance(run, dict) or not isinstance(report, dict):
                    continue

                args = run.get("model_args", {})
                model = record.get("model") or {}
                model_config = model.get("config") or {}
                structure = model.get("structure") or {}
                ansatz_structure = structure.get("ansatz") or {}
                training = model.get("training") or {}

                feature_range = args.get("feature_range", [None, None])
                trainable_layers = args.get("trainable_layers", [])

                if feature_range == [0, np.pi]:
                    feature_range = "0_to_pi"
                elif feature_range == [-np.pi, np.pi]:
                    feature_range = "minus_pi_to_pi"
                else:
                    feature_range = json.dumps(feature_range)

                rows.append({
                    "search_repeat": search_repeat,
                    "run_id": run.get("id"),
                    "dataset": Path(run.get("dataset", "")).name,

                    "ansatz": args.get("ansatz"),
                    "n_qubits": args.get("n_qubits"),
                    "feature_density": args.get("feature_density"),
                    "feature_range": feature_range,
                    "measurement_mode": args.get("measurement_mode"),

                    "feats_per_qubit": args.get("feats_per_qubit"),
                    "reuploads": args.get("reuploads"),
                    "encoding_style": args.get("encoding_style"),
                    "feature_strategy": args.get("feature_strategy"),
                    "trainable_layers": json.dumps(
                        trainable_layers,
                        separators=(",", ":"),
                    ),
                    "n_trainable_layers": sum(trainable_layers),
                    "entangling_pattern": args.get("entangling_pattern"),
                    "entangler": args.get("entangler"),

                    "n_classes": model_config.get("n_classes"),
                    "n_features": model_config.get("n_features"),

                    "n_params": structure.get("n_params"),
                    "n_layers": ansatz_structure.get("n_layers"),
                    "trainable_params": ansatz_structure.get(
                        "trainable_params"
                    ),
                    "n_used_features": ansatz_structure.get(
                        "n_used_features"
                    ),
                    "feature_coverage": ansatz_structure.get(
                        "feature_coverage"
                    ),

                    "accuracy": report.get("accuracy"),
                    "macro_precision": (
                        report
                        .get("macro avg", {})
                        .get("precision")
                    ),
                    "macro_recall": (
                        report
                        .get("macro avg", {})
                        .get("recall")
                    ),
                    "macro_f1": (
                        report
                        .get("macro avg", {})
                        .get("f1-score")
                    ),
                    "weighted_f1": (
                        report
                        .get("weighted avg", {})
                        .get("f1-score")
                    ),

                    "training_time": training.get("training_time"),
                    "final_train_loss": training.get("final_train_loss"),
                    "best_train_loss": training.get("best_train_loss"),
                    "final_val_loss": training.get("final_val_loss"),
                    "best_val_loss": training.get("best_val_loss"),
                    "n_train_epochs": training.get("n_train_epochs"),
                })

                file_results += 1

                if line_number % progress_interval == 0:
                    print(
                        f"{path.name}: "
                        f"{line_number:,} lines | "
                        f"{file_results:,} valid results"
                    )

                if test_mode and len(rows) >= n_test_results:
                    break

        print(f"Loaded {file_results:,} results from {path.name}")

        if test_mode and len(rows) >= n_test_results:
            break

    runs = pd.DataFrame(rows)

    float_columns = [
        "feature_density",
        "feature_coverage",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "training_time",
        "final_train_loss",
        "best_train_loss",
        "final_val_loss",
        "best_val_loss",
    ]

    for column in float_columns:
        runs[column] = pd.to_numeric(
            runs[column],
            errors="coerce",
        ).astype("float32")

    int_columns = [
        "search_repeat",
        "n_qubits",
        "feats_per_qubit",
        "reuploads",
        "n_trainable_layers",
        "n_classes",
        "n_features",
        "n_params",
        "n_layers",
        "trainable_params",
        "n_used_features",
        "n_train_epochs",
    ]

    for column in int_columns:
        runs[column] = pd.to_numeric(
            runs[column],
            errors="coerce",
        ).astype("Int32")

    category_columns = [
        "dataset",
        "ansatz",
        "feature_range",
        "measurement_mode",
        "encoding_style",
        "feature_strategy",
        "trainable_layers",
        "entangling_pattern",
        "entangler",
    ]

    for column in category_columns:
        runs[column] = runs[column].astype("category")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        runs.to_parquet(
            output_path,
            compression="zstd",
            index=False,
        )

        print(f"\nSaved {len(runs):,} compact runs to {output_path}")

    return runs


def aggregate_results(runs, output_path=None):
    results = runs.groupby(
        "run_id",
        observed=True,
    ).agg(
        dataset=("dataset", "first"),

        ansatz=("ansatz", "first"),
        n_qubits=("n_qubits", "first"),
        feature_density=("feature_density", "first"),
        feature_range=("feature_range", "first"),
        measurement_mode=("measurement_mode", "first"),

        feats_per_qubit=("feats_per_qubit", "first"),
        reuploads=("reuploads", "first"),
        encoding_style=("encoding_style", "first"),
        feature_strategy=("feature_strategy", "first"),
        trainable_layers=("trainable_layers", "first"),
        n_trainable_layers=("n_trainable_layers", "first"),
        entangling_pattern=("entangling_pattern", "first"),
        entangler=("entangler", "first"),

        n_classes=("n_classes", "first"),
        n_features=("n_features", "first"),
        n_params=("n_params", "first"),
        n_layers=("n_layers", "first"),
        trainable_params=("trainable_params", "first"),
        n_used_features=("n_used_features", "first"),
        feature_coverage=("feature_coverage", "first"),

        repetitions=("macro_f1", "size"),

        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        accuracy_min=("accuracy", "min"),
        accuracy_max=("accuracy", "max"),

        macro_precision_mean=("macro_precision", "mean"),
        macro_recall_mean=("macro_recall", "mean"),

        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        macro_f1_min=("macro_f1", "min"),
        macro_f1_max=("macro_f1", "max"),

        weighted_f1_mean=("weighted_f1", "mean"),
        weighted_f1_std=("weighted_f1", "std"),

        training_time_mean=("training_time", "mean"),
        training_time_std=("training_time", "std"),

        final_train_loss_mean=("final_train_loss", "mean"),
        best_train_loss_mean=("best_train_loss", "mean"),
        final_val_loss_mean=("final_val_loss", "mean"),
        best_val_loss_mean=("best_val_loss", "mean"),
        n_train_epochs_mean=("n_train_epochs", "mean"),
    ).reset_index()

    results["rank"] = (
        results
        .groupby("dataset", observed=True)["macro_f1_mean"]
        .rank(method="min", ascending=False)
        .astype("Int32")
    )

    results["dataset_configurations"] = (
        results
        .groupby("dataset", observed=True)["run_id"]
        .transform("size")
        .astype("Int32")
    )

    results["rank_percentile"] = (
        1
        - (
            (results["rank"] - 1)
            / (results["dataset_configurations"] - 1)
        )
    ).astype("float32")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results.to_parquet(
            output_path,
            compression="zstd",
            index=False,
        )

        print(
            f"Saved {len(results):,} aggregated results "
            f"to {output_path}"
        )

    return results


def rank_results(results, output_path=None):
    config_columns = [
        "ansatz",
        "n_qubits",
        "feature_density",
        "feature_range",
        "measurement_mode",
        "feats_per_qubit",
        "reuploads",
        "encoding_style",
        "feature_strategy",
        "trainable_layers",
        "n_trainable_layers",
        "entangling_pattern",
        "entangler",
    ]

    rankings = results.groupby(
        config_columns,
        observed=True,
        dropna=False,
    ).agg(
        datasets=("dataset", "nunique"),

        average_rank=("rank", "mean"),
        median_rank=("rank", "median"),
        best_rank=("rank", "min"),
        worst_rank=("rank", "max"),

        average_percentile=("rank_percentile", "mean"),
        minimum_percentile=("rank_percentile", "min"),
        maximum_percentile=("rank_percentile", "max"),

        macro_f1_mean=("macro_f1_mean", "mean"),
        macro_f1_dataset_std=("macro_f1_mean", "std"),
        macro_f1_repeat_std=("macro_f1_std", "mean"),

        accuracy_mean=("accuracy_mean", "mean"),
        weighted_f1_mean=("weighted_f1_mean", "mean"),

        n_params=("n_params", "mean"),
        n_layers=("n_layers", "mean"),
        trainable_params=("trainable_params", "mean"),
        training_time_mean=("training_time_mean", "mean"),
    ).reset_index()

    rankings = rankings.sort_values(
        [
            "average_rank",
            "average_percentile",
            "macro_f1_repeat_std",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).reset_index(drop=True)

    rankings.insert(
        0,
        "overall_rank",
        np.arange(
            1,
            len(rankings) + 1,
            dtype=np.int32,
        ),
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rankings.to_parquet(
            output_path,
            compression="zstd",
            index=False,
        )

        print(
            f"Saved {len(rankings):,} overall rankings "
            f"to {output_path}"
        )

    return rankings