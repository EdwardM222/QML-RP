import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


QUANTUM_STACK_META_QUBITS = 12

MODEL_FAMILIES = [
    "Quantum StackedECOC",
    "QuantumECOC",
    "StackedECOC",
    "CoherentECOC",
    "Random Forest",
    "VQC",
    "SVC",
]

ENSEMBLE_FAMILIES = {
    "QuantumECOC",
    "StackedECOC",
    "Quantum StackedECOC",
    "CoherentECOC",
}


def find_result_files(results_dir, file_prefix="ENSEMBLE"):
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


def split_model_name(model_name):
    for family in MODEL_FAMILIES:
        if model_name == family:
            return family, ""
        if model_name.startswith(f"{family} "):
            return family, model_name[len(family) + 1:]

    return model_name.split(" ", 1)[0], model_name


def normalise_feature_range(feature_range):
    if feature_range == [0, np.pi]:
        return "0_to_pi"
    if feature_range == [-np.pi, np.pi]:
        return "minus_pi_to_pi"
    if feature_range is None:
        return None
    return json.dumps(feature_range, separators=(",", ":"))


def parse_ansatz_name(name):
    if not isinstance(name, str):
        return {}

    match = re.match(
        r"^(?P<strategy>[bc])-(?P<encoding>[ap])-[^_]+_"
        r"f(?P<fpq>\d+)_r(?P<reuploads>\d+)_t(?P<layout>[0-9-]+)_"
        r"e[^_]+_(?P<pattern>[lp])-(?P<entangler>[^-]+)-",
        name,
    )

    if match is None:
        return {}

    strategy_map = {"b": "block", "c": "cyclic"}
    encoding_map = {"a": "angle", "p": "parallel_pairwise"}
    pattern_map = {"l": "linear", "p": "parallel"}
    layout = [int(value) for value in match.group("layout").split("-")]

    return {
        "feature_strategy": strategy_map.get(match.group("strategy"), match.group("strategy")),
        "encoding_style": encoding_map.get(match.group("encoding"), match.group("encoding")),
        "feats_per_qubit": int(match.group("fpq")),
        "reuploads": int(match.group("reuploads")),
        "trainable_layers": layout,
        "n_trainable_layers": sum(layout),
        "entangling_pattern": pattern_map.get(match.group("pattern"), match.group("pattern")),
        "entangler": match.group("entangler"),
    }


def parse_ensemble_suffix(suffix):
    meta_design = None
    meta_layout = None
    meta_entangler = None

    if "_meta-" in suffix:
        base_suffix, meta_suffix = suffix.split("_meta-", 1)
        meta_parts = meta_suffix.rsplit("_", 2)
        if len(meta_parts) == 3:
            meta_design, meta_layout, meta_entangler = meta_parts
    else:
        base_suffix = suffix

    parts = base_suffix.split("_")
    if len(parts) < 7:
        return {
            "base_config_id": base_suffix,
            "meta_design": meta_design,
            "meta_layout": meta_layout,
            "meta_entangler": meta_entangler,
        }

    return {
        "base_config_id": base_suffix,
        "n_learners": int(parts[0]),
        "layout_group": parts[1],
        "encoding_condition": parts[2],
        "fpq_condition": parts[3],
        "feature_density_condition": parts[4],
        "feature_strategy": parts[5],
        "entangler_condition": parts[6],
        "meta_design": meta_design,
        "meta_layout": meta_layout,
        "meta_entangler": meta_entangler,
    }


def parse_vqc_suffix(suffix):
    match = re.match(r"^(\[[^\]]+\])_(.+)$", suffix)
    if match is None:
        return {"base_config_id": suffix}

    layout_text = match.group(1)
    layout = [int(value) for value in re.findall(r"\d+", layout_text)]
    parts = match.group(2).rsplit("_", 4)

    if len(parts) != 5:
        return {
            "base_config_id": suffix,
            "trainable_layers": layout,
        }

    encoding, fpq, density, strategy, entangler = parts

    return {
        "base_config_id": suffix,
        "layout_group": "single-vqc",
        "encoding_condition": encoding,
        "fpq_condition": fpq,
        "feature_density_condition": density,
        "feature_strategy": strategy,
        "entangler_condition": entangler,
        "trainable_layers": layout,
        "n_trainable_layers": sum(layout),
    }


def extract_runs(
    results_dir,
    output_path=None,
    file_prefix="ENSEMBLE",
    test_mode=False,
    n_test_results=1000,
    progress_interval=10_000,
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
                report = record.get("metrics", {}).get("classification_report")

                if not isinstance(run, dict) or not isinstance(report, dict):
                    continue

                model_name = run.get("model_name", "")
                model_family, suffix = split_model_name(model_name)
                args = run.get("model_args", {})
                model = record.get("model") or {}
                model_config = model.get("config") or {}
                dataset_config = model.get("dataset") or {}
                structure = model.get("structure") or {}
                ansatz_structure = structure.get("ansatz") or {}
                training = model.get("training") or {}
                meta_learner = model.get("meta_learner") or {}
                stacking = model.get("stacking") or {}
                base_learners = model.get("base_learners") or {}

                parsed = {}
                if model_family in ENSEMBLE_FAMILIES:
                    parsed = parse_ensemble_suffix(suffix)
                elif model_family == "VQC":
                    parsed = parse_vqc_suffix(suffix)
                else:
                    parsed["base_config_id"] = model_family

                templates = args.get("templates") or model_config.get("templates") or []
                template_details = [parse_ansatz_name(template) for template in templates]
                template_details = [details for details in template_details if details]

                unique_templates = len(set(templates)) if templates else 0
                template_count = len(templates)

                def unique_values(key):
                    values = []
                    for details in template_details:
                        value = details.get(key)
                        if isinstance(value, list):
                            value = tuple(value)
                        if value is not None and value not in values:
                            values.append(value)
                    return values

                template_fpqs = unique_values("feats_per_qubit")
                template_reuploads = unique_values("reuploads")
                template_encodings = unique_values("encoding_style")
                template_layouts = unique_values("trainable_layers")
                template_patterns = unique_values("entangling_pattern")
                template_entanglers = unique_values("entangler")

                if model_family == "VQC":
                    ansatz = args.get("ansatz")
                    ansatz_details = parse_ansatz_name(ansatz)
                    parsed.update({
                        key: value
                        for key, value in ansatz_details.items()
                        if parsed.get(key) is None
                    })
                    compatibility_ansatz = ansatz
                elif model_family in ENSEMBLE_FAMILIES:
                    compatibility_ansatz = f"{model_family} | {parsed.get('layout_group', 'unknown')}"
                else:
                    compatibility_ansatz = model_family

                class_rows = [
                    metrics
                    for key, metrics in report.items()
                    if key not in {"accuracy", "macro avg", "weighted avg"}
                    and isinstance(metrics, dict)
                    and "recall" in metrics
                ]
                class_recalls = [metrics["recall"] for metrics in class_rows]

                ecoc = structure.get("ecoc") or []
                ecoc_rows = [tuple(row) for row in ecoc]
                ecoc_unique_rows = len(set(ecoc_rows)) if ecoc_rows else 0

                base_f1s = base_learners.get("macro-f1s") or []
                base_qubits_total = structure.get("total_base_qubits")
                base_params_total = structure.get("total_base_params")
                n_learners = model_config.get("n_learners", parsed.get("n_learners"))

                base_qubits_per_learner = None
                if base_qubits_total is not None and n_learners:
                    base_qubits_per_learner = base_qubits_total / n_learners

                meta_learner_type = meta_learner.get("type")
                meta_qubits = None
                meta_qubits_inferred = False

                if model_family == "Quantum StackedECOC" and meta_learner_type == "VQC":
                    meta_qubits = QUANTUM_STACK_META_QUBITS
                    meta_qubits_inferred = True

                if model_family == "VQC":
                    max_circuit_qubits = args.get("n_qubits", model_config.get("n_qubits"))
                elif model_family == "CoherentECOC":
                    max_circuit_qubits = base_qubits_total
                elif model_family in ENSEMBLE_FAMILIES:
                    widths = [value for value in [base_qubits_per_learner, meta_qubits] if value is not None]
                    max_circuit_qubits = max(widths) if widths else None
                else:
                    max_circuit_qubits = 0

                if model_family == "VQC":
                    n_qubits = args.get("n_qubits", model_config.get("n_qubits"))
                    n_params = structure.get("n_params")
                elif model_family in ENSEMBLE_FAMILIES:
                    n_qubits = base_qubits_total
                    n_params = base_params_total
                else:
                    n_qubits = 0
                    n_params = None

                if model_family == "VQC":
                    feature_range = normalise_feature_range(args.get("feature_range"))
                elif model_family in ENSEMBLE_FAMILIES:
                    feature_range = normalise_feature_range(model_config.get("scaler_range"))
                else:
                    feature_range = "classical"

                feature_density_condition = parsed.get("feature_density_condition")
                if feature_density_condition is None and args.get("feature_density") is not None:
                    feature_density_condition = str(args.get("feature_density"))

                fpq_condition = parsed.get("fpq_condition")
                if fpq_condition is None and parsed.get("feats_per_qubit") is not None:
                    fpq_condition = str(parsed.get("feats_per_qubit"))

                encoding_condition = parsed.get("encoding_condition") or parsed.get("encoding_style")
                entangler_condition = parsed.get("entangler_condition") or parsed.get("entangler")

                trainable_layers = parsed.get("trainable_layers")
                if trainable_layers is None and len(template_layouts) == 1:
                    trainable_layers = list(template_layouts[0])

                reuploads = parsed.get("reuploads")
                if reuploads is None and len(template_reuploads) == 1:
                    reuploads = template_reuploads[0]

                n_trainable_layers = parsed.get("n_trainable_layers")
                if n_trainable_layers is None and trainable_layers is not None:
                    n_trainable_layers = sum(trainable_layers)

                entangling_pattern = parsed.get("entangling_pattern")
                if entangling_pattern is None and len(template_patterns) == 1:
                    entangling_pattern = template_patterns[0]

                n_classes = dataset_config.get("n_classes", model_config.get("n_classes"))
                if n_classes is None:
                    n_classes = len(class_rows)

                n_features = dataset_config.get("n_features", model_config.get("n_features"))
                implementation_type = model.get("model_type") or model_family

                configuration_id = f"{model_family}|{suffix}" if suffix else model_family
                base_config_id = parsed.get("base_config_id") or suffix or model_family

                rows.append({
                    "search_repeat": search_repeat,
                    "run_id": run.get("id"),
                    "dataset": Path(run.get("dataset", "")).name,
                    "model_name": model_name,
                    "model_family": model_family,
                    "implementation_type": implementation_type,
                    "configuration_id": configuration_id,
                    "base_config_id": base_config_id,

                    "ansatz": compatibility_ansatz,
                    "layout_group": parsed.get("layout_group"),
                    "n_learners": n_learners,
                    "feature_density": str(feature_density_condition) if feature_density_condition is not None else None,
                    "feature_density_condition": str(feature_density_condition) if feature_density_condition is not None else None,
                    "feature_range": feature_range,
                    "measurement_mode": model_config.get("measurement_mode"),
                    "feats_per_qubit": str(fpq_condition) if fpq_condition is not None else None,
                    "fpq_condition": str(fpq_condition) if fpq_condition is not None else None,
                    "reuploads": reuploads,
                    "encoding_style": encoding_condition,
                    "encoding_condition": encoding_condition,
                    "feature_strategy": parsed.get("feature_strategy"),
                    "trainable_layers": (
                        json.dumps(trainable_layers, separators=(",", ":"))
                        if isinstance(trainable_layers, list)
                        else parsed.get("layout_group")
                    ),
                    "n_trainable_layers": n_trainable_layers,
                    "entangling_pattern": entangling_pattern,
                    "entangler": entangler_condition,
                    "entangler_condition": entangler_condition,

                    "template_count": template_count,
                    "unique_template_count": unique_templates,
                    "template_diversity": unique_templates / template_count if template_count else None,
                    "template_fpq_values": json.dumps(template_fpqs, separators=(",", ":")),
                    "template_reuploads": json.dumps(template_reuploads, separators=(",", ":")),
                    "template_encodings": json.dumps(template_encodings, separators=(",", ":")),
                    "template_layouts": json.dumps([list(value) for value in template_layouts], separators=(",", ":")),
                    "template_entanglers": json.dumps(template_entanglers, separators=(",", ":")),

                    "n_classes": n_classes,
                    "n_features": n_features,
                    "n_qubits": n_qubits,
                    "n_params": n_params,
                    "n_layers": ansatz_structure.get("n_layers"),
                    "trainable_params": ansatz_structure.get("trainable_params"),
                    "n_used_features": ansatz_structure.get("n_used_features"),
                    "feature_coverage": ansatz_structure.get("feature_coverage"),

                    "base_qubits_total": base_qubits_total,
                    "base_qubits_per_learner": base_qubits_per_learner,
                    "base_params_total": base_params_total,
                    "meta_learner_type": meta_learner_type,
                    "meta_feature_dim": stacking.get("meta_feature_dim"),
                    "meta_qubits": meta_qubits,
                    "meta_qubits_inferred": meta_qubits_inferred,
                    "max_circuit_qubits": max_circuit_qubits,
                    "meta_design": parsed.get("meta_design"),
                    "meta_layout": parsed.get("meta_layout"),
                    "meta_entangler": parsed.get("meta_entangler"),

                    "ecoc_unique_rows": ecoc_unique_rows if ecoc_rows else None,
                    "ecoc_duplicate_rows": len(ecoc_rows) - ecoc_unique_rows if ecoc_rows else None,
                    "ecoc_unique_fraction": ecoc_unique_rows / len(ecoc_rows) if ecoc_rows else None,

                    "base_f1_mean": np.mean(base_f1s) if base_f1s else None,
                    "base_f1_std": np.std(base_f1s) if base_f1s else None,
                    "base_f1_min": np.min(base_f1s) if base_f1s else None,
                    "base_f1_max": np.max(base_f1s) if base_f1s else None,

                    "accuracy": report.get("accuracy"),
                    "macro_precision": report.get("macro avg", {}).get("precision"),
                    "macro_recall": report.get("macro avg", {}).get("recall"),
                    "macro_f1": report.get("macro avg", {}).get("f1-score"),
                    "weighted_f1": report.get("weighted avg", {}).get("f1-score"),
                    "minimum_class_recall": min(class_recalls) if class_recalls else None,
                    "zero_recall_classes": sum(recall == 0 for recall in class_recalls),

                    "training_time": (1.0 if model_family in {"SVC", "Random Forest"} else training.get("training_time")),
                    "final_train_loss": training.get("final_train_loss"),
                    "best_train_loss": training.get("best_train_loss"),
                    "final_val_loss": training.get("final_val_loss"),
                    "best_val_loss": training.get("best_val_loss"),
                    "n_train_epochs": training.get("n_train_epochs"),
                    "base_final_train_loss_mean": training.get("mean_final_train_loss"),
                    "base_best_train_loss_mean": training.get("mean_best_train_loss"),
                    "base_final_val_loss_mean": training.get("mean_final_val_loss"),
                    "base_best_val_loss_mean": training.get("mean_best_val_loss"),
                })

                file_results += 1

                if line_number % progress_interval == 0:
                    print(f"{path.name}: {line_number:,} lines | {file_results:,} valid results")

                if test_mode and len(rows) >= n_test_results:
                    break

        print(f"Loaded {file_results:,} results from {path.name}")

        if test_mode and len(rows) >= n_test_results:
            break

    runs = pd.DataFrame(rows)

    float_columns = [
        "feature_coverage",
        "template_diversity",
        "base_qubits_per_learner",
        "ecoc_unique_fraction",
        "base_f1_mean",
        "base_f1_std",
        "base_f1_min",
        "base_f1_max",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "minimum_class_recall",
        "training_time",
        "final_train_loss",
        "best_train_loss",
        "final_val_loss",
        "best_val_loss",
        "base_final_train_loss_mean",
        "base_best_train_loss_mean",
        "base_final_val_loss_mean",
        "base_best_val_loss_mean",
    ]

    int_columns = [
        "search_repeat",
        "n_learners",
        "reuploads",
        "n_trainable_layers",
        "template_count",
        "unique_template_count",
        "n_classes",
        "n_features",
        "n_qubits",
        "n_params",
        "n_layers",
        "trainable_params",
        "n_used_features",
        "base_qubits_total",
        "base_params_total",
        "meta_feature_dim",
        "meta_qubits",
        "max_circuit_qubits",
        "ecoc_unique_rows",
        "ecoc_duplicate_rows",
        "zero_recall_classes",
        "n_train_epochs",
    ]

    category_columns = [
        "dataset",
        "model_family",
        "implementation_type",
        "configuration_id",
        "base_config_id",
        "ansatz",
        "layout_group",
        "feature_density",
        "feature_density_condition",
        "feature_range",
        "measurement_mode",
        "feats_per_qubit",
        "fpq_condition",
        "encoding_style",
        "encoding_condition",
        "feature_strategy",
        "trainable_layers",
        "entangling_pattern",
        "entangler",
        "entangler_condition",
        "meta_learner_type",
        "meta_design",
        "meta_layout",
        "meta_entangler",
    ]

    for column in float_columns:
        runs[column] = pd.to_numeric(runs[column], errors="coerce").astype("float32")

    for column in int_columns:
        runs[column] = pd.to_numeric(runs[column], errors="coerce").astype("Int32")

    for column in category_columns:
        runs[column] = runs[column].astype("category")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        runs.to_parquet(output_path, compression="zstd", index=False)
        print(f"\nSaved {len(runs):,} compact runs to {output_path}")

    return runs


def aggregate_results(runs, output_path=None):
    first_columns = [
        "dataset",
        "model_name",
        "model_family",
        "implementation_type",
        "configuration_id",
        "base_config_id",
        "ansatz",
        "layout_group",
        "n_learners",
        "feature_density",
        "feature_density_condition",
        "feature_range",
        "measurement_mode",
        "feats_per_qubit",
        "fpq_condition",
        "reuploads",
        "encoding_style",
        "encoding_condition",
        "feature_strategy",
        "trainable_layers",
        "n_trainable_layers",
        "entangling_pattern",
        "entangler",
        "entangler_condition",
        "template_count",
        "unique_template_count",
        "template_diversity",
        "template_fpq_values",
        "template_reuploads",
        "template_encodings",
        "template_layouts",
        "template_entanglers",
        "n_classes",
        "n_features",
        "n_qubits",
        "n_params",
        "n_layers",
        "trainable_params",
        "n_used_features",
        "feature_coverage",
        "base_qubits_total",
        "base_qubits_per_learner",
        "base_params_total",
        "meta_learner_type",
        "meta_feature_dim",
        "meta_qubits",
        "meta_qubits_inferred",
        "max_circuit_qubits",
        "meta_design",
        "meta_layout",
        "meta_entangler",
        "ecoc_unique_rows",
        "ecoc_duplicate_rows",
        "ecoc_unique_fraction",
    ]

    aggregations = {column: (column, "first") for column in first_columns}
    aggregations.update({
        "repetitions": ("macro_f1", "size"),
        "accuracy_mean": ("accuracy", "mean"),
        "accuracy_std": ("accuracy", "std"),
        "accuracy_min": ("accuracy", "min"),
        "accuracy_max": ("accuracy", "max"),
        "macro_precision_mean": ("macro_precision", "mean"),
        "macro_recall_mean": ("macro_recall", "mean"),
        "macro_f1_mean": ("macro_f1", "mean"),
        "macro_f1_std": ("macro_f1", "std"),
        "macro_f1_min": ("macro_f1", "min"),
        "macro_f1_max": ("macro_f1", "max"),
        "weighted_f1_mean": ("weighted_f1", "mean"),
        "weighted_f1_std": ("weighted_f1", "std"),
        "minimum_class_recall_mean": ("minimum_class_recall", "mean"),
        "minimum_class_recall_min": ("minimum_class_recall", "min"),
        "zero_recall_classes_mean": ("zero_recall_classes", "mean"),
        "zero_recall_classes_max": ("zero_recall_classes", "max"),
        "training_time_mean": ("training_time", "mean"),
        "training_time_std": ("training_time", "std"),
        "base_f1_mean": ("base_f1_mean", "mean"),
        "base_f1_std": ("base_f1_std", "mean"),
        "base_f1_min": ("base_f1_min", "mean"),
        "base_f1_max": ("base_f1_max", "mean"),
        "final_train_loss_mean": ("final_train_loss", "mean"),
        "best_train_loss_mean": ("best_train_loss", "mean"),
        "final_val_loss_mean": ("final_val_loss", "mean"),
        "best_val_loss_mean": ("best_val_loss", "mean"),
        "n_train_epochs_mean": ("n_train_epochs", "mean"),
        "base_final_train_loss_mean": ("base_final_train_loss_mean", "mean"),
        "base_best_train_loss_mean": ("base_best_train_loss_mean", "mean"),
        "base_final_val_loss_mean": ("base_final_val_loss_mean", "mean"),
        "base_best_val_loss_mean": ("base_best_val_loss_mean", "mean"),
    })

    results = runs.groupby("run_id", observed=True).agg(**aggregations).reset_index()

    results["global_rank"] = (
        results.groupby("dataset", observed=True)["macro_f1_mean"]
        .rank(method="min", ascending=False)
        .astype("Int32")
    )
    results["dataset_configurations"] = (
        results.groupby("dataset", observed=True)["run_id"]
        .transform("size")
        .astype("Int32")
    )
    results["global_rank_percentile"] = np.where(
        results["dataset_configurations"] <= 1,
        1.0,
        1 - (results["global_rank"] - 1) / (results["dataset_configurations"] - 1),
    ).astype("float32")

    results["family_rank"] = (
        results.groupby(["dataset", "model_family"], observed=True)["macro_f1_mean"]
        .rank(method="min", ascending=False)
        .astype("Int32")
    )
    results["family_configurations"] = (
        results.groupby(["dataset", "model_family"], observed=True)["run_id"]
        .transform("size")
        .astype("Int32")
    )
    results["family_rank_percentile"] = np.where(
        results["family_configurations"] <= 1,
        1.0,
        1 - (results["family_rank"] - 1) / (results["family_configurations"] - 1),
    ).astype("float32")

    # Keep the original dashboard column names as aliases for global ranking.
    results["rank"] = results["global_rank"]
    results["rank_percentile"] = results["global_rank_percentile"]

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_parquet(output_path, compression="zstd", index=False)
        print(f"Saved {len(results):,} aggregated results to {output_path}")

    return results


def rank_results(results, output_path=None):
    first_columns = [
        "model_name",
        "model_family",
        "implementation_type",
        "base_config_id",
        "ansatz",
        "layout_group",
        "n_learners",
        "feature_density",
        "feature_density_condition",
        "feature_range",
        "measurement_mode",
        "feats_per_qubit",
        "fpq_condition",
        "reuploads",
        "encoding_style",
        "encoding_condition",
        "feature_strategy",
        "trainable_layers",
        "n_trainable_layers",
        "entangling_pattern",
        "entangler",
        "entangler_condition",
        "template_count",
        "unique_template_count",
        "template_diversity",
        "n_qubits",
        "n_params",
        "n_layers",
        "trainable_params",
        "base_qubits_total",
        "base_qubits_per_learner",
        "base_params_total",
        "meta_learner_type",
        "meta_feature_dim",
        "meta_qubits",
        "max_circuit_qubits",
        "meta_design",
        "meta_layout",
        "meta_entangler",
    ]

    aggregations = {column: (column, "first") for column in first_columns}
    aggregations.update({
        "datasets": ("dataset", "nunique"),
        "average_rank": ("global_rank", "mean"),
        "median_rank": ("global_rank", "median"),
        "best_rank": ("global_rank", "min"),
        "worst_rank": ("global_rank", "max"),
        "average_percentile": ("global_rank_percentile", "mean"),
        "minimum_percentile": ("global_rank_percentile", "min"),
        "maximum_percentile": ("global_rank_percentile", "max"),
        "average_family_rank": ("family_rank", "mean"),
        "median_family_rank": ("family_rank", "median"),
        "average_family_percentile": ("family_rank_percentile", "mean"),
        "minimum_family_percentile": ("family_rank_percentile", "min"),
        "macro_f1_mean": ("macro_f1_mean", "mean"),
        "macro_f1_dataset_std": ("macro_f1_mean", "std"),
        "macro_f1_repeat_std": ("macro_f1_std", "mean"),
        "accuracy_mean": ("accuracy_mean", "mean"),
        "weighted_f1_mean": ("weighted_f1_mean", "mean"),
        "minimum_class_recall_mean": ("minimum_class_recall_mean", "mean"),
        "zero_recall_classes_mean": ("zero_recall_classes_mean", "mean"),
        "training_time_mean": ("training_time_mean", "mean"),
    })

    rankings = results.groupby("configuration_id", observed=True).agg(**aggregations).reset_index()

    total_datasets = results["dataset"].nunique()
    rankings["dataset_coverage"] = (rankings["datasets"] / total_datasets).astype("float32")
    rankings["complete_dataset_coverage"] = rankings["datasets"] == total_datasets

    rankings = rankings.sort_values(
        ["complete_dataset_coverage", "average_rank", "average_percentile", "macro_f1_repeat_std"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)

    rankings.insert(0, "overall_rank", np.arange(1, len(rankings) + 1, dtype=np.int32))
    rankings["family_overall_rank"] = (
        rankings.groupby("model_family", observed=True)["average_family_rank"]
        .rank(method="min", ascending=True)
        .astype("Int32")
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rankings.to_parquet(output_path, compression="zstd", index=False)
        print(f"Saved {len(rankings):,} overall rankings to {output_path}")

    return rankings


def aggregate_ansatze(results, overall_output_path=None, dataset_output_path=None):
    ansatz_columns = [
        "model_family",
        "ansatz",
        "feats_per_qubit",
        "reuploads",
        "encoding_style",
        "feature_strategy",
        "trainable_layers",
        "n_trainable_layers",
        "entangling_pattern",
        "entangler",
    ]

    ansatz_dataset_results = results.groupby(
        ["dataset"] + ansatz_columns,
        observed=True,
        dropna=False,
    ).agg(
        configurations=("run_id", "size"),
        qubit_counts=("n_qubits", "nunique"),
        feature_densities=("feature_density", "nunique"),
        feature_ranges=("feature_range", "nunique"),
        average_percentile=("global_rank_percentile", "mean"),
        median_percentile=("global_rank_percentile", "median"),
        percentile_std=("global_rank_percentile", "std"),
        minimum_percentile=("global_rank_percentile", "min"),
        maximum_percentile=("global_rank_percentile", "max"),
        average_family_percentile=("family_rank_percentile", "mean"),
        macro_f1_mean=("macro_f1_mean", "mean"),
        macro_f1_std=("macro_f1_mean", "std"),
        repeat_std_mean=("macro_f1_std", "mean"),
        accuracy_mean=("accuracy_mean", "mean"),
        weighted_f1_mean=("weighted_f1_mean", "mean"),
        training_time_mean=("training_time_mean", "mean"),
        n_qubits_mean=("n_qubits", "mean"),
        n_params_mean=("n_params", "mean"),
        n_params_min=("n_params", "min"),
        n_params_max=("n_params", "max"),
    ).reset_index()

    ansatz_dataset_results["ansatz_rank"] = (
        ansatz_dataset_results.groupby("dataset", observed=True)["average_percentile"]
        .rank(method="min", ascending=False)
        .astype("Int32")
    )
    ansatz_dataset_results["dataset_ansatze"] = (
        ansatz_dataset_results.groupby("dataset", observed=True)["ansatz"]
        .transform("size")
        .astype("Int32")
    )
    ansatz_dataset_results["ansatz_rank_percentile"] = np.where(
        ansatz_dataset_results["dataset_ansatze"] <= 1,
        1.0,
        1 - (
            (ansatz_dataset_results["ansatz_rank"] - 1)
            / (ansatz_dataset_results["dataset_ansatze"] - 1)
        ),
    ).astype("float32")

    ansatz_results = ansatz_dataset_results.groupby(
        ansatz_columns,
        observed=True,
        dropna=False,
    ).agg(
        datasets=("dataset", "nunique"),
        configurations=("configurations", "sum"),
        average_rank=("ansatz_rank", "mean"),
        median_rank=("ansatz_rank", "median"),
        best_rank=("ansatz_rank", "min"),
        worst_rank=("ansatz_rank", "max"),
        average_percentile=("ansatz_rank_percentile", "mean"),
        minimum_percentile=("ansatz_rank_percentile", "min"),
        maximum_percentile=("ansatz_rank_percentile", "max"),
        configuration_percentile_mean=("average_percentile", "mean"),
        configuration_percentile_std=("average_percentile", "std"),
        macro_f1_mean=("macro_f1_mean", "mean"),
        macro_f1_dataset_std=("macro_f1_mean", "std"),
        macro_f1_repeat_std=("repeat_std_mean", "mean"),
        accuracy_mean=("accuracy_mean", "mean"),
        weighted_f1_mean=("weighted_f1_mean", "mean"),
        training_time_mean=("training_time_mean", "mean"),
        n_qubits_mean=("n_qubits_mean", "mean"),
        n_params_mean=("n_params_mean", "mean"),
        n_params_min=("n_params_min", "min"),
        n_params_max=("n_params_max", "max"),
    ).reset_index()

    ansatz_results = ansatz_results.sort_values(
        ["average_rank", "average_percentile", "macro_f1_repeat_std"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    ansatz_results.insert(0, "overall_rank", np.arange(1, len(ansatz_results) + 1, dtype=np.int32))

    if dataset_output_path is not None:
        dataset_output_path = Path(dataset_output_path)
        dataset_output_path.parent.mkdir(parents=True, exist_ok=True)
        ansatz_dataset_results.to_parquet(dataset_output_path, compression="zstd", index=False)
        print(f"Saved {len(ansatz_dataset_results):,} dataset ansatz results to {dataset_output_path}")

    if overall_output_path is not None:
        overall_output_path = Path(overall_output_path)
        overall_output_path.parent.mkdir(parents=True, exist_ok=True)
        ansatz_results.to_parquet(overall_output_path, compression="zstd", index=False)
        print(f"Saved {len(ansatz_results):,} overall ansatz results to {overall_output_path}")

    return ansatz_results, ansatz_dataset_results


def build_paired_comparisons(results, output_path=None):
    families = ["QuantumECOC", "StackedECOC", "Quantum StackedECOC"]
    ensemble_results = results[results["model_family"].astype(str).isin(families)].copy()

    factor_columns = [
        "dataset",
        "base_config_id",
        "n_learners",
        "layout_group",
        "encoding_condition",
        "fpq_condition",
        "feature_density_condition",
        "feature_strategy",
        "entangler_condition",
        "base_qubits_total",
        "base_params_total",
    ]

    factors = ensemble_results.groupby(
        ["dataset", "base_config_id"], observed=True
    )[factor_columns[2:]].first().reset_index()

    metric_columns = [
        "macro_f1_mean",
        "accuracy_mean",
        "training_time_mean",
        "minimum_class_recall_mean",
        "zero_recall_classes_mean",
    ]

    paired = factors
    family_prefixes = {
        "QuantumECOC": "direct",
        "StackedECOC": "classical_stack",
        "Quantum StackedECOC": "quantum_stack",
    }

    for family, prefix in family_prefixes.items():
        family_results = ensemble_results[
            ensemble_results["model_family"].astype(str) == family
        ][["dataset", "base_config_id"] + metric_columns].copy()

        family_results = family_results.rename(columns={
            column: f"{prefix}_{column}" for column in metric_columns
        })

        paired = paired.merge(
            family_results,
            on=["dataset", "base_config_id"],
            how="outer",
        )

    paired["classical_stacking_gain_macro_f1"] = (
        paired["classical_stack_macro_f1_mean"] - paired["direct_macro_f1_mean"]
    )
    paired["quantum_stacking_gain_macro_f1"] = (
        paired["quantum_stack_macro_f1_mean"] - paired["classical_stack_macro_f1_mean"]
    )
    paired["quantum_vs_direct_gain_macro_f1"] = (
        paired["quantum_stack_macro_f1_mean"] - paired["direct_macro_f1_mean"]
    )

    paired["classical_stacking_gain_accuracy"] = (
        paired["classical_stack_accuracy_mean"] - paired["direct_accuracy_mean"]
    )
    paired["quantum_stacking_gain_accuracy"] = (
        paired["quantum_stack_accuracy_mean"] - paired["classical_stack_accuracy_mean"]
    )
    paired["quantum_vs_direct_gain_accuracy"] = (
        paired["quantum_stack_accuracy_mean"] - paired["direct_accuracy_mean"]
    )

    paired["classical_stack_time_multiplier"] = (
        paired["classical_stack_training_time_mean"] / paired["direct_training_time_mean"]
    )
    paired["quantum_stack_time_multiplier"] = (
        paired["quantum_stack_training_time_mean"] / paired["direct_training_time_mean"]
    )

    paired["paired_complete"] = paired[[
        "direct_macro_f1_mean",
        "classical_stack_macro_f1_mean",
        "quantum_stack_macro_f1_mean",
    ]].notna().all(axis=1)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        paired.to_parquet(output_path, compression="zstd", index=False)
        print(f"Saved {len(paired):,} paired comparisons to {output_path}")

    return paired
