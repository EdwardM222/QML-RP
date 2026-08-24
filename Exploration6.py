from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import time
import os
import pandas as pd
import numpy as np
from QuECOC import QuantumECOC, StackedECOC, CoherentECOC, VQC, TimeInt
from Ansatze import get_ansatze_configs, build_ansatz
import traceback
from sklearn.model_selection import ParameterGrid
import argparse
import json
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import hashlib
from itertools import product
import random

DEVICE = "cpu"

def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, pd.DataFrame):
        return value.reset_index().rename(columns={"index": "label"}).to_dict(orient="records")

    if isinstance(value, pd.Series):
        return value.to_dict()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, TimeInt):
        return float(value)

    return value

def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(record)) + "\n")

def replace_pending_job(path, job_id, record):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    replaced = False

    with (
        path.open("r", encoding="utf-8") as source,
        temporary_path.open("w", encoding="utf-8") as destination,
    ):
        for line in source:
            try:
                existing = json.loads(line)
            except json.JSONDecodeError:
                destination.write(line)
                continue

            run = existing.get("run", {})

            if (
                not replaced
                and run.get("id") == job_id
                and run.get("status") == "pending"
            ):
                destination.write(json.dumps(json_safe(record)) + "\n")
                replaced = True
            else:
                destination.write(line)

    temporary_path.replace(path)

    if not replaced:
        raise RuntimeError(
            f"Pending job '{job_id}' was not found in {path}."
        )

def remove_pending_job(path, job_id):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    removed = False

    with (
        path.open("r", encoding="utf-8") as source,
        temporary_path.open("w", encoding="utf-8") as destination,
    ):
        for line in source:
            try:
                existing = json.loads(line)
            except json.JSONDecodeError:
                destination.write(line)
                continue

            run = existing.get("run", {})

            if (
                not removed
                and run.get("id") == job_id
                and run.get("status") == "pending"
            ):
                removed = True
                continue

            destination.write(line)

    temporary_path.replace(path)


def train_model(name, dataset, model_args, fit_args=None, job_id=None):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
    sys.stderr.reconfigure(line_buffering=True, write_through=True)
    
    if fit_args is None:
        fit_args = {}

    print(f"Training {name}...")

    X_train = pd.read_csv(f"{dataset}/train.csv")
    y_train = X_train['target']
    X_train = X_train.drop('target', axis=1)

    X_test = pd.read_csv(f"{dataset}/test.csv")
    y_test = X_test['target']
    X_test = X_test.drop('target', axis=1)

    if name.startswith("SVC"):
        model = SVC(**model_args)
        model.fit(X_train, y_train, **fit_args)
    elif name.startswith("Random Forest"):
        model = RandomForestClassifier(**model_args)
        model.fit(X_train, y_train, **fit_args)
    elif name.startswith("VQC"):
        c = len(np.unique(y_train))
        feature_range = model_args.pop('feature_range')
        scaler = MinMaxScaler(feature_range=feature_range)

        model = VQC.from_template(
            template=model_args,
            n_classes=c,
            n_total_features=X_train.shape[1],
            cuda_device=DEVICE,
        )

        X_tr = scaler.fit_transform(X_train.iloc[:, model.feats])
        X_te = scaler.transform(X_test.iloc[:, model.feats])

        y_tr = y_train.map({label: idx for idx, label in enumerate(sorted(y_train.unique()))}).values
        y_te = y_test.map({label: idx for idx, label in enumerate(sorted(y_test.unique()))}).values

        model.fit(X_tr, y_tr, X_te, y_te, **fit_args)
    elif name.startswith("QuantumECOC"):
        model = QuantumECOC(**model_args).to(DEVICE)
        model.fit(X_train, y_train, X_test, y_test, verbosity=1, **fit_args)
    elif name.startswith("StackedECOC"):
        model = StackedECOC(**model_args).to(DEVICE)
        model.fit(X_train, y_train, X_test, y_test, k_folds=1, verbosity=1, **fit_args)
    elif name.startswith("Quantum StackedECOC"):
        c = len(np.unique(y_train))
        metaVQC = VQC.from_template(
            template={
                "n_qubits": 12,
                "feature_density": 1.0,
                "ansatz": build_ansatz(
                    feats_per_qubit=5,
                    reuploads=3,
                    encoding_style="angle",
                    feature_strategy="cyclic",
                    trainable_layers=[1, 2, 3],
                    entangling_uploads="all",
                    entangling_layers="all",
                    entangling_pattern="parallel",
                    entangler="cz",
                )["ansatz"]
            },
            n_classes=c,
            n_total_features=len(model_args["templates"])
        ).to("cpu")
        model = StackedECOC(meta_learner=metaVQC, **model_args).to(DEVICE)
        model.fit(X_train, y_train, X_test, y_test, verbosity=1, **fit_args)
    elif name.startswith("CoherentECOC"):
        model = CoherentECOC(**model_args).to(DEVICE)
        model.fit(X_train, y_train, X_test, y_test, k_folds=1, tune_size=0, verbosity=1, **fit_args)

    if name.startswith("VQC"):
        report = model.val_report
        model_args["ansatz"] = next(iter(model_args["ansatz"].keys()))
        # for key, val in model_args["config"].items():
        #     model_args[key] = val
        model_args["feature_range"] = feature_range
        # print(model_args) # New updaed fixed model args
    else:
        preds = model.predict(X_test)
        report = classification_report(
            y_test,
            preds,
            zero_division=0,
            output_dict=True,
        )

    if model_args.get("templates"):
        for i, template in enumerate(model_args["templates"]):
            if isinstance(template, dict):
                model_args["templates"][i] = next(iter(template["ansatz"].keys()))

    model_results = None
    if hasattr(model, "get_results"):
        model_results = model.get_results()

    result = {
        "run": {
            "id": job_id,
            "model_name": name,
            "dataset": dataset,
            "model_args": model_args,
            "fit_args": fit_args,
            "device": DEVICE,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        },
        "metrics": {
            "classification_report": report,
        },
        "model": model_results,
    }

    # print(f"Finished {name}.")

    return result

def search(model_name, dataset_path, param_grid, fit_args=None):
    results = []
    for params in ParameterGrid(param_grid):
        name = f"{model_name} - {params}"
        result = train_model(name, dataset_path, params, fit_args)
        results.append(result)

    return results

def make_job_id(model_name, dataset_path, model_args, fit_args=None):
    return hashlib.md5(
        json.dumps(
            json_safe({
                "model_name": model_name,
                "dataset": dataset_path,
                # "model_args": model_args,
                "fit_args": fit_args or {},
            }),
        sort_keys=True,
    ).encode()).hexdigest()

def create_search_jobs(model_name, dataset_path, param_grid, fit_args=None, existing_ids=None):
    jobs = []

    skipped = []
    for params in ParameterGrid(param_grid):
        job_id = make_job_id(model_name, dataset_path, params, fit_args)

        if job_id in existing_ids:
            skipped.append(job_id)
            continue

        jobs.append((f"{model_name}", dataset_path, params, fit_args, job_id))

    if len(skipped) > 0:
        print(f"Skipped {len(skipped)} jobs for {model_name} on {dataset_path} due to existing results.")
        # print(f"Skipped job IDs: {skipped}")

    return jobs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        default="default",
        help="Identifier used in the output filename",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--datasets",
        default="main",
        help="List of datasets to use.",
    )
    parser.add_argument(
        "--split",
        default="0",
        help="Job split to run.",
    )
    args = parser.parse_args()
    results_path = Path(f"results/{args.run_id}.jsonl")

    existing_ids = set()

    if results_path is not None and Path(results_path).exists():
        with Path(results_path).open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    job_id = record.get("run", {}).get("id")

                    if job_id is not None:
                        existing_ids.add(job_id)
                except json.JSONDecodeError:
                    pass
    
    main_datasets = [
        "iris",
        "balance-scale",
        "contraceptive-method-choice",
        "heart-disease",
        "obesity",
        "image-segmentation",
        "steel-plates",
        "yeast",
    ]
    stress_datasets = [
        "waveform",
        "ctg-10classes",
        "wine-quality",
        "nursery",
        "optical",
        "letter",
    ]

    train_layers = [
        [3, 3, 3],
        [2, 2, 2],
        [1, 2, 3],
        [3, 0, 3],
        [3, 2, 1],
        [3, 3],
        [2, 3],
        [3, 2],
        [1, 3],
    ]

    train_layer_groups = {
        "same-333": [[3, 3, 3]],
        # "same-33": [[3, 3]],
        # "same-303": [[3, 0, 3]],

        "uniform": [
            [3, 3, 3],
            [2, 2, 2],
            [3, 3],
            [2, 2]
        ],

        # "directional": [
        #     [1, 2, 3],
        #     [3, 2, 1],
        #     [1, 3],
        # ],

        # "dense": [
        #     [3, 3, 3],
        #     [2, 2, 2],
        #     [1, 2, 3],
        #     [3, 2, 1],
        #     [3, 3],
        # ],

        "representative": [
            [2, 2, 2],
            [1, 2, 3],
            [3, 0, 3],
            [1, 3],
            [2, 2]
        ],

        "two-layer": [
            [3, 3],
            [2, 3],
            [3, 2],
            [1, 3],
        ],

        "three-layer": [
            [3, 3, 3],
            [2, 2, 2],
            [1, 2, 3],
            [3, 0, 3],
            [3, 2, 1],
        ],

        "all-layouts": train_layers,
    }

    # ansatze = []
    # ids = []
    # for ansatz in get_ansatze_configs():
    #     if ansatz["id"] not in ids:
    #         ansatze.append(ansatz["ansatz"])
    #         ids.append(ansatz["id"])

    jobs = []

    feature_density_options = [0.75, "mixed"]
    encoding_options = ["angle"]
    n_learners_options = [6, 10]

    feat_strategy_options = ["cyclic"]
    entangler_options = ["cz"]
    fpq_options = ["mixed"]

    primary_configs = list(product(
        feature_density_options,
        encoding_options,
        n_learners_options,
    ))

    secondary_configs = list(product(
        feat_strategy_options,
        entangler_options,
        fpq_options,
    ))

    meta_configs = list(product(
        ["main"],
        # train_layers,
        [[1, 3]],
    ))

    vqc_primary_configs = list(product(
        feature_density_options,
        encoding_options,
    ))

    entangling_pattern = "linear"

    for dataset in sorted(os.listdir("datasets/")):
        path = os.path.join("datasets/", dataset)

        if args.datasets == "main" and dataset not in main_datasets:
            continue
        elif args.datasets == "stress" and dataset not in stress_datasets:
            continue
        elif dataset not in main_datasets and dataset not in stress_datasets:
            continue

        # jobs.extend(create_search_jobs("SVC", path, {
        #     "kernel": ["rbf"],
        #     "class_weight": ["balanced"],
        #     "random_state": [2],
        # }, existing_ids=existing_ids))

        # jobs.extend(create_search_jobs("Random Forest", path, {
        #     "n_estimators": [100],
        #     "class_weight": ["balanced"],
        #     "random_state": [2],
        # }, existing_ids=existing_ids))

        for group_index, (group, layouts) in enumerate(train_layer_groups.items()):
            secondary_order = secondary_configs.copy()

            random.Random(2 + group_index).shuffle(secondary_order)

            assigned_secondary = [
                secondary_order[config_index % len(secondary_order)]
                for config_index in range(len(primary_configs))
            ]

            assigned_meta = [
                meta_configs[(group_index * len(primary_configs) + config_index) % len(meta_configs)]
                for config_index in range(len(primary_configs))
            ]

            for config_index, (primary, secondary, meta) in enumerate(zip(
                primary_configs,
                assigned_secondary,
                assigned_meta,
            )):
                feature_density, encoding_style, n_learners = primary
                feat_strategy, entangler, fpq = secondary
                meta_design, meta_layout = meta

                suffix = (
                    f"{n_learners}_{group}_{encoding_style}_{fpq}_{feature_density}_{feat_strategy}_{entangler}"
                )

                templates = []

                for i in range(n_learners):
                    layout = layouts[(i + config_index) % len(layouts)]

                    if feature_density == "mixed":
                        learner_density = [0.25, 0.5, 0.75][(i + group_index + config_index) % 3]
                    else:
                        learner_density = feature_density

                    if encoding_style == "mixed":
                        encoding = ["angle", "angle", "parallel_pairwise"][(i + group_index + config_index + 1) % 3]
                    else:
                        encoding = "angle"

                    if entangler == "mixed":
                        entangler_gate = ["cz", "rzz"][(i + group_index + config_index)  % 2]
                    else:
                        entangler_gate = entangler

                    if fpq == "mixed":
                        fpq_value = [3, 4, 5][(i + group_index + config_index + 2) % 3]
                    else:
                        fpq_value = fpq

                    templates.append({
                        "n_qubits": 2,
                        "feature_density": learner_density,
                        "ansatz": build_ansatz(
                            feats_per_qubit=fpq_value,
                            reuploads=len(layout),
                            encoding_style=encoding,
                            feature_strategy=feat_strategy,
                            trainable_layers=layout,
                            entangling_uploads="all",
                            entangling_layers="all",
                            entangling_pattern=entangling_pattern,
                            entangler=entangler_gate,
                        )["ansatz"],
                    })

                for model_name in [
                    # "QuantumECOC",
                    "StackedECOC-1F",
                    # "Quantum StackedECOC",
                ]:
                    jobs.extend(create_search_jobs(
                        f"{model_name} {suffix}",
                        path,
                        {
                            "templates": [templates],
                        },
                        existing_ids=existing_ids,
                    ))

                if entangler == "mixed":
                    meta_entangler = ["cz", "rzz"][(group_index + config_index) % 2]
                else:
                    meta_entangler = entangler

                meta_template = build_ansatz(
                    feats_per_qubit=1,
                    reuploads=len(meta_layout),
                    encoding_style="none",
                    feature_strategy=feat_strategy,
                    trainable_layers=meta_layout,
                    entangling_uploads="all",
                    entangling_layers="all",
                    entangling_pattern=entangling_pattern,
                    entangler=meta_entangler,
                )["ansatz"]

                meta_layout_label = "-".join(
                    str(value)
                    for value in meta_layout
                )

                meta_suffix = (
                    f"{suffix}_meta-{meta_design}_"
                    f"{meta_layout_label}_{meta_entangler}"
                )

                jobs.extend(create_search_jobs(
                    f"CoherentECOC-1F {meta_suffix}",
                    path,
                    {
                        "templates": [templates],
                        "meta_design": [meta_design],
                        "meta_template": [meta_template],
                    },
                    existing_ids=existing_ids,
                ))

            # if group != "representative":
            #     continue

            # vqc_secondary_order = secondary_configs.copy()
            # random.Random(100 + group_index).shuffle(vqc_secondary_order)

            # assigned_vqc_secondary = [
            #     vqc_secondary_order[config_index % len(vqc_secondary_order)]
            #     for config_index in range(len(vqc_primary_configs))
            # ]

            # for config_index, (primary, secondary) in enumerate(zip(
            #     vqc_primary_configs,
            #     assigned_vqc_secondary,
            # )):
            #     feature_density, encoding_style = primary
            #     feat_strategy, entangler, fpq = secondary

            #     if feature_density == "mixed":
            #         vqc_densities = [0.25, 0.5, 0.75]
            #     else:
            #         vqc_densities = [feature_density]

            #     if encoding_style == "mixed":
            #         vqc_encoding = "parallel_pairwise"
            #     else:
            #         vqc_encoding = "angle"

            #     if entangler == "mixed":
            #         vqc_entangler = ["cz", "rzz"][config_index % 2]
            #     else:
            #         vqc_entangler = entangler

            #     if fpq == "mixed":
            #         vqc_fpq = [3, 4, 5][config_index % 3]
            #     else:
            #         vqc_fpq = fpq

            #     for layout in layouts:
            #         for vqc_density in vqc_densities:
            #             suffix = (
            #                 f"{layout}_{vqc_encoding}_{vqc_fpq}_{vqc_density}_{feat_strategy}_{vqc_entangler}"
            #             )

            #             jobs.extend(create_search_jobs(
            #                 f"VQC {suffix}",
            #                 path,
            #                 {
            #                     "n_qubits": [12],
            #                     "ansatz": [build_ansatz(
            #                         feats_per_qubit=vqc_fpq,
            #                         reuploads=len(layout),
            #                         encoding_style=vqc_encoding,
            #                         feature_strategy=feat_strategy,
            #                         trainable_layers=layout,
            #                         entangling_uploads="all",
            #                         entangling_layers="all",
            #                         entangling_pattern=entangling_pattern,
            #                         entangler=vqc_entangler,
            #                     )["ansatz"]],
            #                     "feature_density": [vqc_density],
            #                     "feature_range": [(0, np.pi)],
            #                 },
            #                 existing_ids=existing_ids,
            #             ))

    print(f"Total jobs: {len(jobs)}\n")
    if args.split != "0":
        split_size = len(jobs) // int(args.split)
        if len(jobs) > split_size * 1.5:
            jobs = jobs[:split_size]
        
    print(f"Jobs running in this split ({args.split}): {len(jobs)}\n")

    for job in jobs:
        append_jsonl(results_path, {
            "run": {
                "id": job[-1],
                "status": "pending",
            }
        })

    # jobs = jobs[:10]
    # exit()

    start_time = time.time()
    results = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(train_model, *job): job
            for job in jobs
        }

        for future in as_completed(futures):
            job = futures[future]
            job_name, dataset_path, model_args, fit_args, job_id = job

            try:
                result = future.result()
                replace_pending_job(results_path, job_id, result)
                results += 1

                # metrics = result.get("metrics", {})
                # training_time = result.get("timing", {}).get("training_time")

                # print(
                #     f"Saved result: {job_name} | {dataset_path} | "
                #     f"macro_f1={metrics.get('macro_f1')} | "
                #     f"time={training_time}"
                # )
            except Exception:
                error_record = {
                    "status": "error",
                    "run": {
                        "model_name": job_name,
                        "dataset": dataset_path,
                        "model_args": model_args,
                        "fit_args": fit_args,
                    },
                    "error": traceback.format_exc(),
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                }

                remove_pending_job(results_path, job_id)
                # append_jsonl(results_path, error_record)
                print(f"Error occurred while processing {job_name}: {error_record['error']}")

    final_time = TimeInt(time.time() - start_time)

    summary = {
        "status": "complete",
        "run_id": args.run_id,
        "n_jobs": len(jobs),
        "n_success": results,
        "total_execution_time": final_time,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }

    append_jsonl(results_path, summary)

    print(f"\nTotal execution time: {final_time}")
    print(f"Saved {results}/{len(jobs)} successful results to {results_path}")