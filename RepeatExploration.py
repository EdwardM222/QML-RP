from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from pathlib import Path
import argparse
import json
import time
import traceback

import numpy as np
import pandas as pd

from Ansatze import build_ansatz
from Exploration import append_jsonl, train_model
from QuECOC import TimeInt


ANALYSIS_RESULTS_PATH = Path("results/analysis/ensemble_results.parquet")
RANKINGS_PATH = Path("results/analysis/ensemble_rankings.parquet")
PRIORITY_PATH = Path("results/analysis/ensemble_repeat_priority.parquet")

SORT_COLUMN = "average_percentile"
REQUIRE_COMPLETE_DATASET_COVERAGE = True
MIN_PERCENTILE = 0.0
MAX_CONFIGURATIONS = 0
RESET_PRIORITY = False

SKIP_FAMILIES = {"SVC", "Random Forest"}

feature_density_options = [0.25, 0.5, 0.75, "mixed"]
encoding_options = ["angle", "mixed"]
n_learners_options = [6, 10, 14]

primary_configs = list(product(
    feature_density_options,
    encoding_options,
    n_learners_options,
))

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
    [2, 0, 2],
    [2, 2],
    [3, 1],
    [0, 3, 0],
    [3, 0, 0],
]

train_layer_groups = {
    "same-333": [[3, 3, 3]],
    "same-222": [[2, 2, 2]],
    "same-303": [[3, 0, 3]],
    "same-22": [[2, 2]],

    "uniform": [
        [3, 3, 3],
        [2, 2, 2],
        [3, 3],
        [2, 2],
    ],

    "directional": [
        [1, 2, 3],
        [3, 2, 1],
        [1, 3],
        [3, 1],
    ],

    "sparse": [
        [3, 0, 3],
        [2, 0, 2],
        [0, 3, 0],
        [3, 0, 0],
    ],

    "dense": [
        [3, 3, 3],
        [2, 2, 2],
        [1, 2, 3],
        [3, 2, 1],
        [3, 3],
        [2, 2],
    ],

    "representative": [
        [3, 3, 3],
        [1, 2, 3],
        [3, 0, 3],
        [1, 3],
        [2, 2],
        [3, 1],
    ],

    "two-layer": [
        [3, 3],
        [2, 3],
        [3, 2],
        [1, 3],
        [2, 2],
        [3, 1],
    ],

    "three-layer": [
        [3, 3, 3],
        [2, 2, 2],
        [1, 2, 3],
        [3, 0, 3],
        [3, 2, 1],
        [2, 0, 2],
        [0, 3, 0],
        [3, 0, 0],
    ],

    "all-layouts": train_layers,
}

entangling_pattern = "linear"


def parse_number(value):
    value = str(value)

    if value == "mixed":
        return value

    number = float(value)
    return int(number) if number.is_integer() else number


def build_repeat_model_args(row):
    model_family = str(row["model_family"])

    if model_family == "VQC":
        layout = json.loads(str(row["trainable_layers"]))
        fpq = int(parse_number(row["fpq_condition"]))
        density = float(parse_number(row["feature_density_condition"]))
        encoding = str(row["encoding_condition"])
        feature_strategy = str(row["feature_strategy"])
        entangler = str(row["entangler_condition"])

        feature_range = (0, np.pi)
        if str(row["feature_range"]) == "minus_pi_to_pi":
            feature_range = (-np.pi, np.pi)

        return {
            "n_qubits": int(row["n_qubits"]),
            "ansatz": build_ansatz(
                feats_per_qubit=fpq,
                reuploads=len(layout),
                encoding_style=encoding,
                feature_strategy=feature_strategy,
                trainable_layers=layout,
                entangling_uploads="all",
                entangling_layers="all",
                entangling_pattern=entangling_pattern,
                entangler=entangler,
            )["ansatz"],
            "feature_density": density,
            "feature_range": feature_range,
        }

    n_learners = int(row["n_learners"])
    group = str(row["layout_group"])
    encoding_style = str(row["encoding_condition"])
    fpq = parse_number(row["fpq_condition"])
    feature_density = parse_number(row["feature_density_condition"])
    feature_strategy = str(row["feature_strategy"])
    entangler = str(row["entangler_condition"])

    group_index = list(train_layer_groups).index(group)
    layouts = train_layer_groups[group]
    config_index = primary_configs.index((
        feature_density,
        encoding_style,
        n_learners,
    ))

    templates = []

    for i in range(n_learners):
        layout = layouts[(i + config_index) % len(layouts)]

        if feature_density == "mixed":
            learner_density = [0.25, 0.5, 0.75][
                (i + group_index + config_index) % 3
            ]
        else:
            learner_density = feature_density

        if encoding_style == "mixed":
            encoding = ["angle", "angle", "parallel_pairwise"][
                (i + group_index + config_index + 1) % 3
            ]
        else:
            encoding = "angle"

        if entangler == "mixed":
            entangler_gate = ["cz", "rzz"][
                (i + group_index + config_index) % 2
            ]
        else:
            entangler_gate = entangler

        if fpq == "mixed":
            fpq_value = [3, 4, 5][
                (i + group_index + config_index + 2) % 3
            ]
        else:
            fpq_value = fpq

        templates.append({
            "n_qubits": 2,
            "feature_density": learner_density,
            "ansatz": build_ansatz(
                feats_per_qubit=fpq_value,
                reuploads=len(layout),
                encoding_style=encoding,
                feature_strategy=feature_strategy,
                trainable_layers=layout,
                entangling_uploads="all",
                entangling_layers="all",
                entangling_pattern=entangling_pattern,
                entangler=entangler_gate,
            )["ansatz"],
        })

    model_args = {"templates": templates}

    if model_family == "CoherentECOC":
        meta_layout = [
            int(value)
            for value in str(row["meta_layout"]).split("-")
        ]

        model_args["meta_design"] = str(row["meta_design"])
        model_args["meta_template"] = build_ansatz(
            feats_per_qubit=1,
            reuploads=len(meta_layout),
            encoding_style="none",
            feature_strategy=feature_strategy,
            trainable_layers=meta_layout,
            entangling_uploads="all",
            entangling_layers="all",
            entangling_pattern=entangling_pattern,
            entangler=str(row["meta_entangler"]),
        )["ansatz"]

    return model_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        required=True,
        help="Output repeat name, for example ENSEMBLE_2",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--split",
        default="0",
        help="0 for all jobs, or 1/2/3 for a round-robin third.",
    )
    args = parser.parse_args()

    output_path = Path(f"results/{args.run_id}.jsonl")

    existing_ids = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    record = json.loads(line)
                    job_id = record.get("run", {}).get("id")
                    if job_id is not None:
                        existing_ids.add(job_id)
                except json.JSONDecodeError:
                    pass

    analysis_results = pd.read_parquet(ANALYSIS_RESULTS_PATH)

    if PRIORITY_PATH.exists() and not RESET_PRIORITY:
        priority = pd.read_parquet(PRIORITY_PATH)
        print(f"Loading fixed repeat priority from {PRIORITY_PATH}")
    else:
        rankings = pd.read_parquet(RANKINGS_PATH)
        priority = rankings[
            ~rankings["model_family"].astype(str).isin(SKIP_FAMILIES)
        ].copy()

        if REQUIRE_COMPLETE_DATASET_COVERAGE:
            priority = priority[priority["complete_dataset_coverage"]]

        priority = priority[
            pd.to_numeric(priority[SORT_COLUMN], errors="coerce") >= MIN_PERCENTILE
        ]

        priority = priority.sort_values(
            [SORT_COLUMN, "macro_f1_mean", "training_time_mean"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

        priority.insert(0, "repeat_priority", range(1, len(priority) + 1))

        PRIORITY_PATH.parent.mkdir(parents=True, exist_ok=True)
        priority.to_parquet(PRIORITY_PATH, compression="zstd", index=False)
        print(f"Saved fixed repeat priority to {PRIORITY_PATH}")

    if MAX_CONFIGURATIONS > 0:
        priority = priority.head(MAX_CONFIGURATIONS)

    jobs = []
    manifest = []

    for _, ranking in priority.iterrows():
        configuration_id = str(ranking["configuration_id"])

        config_results = analysis_results[
            analysis_results["configuration_id"].astype(str) == configuration_id
        ].sort_values("dataset")

        for _, row in config_results.iterrows():
            job_id = str(row["run_id"])

            if job_id in existing_ids:
                continue

            model_name = str(row["model_name"])
            dataset_path = str(Path("datasets") / str(row["dataset"]))
            model_args = build_repeat_model_args(row)
            fit_args = {}

            jobs.append((
                model_name,
                dataset_path,
                model_args,
                fit_args,
                job_id,
            ))

            manifest.append({
                "repeat_priority": int(ranking["repeat_priority"]),
                "average_percentile": float(ranking[SORT_COLUMN]),
                "model_family": str(ranking["model_family"]),
                "configuration_id": configuration_id,
                "dataset": str(row["dataset"]),
                "run_id": job_id,
            })

    print(
        f"\nSelected {len(priority):,} ranked configurations "
        f"and {len(jobs):,} unfinished repeat jobs."
    )

    if manifest:
        manifest_path = Path(f"results/{args.run_id}_jobs.csv")
        pd.DataFrame(manifest).to_csv(manifest_path, index=False)
        print(f"Saved job order to {manifest_path}")

        print("\nHighest-priority jobs:")
        print(
            pd.DataFrame(manifest)[
                [
                    "repeat_priority",
                    "average_percentile",
                    "model_family",
                    "configuration_id",
                    "dataset",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    if args.split != "0":
        split_index = int(args.split)

        if split_index not in {1, 2, 3}:
            raise ValueError("--split must be 0, 1, 2 or 3")

        # Round-robin keeps all three splits focused on the top of the ranking.
        jobs = jobs[split_index - 1::3]

    print(f"\nJobs running in this split ({args.split}): {len(jobs):,}\n")

    start_time = time.time()
    completed = 0

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
                append_jsonl(output_path, result)
                completed += 1

            except Exception:
                error_record = {
                    "status": "error",
                    "run": {
                        "id": job_id,
                        "model_name": job_name,
                        "dataset": dataset_path,
                        "model_args": model_args,
                        "fit_args": fit_args,
                    },
                    "error": traceback.format_exc(),
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                }

                print(
                    f"Error occurred while processing "
                    f"{job_name}: {error_record['error']}"
                )

    final_time = TimeInt(time.time() - start_time)

    summary = {
        "status": "complete",
        "run_id": args.run_id,
        "n_jobs": len(jobs),
        "n_success": completed,
        "total_execution_time": final_time,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }

    append_jsonl(output_path, summary)

    print(f"\nTotal execution time: {final_time}")
    print(f"Saved {completed}/{len(jobs)} successful results to {output_path}")
