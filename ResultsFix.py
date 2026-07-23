from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import ParameterGrid

from Ansatze import get_ansatze_configs

def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, Path):
        return str(value)

    return value

def make_job_id(model_name, dataset_path, model_args, fit_args=None):
    return hashlib.md5(
        json.dumps(
            json_safe({
                "model_name": model_name,
                "dataset": dataset_path,
                "model_args": model_args,
                "fit_args": fit_args or {},
        }),
        sort_keys=True,
    ).encode()).hexdigest()

if __name__ == "__main__":
    RESULTS_DIR = Path("results")
    OUTPUT_DIR = RESULTS_DIR / "vqcs"

    INPUT_FILES = [
        RESULTS_DIR / f"VQCS_{i}.jsonl"
        for i in range(1, 6)
    ]

    MAIN_DATASETS = [
        "iris",
        "balance-scale",
        "contraceptive-method-choice",
        "heart-disease",
        "obesity",
        "image-segmentation",
        "steel-plates",
        "yeast",
    ]

    TEST_MODE = False
    N_TEST_SAMPLES = 50

    missing_files = [
        path
        for path in INPUT_FILES
        if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing result files: "
            + ", ".join(
                str(path)
                for path in missing_files
            )
        )

    print("Building job lookup...")
    
    ansatze = []
    ids = []
    for ansatz in get_ansatze_configs():
        if ansatz["id"] not in ids:
            name = next(iter(ansatz["ansatz"].keys()))
            ansatze.append({
                'ansatz': {"vqc": ansatz["ansatz"][name]},
                'name': name,
                'config': ansatz["config"]
            })
            ids.append(ansatz["id"])
    ids = None

    base_grid = {
        "n_qubits": [2, 4, 6, 8],
        "measurement_mode": ["min"],
        "feature_density": [0.25, 0.5, 0.75, 1.0, 2, 4, 6, 8],
        "feature_range": [(0, np.pi), (-np.pi, np.pi)],
    }

    lookup = {}
    for dataset in MAIN_DATASETS:
        dataset_path = os.path.join("datasets/", dataset)

        for params in ParameterGrid(base_grid):
            for ansatz_entry in ansatze:
                params["ansatz"] = ansatz_entry["ansatz"]
                
                # The ID must be calculated before fixing model_args.
                job_id = make_job_id("VQC", dataset_path, params, {})

                new_args = {
                    'name': ansatz_entry['name'],
                    'config': ansatz_entry['config']
                }

                lookup[job_id] = new_args

    print(f"Generated {len(lookup):,} job mappings.\n")
     
    passed = 0
    failed = 0
    for input_path in INPUT_FILES:
        output_path = OUTPUT_DIR / input_path.name
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

        print(f"Processing {input_path}...")

        with (
            input_path.open("r", encoding="utf-8") as source,
            temporary_path.open("w", encoding="utf-8") as destination
        ):
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    destination.write(line)
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    destination.write(line)
                    continue

                run = record.get("run")

                if not isinstance(run, dict):
                    destination.write(json.dumps(json_safe(record), separators=(",", ":")) + "\n")
                    continue

                job_id = run.get("id")

                new_model_args = lookup.get(job_id)

                if new_model_args is None:
                    failed += 1
                    continue
                
                if TEST_MODE:
                    print("Old model_args: ", run.get("model_args"))
                    print("New model_args: ", new_model_args)

                run["model_args"]["ansatz"] = new_model_args["name"]
                for key, value in new_model_args["config"].items():
                    run["model_args"][key] = value
                
                passed += 1

                destination.write(json.dumps(json_safe(record), separators=(",", ":")) + "\n")

                if line_number % 50_000 == 0:
                    print(f"{input_path.name}: {line_number:,} lines processed | passed={passed:,} | failed={failed:,}")

                if passed >= N_TEST_SAMPLES and TEST_MODE:
                    print(f"Test mode: processed {N_TEST_SAMPLES} samples, stopping early.")
                    break

        temporary_path.replace(output_path)

    print(f"Finished processing {len(INPUT_FILES)} files. Total passed={passed:,}, total failed={failed:,}.")