from concurrent.futures import ProcessPoolExecutor, as_completed
from qiskit_ibm_runtime import results
from qiskit import result
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import time
import os
import pandas as pd
import numpy as np
from QuECOC import QuantumECOC, StackedECOC, VQC, TimeInt
import traceback
from sklearn.model_selection import ParameterGrid
import argparse
import json
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

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

def train_model(name, dataset, model_args, fit_args=None):
    if fit_args is None:
        fit_args = {}

    print(f"Training {name}...")

    X = pd.read_csv(dataset)
    y = X['target']
    X = X.drop('target', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)

    if name.startswith("SVC"):
        model = SVC(**model_args)
        model.fit(X_train, y_train, **fit_args)
    elif name.startswith("Random Forest"):
        model = RandomForestClassifier(**model_args)
        model.fit(X_train, y_train, **fit_args)
    elif name.startswith("VQC"):
        c = len(np.unique(y_train))
        scaler = MinMaxScaler(feature_range=model_args.pop('feature_range'))

        model = VQC.from_template(
            template=model_args,
            n_classes=c,
            n_total_features=X_train.shape[1],
            cuda_device=DEVICE,
        )

        X_tr = scaler.fit_transform(X_train.iloc[:, model.feats])
        X_te = scaler.transform(X_test.iloc[:, model.feats])

        y_tr = y_train.map({label: idx for idx, label in enumerate(sorted(y.unique()))}).values
        y_te = y_test.map({label: idx for idx, label in enumerate(sorted(y.unique()))}).values

        model.fit(X_tr, y_tr, X_te, y_te, **fit_args)
    elif name.startswith("QuantumECOC"):
        model = QuantumECOC(**model_args).to(DEVICE)
        model.fit(X_train, y_train, X_test, y_test, **fit_args)
    elif name.startswith("StackedECOC"):
        model = StackedECOC(**model_args).to(DEVICE)
        model.fit(X_train, y_train, X_test, y_test, **fit_args)
    elif name.startswith("Quantum StackedECOC"):
        c = len(np.unique(y_train))
        metaVQC = VQC.from_template(
            template='meta',
            n_classes=c,
            n_total_features=X_train.shape[1]
        ).to("cpu")
        model = StackedECOC(meta_learner=metaVQC, **model_args).to(DEVICE)
        model.fit(X_train, y_train, X_test, y_test, **fit_args)

    if name.startswith("VQC"):
        report = model.val_report
    else:
        preds = model.predict(X_test)
        report = classification_report(
            y_test,
            preds,
            zero_division=0,
            output_dict=True,
        )

    report_df = pd.DataFrame(report).transpose()
    report_df.loc["accuracy"] = [
        np.nan,
        np.nan,
        report_df.loc["accuracy", "f1-score"],
        report_df.loc["macro avg", "support"],
    ]

    model_results = None
    if hasattr(model, "get_results"):
        model_results = model.get_results()

    result = {
        "run": {
            "model_name": name,
            "dataset": dataset,
            "model_args": model_args,
            "fit_args": fit_args,
            "device": DEVICE,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        },
        "timing": {
            "training_time": getattr(model, "training_time", None),
        },
        "metrics": {
            "classification_report": report,
            "accuracy": report.get("accuracy"),
            "macro_f1": report.get("macro avg", {}).get("f1-score"),
            "weighted_f1": report.get("weighted avg", {}).get("f1-score"),
        },
        "report_table": report_df,
        "model": model_results,
    }

    print(f"Finished {name}.")

    return result

def search(model_name, dataset_path, param_grid, fit_args=None):
    results = []
    for params in ParameterGrid(param_grid):
        name = f"{model_name} - {params}"
        result = train_model(name, dataset_path, params, fit_args)
        results.append(result)

    return results

def create_search_jobs(model_name, dataset_path, param_grid, fit_args=None):
    jobs = []
    for params in ParameterGrid(param_grid):
        jobs.append((f"{model_name}", dataset_path, params, fit_args))

    return jobs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        default="default",
        help="Identifier used in the output filename",
    )
    args = parser.parse_args()
    results_path = Path(f"results/search_results_{args.run_id}.jsonl")

    jobs = []
    for tier in [0]:
        for dataset in sorted(os.listdir(f"datasets/{tier}")):
            path = os.path.join(f"datasets/{tier}", dataset)

            # jobs.extend(create_search_jobs("SVC", path, {
            #     'kernel': ['rbf'],
            #     'class_weight': ['balanced'],
            #     'random_state': [2]
            # }))

            # jobs.extend(create_search_jobs("Random Forest", path, {
            #     'n_estimators': [100],
            #     'class_weight': ['balanced'],
            #     'random_state': [2]
            # }))

            jobs.extend(create_search_jobs("VQC", path, {
                "n_qubits": [2, 4, 6, 8, 12, 16],
                "measurement_mode": ["min"],
                "ansatz": ["default", "hea_cz_ring", "1"],
                "feature_density": [0.25, 0.5, 0.75, 1.0, 2, 4, 6, 8],
                "feature_range": [(0, np.pi)],
            }))

            # jobs.extend(create_search_jobs("QuantumECOC", path, {
            #     'templates': ["1"]
            # }))

            # jobs.extend(create_search_jobs("StackedECOC", path, {
            #     'templates': ["1"]
            # }))

            # jobs.extend(create_search_jobs("Quantum StackedECOC", path, {
            #     'templates': ["1"]
            # }))

    jobs = jobs[:10]
    print(f"Total jobs to run: {len(jobs)}\n")

    start_time = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(train_model, *job): job
            for job in jobs
        }

        for future in as_completed(futures):
            job = futures[future]
            job_name, dataset_path, model_args, fit_args = job

            try:
                result = future.result()

                # Add job metadata explicitly, even if train_model already included it.
                result["job"] = {
                    "model_name": job_name,
                    "dataset": dataset_path,
                    "model_args": model_args,
                    "fit_args": fit_args,
                }

                results.append(result)
                append_jsonl(results_path, result)

                metrics = result.get("metrics", {})
                training_time = result.get("timing", {}).get("training_time")

                print(
                    f"Saved result: {job_name} | {dataset_path} | "
                    f"macro_f1={metrics.get('macro_f1')} | "
                    f"time={training_time}"
                )

            except Exception:
                error_record = {
                    "status": "error",
                    "job": {
                        "model_name": job_name,
                        "dataset": dataset_path,
                        "model_args": model_args,
                        "fit_args": fit_args,
                    },
                    "error": traceback.format_exc(),
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                }

                append_jsonl(results_path, error_record)
                print(f"Error occurred while processing {job_name}: {error_record['error']}")

    final_time = TimeInt(time.time() - start_time)

    summary = {
        "status": "complete",
        "run_id": args.run_id,
        "n_jobs": len(jobs),
        "n_success": len(results),
        "total_execution_time": final_time,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }

    append_jsonl(results_path, summary)

    print(f"\nTotal execution time: {final_time}")
    print(f"Saved {len(results)}/{len(jobs)} successful results to {results_path}")