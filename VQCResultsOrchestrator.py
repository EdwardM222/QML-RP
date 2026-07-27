import subprocess
import sys
from pathlib import Path

import pandas as pd

from VQCResults import (
    aggregate_ansatze,
    aggregate_results,
    extract_runs,
    rank_results,
)


RESULTS_DIR = Path("results/vqcs")
OUTPUT_DIR = Path("results/analysis")

RUNS_PATH = OUTPUT_DIR / "vqc_runs.parquet"
RESULTS_PATH = OUTPUT_DIR / "vqc_results.parquet"
RANKINGS_PATH = OUTPUT_DIR / "vqc_rankings.parquet"
ANSATZ_RESULTS_PATH = OUTPUT_DIR / "vqc_ansatz_results.parquet"
ANSATZ_DATASET_RESULTS_PATH = OUTPUT_DIR / "vqc_ansatz_dataset_results.parquet"

DASHBOARD_PATH = Path("VQCDashboard.py")

FILE_PREFIX = "VQCS"
RELOAD_JSONL = False

TEST_MODE = False
N_TEST_RESULTS = 1000

LAUNCH_DASHBOARD = True


def run_vqc_analysis(
    results_dir,
    runs_path,
    results_path,
    rankings_path,
    ansatz_results_path,
    ansatz_dataset_results_path,
    dashboard_path=None,
    file_prefix="VQCS",
    reload_jsonl=False,
    test_mode=False,
    n_test_results=1000,
    launch_dashboard=False,
):
    if reload_jsonl or not Path(runs_path).exists():
        runs = extract_runs(
            results_dir=results_dir,
            output_path=runs_path,
            file_prefix=file_prefix,
            test_mode=test_mode,
            n_test_results=n_test_results,
        )
    else:
        print(f"Loading compact runs from {runs_path}...")
        runs = pd.read_parquet(runs_path)

    print("\nRuns per search repetition:")
    print(
        runs.groupby(
            "search_repeat",
            observed=True,
        ).size()
    )

    print("\nRepetitions per run ID:")
    print(
        runs.groupby(
            "run_id",
            observed=True,
        )
        .size()
        .value_counts()
        .sort_index()
    )

    results = aggregate_results(
        runs,
        output_path=results_path,
    )

    rankings = rank_results(
        results,
        output_path=rankings_path,
    )

    ansatz_results, ansatz_dataset_results = aggregate_ansatze(
        results,
        overall_output_path=ansatz_results_path,
        dataset_output_path=ansatz_dataset_results_path,
    )

    print("\nTop 20 configurations:")
    print(
        rankings[
            [
                "overall_rank",
                "ansatz",
                "n_qubits",
                "feature_density",
                "feature_range",
                "average_rank",
                "average_percentile",
                "best_rank",
                "worst_rank",
                "macro_f1_mean",
                "macro_f1_repeat_std",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    if launch_dashboard:
        if dashboard_path is None:
            raise ValueError(
                "dashboard_path is required when launch_dashboard=True"
            )

        print(f"\nLaunching dashboard from {dashboard_path}...")

        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_path),
            "--",
            "--results-path",
            str(results_path),
            "--rankings-path",
            str(rankings_path),
            "--runs-path",
            str(runs_path),
            "--ansatz-results-path",
            str(ansatz_results_path),
            "--ansatz-dataset-results-path",
            str(ansatz_dataset_results_path),
        ])

    return runs, results, rankings, ansatz_results, ansatz_dataset_results


if __name__ == "__main__":
    run_vqc_analysis(
        results_dir=RESULTS_DIR,
        runs_path=RUNS_PATH,
        results_path=RESULTS_PATH,
        rankings_path=RANKINGS_PATH,
        ansatz_results_path=ANSATZ_RESULTS_PATH,
        ansatz_dataset_results_path=ANSATZ_DATASET_RESULTS_PATH,
        dashboard_path=DASHBOARD_PATH,
        file_prefix=FILE_PREFIX,
        reload_jsonl=RELOAD_JSONL,
        test_mode=TEST_MODE,
        n_test_results=N_TEST_RESULTS,
        launch_dashboard=LAUNCH_DASHBOARD,
    )