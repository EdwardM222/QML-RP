import subprocess
import sys
from pathlib import Path

import pandas as pd

from EnsembleResults import (
    aggregate_ansatze,
    aggregate_results,
    build_paired_comparisons,
    extract_runs,
    rank_results,
)


RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "analysis"

RUNS_PATH = OUTPUT_DIR / "ensemble_runs.parquet"
RESULTS_PATH = OUTPUT_DIR / "ensemble_results.parquet"
RANKINGS_PATH = OUTPUT_DIR / "ensemble_rankings.parquet"
ANSATZ_RESULTS_PATH = OUTPUT_DIR / "ensemble_ansatz_results.parquet"
ANSATZ_DATASET_RESULTS_PATH = OUTPUT_DIR / "ensemble_ansatz_dataset_results.parquet"
PAIRED_RESULTS_PATH = OUTPUT_DIR / "ensemble_paired_results.parquet"

DASHBOARD_PATH = Path("Dashboard.py")

FILE_PREFIX = "combined"
RELOAD_JSONL = True

TEST_MODE = False
N_TEST_RESULTS = 1000

LAUNCH_DASHBOARD = True


def run_ensemble_analysis(
    results_dir,
    runs_path,
    results_path,
    rankings_path,
    ansatz_results_path,
    ansatz_dataset_results_path,
    paired_results_path,
    dashboard_path=None,
    file_prefix="ENSEMBLE",
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
    print(runs.groupby("search_repeat", observed=True).size())

    print("\nRepetitions per run ID:")
    print(
        runs.groupby("run_id", observed=True)
        .size()
        .value_counts()
        .sort_index()
    )

    print("\nRuns per model family:")
    print(runs.groupby("model_family", observed=True).size().sort_values(ascending=False))

    results = aggregate_results(runs, output_path=results_path)
    rankings = rank_results(results, output_path=rankings_path)

    ansatz_results, ansatz_dataset_results = aggregate_ansatze(
        results,
        overall_output_path=ansatz_results_path,
        dataset_output_path=ansatz_dataset_results_path,
    )

    paired_results = build_paired_comparisons(
        results,
        output_path=paired_results_path,
    )

    print("\nTop 20 configurations:")
    columns = [
        "overall_rank",
        "model_family",
        "base_config_id",
        "datasets",
        "dataset_coverage",
        "average_rank",
        "average_percentile",
        "average_family_rank",
        "average_family_percentile",
        "macro_f1_mean",
        "macro_f1_repeat_std",
        "training_time_mean",
    ]
    columns = [column for column in columns if column in rankings.columns]

    print(rankings[columns].head(20).to_string(index=False))

    print("\nPaired ensemble comparisons:")
    print(
        paired_results[
            [
                "dataset",
                "base_config_id",
                "classical_stacking_gain_macro_f1",
                "quantum_stacking_gain_macro_f1",
                "quantum_vs_direct_gain_macro_f1",
            ]
        ]
        .dropna(how="all", subset=[
            "classical_stacking_gain_macro_f1",
            "quantum_stacking_gain_macro_f1",
            "quantum_vs_direct_gain_macro_f1",
        ])
        .head(20)
        .to_string(index=False)
    )

    if launch_dashboard:
        if dashboard_path is None:
            raise ValueError("dashboard_path is required when launch_dashboard=True")

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

    return (
        runs,
        results,
        rankings,
        ansatz_results,
        ansatz_dataset_results,
        paired_results,
    )


if __name__ == "__main__":
    run_ensemble_analysis(
        results_dir=RESULTS_DIR,
        runs_path=RUNS_PATH,
        results_path=RESULTS_PATH,
        rankings_path=RANKINGS_PATH,
        ansatz_results_path=ANSATZ_RESULTS_PATH,
        ansatz_dataset_results_path=ANSATZ_DATASET_RESULTS_PATH,
        paired_results_path=PAIRED_RESULTS_PATH,
        dashboard_path=DASHBOARD_PATH,
        file_prefix=FILE_PREFIX,
        reload_jsonl=RELOAD_JSONL,
        test_mode=TEST_MODE,
        n_test_results=N_TEST_RESULTS,
        launch_dashboard=LAUNCH_DASHBOARD,
    )
