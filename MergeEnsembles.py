from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Combine five versioned JSONL files into repeat files based on "
            "job ID occurrence, while separating invalid/error rows."
        )
    )

    parser.add_argument(
        "filename",
        type=Path,
        help=(
            "Base filename, e.g. results/Ensembles.jsonl. "
            "Files Ensembles_1.jsonl through Ensembles_5.jsonl will be read."
        ),
    )

    args = parser.parse_args()

    base = args.filename
    input_files = [
        base.with_name(f"{base.stem}_{i}{base.suffix}")
        for i in range(1, 6)
    ]

    missing = [
        path
        for path in input_files
        if not path.exists()
    ]

    if missing:
        print("Missing input files:")
        for path in missing:
            print(f"  {path}")
        raise SystemExit(1)

    output_dir = base.parent
    error_path = output_dir / "combined_err.jsonl"

    if error_path.exists():
        raise FileExistsError(
            f"Output already exists: {error_path}\n"
            "Remove or rename existing combined output files first."
        )

    for path in output_dir.glob("combined_*.jsonl"):
        raise FileExistsError(
            f"Output already exists: {path}\n"
            "Remove or rename existing combined output files first."
        )

    occurrence_counts = Counter()
    output_handles = {}
    output_rows = Counter()

    total_rows = 0
    valid_job_rows = 0
    error_rows = 0
    malformed_rows = 0
    missing_run_rows = 0
    missing_id_rows = 0

    per_file = []

    error_file = error_path.open("w", encoding="utf-8")

    try:
        for input_path in input_files:
            file_rows = 0
            file_valid = 0
            file_errors = 0

            with input_path.open("r", encoding="utf-8") as source:
                for line in source:
                    total_rows += 1
                    file_rows += 1

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        error_file.write(line)
                        error_rows += 1
                        malformed_rows += 1
                        file_errors += 1
                        continue

                    run = record.get("run")

                    if not isinstance(run, dict):
                        error_file.write(line)
                        error_rows += 1
                        missing_run_rows += 1
                        file_errors += 1
                        continue

                    job_id = run.get("id")

                    if job_id is None or job_id == "":
                        error_file.write(line)
                        error_rows += 1
                        missing_id_rows += 1
                        file_errors += 1
                        continue

                    valid_job_rows += 1
                    file_valid += 1

                    occurrence_counts[job_id] += 1
                    occurrence = occurrence_counts[job_id]

                    if occurrence not in output_handles:
                        output_path = output_dir / f"combined_{occurrence}.jsonl"
                        output_handles[occurrence] = output_path.open(
                            "w",
                            encoding="utf-8",
                        )

                    output_handles[occurrence].write(
                        json.dumps(record) + "\n"
                    )
                    output_rows[occurrence] += 1

            per_file.append({
                "path": input_path,
                "rows": file_rows,
                "valid": file_valid,
                "errors": file_errors,
            })

    finally:
        error_file.close()

        for handle in output_handles.values():
            handle.close()

    unique_job_ids = len(occurrence_counts)
    duplicate_rows = valid_job_rows - unique_job_ids
    duplicate_ids = sum(
        1
        for count in occurrence_counts.values()
        if count > 1
    )
    max_occurrences = max(
        occurrence_counts.values(),
        default=0,
    )

    frequency_distribution = Counter(
        occurrence_counts.values()
    )

    print("\nInput files:")
    for path in input_files:
        print(f"  {path}")

    print("\nPer-file statistics:")
    for stats in per_file:
        print(
            f"  {stats['path'].name}: "
            f"rows={stats['rows']:,} | "
            f"valid_jobs={stats['valid']:,} | "
            f"errors={stats['errors']:,}"
        )

    print("\nCombined output files:")
    for occurrence in sorted(output_rows):
        output_path = output_dir / f"combined_{occurrence}.jsonl"
        print(
            f"  {output_path.name}: "
            f"{output_rows[occurrence]:,} rows"
        )

    print(
        f"  {error_path.name}: "
        f"{error_rows:,} rows"
    )

    print("\nOverall statistics:")
    print(f"  Total input rows:          {total_rows:,}")
    print(f"  Valid job rows:            {valid_job_rows:,}")
    print(f"  Unique job IDs:            {unique_job_ids:,}")
    print(f"  Duplicate job rows:        {duplicate_rows:,}")
    print(f"  Job IDs with duplicates:   {duplicate_ids:,}")
    print(f"  Maximum repeats of one ID: {max_occurrences:,}")
    print(f"  Error rows:                {error_rows:,}")
    print(f"    Malformed JSON:           {malformed_rows:,}")
    print(f"    Missing/invalid run:      {missing_run_rows:,}")
    print(f"    Missing/blank job ID:     {missing_id_rows:,}")

    print("\nJob ID repeat distribution:")
    for repeats in sorted(frequency_distribution):
        print(
            f"  {repeats} occurrence(s): "
            f"{frequency_distribution[repeats]:,} job ID(s)"
        )

    print(
        "\nNo source files were modified."
    )


if __name__ == "__main__":
    main()