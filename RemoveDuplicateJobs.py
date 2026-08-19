from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def get_versioned_files(filename: Path) -> list[Path]:
    suffix = filename.suffix
    stem = filename.stem

    pattern = re.compile(
        rf"^{re.escape(stem)}_(\d+){re.escape(suffix)}$"
    )

    files = []
    for path in filename.parent.iterdir():
        match = pattern.match(path.name)
        if match:
            files.append((int(match.group(1)), path))

    if not files:
        raise FileNotFoundError(
            f"No versioned files found matching "
            f"'{stem}_x{suffix}' in '{filename.parent}'."
        )

    files.sort(key=lambda item: item[0])

    return [
        path
        for _, path in files
    ]


def get_job_id(line: str) -> str | None:
    if not line.strip():
        return None

    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None

    run = record.get("run")

    if not isinstance(run, dict):
        return None

    return run.get("id")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Remove duplicate job IDs across versioned JSONL files, "
            "keeping only the latest occurrence."
        )
    )

    parser.add_argument(
        "filename",
        type=Path,
        help=(
            "Base filename, e.g. results/VQCS.jsonl. "
            "Files VQCS_1.jsonl, VQCS_2.jsonl, ... will be checked."
        ),
    )

    args = parser.parse_args()

    input_files = get_versioned_files(args.filename)

    print("Files:")
    for path in input_files:
        print(f"  {path}")

    latest_occurrence = {}
    total_jobs = 0

    print("\nScanning job IDs...")

    for file_index, input_path in enumerate(input_files):
        with input_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                job_id = get_job_id(line)

                if job_id is None:
                    continue

                total_jobs += 1

                latest_occurrence[job_id] = (
                    file_index,
                    line_number,
                )

    duplicate_count = total_jobs - len(latest_occurrence)

    print(f"Total job records: {total_jobs:,}")
    print(f"Unique job IDs: {len(latest_occurrence):,}")
    print(f"Duplicates to remove: {duplicate_count:,}")

    if duplicate_count == 0:
        print("No duplicate job IDs found.")
        return

    removed = 0

    print("\nRemoving older duplicates...")

    for file_index, input_path in enumerate(input_files):
        temporary_path = input_path.with_suffix(
            input_path.suffix + ".tmp"
        )

        kept_in_file = 0
        removed_in_file = 0

        with (
            input_path.open("r", encoding="utf-8") as source,
            temporary_path.open("w", encoding="utf-8") as destination,
        ):
            for line_number, line in enumerate(source, start=1):
                job_id = get_job_id(line)

                if job_id is None:
                    destination.write(line)
                    continue

                if latest_occurrence[job_id] != (
                    file_index,
                    line_number,
                ):
                    removed += 1
                    removed_in_file += 1
                    continue

                destination.write(line)
                kept_in_file += 1

        temporary_path.replace(input_path)

        print(
            f"{input_path.name}: "
            f"kept={kept_in_file:,} | "
            f"removed={removed_in_file:,}"
        )

    print(
        f"\nFinished processing {len(input_files)} files. "
        f"Removed {removed:,} duplicate job records."
    )


if __name__ == "__main__":
    main()
