from pathlib import Path
import json

RESULTS_DIR = Path("results")
FILE_PREFIX = "ensemble/Ensemble_S3"

for path in sorted(RESULTS_DIR.glob(f"{FILE_PREFIX}_*.jsonl")):
    kept = []
    removed = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            if record.get("run", {}).get("status") == "pending":
                removed += 1
                continue

            kept.append(line)

    with path.open("w", encoding="utf-8") as f:
        f.writelines(kept)

    print(f"{path}: removed {removed} pending lines")