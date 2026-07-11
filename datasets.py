# from ucimlrepo import fetch_ucirepo
# import pandas as pd
# import os
# import sys
# from pathlib import Path

# UCI_DATASETS = {
#     # "balance-scale": 12,
#     # "contraceptive-method-choice": 30,
#     # "credit-approval": 27,
#     # "ctg-10classes": 193,
#     # "heart-disease": 45,
#     "image-segmentation": 50,
#     # "iris": 53,
#     # "letter": 59,
#     # "nursery": 76,
#     # "obesity": 544,
#     # "optical": 80,
#     # "pendigits": 81,
#     # "steel-plates": 198,
#     # "teaching-assistant-evaluation": 0,
#     # "waveform": 107,
#     # "wine-quality": 186,
#     # "wine": 109,
#     # "yeast": 110,
# }

# def classify_feature_type(series: pd.Series) -> str:
#     """
#     Classify a feature more usefully than pandas dtype alone.
#     """
#     if pd.api.types.is_bool_dtype(series):
#         return "boolean"

#     if pd.api.types.is_numeric_dtype(series):
#         unique_count = series.nunique(dropna=True)

#         if pd.api.types.is_integer_dtype(series):
#             if unique_count <= 2:
#                 return "binary integer"
#             if unique_count <= 15:
#                 return "discrete/ordinal integer"
#             return "integer"

#         if unique_count <= 2:
#             return "binary numeric"
#         return "continuous numeric"

#     unique_count = series.nunique(dropna=True)

#     if unique_count <= 2:
#         return "binary categorical"

#     return "categorical"

# DATASET_ROOT = Path("/home/edwar/QML-RP/original-datasets")
# TARGET_COLUMN = "target"
# OUTPUT_PATH = Path("dataset_quality_report.csv")

# summary_rows = []
# for dataset_name, uci_id in UCI_DATASETS.items():
#     dataset = fetch_ucirepo(id=uci_id)

#     for key, value in dataset.items():
#         print(f" --- {key}: {value}")
#         print("\n --- \n")

#     X = dataset.data.features
#     y = dataset.data.targets.iloc[:, 0]

#     if dataset_name == "steel-plates":
#         y = pd.Series(dataset.data.targets.apply(lambda row:dataset.data.targets.columns[row.argmax()], axis=1))

#     df = pd.concat([X, y.rename("target")], axis=1)

#     dataset_path = DATASET_ROOT / f"{dataset_name}.csv"

#     df.to_csv(dataset_path, index=False)

#     print("\n" + "=" * 100)
#     print(f"Dataset: {dataset_path}")
#     print("=" * 100)

#     try:
#         df = pd.read_csv(
#             dataset_path,
#             na_values=["?", "NA", "N/A", "null", "None", ""],
#             keep_default_na=True,
#         )
#     except Exception as exc:
#         print(f"Could not load file: {exc}")

#         summary_rows.append({
#             "dataset": dataset_path.stem,
#             "path": str(dataset_path),
#             "status": "load_error",
#             "error": str(exc),
#         })
#         continue

#     print(f"Rows: {len(df)}")
#     print(f"Columns: {len(df.columns)}")

#     if TARGET_COLUMN not in df.columns:
#         print(f"WARNING: target column '{TARGET_COLUMN}' not found.")

#     total_missing = int(df.isna().sum().sum())
#     rows_with_missing = int(df.isna().any(axis=1).sum())
#     missing_fraction = (
#         rows_with_missing / len(df)
#         if len(df) > 0
#         else 0
#     )

#     print(f"Total missing values: {total_missing}")
#     print(f"Rows with at least one missing value: {rows_with_missing}")
#     print(f"Fraction of rows with missing values: {missing_fraction:.4f}")

#     missing_by_column = df.isna().sum()
#     missing_by_column = missing_by_column[missing_by_column > 0]

#     if missing_by_column.empty:
#         print("Missing values by column: none")
#     else:
#         print("\nMissing values by column:")
#         for column, count in missing_by_column.items():
#             percentage = count / len(df) * 100
#             print(f"  {column}: {count} ({percentage:.2f}%)")

#     feature_columns = [
#         column
#         for column in df.columns
#         if column != TARGET_COLUMN
#     ]

#     feature_type_rows = []

#     for column in feature_columns:
#         series = df[column]

#         feature_type_rows.append({
#             "column": column,
#             "pandas_dtype": str(series.dtype),
#             "feature_type": classify_feature_type(series),
#             "unique_values": int(series.nunique(dropna=True)),
#             "missing_values": int(series.isna().sum()),
#             "missing_percentage": float(series.isna().mean() * 100),
#         })

#     feature_types_df = pd.DataFrame(feature_type_rows)

#     print("\nFeature datatypes:")
#     if feature_types_df.empty:
#         print("  No feature columns found.")
#     else:
#         print(
#             feature_types_df.to_string(
#                 index=False,
#                 justify="left",
#             )
#         )

#     type_counts = (
#         feature_types_df["feature_type"].value_counts().to_dict()
#         if not feature_types_df.empty
#         else {}
#     )

#     numeric_feature_count = sum(
#         "numeric" in feature_type or "integer" in feature_type
#         for feature_type in feature_types_df.get("feature_type", [])
#     )

#     categorical_feature_count = sum(
#         "categorical" in feature_type
#         for feature_type in feature_types_df.get("feature_type", [])
#     )

#     boolean_feature_count = sum(
#         feature_type == "boolean"
#         for feature_type in feature_types_df.get("feature_type", [])
#     )

#     target_classes = None
#     target_missing = None
#     target_dtype = None
#     target_class_counts = None

#     if TARGET_COLUMN in df.columns:
#         target = df[TARGET_COLUMN]

#         target_classes = int(target.nunique(dropna=True))
#         target_missing = int(target.isna().sum())
#         target_dtype = str(target.dtype)
#         target_class_counts = target.value_counts(dropna=False).to_dict()

#         print("\nTarget:")
#         print(f"  dtype: {target_dtype}")
#         print(f"  classes: {target_classes}")
#         print(f"  missing values: {target_missing}")
#         print("  class counts:")

#         for label, count in target.value_counts(dropna=False).items():
#             print(f"    {label}: {count}")

#     summary_rows.append({
#         "dataset": dataset_path.stem,
#         "tier": dataset_path.parent.name,
#         "path": str(dataset_path),
#         "status": "ok",
#         "rows": len(df),
#         "columns": len(df.columns),
#         "features": len(feature_columns),
#         "numeric_features": numeric_feature_count,
#         "categorical_features": categorical_feature_count,
#         "boolean_features": boolean_feature_count,
#         "feature_type_counts": str(type_counts),
#         "total_missing_values": total_missing,
#         "rows_with_missing_values": rows_with_missing,
#         "missing_row_fraction": missing_fraction,
#         "target_present": TARGET_COLUMN in df.columns,
#         "target_dtype": target_dtype,
#         "target_classes": target_classes,
#         "target_missing_values": target_missing,
#         "target_class_counts": str(target_class_counts),
#     })


# summary_df = pd.DataFrame(summary_rows)

# summary_df.to_csv(
#     OUTPUT_PATH,
#     index=False,
# )

# print("\n" + "=" * 100)
# print("Overall summary")
# print("=" * 100)

# if summary_df.empty:
#     print("No CSV files found.")
# else:
#     display_columns = [
#         "dataset",
#         "tier",
#         "rows",
#         "features",
#         "numeric_features",
#         "categorical_features",
#         "total_missing_values",
#         "rows_with_missing_values",
#         "target_classes",
#     ]

#     available_columns = [
#         column
#         for column in display_columns
#         if column in summary_df.columns
#     ]

#     print(summary_df[available_columns].to_string(index=False))
#     print(f"\nSaved full report to: {OUTPUT_PATH.resolve()}")

from pathlib import Path

import pandas as pd


DATA_DIR = Path("/home/edwar/QML-RP")

TRAIN_PATH = DATA_DIR / "segmentation.data"
TEST_PATH = DATA_DIR / "segmentation.test"

OUTPUT_PATH = Path(
    "/home/edwar/QML-RP/original-datasets/image-segmentation.csv"
)

COLUMNS = [
    "target",
    "region-centroid-col",
    "region-centroid-row",
    "region-pixel-count",
    "short-line-density-5",
    "short-line-density-2",
    "vedge-mean",
    "vedge-sd",
    "hedge-mean",
    "hedge-sd",
    "intensity-mean",
    "rawred-mean",
    "rawblue-mean",
    "rawgreen-mean",
    "exred-mean",
    "exblue-mean",
    "exgreen-mean",
    "value-mean",
    "saturation-mean",
    "hue-mean",
]


def load_segmentation_file(path: Path) -> pd.DataFrame:
    """
    Load a UCI Image Segmentation data file.

    Comment lines and blank lines are ignored.
    """
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        sep=",",
        comment="|",
        skip_blank_lines=True,
        skipinitialspace=True,
    )

    # Remove rows that may have been parsed from non-data text.
    df = df.dropna(how="all")

    if len(df.columns) != len(COLUMNS):
        raise ValueError(
            f"{path} produced {len(df.columns)} columns; "
            f"expected {len(COLUMNS)}."
        )

    return df


train_df = load_segmentation_file(TRAIN_PATH)
test_df = load_segmentation_file(TEST_PATH)

print(f"Training rows: {len(train_df)}")
print(f"Test rows: {len(test_df)}")

combined_df = pd.concat(
    [train_df, test_df],
    ignore_index=True,
)

# Clean whitespace from class labels.
combined_df["target"] = (
    combined_df["target"]
    .astype(str)
    .str.strip()
)

# Ensure all feature columns are numeric.
feature_columns = [
    column
    for column in combined_df.columns
    if column != "target"
]

for column in feature_columns:
    combined_df[column] = pd.to_numeric(
        combined_df[column],
        errors="raise",
    )

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

combined_df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print(f"Combined rows: {len(combined_df)}")
print(f"Columns: {len(combined_df.columns)}")
print("\nClass counts:")
print(combined_df["target"].value_counts())
print(f"\nSaved to: {OUTPUT_PATH}")