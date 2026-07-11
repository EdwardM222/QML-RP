from __future__ import annotations

import json
import warnings
from math import log2
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# =============================================================================
# Paths and global settings
# =============================================================================

DATASET_ROOT = Path("/home/edwar/QML-RP/original-datasets")
OUTPUT_ROOT = Path("/home/edwar/QML-RP/datasets")

TARGET_COLUMN = "target"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Text values that should be treated as missing.
#
# Do not add "-" globally because it may be a legitimate categorical value in
# some UCI datasets.
MISSING_TOKENS = [
    "?",
    "NA",
    "N/A",
    "na",
    "n/a",
    "NULL",
    "null",
    "None",
    "",
]


# =============================================================================
# Dataset-specific schema
# =============================================================================
#
# categorical:
#     Nominal variables that should be one-hot encoded.
#
# ordinal:
#     Ordered categorical variables that should remain numeric.
#
# drop:
#     Explicitly redundant or unsuitable columns to remove.
#
# domain, description, collection_notes and manual_tags:
#     Metadata that cannot be safely inferred. Blank values are deliberately
#     saved and warnings are printed so they can be completed manually.
#
# =============================================================================

DATASET_CONFIG: dict[str, dict[str, Any]] = {
    "balance-scale": {
        "categorical": [],
        "ordinal": [
            "right-distance",
            "right-weight",
            "left-distance",
            "left-weight",
        ],
        "drop": [],
        "domain": "",
        "description": "",
        "collection_notes": "",
        "manual_tags": [],
    },

    "contraceptive-method-choice": {
        "categorical": [
            "wife_religion",
            "wife_working",
            "husband_occupation",
            "media_exposure",
        ],
        "ordinal": [
            "wife_edu",
            "husband_edu",
            "standard_of_living_index",
        ],
        "drop": [],
        "domain": "",
        "description": "",
        "collection_notes": "",
        "manual_tags": [],
    },

    "credit-approval": {
        "categorical": [
            "A1",
            "A4",
            "A5",
            "A6",
            "A7",
            "A9",
            "A10",
            "A12",
            "A13",
        ],
        "ordinal": [],
        "drop": [],
        "domain": "",
        "description": "",
        "collection_notes": "",
        "manual_tags": [],
    },

    "ctg-10classes": {
        "categorical": [],
        "ordinal": [
            "Tendency",
        ],
        "drop": [],
        "domain": "",
        "description": "",
        "collection_notes": "",
        "manual_tags": [],
    },

    "heart-disease": {
        "categorical": [
            "sex",
            "cp",
            "fbs",
            "restecg",
            "exang",
            "slope",
            "ca",
            "thal",
        ],
        "ordinal": [],
        "drop": [],
        "domain": "",
        "description": "",
        "collection_notes": "",
        "manual_tags": [],
    },

    "image-segmentation": {
        "categorical": [],
        "ordinal": [],
        "drop": [],
        "domain": "",
        "description": "",
        "collection_notes": "",
        "manual_tags": [],
    },

    "iris": {
        "categorical": [],
        "ordinal": [],
        "drop": [],
        "domain": "",
        "description": "",
        "collection_notes": "",
        "manual_tags": [],
    },

    "letter": {
        "categorical": [],
        "ordinal": [],
        "drop": [],
        "domain": "",
        "description": "",
        "collection_notes": "",
        "manual_tags": [],
    },

    "nursery": {
        "categorical": [
            "parents",
            "has_nurs",
            "form",
            "children",
            "housing",
            "finance",
            "social",
            "health",
        ],
        "ordinal": [],
        "drop": [],
        "target_merge": {
            "very_recom": "recommend",
        },
        "domain": "",
        "description": "",
        "collection_notes": "",
        "manual_tags": [],
    },

    "obesity": {
        "categorical": [
            "Gender",
            "family_history_with_overweight",
            "FAVC",
            "CAEC",
            "SMOKE",
            "SCC",
            "CALC",
            "MTRANS",
        ],
        "ordinal": [],
        "drop": [],
        "domain": "",
        "description": "",
        "collection_notes": "",
        "manual_tags": [],
    },

    "optical": {
        "categorical": [],
        "ordinal": [],
        "drop": [],
        "domain": "",
        "description": "",
        "collection_notes": "",
        "manual_tags": [],
    },

    "pendigits": {
        "categorical": [],
        "ordinal": [],
        "drop": [],
        "domain": "",
        "description": "",
        "collection_notes": "",
        "manual_tags": [],
    },

    "steel-plates": {
        "categorical": [
            "Outside_Global_Index",
        ],
        "ordinal": [],
        "drop": [
            # These two variables are complementary indicators.
            # Keeping one is enough to preserve the information.
            "TypeOfSteel_A400",
        ],
        "domain": "",
        "description": "",
        "collection_notes": "",
        "manual_tags": [],
    },

    "teaching-assistant-evaluation": {
        "categorical": [
            "native english speaking",
            "instructor",
            "course",
            "semester",
        ],
        "ordinal": [],
        "drop": [],
        "domain": "",
        "description": "",
        "collection_notes": "",
        "manual_tags": [],
    },

    "waveform": {
        "categorical": [],
        "ordinal": [],
        "drop": [],
        "domain": "",
        "description": "",
        "collection_notes": "",
        "manual_tags": [],
    },

    "wine-quality": {
        "categorical": [],
        "ordinal": [],
        "drop": [],
        "domain": "",
        "description": "",
        "collection_notes": "",
        "manual_tags": [],
    },

    "wine": {
        "categorical": [],
        "ordinal": [],
        "drop": [],
        "domain": "",
        "description": "",
        "collection_notes": "",
        "manual_tags": [],
    },

    "yeast": {
        "categorical": [],
        "ordinal": [],
        "drop": [],
        "domain": "",
        "description": "",
        "collection_notes": "",
        "manual_tags": [],
    },
}


# Fields deliberately intended for manual completion.
MANUAL_METADATA_FIELDS = [
    "domain",
    "description",
    "collection_notes",
    "manual_tags",
]


# =============================================================================
# Utility functions
# =============================================================================

def json_safe(value: Any) -> Any:
    """
    Convert common NumPy and pandas values into JSON-serialisable Python values.
    """
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if pd.isna(value):
        return None

    return value


def normalise_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace common textual missing-value markers with NaN.
    """
    return df.replace(MISSING_TOKENS, np.nan)


def remove_constant_columns(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove feature columns containing no variation.

    Missing values are ignored when counting distinct values.
    """
    feature_columns = [
        column
        for column in df.columns
        if column != target_column
    ]

    constant_columns = [
        column
        for column in feature_columns
        if df[column].nunique(dropna=True) <= 1
    ]

    return df.drop(columns=constant_columns), constant_columns


def normalised_class_entropy(y: pd.Series) -> float:
    """
    Calculate class entropy normalised to the range [0, 1].

    Higher values indicate a more even class distribution.
    """
    probabilities = y.value_counts(normalize=True).to_numpy(dtype=float)

    if len(probabilities) <= 1:
        return 0.0

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    maximum_entropy = log2(len(probabilities))

    if maximum_entropy == 0:
        return 0.0

    return float(entropy / maximum_entropy)


def sample_size_group(n_samples: int) -> str:
    if n_samples < 500:
        return "small"

    if n_samples < 5_000:
        return "medium"

    return "large"


def feature_count_group(n_features: int) -> str:
    if n_features <= 8:
        return "low"

    if n_features <= 20:
        return "medium"

    if n_features <= 40:
        return "high"

    return "very_high"


def class_count_group(n_classes: int) -> str:
    if n_classes <= 3:
        return "low"

    if n_classes <= 7:
        return "medium"

    if n_classes <= 10:
        return "high"

    return "very_high"


def imbalance_group(imbalance_ratio: float) -> str:
    if imbalance_ratio <= 1.5:
        return "balanced"

    if imbalance_ratio <= 3:
        return "moderate"

    if imbalance_ratio <= 10:
        return "high"

    return "extreme"


def datatype_group(
    numeric_feature_count: int,
    categorical_feature_count: int,
) -> str:
    if numeric_feature_count > 0 and categorical_feature_count > 0:
        return "mixed"

    if categorical_feature_count > 0:
        return "categorical"

    return "numeric"


def detect_binary_columns(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> list[str]:
    """
    Return all features containing exactly two distinct non-missing values.
    """
    return [
        column
        for column in feature_columns
        if df[column].nunique(dropna=True) == 2
    ]


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    """
    Build the leakage-safe preprocessing pipeline.

    Numeric:
        median imputation

    Categorical:
        most-frequent imputation
        one-hot encoding
        first category dropped
    """
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ])

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            )
        )

    if categorical_columns:
        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float32,
                ),
            ),
        ])

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError(
            "No numeric or categorical features were available."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def validate_dataset_config(
    dataset_name: str,
    df: pd.DataFrame,
    categorical_columns: list[str],
    ordinal_columns: list[str],
) -> None:
    """
    Confirm configured feature names exist and do not overlap incorrectly.
    """
    configured_columns = categorical_columns + ordinal_columns

    missing_columns = [
        column
        for column in configured_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name}: configured columns are absent from the CSV: "
            f"{missing_columns}"
        )

    overlap = sorted(
        set(categorical_columns).intersection(ordinal_columns)
    )

    if overlap:
        raise ValueError(
            f"{dataset_name}: columns cannot be both categorical and ordinal: "
            f"{overlap}"
        )


def find_blank_manual_metadata(
    dataset_name: str,
    config: dict[str, Any],
) -> list[str]:
    """
    Find metadata fields that still require manual completion.
    """
    blank_fields = []

    for field in MANUAL_METADATA_FIELDS:
        value = config.get(field)

        if value is None:
            blank_fields.append(field)

        elif isinstance(value, str) and not value.strip():
            blank_fields.append(field)

        elif isinstance(value, list) and len(value) == 0:
            blank_fields.append(field)

    if blank_fields:
        print(
            f"WARNING: {dataset_name} has blank manual metadata fields: "
            f"{blank_fields}"
        )

    return blank_fields


def warn_about_class_distribution(
    dataset_name: str,
    y: pd.Series,
) -> list[str]:
    """
    Produce warnings for rare target classes.
    """
    warning_messages: list[str] = []
    class_counts = y.value_counts()
    minimum_class_size = int(class_counts.min())

    if minimum_class_size < 2:
        warning_messages.append(
            f"{dataset_name}: at least one target class has fewer than two "
            "samples; stratified splitting is impossible."
        )

    elif minimum_class_size < 5:
        warning_messages.append(
            f"{dataset_name}: at least one target class has only "
            f"{minimum_class_size} samples; evaluation will be unstable."
        )

    elif minimum_class_size < 10:
        warning_messages.append(
            f"{dataset_name}: at least one target class has only "
            f"{minimum_class_size} samples; cross-validation options are "
            "limited."
        )

    for message in warning_messages:
        print(f"WARNING: {message}")

    return warning_messages


def train_test_split_with_checks(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create a stratified split and fail with an informative error if a class is
    too rare for the requested test size.
    """
    class_counts = y.value_counts()

    if class_counts.min() < 2:
        raise ValueError(
            "Stratified splitting requires at least two samples in every "
            "target class."
        )

    n_classes = y.nunique()
    requested_test_samples = int(np.ceil(len(y) * TEST_SIZE))

    if requested_test_samples < n_classes:
        raise ValueError(
            f"The requested test split would contain approximately "
            f"{requested_test_samples} rows, but the dataset has "
            f"{n_classes} classes. Increase TEST_SIZE or handle this dataset "
            "manually."
        )

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


# =============================================================================
# Dataset processing
# =============================================================================

def process_dataset(
    dataset_path: Path,
    output_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    dataset_name = dataset_path.stem

    if dataset_name not in DATASET_CONFIG:
        raise KeyError(
            f"No DATASET_CONFIG entry exists for '{dataset_name}'."
        )

    config = DATASET_CONFIG[dataset_name]
    dataset_warnings: list[str] = []

    print("\n" + "=" * 110)
    print(f"Processing dataset: {dataset_name}")
    print(f"Source: {dataset_path}")
    print("=" * 110)

    blank_metadata_fields = find_blank_manual_metadata(
        dataset_name,
        config,
    )

    if blank_metadata_fields:
        dataset_warnings.append(
            f"{dataset_name}: blank manual metadata fields: "
            f"{blank_metadata_fields}"
        )

    df = pd.read_csv(dataset_path)
    df = normalise_missing_values(df)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"{dataset_name}: required target column "
            f"'{TARGET_COLUMN}' was not found."
        )
    
    target_merge = config.get("target_merge", {})

    original_target_counts = (
        df[TARGET_COLUMN]
        .value_counts(dropna=False)
        .to_dict()
    )

    if target_merge:
        print(f"Applying target class merge: {target_merge}")

        df[TARGET_COLUMN] = df[TARGET_COLUMN].replace(
            target_merge
        )

    processed_target_counts = (
        df[TARGET_COLUMN]
        .value_counts(dropna=False)
        .to_dict()
    )

    source_row_count = len(df)
    source_column_count = len(df.columns)

    missing_target_count = int(
        df[TARGET_COLUMN].isna().sum()
    )

    if missing_target_count:
        message = (
            f"{dataset_name}: dropping {missing_target_count} rows with "
            "missing target values."
        )

        print(f"WARNING: {message}")
        dataset_warnings.append(message)

        df = (
            df.dropna(subset=[TARGET_COLUMN])
            .reset_index(drop=True)
        )

    explicit_drop = [
        column
        for column in config.get("drop", [])
        if column in df.columns
    ]

    configured_but_absent_drop_columns = [
        column
        for column in config.get("drop", [])
        if column not in df.columns
    ]

    if configured_but_absent_drop_columns:
        message = (
            f"{dataset_name}: configured drop columns were not found: "
            f"{configured_but_absent_drop_columns}"
        )

        print(f"WARNING: {message}")
        dataset_warnings.append(message)

    if explicit_drop:
        print(f"Explicitly dropping columns: {explicit_drop}")
        df = df.drop(columns=explicit_drop)

    df, constant_columns = remove_constant_columns(
        df,
        TARGET_COLUMN,
    )

    if constant_columns:
        print(f"Dropping constant columns: {constant_columns}")

    categorical_columns = list(
        config.get("categorical", [])
    )

    ordinal_columns = list(
        config.get("ordinal", [])
    )

    # Remove any configured columns that have already been explicitly or
    # automatically dropped.
    categorical_columns = [
        column
        for column in categorical_columns
        if column in df.columns
    ]

    ordinal_columns = [
        column
        for column in ordinal_columns
        if column in df.columns
    ]

    validate_dataset_config(
        dataset_name,
        df,
        categorical_columns,
        ordinal_columns,
    )

    feature_columns = [
        column
        for column in df.columns
        if column != TARGET_COLUMN
    ]

    numeric_columns = [
        column
        for column in feature_columns
        if column not in categorical_columns
    ]

    # Ensure integer-coded nominal variables are handled as categorical.
    for column in categorical_columns:
        df[column] = df[column].astype("object")

    X = df[feature_columns].copy()
    y_raw = df[TARGET_COLUMN].copy()

    class_warning_messages = warn_about_class_distribution(
        dataset_name,
        y_raw,
    )

    dataset_warnings.extend(class_warning_messages)

    # -------------------------------------------------------------------------
    # Characteristics before processing
    # -------------------------------------------------------------------------

    missing_values_by_column = X.isna().sum()
    missing_fraction_by_column = X.isna().mean()

    total_missing_values = int(
        missing_values_by_column.sum()
    )

    rows_with_missing_values = int(
        X.isna().any(axis=1).sum()
    )

    total_feature_cells = int(
        X.shape[0] * X.shape[1]
    )

    columns_with_missing_values = int(
        (missing_values_by_column > 0).sum()
    )

    binary_columns = detect_binary_columns(
        X,
        feature_columns,
    )

    continuous_columns = [
        column
        for column in numeric_columns
        if column not in ordinal_columns
        and column not in binary_columns
    ]

    discrete_numeric_columns = [
        column
        for column in numeric_columns
        if column not in ordinal_columns
        and column not in binary_columns
        and pd.api.types.is_integer_dtype(X[column])
    ]

    class_counts = y_raw.value_counts()
    n_classes = int(y_raw.nunique())

    smallest_class_size = int(class_counts.min())
    largest_class_size = int(class_counts.max())

    imbalance_ratio = float(
        largest_class_size / smallest_class_size
    )

    majority_class_fraction = float(
        largest_class_size / len(y_raw)
    )

    # -------------------------------------------------------------------------
    # Split before fitting preprocessing
    # -------------------------------------------------------------------------

    X_train, X_test, y_train_raw, y_test_raw = (
        train_test_split_with_checks(
            X,
            y_raw,
        )
    )

    preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    processed_feature_names = (
        preprocessor.get_feature_names_out().tolist()
    )

    # Fit the label encoder on the complete target so the mapping is an
    # explicit property of the dataset rather than of one split.
    label_encoder = LabelEncoder()
    label_encoder.fit(y_raw)

    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    train_df = pd.DataFrame(
        X_train_processed,
        columns=processed_feature_names,
    )

    test_df = pd.DataFrame(
        X_test_processed,
        columns=processed_feature_names,
    )

    train_df[TARGET_COLUMN] = y_train
    test_df[TARGET_COLUMN] = y_test

    processed_feature_count = len(
        processed_feature_names
    )

    raw_feature_count = len(feature_columns)

    categorical_expansion_factor = float(
        processed_feature_count / raw_feature_count
        if raw_feature_count
        else 0.0
    )

    target_mapping = {
        str(original_label): int(encoded_label)
        for encoded_label, original_label
        in enumerate(label_encoder.classes_)
    }

    # -------------------------------------------------------------------------
    # Output paths
    # -------------------------------------------------------------------------

    dataset_output_dir = output_root / dataset_name
    dataset_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = dataset_output_dir / "train.csv"
    test_path = dataset_output_dir / "test.csv"
    preprocessor_path = (
        dataset_output_dir / "preprocessor.joblib"
    )
    metadata_path = dataset_output_dir / "metadata.json"

    train_df.to_csv(
        train_path,
        index=False,
    )

    test_df.to_csv(
        test_path,
        index=False,
    )

    joblib.dump(
        {
            "preprocessor": preprocessor,
            "label_encoder": label_encoder,
            "raw_feature_columns": feature_columns,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "ordinal_columns": ordinal_columns,
            "processed_feature_names": processed_feature_names,
            "target_column": TARGET_COLUMN,
        },
        preprocessor_path,
    )

    # -------------------------------------------------------------------------
    # Complete metadata
    # -------------------------------------------------------------------------

    metadata: dict[str, Any] = {
        "identity": {
            "dataset": dataset_name,
            "source": "UCI",
            "source_path": str(dataset_path),
            "domain": config.get("domain", ""),
            "description": config.get("description", ""),
            "collection_notes": config.get(
                "collection_notes",
                "",
            ),
            "manual_tags": config.get(
                "manual_tags",
                [],
            ),
            "blank_manual_metadata_fields": (
                blank_metadata_fields
            ),
        },

        "source_structure": {
            "source_rows": int(source_row_count),
            "source_columns": int(source_column_count),
            "usable_rows": int(len(df)),
            "raw_feature_count": int(
                raw_feature_count
            ),
            "processed_feature_count": int(
                processed_feature_count
            ),
            "categorical_expansion_factor": (
                categorical_expansion_factor
            ),
        },

        "split": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "stratified": True,
        },

        "feature_characteristics": {
            "numeric_feature_count": int(
                len(numeric_columns)
            ),
            "categorical_feature_count": int(
                len(categorical_columns)
            ),
            "ordinal_feature_count": int(
                len(ordinal_columns)
            ),
            "binary_feature_count": int(
                len(binary_columns)
            ),
            "continuous_feature_count": int(
                len(continuous_columns)
            ),
            "discrete_numeric_feature_count": int(
                len(discrete_numeric_columns)
            ),

            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "ordinal_columns": ordinal_columns,
            "binary_columns": binary_columns,
            "continuous_columns": continuous_columns,
            "discrete_numeric_columns": (
                discrete_numeric_columns
            ),
            "raw_feature_columns": feature_columns,
            "processed_feature_names": (
                processed_feature_names
            ),

            "raw_feature_unique_counts": {
                column: int(
                    X[column].nunique(dropna=True)
                )
                for column in feature_columns
            },
        },

        "target_characteristics": {
            "class_count": n_classes,
            "target_dtype": str(y_raw.dtype),
            "target_mapping": target_mapping,
            "target_class_counts": {
                str(label): int(count)
                for label, count
                in class_counts.items()
            },
            "smallest_class_size": (
                smallest_class_size
            ),
            "largest_class_size": (
                largest_class_size
            ),
            "imbalance_ratio": imbalance_ratio,
            "majority_class_fraction": (
                majority_class_fraction
            ),
            "normalised_class_entropy": (
                normalised_class_entropy(y_raw)
            ),
            "merged_classes": target_merge,
            "original_class_count": int(
                len(original_target_counts)
            ),
            "processed_class_count": int(
                len(processed_target_counts)
            ),
            "original_class_counts": {
                str(label): int(count)
                for label, count in original_target_counts.items()
            },
            "processed_class_counts": {
                str(label): int(count)
                for label, count in processed_target_counts.items()
            },
        },

        "missing_data": {
            "missing_target_count": (
                missing_target_count
            ),
            "total_missing_feature_values": (
                total_missing_values
            ),
            "rows_with_missing_feature_values": (
                rows_with_missing_values
            ),
            "missing_row_fraction": float(
                rows_with_missing_values / len(X)
                if len(X)
                else 0.0
            ),
            "missing_cell_fraction": float(
                total_missing_values
                / total_feature_cells
                if total_feature_cells
                else 0.0
            ),
            "columns_with_missing_values": (
                columns_with_missing_values
            ),
            "maximum_feature_missing_fraction": float(
                missing_fraction_by_column.max()
                if len(missing_fraction_by_column)
                else 0.0
            ),
            "missing_values_by_column": {
                column: int(count)
                for column, count
                in missing_values_by_column.items()
                if count > 0
            },
            "missing_fraction_by_column": {
                column: float(fraction)
                for column, fraction
                in missing_fraction_by_column.items()
                if fraction > 0
            },
        },

        "preprocessing": {
            "numeric_imputation": "median",
            "categorical_imputation": (
                "most_frequent"
            ),
            "categorical_encoding": "one_hot",
            "one_hot_drop": "first",
            "unknown_category_handling": "ignore",
            "feature_scaling": (
                "not applied during dataset processing"
            ),
            "explicitly_dropped_columns": (
                explicit_drop
            ),
            "configured_but_absent_drop_columns": (
                configured_but_absent_drop_columns
            ),
            "constant_columns_removed": (
                constant_columns
            ),
        },

        "analysis_groups": {
            "sample_size_group": sample_size_group(
                len(df)
            ),
            "raw_feature_count_group": (
                feature_count_group(
                    raw_feature_count
                )
            ),
            "processed_feature_count_group": (
                feature_count_group(
                    processed_feature_count
                )
            ),
            "class_count_group": (
                class_count_group(n_classes)
            ),
            "datatype_group": datatype_group(
                len(numeric_columns),
                len(categorical_columns),
            ),
            "imbalance_group": imbalance_group(
                imbalance_ratio
            ),
        },

        "warnings": dataset_warnings,
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
            default=json_safe,
        )

    print(f"Raw rows:              {len(df)}")
    print(f"Train rows:            {len(train_df)}")
    print(f"Test rows:             {len(test_df)}")
    print(f"Raw features:          {raw_feature_count}")
    print(
        f"Processed features:    "
        f"{processed_feature_count}"
    )
    print(
        f"Categorical features:  "
        f"{len(categorical_columns)}"
    )
    print(
        f"Ordinal features:      "
        f"{len(ordinal_columns)}"
    )
    print(
        f"Missing feature cells: "
        f"{total_missing_values}"
    )
    print(f"Saved train data:      {train_path}")
    print(f"Saved test data:       {test_path}")
    print(f"Saved preprocessor:    {preprocessor_path}")
    print(f"Saved metadata:        {metadata_path}")

    return metadata, dataset_warnings


# =============================================================================
# Catalogue generation
# =============================================================================

def flatten_metadata_for_catalogue(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert nested metadata into one flat catalogue row suitable for CSV and
    later joining with MLflow results.
    """
    identity = metadata["identity"]
    structure = metadata["source_structure"]
    split = metadata["split"]
    features = metadata["feature_characteristics"]
    target = metadata["target_characteristics"]
    missing = metadata["missing_data"]
    preprocessing = metadata["preprocessing"]
    groups = metadata["analysis_groups"]

    return {
        "dataset": identity["dataset"],
        "source": identity["source"],
        "domain": identity["domain"],
        "description": identity["description"],
        "collection_notes": identity[
            "collection_notes"
        ],
        "manual_tags": json.dumps(
            identity["manual_tags"]
        ),
        "blank_manual_metadata_fields": json.dumps(
            identity[
                "blank_manual_metadata_fields"
            ]
        ),

        "source_rows": structure["source_rows"],
        "usable_rows": structure["usable_rows"],
        "train_rows": split["train_rows"],
        "test_rows": split["test_rows"],

        "raw_feature_count": structure[
            "raw_feature_count"
        ],
        "processed_feature_count": structure[
            "processed_feature_count"
        ],
        "categorical_expansion_factor": structure[
            "categorical_expansion_factor"
        ],

        "numeric_feature_count": features[
            "numeric_feature_count"
        ],
        "categorical_feature_count": features[
            "categorical_feature_count"
        ],
        "ordinal_feature_count": features[
            "ordinal_feature_count"
        ],
        "binary_feature_count": features[
            "binary_feature_count"
        ],
        "continuous_feature_count": features[
            "continuous_feature_count"
        ],
        "discrete_numeric_feature_count": features[
            "discrete_numeric_feature_count"
        ],

        "class_count": target["class_count"],
        "smallest_class_size": target[
            "smallest_class_size"
        ],
        "largest_class_size": target[
            "largest_class_size"
        ],
        "imbalance_ratio": target[
            "imbalance_ratio"
        ],
        "majority_class_fraction": target[
            "majority_class_fraction"
        ],
        "normalised_class_entropy": target[
            "normalised_class_entropy"
        ],

        "total_missing_feature_values": missing[
            "total_missing_feature_values"
        ],
        "rows_with_missing_feature_values": missing[
            "rows_with_missing_feature_values"
        ],
        "missing_row_fraction": missing[
            "missing_row_fraction"
        ],
        "missing_cell_fraction": missing[
            "missing_cell_fraction"
        ],
        "columns_with_missing_values": missing[
            "columns_with_missing_values"
        ],
        "maximum_feature_missing_fraction": missing[
            "maximum_feature_missing_fraction"
        ],

        "sample_size_group": groups[
            "sample_size_group"
        ],
        "raw_feature_count_group": groups[
            "raw_feature_count_group"
        ],
        "processed_feature_count_group": groups[
            "processed_feature_count_group"
        ],
        "class_count_group": groups[
            "class_count_group"
        ],
        "datatype_group": groups[
            "datatype_group"
        ],
        "imbalance_group": groups[
            "imbalance_group"
        ],

        "numeric_imputation": preprocessing[
            "numeric_imputation"
        ],
        "categorical_imputation": preprocessing[
            "categorical_imputation"
        ],
        "categorical_encoding": preprocessing[
            "categorical_encoding"
        ],
        "one_hot_drop": preprocessing[
            "one_hot_drop"
        ],
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_paths = sorted(
        DATASET_ROOT.glob("*.csv")
    )

    if not dataset_paths:
        raise FileNotFoundError(
            f"No CSV files were found in {DATASET_ROOT}"
        )

    metadata_records: list[dict[str, Any]] = []
    all_warnings: list[str] = []
    failures: list[dict[str, str]] = []

    configured_names = set(DATASET_CONFIG)
    discovered_names = {
        path.stem
        for path in dataset_paths
    }

    missing_configurations = sorted(
        discovered_names - configured_names
    )

    unused_configurations = sorted(
        configured_names - discovered_names
    )

    if missing_configurations:
        print(
            "WARNING: CSV files without DATASET_CONFIG entries: "
            f"{missing_configurations}"
        )

    if unused_configurations:
        print(
            "WARNING: DATASET_CONFIG entries without matching CSV files: "
            f"{unused_configurations}"
        )

    for dataset_path in dataset_paths:
        try:
            metadata, dataset_warnings = process_dataset(
                dataset_path,
                OUTPUT_ROOT,
            )

            metadata_records.append(metadata)
            all_warnings.extend(dataset_warnings)

        except Exception as exc:
            message = (
                f"{dataset_path.stem}: processing failed: {exc}"
            )

            print("\n" + "!" * 110)
            print(f"FAILED: {message}")
            print("!" * 110)

            failures.append({
                "dataset": dataset_path.stem,
                "path": str(dataset_path),
                "error": str(exc),
            })

            all_warnings.append(message)

    # -------------------------------------------------------------------------
    # Save global catalogue
    # -------------------------------------------------------------------------

    catalogue_rows = [
        flatten_metadata_for_catalogue(metadata)
        for metadata in metadata_records
    ]

    catalogue_df = pd.DataFrame(
        catalogue_rows
    )

    catalogue_path = (
        OUTPUT_ROOT / "dataset_catalogue.csv"
    )

    catalogue_df.to_csv(
        catalogue_path,
        index=False,
    )

    # Also save the catalogue as JSON to retain data types more reliably.
    catalogue_json_path = (
        OUTPUT_ROOT / "dataset_catalogue.json"
    )

    with catalogue_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata_records,
            file,
            indent=2,
            ensure_ascii=False,
            default=json_safe,
        )

    # -------------------------------------------------------------------------
    # Save warnings and failures
    # -------------------------------------------------------------------------

    warnings_path = (
        OUTPUT_ROOT / "processing_warnings.txt"
    )

    with warnings_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        if all_warnings:
            for warning_message in all_warnings:
                file.write(
                    f"{warning_message}\n"
                )
        else:
            file.write(
                "No processing warnings were generated.\n"
            )

    failures_path = (
        OUTPUT_ROOT / "processing_failures.json"
    )

    with failures_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            failures,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    print("\n" + "=" * 110)
    print("PROCESSING COMPLETE")
    print("=" * 110)

    print(
        f"Successfully processed: "
        f"{len(metadata_records)}"
    )

    print(
        f"Failed datasets:        "
        f"{len(failures)}"
    )

    print(
        f"Warnings generated:     "
        f"{len(all_warnings)}"
    )

    print(f"Dataset catalogue CSV:  {catalogue_path}")
    print(f"Dataset catalogue JSON: {catalogue_json_path}")
    print(f"Warnings file:          {warnings_path}")
    print(f"Failures file:          {failures_path}")

    if not catalogue_df.empty:
        display_columns = [
            "dataset",
            "usable_rows",
            "raw_feature_count",
            "processed_feature_count",
            "class_count",
            "datatype_group",
            "imbalance_group",
            "missing_cell_fraction",
        ]

        print("\nDataset catalogue summary:")
        print(
            catalogue_df[
                display_columns
            ].to_string(index=False)
        )

    blank_metadata_datasets = [
        metadata["identity"]["dataset"]
        for metadata in metadata_records
        if metadata["identity"][
            "blank_manual_metadata_fields"
        ]
    ]

    if blank_metadata_datasets:
        print("\n" + "!" * 110)
        print("MANUAL METADATA STILL REQUIRED")
        print("!" * 110)

        for metadata in metadata_records:
            dataset_name = metadata[
                "identity"
            ]["dataset"]

            blank_fields = metadata[
                "identity"
            ]["blank_manual_metadata_fields"]

            if blank_fields:
                print(
                    f"{dataset_name}: "
                    f"{', '.join(blank_fields)}"
                )


if __name__ == "__main__":
    # Make sklearn warnings visible during preprocessing, particularly warnings
    # about categories appearing in test data but not training data.
    warnings.simplefilter("default")

    main()