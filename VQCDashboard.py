import argparse

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


parser = argparse.ArgumentParser()
parser.add_argument("--results-path", required=True)
parser.add_argument("--rankings-path", required=True)
parser.add_argument("--runs-path", required=True)
parser.add_argument("--ansatz-results-path", required=True)
parser.add_argument("--ansatz-dataset-results-path", required=True)
args = parser.parse_args()


@st.cache_data
def load_data(
    results_path,
    rankings_path,
    runs_path,
    ansatz_results_path,
    ansatz_dataset_results_path,
):
    return (
        pd.read_parquet(results_path),
        pd.read_parquet(rankings_path),
        pd.read_parquet(runs_path),
        pd.read_parquet(ansatz_results_path),
        pd.read_parquet(ansatz_dataset_results_path),
    )



def calculate_feature_importance(
    data,
    feature_columns,
    target_column,
    n_estimators,
    max_depth,
    min_samples_leaf,
    max_features,
    permutation_repeats,
    max_rows_per_dataset,
    random_state,
    n_jobs,
    progress_bar,
    progress_status,
):
    sampled = []

    for _, dataset_data in data.groupby("dataset", observed=True):
        if len(dataset_data) > max_rows_per_dataset:
            dataset_data = dataset_data.sample(max_rows_per_dataset, random_state=random_state)

        sampled.append(dataset_data)

    data = pd.concat(sampled, ignore_index=True)
    X = data[feature_columns]
    y = pd.to_numeric(data[target_column], errors="coerce")
    groups = data["dataset"].astype(str)

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]
    groups = groups.loc[valid]

    categorical_columns = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(X[column])
    ]
    numeric_columns = [
        column for column in feature_columns if column not in categorical_columns
    ]

    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("numeric", "passthrough", numeric_columns),
    ])

    forest = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None if max_depth == 0 else max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("forest", forest),
    ])

    fold_scores = []
    permutation_rows = []
    impurity_rows = []
    splitter = LeaveOneGroupOut()

    splits = list(splitter.split(X, y, groups))
    total_steps = len(splits) * 2
    completed_steps = 0

    for fold, (train_indices, test_indices) in enumerate(splits, start=1):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]
        held_out_dataset = groups.iloc[test_indices].iloc[0]

        progress_status.write(
            f"Fold {fold}/{len(splits)}: training with "
            f"{held_out_dataset} held out..."
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        completed_steps += 1
        progress_bar.progress(completed_steps / total_steps)

        fold_scores.append({
            "fold": fold,
            "held_out_dataset": held_out_dataset,
            "train_rows": len(train_indices),
            "test_rows": len(test_indices),
            "r2": r2_score(y_test, predictions),
            "mae": mean_absolute_error(y_test, predictions),
        })

        progress_status.write(
            f"Fold {fold}/{len(splits)}: calculating permutation importance "
            f"for {held_out_dataset}..."
        )

        permutation = permutation_importance(
            pipeline,
            X_test,
            y_test,
            scoring="r2",
            n_repeats=permutation_repeats,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        completed_steps += 1
        progress_bar.progress(completed_steps / total_steps)

        for feature, mean, std in zip(
            feature_columns,
            permutation.importances_mean,
            permutation.importances_std,
        ):
            permutation_rows.append({
                "fold": fold,
                "held_out_dataset": held_out_dataset,
                "feature": feature,
                "importance": mean,
                "importance_std": std,
            })

        fitted_preprocessor = pipeline.named_steps["preprocessor"]
        encoded_names = fitted_preprocessor.get_feature_names_out()
        encoded_importances = pipeline.named_steps["forest"].feature_importances_

        for feature in feature_columns:
            prefix = (
                f"categorical__{feature}_"
                if feature in categorical_columns
                else f"numeric__{feature}"
            )

            importance = sum(
                value
                for name, value in zip(encoded_names, encoded_importances)
                if name == prefix or name.startswith(prefix)
            )

            impurity_rows.append({
                "fold": fold,
                "held_out_dataset": held_out_dataset,
                "feature": feature,
                "importance": importance,
            })

    fold_scores = pd.DataFrame(fold_scores)
    permutation_results = pd.DataFrame(permutation_rows)
    impurity_results = pd.DataFrame(impurity_rows)

    permutation_summary = (
        permutation_results
        .groupby("feature", observed=True)
        .agg(
            importance_mean=("importance", "mean"),
            importance_std=("importance", "std"),
            minimum_importance=("importance", "min"),
            maximum_importance=("importance", "max"),
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index()
    )

    impurity_summary = (
        impurity_results
        .groupby("feature", observed=True)
        .agg(
            importance_mean=("importance", "mean"),
            importance_std=("importance", "std"),
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index()
    )

    return (
        fold_scores,
        permutation_results,
        permutation_summary,
        impurity_results,
        impurity_summary,
        len(data),
    )


def plot_size_controls(key_prefix, default_height=650):
    st.sidebar.subheader("Plot size")

    stretch_width = st.sidebar.checkbox(
        "Stretch plot width",
        value=True,
        key=f"{key_prefix}_stretch_width",
    )

    plot_width = "stretch"

    if not stretch_width:
        plot_width = st.sidebar.slider(
            "Plot width",
            500,
            1800,
            1100,
            50,
            key=f"{key_prefix}_plot_width",
        )

    plot_height = st.sidebar.slider(
        "Plot height",
        300,
        1400,
        default_height,
        50,
        key=f"{key_prefix}_plot_height",
    )

    return plot_width, plot_height

def normalise(series, reverse=False):
    series = pd.to_numeric(series, errors="coerce")
    minimum = series.min()
    maximum = series.max()

    if pd.isna(minimum) or maximum == minimum:
        values = pd.Series(1.0, index=series.index)
    else:
        values = (series - minimum) / (maximum - minimum)

    return 1 - values if reverse else values


def add_tradeoff_score(
    data,
    performance_column,
    stability_column,
    performance_weight,
    stability_weight,
    qubit_weight,
    parameter_weight,
    time_weight,
):
    data = data.copy()

    data["performance_score"] = normalise(data[performance_column])
    data["stability_score"] = normalise(data[stability_column], reverse=True)
    data["qubit_score"] = normalise(data["n_qubits"], reverse=True)
    data["parameter_score"] = normalise(data["n_params"], reverse=True)
    data["time_score"] = normalise(data["training_time_mean"], reverse=True)

    total_weight = (
        performance_weight
        + stability_weight
        + qubit_weight
        + parameter_weight
        + time_weight
    )

    if total_weight == 0:
        total_weight = 1

    data["tradeoff_score"] = (
        performance_weight * data["performance_score"]
        + stability_weight * data["stability_score"]
        + qubit_weight * data["qubit_score"]
        + parameter_weight * data["parameter_score"]
        + time_weight * data["time_score"]
    ) / total_weight

    return data


def pareto_frontier(data, performance_column, stability_column, pool_size):
    if data.empty:
        return data

    data = data.nlargest(min(pool_size, len(data)), performance_column).copy()

    pareto_values = pd.DataFrame({
        "performance": normalise(data[performance_column]),
        "stability": normalise(data[stability_column], reverse=True),
        "qubits": normalise(data["n_qubits"], reverse=True),
        "parameters": normalise(data["n_params"], reverse=True),
    }).fillna(0).to_numpy()

    keep = np.ones(len(data), dtype=bool)

    for i, point in enumerate(pareto_values):
        dominates = (
            np.all(pareto_values >= point, axis=1)
            & np.any(pareto_values > point, axis=1)
        )

        if dominates.any():
            keep[i] = False

    return data.loc[keep].copy()


def apply_filters(data, filter_columns, key_prefix):
    data = data.copy()

    active_filters = st.sidebar.multiselect(
        "Active filters",
        [column for column in filter_columns if column in data.columns],
        key=f"{key_prefix}_active_filters",
    )

    for column in active_filters:
        series = data[column]

        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").dropna()

            if values.empty:
                continue

            minimum = float(values.min())
            maximum = float(values.max())

            if minimum == maximum:
                st.sidebar.caption(f"{column}: {minimum:g}")
                continue

            selected_range = st.sidebar.slider(
                column,
                min_value=minimum,
                max_value=maximum,
                value=(minimum, maximum),
                key=f"{key_prefix}_{column}",
            )

            data = data[
                pd.to_numeric(data[column], errors="coerce").between(
                    selected_range[0],
                    selected_range[1],
                )
            ]

        else:
            options = sorted(data[column].dropna().astype(str).unique().tolist())

            selected_values = st.sidebar.multiselect(
                column,
                options,
                default=options,
                key=f"{key_prefix}_{column}",
            )

            data = data[data[column].astype(str).isin(selected_values)]

    return data


def select_rows(data, method, metric, n_rows, ascending, random_state=2):
    if data.empty or method == "All filtered":
        return data

    n_rows = min(n_rows, len(data))

    if method == "Top by selection metric":
        return data.sort_values(metric, ascending=ascending).head(n_rows)

    if method == "Bottom by selection metric":
        return data.sort_values(metric, ascending=not ascending).head(n_rows)

    if method == "Random sample":
        return data.sample(n_rows, random_state=random_state)

    if method == "Even sample":
        ordered = data.sort_values(metric, ascending=ascending)

        if n_rows == 1:
            return ordered.head(1)

        indices = np.linspace(0, len(ordered) - 1, n_rows).astype(int)
        return ordered.iloc[indices]

    return data.head(n_rows)


def add_configuration_label(data):
    data = data.copy()

    if "feature_density" in data.columns:
        data["configuration"] = (
            data["ansatz"].astype(str)
            + " | q="
            + data["n_qubits"].astype(str)
            + " | density="
            + data["feature_density"].astype(str)
            + " | "
            + data["feature_range"].astype(str)
        )
    elif "configurations" in data.columns:
        data["configuration"] = (
            data["ansatz"].astype(str)
            + " | configs="
            + data["configurations"].astype(str)
        )
    else:
        data["configuration"] = data["ansatz"].astype(str)

    return data


results, rankings, runs, ansatz_results, ansatz_dataset_results = load_data(
    args.results_path,
    args.rankings_path,
    args.runs_path,
    args.ansatz_results_path,
    args.ansatz_dataset_results_path,
)


st.set_page_config(page_title="VQC Grid Search", layout="wide")
st.title("VQC Grid Search Analysis")

page = st.sidebar.radio(
    "Section",
    ["Rankings", "Comparison", "Distributions", "Feature importance"],
)


# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------

if page == "Rankings":
    st.header("Candidate rankings")

    datasets = sorted(results["dataset"].dropna().astype(str).unique())

    ranking_view = st.sidebar.selectbox(
        "Ranking scope",
        ["Overall configurations", "Overall ansatze"]
        + datasets
        + [f"Ansatze: {dataset}" for dataset in datasets],
    )

    if ranking_view == "Overall configurations":
        ranking_data = rankings.copy()

        metric_options = {
            "Average percentile": ("average_percentile", False),
            "Average rank": ("average_rank", True),
            "Median rank": ("median_rank", True),
            "Worst rank": ("worst_rank", True),
            "Mean macro-F1": ("macro_f1_mean", False),
            "Repeat variation": ("macro_f1_repeat_std", True),
            "Training time": ("training_time_mean", True),
            "Parameter count": ("n_params", True),
        }

        performance_column = "average_percentile"
        stability_column = "macro_f1_repeat_std"

    elif ranking_view == "Overall ansatze":
        ranking_data = ansatz_results.copy()

        metric_options = {
            "Average percentile": ("average_percentile", False),
            "Average rank": ("average_rank", True),
            "Median rank": ("median_rank", True),
            "Worst rank": ("worst_rank", True),
            "Mean macro-F1": ("macro_f1_mean", False),
            "Repeat variation": ("macro_f1_repeat_std", True),
            "Training time": ("training_time_mean", True),
        }

        performance_column = "average_percentile"
        stability_column = "macro_f1_repeat_std"

    elif ranking_view.startswith("Ansatze: "):
        dataset = ranking_view.removeprefix("Ansatze: ")
        ranking_data = ansatz_dataset_results[
            ansatz_dataset_results["dataset"].astype(str) == dataset
        ].copy()

        metric_options = {
            "Rank percentile": ("ansatz_rank_percentile", False),
            "Dataset rank": ("ansatz_rank", True),
            "Mean configuration percentile": ("average_percentile", False),
            "Mean macro-F1": ("macro_f1_mean", False),
            "Macro-F1 variation": ("macro_f1_std", True),
            "Repeat variation": ("repeat_std_mean", True),
            "Training time": ("training_time_mean", True),
        }

        performance_column = "ansatz_rank_percentile"
        stability_column = "repeat_std_mean"

    else:
        ranking_data = results[
            results["dataset"].astype(str) == ranking_view
        ].copy()

        metric_options = {
            "Rank percentile": ("rank_percentile", False),
            "Dataset rank": ("rank", True),
            "Mean macro-F1": ("macro_f1_mean", False),
            "Macro-F1 variation": ("macro_f1_std", True),
            "Mean accuracy": ("accuracy_mean", False),
            "Weighted F1": ("weighted_f1_mean", False),
            "Training time": ("training_time_mean", True),
            "Parameter count": ("n_params", True),
        }

        performance_column = "rank_percentile"
        stability_column = "macro_f1_std"

    filter_columns = [
        "datasets",
        "repetitions",
        "n_qubits",
        "feature_density",
        "feature_range",
        "measurement_mode",
        "feats_per_qubit",
        "reuploads",
        "encoding_style",
        "feature_strategy",
        "trainable_layers",
        "n_trainable_layers",
        "entangling_pattern",
        "entangler",
        "n_params",
        "n_layers",
        "trainable_params",
        "training_time_mean",
        performance_column,
        stability_column,
    ]

    with st.sidebar.expander("Candidate filters", expanded=True):
        ranking_data = apply_filters(
            ranking_data,
            filter_columns,
            "rankings",
        )

    if "n_qubits" not in ranking_data and "n_qubits_mean" in ranking_data:
        ranking_data["n_qubits"] = ranking_data["n_qubits_mean"]
    if "n_params" not in ranking_data and "n_params_mean" in ranking_data:
        ranking_data["n_params"] = ranking_data["n_params_mean"]

    selection_mode = st.sidebar.selectbox(
        "Candidate selection",
        [
            "Best performance",
            "Best stability",
            "Best trade-off",
            "Pareto frontier",
            "Custom metric",
        ],
    )

    performance_weight = 0.55
    stability_weight = 0.20
    qubit_weight = 0.15
    parameter_weight = 0.10
    time_weight = 0.00

    if selection_mode in ["Best trade-off", "Pareto frontier"]:
        st.sidebar.subheader("Trade-off weights")

        performance_weight = st.sidebar.slider(
            "Performance",
            0.0,
            1.0,
            performance_weight,
            0.05,
        )

        stability_weight = st.sidebar.slider(
            "Stability",
            0.0,
            1.0,
            stability_weight,
            0.05,
        )

        qubit_weight = st.sidebar.slider(
            "Fewer qubits",
            0.0,
            1.0,
            qubit_weight,
            0.05,
        )

        parameter_weight = st.sidebar.slider(
            "Fewer parameters",
            0.0,
            1.0,
            parameter_weight,
            0.05,
        )

        time_weight = st.sidebar.slider(
            "Lower training time",
            0.0,
            1.0,
            time_weight,
            0.05,
        )

        ranking_data = add_tradeoff_score(
            ranking_data,
            performance_column,
            stability_column,
            performance_weight,
            stability_weight,
            qubit_weight,
            parameter_weight,
            time_weight,
        )

    if selection_mode == "Best performance":
        selection_metric = performance_column
        selection_ascending = False

    elif selection_mode == "Best stability":
        selection_metric = stability_column
        selection_ascending = True

    elif selection_mode == "Best trade-off":
        selection_metric = "tradeoff_score"
        selection_ascending = False

    elif selection_mode == "Pareto frontier":
        pareto_pool_size = st.sidebar.number_input(
            "Pareto candidate pool",
            min_value=100,
            max_value=max(100, len(ranking_data)),
            value=min(5000, max(100, len(ranking_data))),
            step=100,
        )

        ranking_data = pareto_frontier(
            ranking_data,
            performance_column,
            stability_column,
            pareto_pool_size,
        )

        ranking_data = add_tradeoff_score(
            ranking_data,
            performance_column,
            stability_column,
            performance_weight,
            stability_weight,
            qubit_weight,
            parameter_weight,
            time_weight,
        )

        selection_metric = "tradeoff_score"
        selection_ascending = False

    else:
        metric_name = st.sidebar.selectbox(
            "Ranking metric",
            list(metric_options),
        )

        selection_metric, selection_ascending = metric_options[metric_name]

    ranking_data = ranking_data.sort_values(
        selection_metric,
        ascending=selection_ascending,
    )

    st.sidebar.subheader("Shortlist")

    n_candidates = st.sidebar.number_input(
        "Configurations to shortlist",
        min_value=1,
        max_value=max(1, len(ranking_data)),
        value=min(50, max(1, len(ranking_data))),
        step=10,
    )

    use_diversity_limit = st.sidebar.checkbox(
        "Limit similar configurations",
        value=True,
    )

    shortlist = ranking_data.copy()

    if use_diversity_limit and not shortlist.empty:
        diversity_column = st.sidebar.selectbox(
            "Diversity grouping",
            [
                "ansatz",
                "encoding_style",
                "n_qubits",
                "reuploads",
                "entangler",
                "feature_density",
            ],
        )

        max_per_group = st.sidebar.number_input(
            "Maximum per group",
            min_value=1,
            max_value=n_candidates,
            value=min(3, n_candidates),
        )

        shortlist = (
            shortlist
            .groupby(diversity_column, observed=True, dropna=False)
            .head(max_per_group)
        )

    shortlist = shortlist.head(n_candidates).copy()
    shortlist = add_configuration_label(shortlist)

    metric_columns = [
        column
        for column in ranking_data.select_dtypes(include="number").columns
        if column not in ["overall_rank"]
    ]

    colour_options = [
        "None",
        "encoding_style",
        "n_qubits",
        "reuploads",
        "entangler",
        "feature_range",
        "feature_density",
    ]

    st.sidebar.subheader("Ranking plot")

    show_plot = st.sidebar.checkbox("Show candidate plot", value=True)

    if show_plot and not shortlist.empty:
        x_column = st.sidebar.selectbox(
            "X-axis",
            metric_columns,
            index=metric_columns.index("n_params")
            if "n_params" in metric_columns else 0,
        )

        y_default = (
            "tradeoff_score"
            if "tradeoff_score" in metric_columns
            else performance_column
        )

        y_column = st.sidebar.selectbox(
            "Y-axis",
            metric_columns,
            index=metric_columns.index(y_default),
        )

        colour_column = st.sidebar.selectbox(
            "Colour",
            colour_options,
        )

        size_options = ["None"] + metric_columns
        size_column = st.sidebar.selectbox("Point size", size_options)

        plot_limit = st.sidebar.number_input(
            "Configurations to plot",
            min_value=1,
            max_value=max(1, len(shortlist)),
            value=min(100, max(1, len(shortlist))),
        )

        plot_data = shortlist.head(plot_limit)

        plot_width, plot_height = plot_size_controls(
            "rankings",
            default_height=650,
        )

        figure = px.scatter(
            plot_data,
            x=x_column,
            y=y_column,
            color=None if colour_column == "None" else colour_column,
            size=None if size_column == "None" else size_column,
            hover_name="configuration",
            hover_data=[
                performance_column,
                stability_column,
                "n_qubits",
                "n_params",
                "training_time_mean",
            ],
            render_mode="webgl",
            opacity=0.7,
            title=f"{y_column} against {x_column}",
        )

        figure.update_layout(height=plot_height)
        st.plotly_chart(figure, width=plot_width)

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    metric_1.metric("Filtered configurations", f"{len(ranking_data):,}")
    metric_2.metric("Shortlisted", f"{len(shortlist):,}")

    if not shortlist.empty:
        metric_3.metric(
            "Best performance",
            f"{shortlist[performance_column].max():.4f}",
        )

        metric_4.metric(
            "Median qubits",
            f"{shortlist['n_qubits'].median():.1f}",
        )

    table_rows = st.sidebar.number_input(
        "Table rows",
        min_value=1,
        max_value=max(1, len(ranking_data)),
        value=min(100, max(1, len(ranking_data))),
        step=25,
    )

    show_all_rows = st.sidebar.checkbox("Show all filtered rows")

    table_height = st.sidebar.slider(
        "Table height",
        300,
        1200,
        700,
        100,
    )

    table_data = (
        ranking_data
        if show_all_rows
        else ranking_data.head(table_rows)
    )

    table_data = add_configuration_label(table_data)

    preferred_columns = [
        "configuration",
        selection_metric,
        performance_column,
        stability_column,
        "n_qubits",
        "n_params",
        "n_layers",
        "trainable_params",
        "training_time_mean",
        "encoding_style",
        "reuploads",
        "entangler",
        "feature_density",
        "feature_range",
    ]

    preferred_columns = list(dict.fromkeys(
        column
        for column in preferred_columns
        if column in table_data.columns
    ))

    st.subheader("Filtered ranking table")

    st.dataframe(
        table_data[preferred_columns],
        width='stretch',
        hide_index=True,
        height=table_height,
    )

    st.download_button(
        "Download shortlist",
        shortlist.to_csv(index=False),
        file_name="vqc_shortlist.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download filtered rankings",
        ranking_data.to_csv(index=False),
        file_name="vqc_filtered_rankings.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

elif page == "Comparison":
    st.header("Configuration comparison")

    comparison_scope = st.sidebar.selectbox(
        "Comparison scope",
        ["Overall rankings", "Dataset results", "Overall ansatze", "Ansatze by dataset"],
    )

    if comparison_scope == "Overall rankings":
        comparison_data = rankings.copy()
        default_y = "average_percentile"
        default_selection_metric = "average_percentile"

    elif comparison_scope == "Dataset results":
        comparison_data = results.copy()

        selected_datasets = st.sidebar.multiselect(
            "Datasets",
            sorted(results["dataset"].dropna().astype(str).unique()),
            default=sorted(results["dataset"].dropna().astype(str).unique()),
        )

        comparison_data = comparison_data[
            comparison_data["dataset"].astype(str).isin(selected_datasets)
        ]

        default_y = "rank_percentile"
        default_selection_metric = "rank_percentile"

    elif comparison_scope == "Overall ansatze":
        comparison_data = ansatz_results.copy()
        comparison_data["n_qubits"] = comparison_data["n_qubits_mean"]
        comparison_data["n_params"] = comparison_data["n_params_mean"]
        default_y = "average_percentile"
        default_selection_metric = "average_percentile"

    else:
        comparison_data = ansatz_dataset_results.copy()

        selected_datasets = st.sidebar.multiselect(
            "Datasets",
            sorted(ansatz_dataset_results["dataset"].dropna().astype(str).unique()),
            default=sorted(ansatz_dataset_results["dataset"].dropna().astype(str).unique()),
        )

        comparison_data = comparison_data[
            comparison_data["dataset"].astype(str).isin(selected_datasets)
        ]
        comparison_data["n_qubits"] = comparison_data["n_qubits_mean"]
        comparison_data["n_params"] = comparison_data["n_params_mean"]
        default_y = "ansatz_rank_percentile"
        default_selection_metric = "ansatz_rank_percentile"

    filter_columns = [
        "dataset",
        "datasets",
        "repetitions",
        "n_qubits",
        "feature_density",
        "feature_range",
        "measurement_mode",
        "feats_per_qubit",
        "reuploads",
        "encoding_style",
        "feature_strategy",
        "trainable_layers",
        "n_trainable_layers",
        "entangling_pattern",
        "entangler",
        "n_params",
        "n_layers",
        "trainable_params",
        "training_time_mean",
        default_y,
    ]

    with st.sidebar.expander("Comparison filters", expanded=True):
        comparison_data = apply_filters(
            comparison_data,
            filter_columns,
            "comparison",
        )

    numeric_columns = comparison_data.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = [
        column
        for column in [
            "dataset",
            "ansatz",
            "n_qubits",
            "feature_density",
            "feature_range",
            "measurement_mode",
            "feats_per_qubit",
            "reuploads",
            "encoding_style",
            "feature_strategy",
            "trainable_layers",
            "n_trainable_layers",
            "entangling_pattern",
            "entangler",
        ]
        if column in comparison_data.columns
    ]

    plot_type = st.sidebar.selectbox(
        "Plot type",
        ["Scatter", "Box", "Violin", "Bar", "Histogram"],
    )

    selection_metric = st.sidebar.selectbox(
        "Point selection metric",
        numeric_columns,
        index=numeric_columns.index(default_selection_metric),
    )

    lower_is_better = st.sidebar.checkbox(
        "Lower selection metric is better",
        value=selection_metric in [
            "rank",
            "average_rank",
            "median_rank",
            "worst_rank",
            "macro_f1_std",
            "macro_f1_repeat_std",
            "training_time_mean",
            "n_params",
        ],
    )

    selection_method = st.sidebar.selectbox(
        "Point selection",
        [
            "Top by selection metric",
            "Bottom by selection metric",
            "Random sample",
            "Even sample",
            "All filtered",
        ],
    )

    max_points = st.sidebar.number_input(
        "Maximum configurations",
        min_value=1,
        max_value=max(1, len(comparison_data)),
        value=min(500, max(1, len(comparison_data))),
        step=100,
    )

    plot_data = select_rows(
        comparison_data,
        selection_method,
        selection_metric,
        max_points,
        ascending=lower_is_better,
    )

    colour_options = ["None"] + categorical_columns
    facet_options = ["None"] + categorical_columns

    colour_column = st.sidebar.selectbox(
        "Colour",
        colour_options,
        index=colour_options.index("encoding_style")
        if "encoding_style" in colour_options else 0,
    )

    facet_row = st.sidebar.selectbox("Facet row", facet_options)
    facet_column = st.sidebar.selectbox("Facet column", facet_options)

    hover_options = [
        column
        for column in [
            "dataset",
            "ansatz",
            "n_qubits",
            "feature_density",
            "feature_range",
            "reuploads",
            "encoding_style",
            "entangler",
            "n_params",
            "n_layers",
            "training_time_mean",
            "macro_f1_mean",
            "macro_f1_std",
            "rank",
            "rank_percentile",
            "average_rank",
            "average_percentile",
        ]
        if column in plot_data.columns
    ]

    hover_columns = st.sidebar.multiselect(
        "Hover details",
        hover_options,
        default=hover_options[: min(8, len(hover_options))],
    )

    if plot_type == "Scatter":
        x_options = numeric_columns + [
            column
            for column in categorical_columns
            if column not in numeric_columns
        ]

        x_column = st.sidebar.selectbox(
            "X-axis",
            x_options,
            index=x_options.index("n_params")
            if "n_params" in x_options else 0,
        )

        y_column = st.sidebar.selectbox(
            "Y-axis",
            numeric_columns,
            index=numeric_columns.index(default_y),
        )

        size_options = ["None"] + numeric_columns
        size_column = st.sidebar.selectbox("Point size", size_options)

        opacity = st.sidebar.slider(
            "Point opacity",
            0.1,
            1.0,
            0.5,
            0.05,
        )

        figure = px.scatter(
            plot_data,
            x=x_column,
            y=y_column,
            color=None if colour_column == "None" else colour_column,
            size=None if size_column == "None" else size_column,
            facet_row=None if facet_row == "None" else facet_row,
            facet_col=None if facet_column == "None" else facet_column,
            hover_data=hover_columns,
            render_mode="webgl",
            opacity=opacity,
            title=f"{y_column} against {x_column}",
        )

    elif plot_type in ["Box", "Violin"]:
        x_column = st.sidebar.selectbox(
            "Category axis",
            categorical_columns,
        )

        y_column = st.sidebar.selectbox(
            "Value axis",
            numeric_columns,
            index=numeric_columns.index(default_y),
        )

        show_points = st.sidebar.selectbox(
            "Individual points",
            ["False", "outliers", "all"],
        )

        points = False if show_points == "False" else show_points

        plot_function = px.box if plot_type == "Box" else px.violin

        figure = plot_function(
            plot_data,
            x=x_column,
            y=y_column,
            color=None if colour_column == "None" else colour_column,
            facet_row=None if facet_row == "None" else facet_row,
            facet_col=None if facet_column == "None" else facet_column,
            points=points,
            hover_data=hover_columns,
            title=f"{y_column} by {x_column}",
        )

    elif plot_type == "Bar":
        x_column = st.sidebar.selectbox(
            "Category axis",
            categorical_columns,
        )

        y_column = st.sidebar.selectbox(
            "Value axis",
            numeric_columns,
            index=numeric_columns.index(default_y),
        )

        aggregation = st.sidebar.selectbox(
            "Aggregation",
            ["mean", "median", "min", "max"],
        )

        group_columns = [x_column]

        if colour_column != "None" and colour_column != x_column:
            group_columns.append(colour_column)

        if aggregation == "mean":
            bar_data = (
                plot_data
                .groupby(group_columns, observed=True, dropna=False)[y_column]
                .mean()
                .reset_index()
            )

        elif aggregation == "median":
            bar_data = (
                plot_data
                .groupby(group_columns, observed=True, dropna=False)[y_column]
                .median()
                .reset_index()
            )

        elif aggregation == "min":
            bar_data = (
                plot_data
                .groupby(group_columns, observed=True, dropna=False)[y_column]
                .min()
                .reset_index()
            )

        else:
            bar_data = (
                plot_data
                .groupby(group_columns, observed=True, dropna=False)[y_column]
                .max()
                .reset_index()
            )

        figure = px.bar(
            bar_data,
            x=x_column,
            y=y_column,
            color=None if colour_column == "None" else colour_column,
            barmode=st.sidebar.selectbox(
                "Bar mode",
                ["group", "stack", "relative"],
            ),
            title=f"{aggregation} {y_column} by {x_column}",
        )

    else:
        x_column = st.sidebar.selectbox(
            "Histogram variable",
            numeric_columns,
            index=numeric_columns.index(default_y),
        )

        n_bins = st.sidebar.slider(
            "Histogram bins",
            5,
            200,
            40,
            5,
        )

        figure = px.histogram(
            plot_data,
            x=x_column,
            color=None if colour_column == "None" else colour_column,
            facet_row=None if facet_row == "None" else facet_row,
            facet_col=None if facet_column == "None" else facet_column,
            nbins=n_bins,
            barmode=st.sidebar.selectbox(
                "Histogram mode",
                ["overlay", "group", "stack"],
            ),
            opacity=st.sidebar.slider(
                "Bar opacity",
                0.1,
                1.0,
                0.6,
                0.05,
            ),
            title=f"Distribution of {x_column}",
        )

    plot_width, plot_height = plot_size_controls(
        "comparison",
        default_height=650,
    )

    figure.update_layout(height=plot_height)

    st.caption(
        f"Plotting {len(plot_data):,} of "
        f"{len(comparison_data):,} filtered configurations."
    )

    st.plotly_chart(figure, width=plot_width)

    show_table = st.sidebar.checkbox("Show comparison table", value=True)

    if show_table:
        table_rows = st.sidebar.number_input(
            "Comparison table rows",
            min_value=1,
            max_value=max(1, len(plot_data)),
            value=min(250, max(1, len(plot_data))),
            step=50,
        )

        st.dataframe(
            plot_data.head(table_rows),
            width='stretch',
            hide_index=True,
            height=st.sidebar.slider(
                "Comparison table height",
                300,
                1200,
                600,
                100,
            ),
        )

    st.download_button(
        "Download plotted configurations",
        plot_data.to_csv(index=False),
        file_name="vqc_comparison.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

elif page == "Distributions":
    st.header("Repeated-run distributions")

    dataset = st.sidebar.selectbox(
        "Dataset",
        sorted(results["dataset"].dropna().astype(str).unique()),
    )

    dataset_results = results[
        results["dataset"].astype(str) == dataset
    ].copy()

    filter_columns = [
        "repetitions",
        "n_qubits",
        "feature_density",
        "feature_range",
        "measurement_mode",
        "feats_per_qubit",
        "reuploads",
        "encoding_style",
        "feature_strategy",
        "trainable_layers",
        "n_trainable_layers",
        "entangling_pattern",
        "entangler",
        "n_params",
        "n_layers",
        "trainable_params",
        "training_time_mean",
        "macro_f1_mean",
        "macro_f1_std",
        "rank",
        "rank_percentile",
    ]

    with st.sidebar.expander("Distribution filters", expanded=True):
        dataset_results = apply_filters(
            dataset_results,
            filter_columns,
            "distributions",
        )

    selection_metrics = {
        "Rank": ("rank", True),
        "Rank percentile": ("rank_percentile", False),
        "Mean macro-F1": ("macro_f1_mean", False),
        "Macro-F1 stability": ("macro_f1_std", True),
        "Accuracy": ("accuracy_mean", False),
        "Weighted F1": ("weighted_f1_mean", False),
        "Parameter count": ("n_params", True),
        "Training time": ("training_time_mean", True),
    }

    selection_name = st.sidebar.selectbox(
        "Configuration selection metric",
        list(selection_metrics),
    )

    selection_metric, selection_ascending = selection_metrics[selection_name]

    n_configurations = st.sidebar.number_input(
        "Configurations to compare",
        min_value=1,
        max_value=max(1, len(dataset_results)),
        value=min(20, max(1, len(dataset_results))),
        step=5,
    )

    selected_results = (
        dataset_results
        .sort_values(selection_metric, ascending=selection_ascending)
        .head(n_configurations)
        .copy()
    )

    selected_runs = runs[
        runs["run_id"].isin(selected_results["run_id"])
    ].copy()

    selected_runs = add_configuration_label(selected_runs)

    metric_options = [
        column
        for column in [
            "macro_f1",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "weighted_f1",
            "training_time",
            "final_train_loss",
            "best_train_loss",
            "final_val_loss",
            "best_val_loss",
        ]
        if column in selected_runs.columns
    ]

    metric = st.sidebar.selectbox("Distribution metric", metric_options)

    plot_type = st.sidebar.selectbox(
        "Distribution plot",
        ["Box", "Violin"],
    )

    orientation = st.sidebar.selectbox(
        "Orientation",
        ["Horizontal", "Vertical"],
    )

    point_display = st.sidebar.selectbox(
        "Individual repeat points",
        ["all", "outliers", "False"],
    )

    points = False if point_display == "False" else point_display

    sort_method = st.sidebar.selectbox(
        "Configuration order",
        ["Selection rank", "Mean metric", "Median metric", "Variation"],
    )

    if sort_method == "Mean metric":
        order = (
            selected_runs
            .groupby("configuration", observed=True)[metric]
            .mean()
            .sort_values(ascending=False)
            .index
            .tolist()
        )

    elif sort_method == "Median metric":
        order = (
            selected_runs
            .groupby("configuration", observed=True)[metric]
            .median()
            .sort_values(ascending=False)
            .index
            .tolist()
        )

    elif sort_method == "Variation":
        order = (
            selected_runs
            .groupby("configuration", observed=True)[metric]
            .std()
            .sort_values()
            .index
            .tolist()
        )

    else:
        selection_order = selected_results["run_id"].tolist()

        order = (
            selected_runs[["run_id", "configuration"]]
            .drop_duplicates()
            .set_index("run_id")
            .reindex(selection_order)["configuration"]
            .dropna()
            .tolist()
        )

    plot_function = px.box if plot_type == "Box" else px.violin

    if orientation == "Horizontal":
        figure = plot_function(
            selected_runs,
            x=metric,
            y="configuration",
            category_orders={"configuration": order[::-1]},
            points=points,
            hover_data=[
                "search_repeat",
                "ansatz",
                "n_qubits",
                "feature_density",
                "feature_range",
                "encoding_style",
                "entangler",
            ],
            title=f"{dataset}: {metric} distributions",
        )

    else:
        figure = plot_function(
            selected_runs,
            x="configuration",
            y=metric,
            category_orders={"configuration": order},
            points=points,
            hover_data=[
                "search_repeat",
                "ansatz",
                "n_qubits",
                "feature_density",
                "feature_range",
                "encoding_style",
                "entangler",
            ],
            title=f"{dataset}: {metric} distributions",
        )

    plot_width, plot_height = plot_size_controls(
        "distributions",
        default_height=800,
    )

    figure.update_layout(height=plot_height)

    st.caption(
        f"Showing {len(selected_results):,} configurations and "
        f"{len(selected_runs):,} repeated runs."
    )

    st.plotly_chart(figure, width=plot_width)

    summary = (
        selected_runs
        .groupby("configuration", observed=True)[metric]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )

    st.subheader("Distribution summary")

    st.dataframe(
        summary,
        width='stretch',
        hide_index=True,
        height=st.sidebar.slider(
            "Summary table height",
            300,
            1000,
            600,
            100,
        ),
    )

    st.download_button(
        "Download selected configurations",
        selected_results.to_csv(index=False),
        file_name=f"{dataset}_vqc_candidates.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download repeated-run data",
        selected_runs.to_csv(index=False),
        file_name=f"{dataset}_vqc_repetitions.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

else:
    st.header("Control feature importance")

    st.caption(
        "A Random Forest predicts configuration performance from original grid "
        "controls only. Each fold trains on all but one dataset and tests on the "
        "held-out dataset. Dataset and ansatz are not used as model features."
    )

    control_columns = [
        "n_qubits",
        "feature_density",
        "feature_range",
        "measurement_mode",
        "feats_per_qubit",
        "reuploads",
        "encoding_style",
        "feature_strategy",
        "trainable_layers",
        "entangling_pattern",
        "entangler",
    ]

    available_controls = [
        column for column in control_columns if column in results.columns
    ]

    selected_controls = st.sidebar.multiselect(
        "Control features",
        available_controls,
        default=available_controls,
    )

    target_options = {
        "Rank percentile": "rank_percentile",
        "Mean macro-F1": "macro_f1_mean",
        "Mean accuracy": "accuracy_mean",
        "Weighted F1": "weighted_f1_mean",
    }

    target_name = st.sidebar.selectbox("Prediction target", list(target_options))
    target_column = target_options[target_name]

    st.sidebar.subheader("Random Forest")

    n_estimators = st.sidebar.slider("Trees", 50, 1000, 300, 50)
    max_depth = st.sidebar.slider(
        "Maximum depth", 0, 50, 0, help="0 means unlimited depth."
    )
    min_samples_leaf = st.sidebar.slider("Minimum samples per leaf", 1, 100, 5)
    max_features = st.sidebar.selectbox(
        "Features considered per split",
        ["sqrt", "log2", 1.0, 0.75, 0.5],
        index=0,
    )

    st.sidebar.subheader("Evaluation")

    permutation_repeats = st.sidebar.slider("Permutation repeats", 3, 30, 10)
    largest_dataset = int(results.groupby("dataset", observed=True).size().max())

    max_rows_per_dataset = st.sidebar.number_input(
        "Maximum rows per dataset",
        min_value=500,
        max_value=max(500, largest_dataset),
        value=min(5000, max(500, largest_dataset)),
        step=500,
        help="Each dataset is sampled independently before validation.",
    )

    n_jobs = st.sidebar.number_input(
        "Parallel workers", min_value=1, max_value=32, value=4
    )
    random_state = st.sidebar.number_input("Random seed", min_value=0, value=2)

    run_analysis = st.sidebar.button(
        "Run feature importance",
        type="primary",
        disabled=not selected_controls,
    )

    if run_analysis:
        progress_status = st.empty()
        progress_bar = st.progress(0)
        progress_status.write("Preparing feature-importance data...")

        st.session_state["feature_importance_results"] = calculate_feature_importance(
            results[["dataset", target_column] + selected_controls].copy(),
            selected_controls,
            target_column,
            n_estimators,
            max_depth,
            min_samples_leaf,
            max_features,
            permutation_repeats,
            max_rows_per_dataset,
            random_state,
            n_jobs,
            progress_bar,
            progress_status,
        )

        progress_bar.progress(1.0)
        progress_status.success("Feature-importance analysis complete.")

        st.session_state["feature_importance_settings"] = {
            "features": selected_controls,
            "target": target_column,
        }

    if "feature_importance_results" not in st.session_state:
        st.info(
            "Choose the controls and model settings, then run the analysis. "
            "The existing Parquet files do not need to be regenerated."
        )
    else:
        (
            fold_scores,
            permutation_results,
            permutation_summary,
            impurity_results,
            impurity_summary,
            sampled_rows,
        ) = st.session_state["feature_importance_results"]

        saved_settings = st.session_state["feature_importance_settings"]

        if (
            saved_settings["features"] != selected_controls
            or saved_settings["target"] != target_column
        ):
            st.warning(
                "The displayed results use the previous feature or target selection. "
                "Run the analysis again to apply the current settings."
            )

        metric_1, metric_2, metric_3, metric_4 = st.columns(4)
        metric_1.metric("Sampled rows", f"{sampled_rows:,}")
        metric_2.metric("Held-out datasets", f"{len(fold_scores):,}")
        metric_3.metric("Mean held-out R²", f"{fold_scores['r2'].mean():.4f}")
        metric_4.metric("Mean held-out MAE", f"{fold_scores['mae'].mean():.4f}")

        importance_method = st.sidebar.selectbox(
            "Importance method",
            ["Permutation", "Random Forest impurity"],
        )

        show_features = st.sidebar.number_input(
            "Features to show",
            min_value=1,
            max_value=max(1, len(permutation_summary)),
            value=len(permutation_summary),
        )

        if importance_method == "Permutation":
            importance_data = permutation_summary.head(show_features).copy()
            title = "Leave-one-dataset-out permutation importance"
            explanation = (
                "Decrease in held-out R² after shuffling each control. Larger "
                "positive values indicate greater predictive importance."
            )
        else:
            importance_data = impurity_summary.head(show_features).copy()
            title = "Random Forest impurity importance"
            explanation = (
                "Mean tree impurity reduction attributed to each original control "
                "after combining its one-hot encoded columns."
            )

        importance_data = importance_data.sort_values(
            "importance_mean", ascending=True
        )

        figure = px.bar(
            importance_data,
            x="importance_mean",
            y="feature",
            orientation="h",
            error_x="importance_std",
            title=title,
        )
        figure.update_layout(
            height=st.sidebar.slider("Importance plot height", 400, 1200, 650, 50)
        )

        st.caption(explanation)
        st.plotly_chart(figure, width="stretch")

        st.subheader("Importance by held-out dataset")

        dataset_method = st.sidebar.selectbox(
            "Dataset importance method",
            ["Permutation", "Random Forest impurity"],
        )
        dataset_importance = (
            permutation_results
            if dataset_method == "Permutation"
            else impurity_results
        )

        heatmap_data = dataset_importance.pivot(
            index="feature",
            columns="held_out_dataset",
            values="importance",
        )
        feature_order = (
            dataset_importance
            .groupby("feature", observed=True)["importance"]
            .mean()
            .sort_values(ascending=False)
            .index
        )
        heatmap_data = heatmap_data.reindex(feature_order)

        heatmap = px.imshow(
            heatmap_data,
            aspect="auto",
            labels={
                "x": "Held-out dataset",
                "y": "Control feature",
                "color": "Importance",
            },
            title=f"{dataset_method} importance by held-out dataset",
        )
        heatmap_width, heatmap_height = plot_size_controls(
            "feature_heatmap",
            default_height=550,
        )

        heatmap.update_layout(height=heatmap_height)
        st.plotly_chart(heatmap, width=heatmap_width)

        st.subheader("Held-out dataset scores")

        score_metric = st.sidebar.selectbox("Score plot", ["r2", "mae"])
        score_data = fold_scores.sort_values(
            score_metric,
            ascending=score_metric == "mae",
        )
        score_figure = px.bar(
            score_data,
            x="held_out_dataset",
            y=score_metric,
            hover_data=["train_rows", "test_rows"],
            title=f"{score_metric.upper()} by held-out dataset",
        )
        score_width, score_height = plot_size_controls(
            "feature_scores",
            default_height=500,
        )

        score_figure.update_layout(height=score_height)
        st.plotly_chart(score_figure, width=score_width)

        table_tabs = st.tabs([
            "Permutation summary",
            "Impurity summary",
            "Fold scores",
            "Permutation by dataset",
        ])

        with table_tabs[0]:
            st.dataframe(permutation_summary, width="stretch", hide_index=True)

        with table_tabs[1]:
            st.dataframe(impurity_summary, width="stretch", hide_index=True)

        with table_tabs[2]:
            st.dataframe(fold_scores, width="stretch", hide_index=True)

        with table_tabs[3]:
            st.dataframe(permutation_results, width="stretch", hide_index=True)

        st.download_button(
            "Download feature importance summary",
            permutation_summary.to_csv(index=False),
            file_name="vqc_feature_importance.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download per-dataset feature importance",
            permutation_results.to_csv(index=False),
            file_name="vqc_feature_importance_by_dataset.csv",
            mime="text/csv",
        )

