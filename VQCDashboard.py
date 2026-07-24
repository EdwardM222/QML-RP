import argparse

import pandas as pd
import plotly.express as px
import streamlit as st


parser = argparse.ArgumentParser()

parser.add_argument("--results-path", required=True)
parser.add_argument("--rankings-path", required=True)
parser.add_argument("--runs-path", required=True)

args = parser.parse_args()


@st.cache_data
def load_data(results_path, rankings_path, runs_path):
    return (
        pd.read_parquet(results_path),
        pd.read_parquet(rankings_path),
        pd.read_parquet(runs_path),
    )


results, rankings, runs = load_data(
    args.results_path,
    args.rankings_path,
    args.runs_path,
)


st.set_page_config(
    page_title="VQC Grid Search",
    layout="wide",
)

st.title("VQC Grid Search Analysis")


page = st.sidebar.radio(
    "View",
    [
        "Rankings",
        "Comparison",
        "Distributions",
    ],
)


if page == "Rankings":
    dataset = st.sidebar.selectbox(
        "Dataset",
        ["Overall"] + sorted(results["dataset"].unique().tolist()),
    )

    top_n = st.sidebar.slider(
        "Number of configurations",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
    )

    if dataset == "Overall":
        ranking_data = rankings.head(top_n).copy()

        ranking_data["configuration"] = (
            ranking_data["ansatz"].astype(str)
            + " | q="
            + ranking_data["n_qubits"].astype(str)
            + " | density="
            + ranking_data["feature_density"].astype(str)
            + " | "
            + ranking_data["feature_range"].astype(str)
        )

        figure = px.bar(
            ranking_data.sort_values(
                "average_rank",
                ascending=False,
            ),
            x="average_rank",
            y="configuration",
            orientation="h",
            hover_data=[
                "overall_rank",
                "average_percentile",
                "best_rank",
                "worst_rank",
                "macro_f1_mean",
                "macro_f1_repeat_std",
            ],
            title="Overall VQC rankings",
        )

        st.plotly_chart(
            figure,
            width='stretch',
        )

        st.dataframe(
            ranking_data,
            width='stretch',
            hide_index=True,
        )

    else:
        ranking_data = (
            results[
                results["dataset"].astype(str) == dataset
            ]
            .sort_values("rank")
            .head(top_n)
            .copy()
        )

        ranking_data["configuration"] = (
            ranking_data["ansatz"].astype(str)
            + " | q="
            + ranking_data["n_qubits"].astype(str)
            + " | density="
            + ranking_data["feature_density"].astype(str)
            + " | "
            + ranking_data["feature_range"].astype(str)
        )

        figure = px.bar(
            ranking_data.sort_values(
                "macro_f1_mean",
                ascending=True,
            ),
            x="macro_f1_mean",
            y="configuration",
            orientation="h",
            error_x="macro_f1_std",
            hover_data=[
                "rank",
                "rank_percentile",
                "accuracy_mean",
                "macro_f1_min",
                "macro_f1_max",
                "repetitions",
            ],
            title=f"{dataset} VQC rankings",
        )

        st.plotly_chart(
            figure,
            width='stretch',
        )

        st.dataframe(
            ranking_data,
            width='stretch',
            hide_index=True,
        )


elif page == "Comparison":
    numeric_columns = [
        column
        for column in results.select_dtypes(
            include="number"
        ).columns
        if column not in [
            "dataset_configurations",
        ]
    ]

    categorical_columns = [
        "dataset",
        "ansatz",
        "n_qubits",
        "feature_density",
        "feature_range",
        "feats_per_qubit",
        "reuploads",
        "encoding_style",
        "feature_strategy",
        "trainable_layers",
        "entangling_pattern",
        "entangler",
    ]

    datasets = st.sidebar.multiselect(
        "Datasets",
        sorted(results["dataset"].unique().tolist()),
        default=sorted(results["dataset"].unique().tolist()),
    )

    x_column = st.sidebar.selectbox(
        "X-axis",
        numeric_columns,
        index=numeric_columns.index("n_params"),
    )

    y_column = st.sidebar.selectbox(
        "Y-axis",
        numeric_columns,
        index=numeric_columns.index("macro_f1_mean"),
    )

    colour_column = st.sidebar.selectbox(
        "Colour",
        categorical_columns,
        index=categorical_columns.index("encoding_style"),
    )

    plot_data = results[
        results["dataset"].isin(datasets)
    ].copy()

    figure = px.scatter(
        plot_data,
        x=x_column,
        y=y_column,
        color=colour_column,
        hover_data=[
            "dataset",
            "ansatz",
            "n_qubits",
            "feature_density",
            "feature_range",
            "macro_f1_std",
            "rank",
        ],
        title=f"{y_column} against {x_column}",
    )

    st.plotly_chart(
        figure,
        width='stretch',
    )

    st.dataframe(
        plot_data,
        width='stretch',
        hide_index=True,
    )


elif page == "Distributions":
    dataset = st.sidebar.selectbox(
        "Dataset",
        sorted(runs["dataset"].unique().tolist()),
    )

    metric = st.sidebar.selectbox(
        "Metric",
        [
            "macro_f1",
            "accuracy",
            "weighted_f1",
            "training_time",
        ],
    )

    top_n = st.sidebar.slider(
        "Top configurations",
        min_value=5,
        max_value=50,
        value=15,
        step=5,
    )

    top_ids = (
        results[
            results["dataset"].astype(str) == dataset
        ]
        .sort_values("rank")
        .head(top_n)["run_id"]
    )

    distribution_data = runs[
        runs["run_id"].isin(top_ids)
    ].copy()

    distribution_data["configuration"] = (
        distribution_data["ansatz"].astype(str)
        + " | q="
        + distribution_data["n_qubits"].astype(str)
        + " | density="
        + distribution_data["feature_density"].astype(str)
    )

    figure = px.box(
        distribution_data,
        x=metric,
        y="configuration",
        orientation="h",
        points="all",
        hover_data=[
            "search_repeat",
            "feature_range",
            "encoding_style",
            "entangler",
        ],
        title=f"{dataset} {metric} distributions",
    )

    st.plotly_chart(
        figure,
        width='stretch',
    )

    st.dataframe(
        distribution_data,
        width='stretch',
        hide_index=True,
    )