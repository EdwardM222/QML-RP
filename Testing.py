from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

UCI_DATASET_TIERS = {
    # Tier 0: tiny datasets for quick pipeline validation
    0: {
        "iris": 53,
        "wine": 109,
    },

    # Tier 1: fast validation on larger multiclass datasets
    1: {
        "balance-scale": 12,
        "image-segmentation": 50,
        "waveform": 107,
        "wine-quality-white": 186,
    },

    # Tier 2: main thesis comparison datasets
    2: {
        # "pendigits": 81,
        # "yeast": 110,
        # "ctg-10classes": 193,
        "steel-plates": 198,
    },

    # Tier 3: stretch/scalability datasets
    3: {
        "letter": 59,
        "nursery": 76,
        "optical": 80,
    },
}

for tier in [2]:
    for dataset_name, uci_id in UCI_DATASET_TIERS[tier].items():
        dataset = fetch_ucirepo(id=uci_id)

        X = dataset.data.features
        y = dataset.data.targets.iloc[:, 0]

        if dataset_name == "steel-plates":
            y = pd.Series(dataset.data.targets.apply(lambda row:dataset.data.targets.columns[row.argmax()], axis=1))

        df = pd.concat([X, y.rename("target")], axis=1)

        if not os.path.exists(f"datasets/{tier}"):
            if not os.path.exists("datasets"):
                os.makedirs("datasets")
            os.makedirs(f"datasets/{tier}")

        df.to_csv(f"datasets/{tier}/{dataset_name}.csv", index=False)

# # df from txt file
# df = pd.read_csv("datasets/misc/seeds_dataset.txt", sep="\t", header=None)
# df.columns = ["area", "perimeter", "compactness", "length_of_kernel", "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove", "target"]
# df.to_csv("datasets/0/seeds.csv", index=False)