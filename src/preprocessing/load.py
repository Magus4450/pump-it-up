import pandas as pd
from .presets import get_preset

def get_processed_data(preset_name: str, n_cols= 10):
    train_values = pd.read_csv("data/train_set_values.csv")
    train_labels = pd.read_csv("data/train_set_labels.csv")
    test_values  = pd.read_csv("data/test_set_values.csv")

    # Merge labels into train only
    train_df = pd.merge(train_values, train_labels, on="id", how="left")

    # preset_name = "log_transform+remove_correlated+feature_engineer"
    pre = get_preset(preset_name, list(train_df.columns), n_cols)

    train_processed = pre.fit_transform(train_df)
    test_processed  = pre.transform(test_values, test=True)

    # print("Train shape:", train_processed.shape)
    # print("Test  shape:", test_processed.shape)

    return train_processed, test_processed, pre
