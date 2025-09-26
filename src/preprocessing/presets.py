from .preprocess import Preprocessor
from typing import List

AVAILABLE_COMBINATION = ["log_transform", "remove_correlated", "feature_engineer"]
def get_preset(preset_name: str, all_columns: List[str]):

    splits = preset_name.split("+")
    for split in splits:
        if split not in AVAILABLE_COMBINATION:
            raise Exception(f"{split} not among {AVAILABLE_COMBINATION}")
    

    preprocessor_kwargs = {
        "all_columns": all_columns,
        "remove_col_after_log": True,
        "cat_col_cut_off": 10,
        "cat_columns": [],
        "log_transform_cols": []
    }

    if "log_transform" in preset_name:
        preprocessor_kwargs["log_transform_cols"] = ["amount_tsh", "population"]
    
    if "remove_correlated" in preset_name:
        preprocessor_kwargs["cat_columns"] = ["installer", "wpt_name", "basin", "public_meeting", "scheme_management",  "permit", "extraction_type", "management", "payment", "water_quality", "quantity", "quantity_group", "waterpoint_type", "recorded_by"]
    else:
        preprocessor_kwargs["cat_columns"] = ["funder", "installer", "wpt_name", "basin", "subvillage", "region", "region_code", "district_code", "lga", "ward", "public_meeting", "scheme_management",  "permit", "extraction_type", "extraction_type_group", "extraction_type_class", "management", "management_group", "payment", "payment_type", "water_quality", "quality_group", "quantity", "quantity_group", "source", "source_type", "source_class", "waterpoint_type", "waterpoint_type_group", "recorded_by"]

    if "feature_engineer" in preset_name:
        preprocessor_kwargs["feature_engineer"] = True

    print(preprocessor_kwargs)
    preprocessor = Preprocessor(**preprocessor_kwargs)
    return preprocessor

if __name__ == "__main__":
    import pandas as pd

    train_values = pd.read_csv("data/train_set_values.csv")
    train_labels = pd.read_csv("data/train_set_labels.csv")
    test_values  = pd.read_csv("data/test_set_values.csv")

    # Merge labels into train only
    train_df = pd.merge(train_values, train_labels, on="id", how="left")

    preset_name = "log_transform+feature_engineer"
    pre = get_preset(preset_name, list(train_df.columns))

    train_processed = pre.fit_transform(train_df)
    test_processed  = pre.transform(test_values)

    print("Train shape:", train_processed.shape)
    print("Test  shape:", test_processed.shape)

    print(sum(train_processed.isna().sum()))
    print(sum(test_processed.isna().sum()))