import pandas as pd
from typing import List, Dict, Any
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class Preprocessor:
    """
    Work similar to adding transformations and imputer in the skleanr pipeline.\
        Fits in the train data, transforms train and test data
    """

    def __init__(
        self,
        log_transform_cols: List[str],
        remove_col_after_log: bool = False,
        cat_columns: List[str] = None,
        cat_col_cut_off: int = 10,
        feature_engineer: bool = False
    ):
        self.log_transform_cols = log_transform_cols
        self.remove_col_after_log = remove_col_after_log
        self.cat_columns = cat_columns
        self.cat_col_cut_off = cat_col_cut_off
        self.feature_engineer = feature_engineer

        # learned during fit()
        self.global_modes: Dict[str, Any] = {}
        self.mode_by_group: Dict[str, Dict[Any, Any]] = {}  # key="x|y"
        self.latlon_mean_by_loc: pd.DataFrame | None = None
        self.latlon_fallback_by_district: pd.DataFrame | None = None
        self.latlon_global_mean: Dict[str, float] = {}
        self.bad_lat = -2.000000e-08
        self.bad_lon = 0.0
        self.col_replacer = {}

        # remember columns removed during fit to mirror on transform
        self.columns_dropped: List[str] = []

    # -------------------- public API -------------------- #

    def fit(self, train_df: pd.DataFrame) -> "Preprocessor":
        df = train_df.copy(deep=True)

        # 1) Learn global modes for simple categorical imputations
        for col in ["funder", "installer", "wpt_name"]:
            if col in df.columns:
                self.global_modes[col] = self._safe_mode(df[col])

        # 2) Learn mode(x) per group y for selected pairs
        self._learn_mode_by_group(df, x="public_meeting", y="management")
        self._learn_mode_by_group(df, x="scheme_management", y="management")
        self._learn_mode_by_group(df, x="permit", y="management")
        self._learn_mode_by_group(df, x="subvillage", y="district_code")

        # 3) Learn lat/lon means per (district_code, ward), with bad coords excluded
        loc_cols = ["district_code", "ward"]
        has_geo = all(c in df.columns for c in ["latitude", "longitude"]) and all(
            c in df.columns for c in loc_cols
        )
        if has_geo:
            mask_bad = (df["latitude"] == self.bad_lat) & (df["longitude"] == self.bad_lon)
            group_means = (
                df.loc[~mask_bad]
                .groupby(loc_cols)[["latitude", "longitude"]]
                .mean()
                .reset_index()
            )
            self.latlon_mean_by_loc = group_means

            # fallback by district_code
            still_ok_mask = ~mask_bad & df["latitude"].notna() & df["longitude"].notna()
            district_means = (
                df.loc[still_ok_mask]
                .groupby(["district_code"])[["latitude", "longitude"]]
                .mean()
                .reset_index()
            )
            self.latlon_fallback_by_district = district_means

            # global mean as last resort
            self.latlon_global_mean = {
                "latitude": df.loc[still_ok_mask, "latitude"].mean(),
                "longitude": df.loc[still_ok_mask, "longitude"].mean(),
            }

        # 4) Decide which columns to drop (mirror at transform)
        #    scheme_name and population were dropped in your original code
        for col in ["scheme_name", "population", "gps_height", "id"]:
            if col in df.columns:
                self.columns_dropped.append(col)

        # 5) For features having more than certain classes, take top k and replace with "Other"
        for col in self.cat_columns:
            self.col_replacer[col] = self._get_col_replacer(df[col])

        # print(self.col_replacer)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy(deep=True)

        # A) Simple categorical imputations with global modes
        for col in ["funder", "installer", "wpt_name"]:
            if col in df.columns and col in self.global_modes:
                df[col].fillna(self.global_modes[col], inplace=True)

        # B) Per-group mode imputations using learned stats (with sensible fallbacks)
        self._apply_mode_by_group(df, x="public_meeting", y="management")
        self._apply_mode_by_group(df, x="scheme_management", y="management")
        self._apply_mode_by_group(df, x="permit", y="management")
        self._apply_mode_by_group(df, x="subvillage", y="district_code")

        # C) Drop high-NA or undesired columns (mirror training decision)
        for col in self.columns_dropped:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # D) Fix lat/lon using learned means
        self._apply_long_lat_fix(df)

        # E) Log transforms (computed directly on current df, column-wise)
        self._log_transform(df)

        # F) One Hot Encode columns

        dummies_lst = []
        for col in self.cat_columns:
            df[col] = df[col].apply(self.col_replacer[col])
            # print(df[col].nunique())

            dummies = pd.get_dummies(df[col], prefix=f"{col}_", drop_first=True)
            dummies_lst.append(dummies)

            df.drop(columns=[col], inplace=True)

        df = pd.concat([df, *dummies_lst], axis=1)

        # G) Feature Engineering
        if self.feature_engineer:
            self._feature_engineering(df)
        return df
    

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)

    # -------------------- internal helpers -------------------- #

    @staticmethod
    def _safe_mode(s: pd.Series):
        try:
            m = s.mode(dropna=True)
            if m.empty:
                return None
            return m.iloc[0]
        except Exception:
            return None

    def _learn_mode_by_group(self, df: pd.DataFrame, x: str, y: str):
        if x not in df.columns or y not in df.columns:
            return
        # compute mode(x) per y
        mode_by_y = (
            df.groupby(y)[x]
            .agg(lambda col: self._safe_mode(col))
            .to_dict()
        )
        # save plus global fallback
        key = f"{x}|{y}"
        self.mode_by_group[key] = mode_by_y
        self.global_modes.setdefault(x, self._safe_mode(df[x]))

    def _apply_mode_by_group(self, df: pd.DataFrame, x: str, y: str):
        if x not in df.columns or y not in df.columns:
            return
        key = f"{x}|{y}"
        mapping = self.mode_by_group.get(key, {})
        global_mode = self.global_modes.get(x, None)

        # only fill NaNs
        def _fill(row):
            if pd.isna(row[x]):
                group_val = row[y]
                return mapping.get(group_val, global_mode)
            return row[x]

        df[x] = df.apply(_fill, axis=1)

    def _apply_long_lat_fix(self, df: pd.DataFrame):
        needed = ["latitude", "longitude", "district_code", "ward"]
        if not all(c in df.columns for c in needed):
            return  # nothing to do

        mask_bad = (df["latitude"] == self.bad_lat) & (df["longitude"] == self.bad_lon)

        # Merge learned per-(district_code, ward) means
        if self.latlon_mean_by_loc is not None:
            df = df.merge(
                self.latlon_mean_by_loc,
                on=["district_code", "ward"],
                how="left",
                suffixes=("", "_mean"),
            )
            # replace only the bad ones
            df.loc[mask_bad, "latitude"] = df.loc[mask_bad, "latitude_mean"]
            df.loc[mask_bad, "longitude"] = df.loc[mask_bad, "longitude_mean"]
            df.drop(columns=["latitude_mean", "longitude_mean"], inplace=True)

        # Any remaining NaNs/bad -> district fallback
        still_bad = df["latitude"].isna() | df["longitude"].isna()
        if still_bad.any() and self.latlon_fallback_by_district is not None:
            df = df.merge(
                self.latlon_fallback_by_district,
                on=["district_code"],
                how="left",
                suffixes=("", "_fallback"),
            )
            df.loc[still_bad, "latitude"] = df.loc[still_bad, "latitude_fallback"]
            df.loc[still_bad, "longitude"] = df.loc[still_bad, "longitude_fallback"]
            df.drop(columns=["latitude_fallback", "longitude_fallback"], inplace=True)

        # Last resort: global mean if still missing
        still_bad = df["latitude"].isna() | df["longitude"].isna()
        if still_bad.any() and self.latlon_global_mean:
            df.loc[still_bad, "latitude"] = df.loc[still_bad, "latitude"].fillna(
                self.latlon_global_mean.get("latitude", np.nan)
            )
            df.loc[still_bad, "longitude"] = df.loc[still_bad, "longitude"].fillna(
                self.latlon_global_mean.get("longitude", np.nan)
            )

        # write back (in case df got re-assigned by merges)
        # (function returns nothing; df is mutated in-place via reference)
        return

    def _log_transform(self, df: pd.DataFrame):
        for col in self.log_transform_cols:
            if col in df.columns:
                df[df[col] < 0 ][col] = 0
                df[f"log_{col}"] = np.log1p(df[col].astype(float))
                if self.remove_col_after_log and col in df.columns:
                    df.drop(columns=[col], inplace=True)

    def _get_col_replacer(self, col: pd.Series):
        vcs = col.value_counts()
        top_vcs = vcs.head(self.cat_col_cut_off)

        replace_w_other = lambda x: x if x in top_vcs.index else 'Other'
        
        return replace_w_other
    

    def _create_age_col(self, df: pd.DataFrame):
        df["construction_year"] = df["construction_year"].astype(int)
        df["date_recorded"] = pd.to_datetime(df["date_recorded"], format="%Y-%m-%d")
        df["age"] = df["construction_year"] - df["date_recorded"].dt.year

        df.drop(columns=["construction_year", "date_recorded"], inplace=True)

    def _feature_engineering(self, df):
        self._create_age_col(df)


# -------------------- Example usage -------------------- #
if __name__ == "__main__":
    # Load
    train_values = pd.read_csv("data/train_set_values.csv")
    train_labels = pd.read_csv("data/train_set_labels.csv")
    test_values  = pd.read_csv("data/test_set_values.csv")

    # Merge labels into train only
    train_df = pd.merge(train_values, train_labels, on="id", how="left")

    # Configure
    pre = Preprocessor(
        log_transform_cols=["amount_tsh", "population"],
        remove_col_after_log=False,   # keep originals unless you want to drop
        cat_columns=["funder", "installer", "wpt_name", "basin", "subvillage", "region", "region_code", "district_code", "lga", "ward", "public_meeting", "scheme_management",  "permit", "extraction_type", "extraction_type_group", "extraction_type_class", "management", "management_group", "payment", "payment_type", "water_quality", "quality_group", "quantity", "quantity_group", "source", "source_type", "source_class", "waterpoint_type", "waterpoint_type_group", "recorded_by"],               # (encoder not implemented here)
        cat_col_cut_off=10,
        feature_engineer=True
    )

    # Fit on TRAIN, then transform both TRAIN and TEST with the same stats
    train_processed = pre.fit_transform(train_df)
    test_processed  = pre.transform(test_values)

    print("Train shape:", train_processed.shape)
    print("Test  shape:", test_processed.shape)

    print(sum(train_processed.isna().sum()))
    print(sum(test_processed.isna().sum()))
