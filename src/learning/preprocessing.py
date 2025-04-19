
import ast
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import logger


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great‑circle distance (km) between *lat/lon* pairs."""
    r = 6_371.0  # Earth radius
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _to_list(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (SyntaxError, ValueError):
            logger.debug("Cannot parse array value: %s", x)
            return []
    return []

class RealValuePreprocessor:
    """High‑level orchestrator: load → clean → feature engineering."""

    def __init__(
        self,
        *,
        dtypes: Dict[str, str] | None = None,
        date_cols: List[str] | None = None,
        array_cols: List[str] | None = None,
        numeric_fillna: Dict[str, float | int] | None = None,
        bool_cols: List[str] | None = None,
        center_lat: float = 55.7558,
        center_lon: float = 37.6173,
    ) -> None:
        self.dtypes = dtypes or {}
        self.date_cols = date_cols or []
        self.array_cols = array_cols or []
        self.numeric_fillna = numeric_fillna or {}
        self.bool_cols = bool_cols or []
        self.center_lat = center_lat
        self.center_lon = center_lon

    def _load(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if path.suffix == ".parquet":
            df = pd.read_parquet(path, engine="pyarrow")
        else:
            df = pd.read_csv(path, low_memory=False)
        return df

    def load(self, path: str | Path) -> pd.DataFrame:
        df = self._load(path)
        for col, dtype in self.dtypes.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype, errors="ignore")
        for col in self.date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        for col in self.array_cols:
            if col in df.columns:
                df[col] = df[col].apply(_to_list)
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning data (fillna, bool cast)")
        for col, value in self.numeric_fillna.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)
        for col in self.bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(bool, errors="ignore")
        if "subway_distances" in df.columns:
            df["subway_distances"] = df["subway_distances"].apply(_to_list)
        return df

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Engineering features …")
        now = datetime.now()

        if "subway_distances" in df.columns:
            df["num_subways"] = df["subway_distances"].apply(len)
            df["subway_min_dist"] = df["subway_distances"].apply(lambda a: min(a) if a else np.nan)
            df["subway_mean_dist"] = df["subway_distances"].apply(lambda a: np.mean(a) if a else np.nan)
        if "subway_names" in df.columns:
            df["primary_subway"] = df["subway_names"].apply(lambda a: (_to_list(a) or ["unknown"])[0])

        if {"latitude", "longitude"}.issubset(df.columns):
            df["center_dist_km"] = df.apply(
                lambda r: haversine(r["latitude"], r["longitude"], self.center_lat, self.center_lon),
                axis=1,
            )
            buckets = [0, 1, 3, 5, 10, np.inf]
            labels = ["<1km", "1-3km", "3-5km", "5-10km", ">10km"]
            df["center_bucket"] = pd.cut(df["center_dist_km"], buckets, labels=labels)

        if {"published_date", "updated_date"}.issubset(df.columns):
            df["days_since_pub"] = (now - df["published_date"]).dt.days
            df["days_since_upd"] = (df["updated_date"] - df["published_date"]).dt.days

        if {"house_floors", "floor"}.issubset(df.columns):
            df["floor_ratio"] = np.where(df["house_floors"] > 0, df["floor"] / df["house_floors"], np.nan)
        if {"rooms", "area"}.issubset(df.columns):
            df["rooms_density"] = np.where(df["area"] > 0, df["rooms"] / df["area"], np.nan)

        if "description" in df.columns:
            df["desc_len"] = df["description"].str.len()
            df["desc_words"] = df["description"].str.split().apply(lambda t: len(t) if isinstance(t, list) else 0)
        return df

    def process(self, path: str | Path, *, include_raw: bool = False) -> pd.DataFrame:
        df = self.load(path)
        df = self.clean(df)
        df = self.add_features(df)
        if not include_raw:
            df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
        return df

    def split_xy(
        self,
        df: pd.DataFrame,
        *,
        features: List[str],
        cat_features: List[str],
        target: str,
    ) -> Tuple[pd.DataFrame, pd.Series, List[int]]:
        req = features + cat_features
        X = df[req].copy()
        y = df[target]
        cat_idx = [X.columns.get_loc(col) for col in cat_features if col in X.columns]
        return X, y, cat_idx