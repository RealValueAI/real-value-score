from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from src.learning.config import CATEGORICAL_FEATURES_DEFAULT, \
    NUMERIC_FEATURES_DEFAULT, TARGETS_DEFAULT
from src.learning.ml_model import RatingModel
from src.learning.preprocessing import RealValuePreprocessor
from src.utils.logger import logger

def _plot_roc(y_true, y_prob, title: str, out_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def train_rating_models(
    *,
    data_path: str | Path,
    labels_csv: str | Path,
    output_dir: str | Path = "models",
    numeric_features: List[str] | None = None,
    categorical_features: List[str] | None = None,
    targets: Sequence[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 777,
    save_plots: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Train CatBoost rating models and, optionally, save ROC‑AUC plots.

    Returns
    -------
    dict : mapping target → {auc, gini}
    """

    numeric_features = numeric_features or NUMERIC_FEATURES_DEFAULT
    categorical_features = categorical_features or CATEGORICAL_FEATURES_DEFAULT
    targets = list(targets or TARGETS_DEFAULT)

    output_dir = Path(output_dir)
    model_dir = output_dir / "models"
    plot_dir = output_dir / "plots"
    model_dir.mkdir(parents=True, exist_ok=True)
    if save_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    prep = RealValuePreprocessor(
        array_cols=["subway_distances", "subway_names"],
        numeric_fillna={
            "price_per_sqm": 0,
            "area": 0,
            "rooms": 0,
            "floor": -1,
            "height": -1,
        },
        bool_cols=["placement_paid"],
    )
    df_feat = prep.process(data_path)

    labels_df = pd.read_csv(labels_csv)

    target_cols = list(targets)
    key_cols = [c for c in ["listing_id"] if
                c in df_feat.columns]
    labels_df = labels_df[key_cols + target_cols]

    df = df_feat.merge(labels_df, on=key_cols, how="inner")

    metrics_out: Dict[str, Dict[str, float]] = {}

    for target in targets:
        X, y, cat_idx = prep.split_xy(df, features=numeric_features, cat_features=categorical_features, target=target)
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

        model = RatingModel()
        model.train(X_tr, y_tr, X_val, y_val, cat_idx=cat_idx)
        y_prob = model.predict_proba(X_val)
        auc = roc_auc_score(y_val, y_prob)
        gini = 2 * auc - 1
        logger.info(f'%s – AUC: %.4f  Gini: %.4f", {target}, {auc}, {gini}')
        metrics_out[target] = {"auc": auc, "gini": gini}

        model_path = model_dir / f"catboost_{target}.cbm"
        model.save(model_path)

        if save_plots:
            plot_path = plot_dir / f"roc_{target}.png"
            _plot_roc(y_val, y_prob, f"ROC curve – {target}", plot_path)

    return metrics_out
