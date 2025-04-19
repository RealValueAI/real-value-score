import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool, CatBoostError
from optuna.integration import CatBoostPruningCallback
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger("rating_model")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

__all__ = ["RatingModel", "gini"]


def gini(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return Gini coefficient (2*AUC-1)."""
    auc = roc_auc_score(y_true, y_prob)
    return 2.0 * auc - 1.0


class RatingModel:
    """CatBoostClassifier wrapper with Optuna tuning for binary rating labels."""

    DEFAULT_PARAMS: Dict[str, Any] = {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": 42,
        "verbose": 0,
        "early_stopping_rounds": 50,
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params: Dict[str, Any] = (params or {}).copy()
        for k, v in RatingModel.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        self.model: Optional[CatBoostClassifier] = None


    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        cat_idx: Optional[List[int]] = None,
    ) -> None:
        self.model = CatBoostClassifier(**self.params)
        if X_val is not None and y_val is not None:
            self.model.fit(
                Pool(X_train, y_train, cat_features=cat_idx),
                eval_set=Pool(X_val, y_val, cat_features=cat_idx),
                verbose=self.params.get("verbose", 100),
            )
        else:
            self.model.fit(X_train, y_train, cat_features=cat_idx, verbose=self.params.get("verbose", 100))

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("Model is not trained")
        y_prob = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        return {"auc": auc, "gini": 2 * auc - 1}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not trained or loaded")
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str | Path):
        if self.model is None:
            raise RuntimeError("Model is not trained")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        logger.info("Model saved to %s", path)

    def load(self, path: str | Path):
        path = Path(path)
        self.model = CatBoostClassifier()
        self.model.load_model(str(path))
        logger.info("Loaded model from %s", path)

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cat_idx: Optional[List[int]] = None,
        *,
        n_trials: int = 50,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

        def objective(trial: optuna.Trial) -> float:
            trial_params = {
                "iterations": trial.suggest_int("iterations", 500, 2000, step=250),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                "random_seed": random_state,
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "verbose": 0,
            }
            model = CatBoostClassifier(**trial_params)
            try:
                model.fit(
                    Pool(X_tr, y_tr, cat_features=cat_idx),
                    eval_set=Pool(X_val, y_val, cat_features=cat_idx),
                    callbacks=[CatBoostPruningCallback(trial, "AUC")],
                )
            except CatBoostError:
                return float("inf")
            pred_val = model.predict_proba(X_val)[:, 1]
            return -roc_auc_score(y_val, pred_val)  # minimize negative AUC

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        logger.info("Best params: %s", study.best_params)
        self.params.update(study.best_params)
        return {"best_params": study.best_params, "best_auc": -study.best_value}
