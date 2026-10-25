"""Classical baseline models (LDA, Random Forest, CatBoost) and cross-validation."""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any

import catboost.utils
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier, CatBoostError
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from hydro.eval.metrics import evaluate_classification_targets
from hydro.paths import project_root

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class SafeGPUCatBoostClassifier(CatBoostClassifier):
    """CatBoost classifier with automatic CPU fallback on CatBoostError or RuntimeError."""

    def fit(self, x: Any, y: Any = None, **kwargs: Any) -> SafeGPUCatBoostClassifier:
        """Fit model, falling back to CPU if GPU execution fails."""
        try:
            return super().fit(x, y=y, **kwargs)
        except (CatBoostError, RuntimeError) as exc:
            if str(self.get_params().get("task_type", "")).upper() == "GPU":
                logger.warning("CatBoost GPU fit failed (%s); refitting on CPU", exc)
                self.set_params(task_type="CPU")
                return super().fit(x, y=y, **kwargs)
            raise


FallbackCatBoostClassifier = SafeGPUCatBoostClassifier


@functools.lru_cache(maxsize=1)
def load_model_config() -> dict[str, Any]:
    """Load model hyperparameters and seeds from configs/model.yaml.

    Returns
    -------
    dict[str, Any]
        Configuration mapping loaded from YAML file.
    """
    config_path = project_root() / "configs" / "model.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_lda_pipeline(config: dict[str, Any]) -> Pipeline:
    """Build LDA pipeline with RobustScaler."""
    params = dict(config.get("lda", {}))
    return Pipeline(
        [
            ("scaler", RobustScaler()),
            ("classifier", LinearDiscriminantAnalysis(**params)),
        ]
    )


def _make_rf_pipeline(config: dict[str, Any], seed: int) -> Pipeline:
    """Build Random Forest pipeline with configured hyperparameters."""
    params = dict(config.get("random_forest", {}))
    params["random_state"] = seed
    return Pipeline([("classifier", RandomForestClassifier(**params))])


def _make_catboost_pipeline(config: dict[str, Any], seed: int) -> Pipeline:
    """Build CatBoost pipeline with configured hyperparameters."""
    params = dict(config.get("catboost", {}))
    params["random_seed"] = seed
    params["allow_writing_files"] = False
    params["thread_count"] = -1
    params.setdefault("border_count", 254)
    params.setdefault("bootstrap_type", "Bernoulli")
    if params["bootstrap_type"] == "Bernoulli":
        params.setdefault("subsample", 0.8)
    use_gpu = catboost.utils.get_gpu_device_count() > 0
    params["task_type"] = "GPU" if use_gpu else "CPU"
    if use_gpu:
        params["devices"] = "0"
    return Pipeline([("classifier", SafeGPUCatBoostClassifier(**params))])


def make_classifier(kind: str, seed: int = 42) -> Pipeline:
    """Create a classification pipeline for the requested model kind.

    Parameters
    ----------
    kind : str
        Classifier kind ('lda', 'rf', or 'catboost').
    seed : int, default=42
        Random seed for model training.

    Returns
    -------
    Pipeline
        Configured scikit-learn Pipeline instance.

    Raises
    ------
    ValueError
        If kind is not one of 'lda', 'rf', 'catboost'.
    """
    config = load_model_config()
    normalized = kind.strip().lower()
    if normalized == "lda":
        return _make_lda_pipeline(config)
    if normalized == "rf":
        return _make_rf_pipeline(config, seed)
    if normalized == "catboost":
        return _make_catboost_pipeline(config, seed)
    raise ValueError(f"Unknown classifier kind: '{kind}'. Expected 'lda', 'rf', or 'catboost'.")


def _validate_inputs(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    target_cols: Sequence[str],
) -> None:
    """Validate shape alignment and column existence."""
    if len(features) != len(targets):
        raise ValueError(f"Length mismatch: features={len(features)}, targets={len(targets)}")
    if len(features) == 0:
        raise ValueError("Features and targets must not be empty")
    for tgt in target_cols:
        if tgt not in targets.columns:
            raise ValueError(f"Target column '{tgt}' missing from targets DataFrame")


def _fit_predict_fold(
    clf: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    x_test: pd.DataFrame,
) -> np.ndarray:
    """Fit a pipeline on training fold and return 1D predictions on test fold."""
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    return np.asarray(preds).ravel()


def _evaluate_target_folds(
    features: pd.DataFrame,
    target_series: pd.Series,
    target_name: str,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    kind: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Fit and predict a single target across all cross-validation folds."""
    oof_target = np.empty(len(features), dtype=target_series.dtype)
    fold_rows: list[dict[str, Any]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        clf = make_classifier(kind)
        y_pred = _fit_predict_fold(
            clf,
            features.iloc[train_idx],
            target_series.iloc[train_idx],
            features.iloc[test_idx],
        )
        oof_target[test_idx] = y_pred
        yt = target_series.iloc[test_idx]
        labels = sorted(set(yt).union(set(y_pred)))
        f1 = float(f1_score(yt, y_pred, labels=labels, average="macro", zero_division=0))
        fold_rows.append({"fold": fold_idx, "target": target_name, "macro_f1": f1})
    return oof_target, fold_rows


def cross_validate_targets(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    kind: str,
    target_cols: Sequence[str] = ("cooler", "valve", "pump", "accumulator"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-validate classifier across multiple targets and return metrics.

    Assumption: Scalers and models are fitted strictly on the training fold.

    Parameters
    ----------
    features : pd.DataFrame
        Input feature matrix of shape (n_cycles, n_features).
    targets : pd.DataFrame
        Target condition labels.
    folds : Sequence[tuple[np.ndarray, np.ndarray]]
        Train and test indices per fold.
    kind : str
        Classifier kind ('lda', 'rf', 'catboost').
    target_cols : Sequence[str], default=('cooler', 'valve', 'pump', 'accumulator')
        Target columns to evaluate.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (per-target out-of-fold metrics, per-fold long table).
    """
    _validate_inputs(features, targets, target_cols)
    oof_dict: dict[str, np.ndarray] = {}
    all_fold_rows: list[dict[str, Any]] = []
    for tgt in target_cols:
        oof_tgt, fold_rows = _evaluate_target_folds(features, targets[tgt], tgt, folds, kind)
        oof_dict[tgt] = oof_tgt
        all_fold_rows.extend(fold_rows)

    oof_df = pd.DataFrame(oof_dict, index=features.index)
    target_df = targets[list(target_cols)]
    metrics_df = evaluate_classification_targets(target_df, oof_df)
    fold_df = pd.DataFrame(all_fold_rows)
    return metrics_df, fold_df


def train_evaluate_baseline(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    kinds: Sequence[str] = ("lda", "rf"),
) -> dict[str, pd.DataFrame]:
    """Train and evaluate baseline classical models across cross-validation folds.

    Parameters
    ----------
    features : pd.DataFrame
        Input feature matrix of shape (n_cycles, n_features).
    targets : pd.DataFrame
        Target condition labels.
    folds : Sequence[tuple[np.ndarray, np.ndarray]]
        Train and test fold index pairs.
    kinds : Sequence[str], default=('lda', 'rf')
        Model kinds to train and evaluate.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping model kind to its per-target evaluation DataFrame.
    """
    results: dict[str, pd.DataFrame] = {}
    for kind in kinds:
        metrics_df, _ = cross_validate_targets(features, targets, folds, kind=kind)
        results[kind] = metrics_df
    return results
