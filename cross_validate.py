"""Slightly adjust and simplify the Pipline API in Sklearn and Imbalance-learn."""
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from joblib import Parallel, delayed
from keras.metrics import AUC


def cross_validate(
    X: pd.DataFrame,
    y,
    estimator,
    sampler,
    transformer,
    scorer,
    num_split: int = 10,
    n_job: int = 2,
    verbose: bool = False,
):
    """Perform cross-validation with transformer and sampler."""
    folds = StratifiedKFold(n_splits=num_split)

    # parallel computing
    parallel = Parallel(n_jobs=n_job, verbose=verbose)
    results = parallel(
        delayed(_fit_and_score)(
            clone(estimator), X, y, train_idx, valid_idx, sampler, transformer, scorer
        )
        for train_idx, valid_idx in folds.split(X, y)
    )
    # results = [
    #     _fit_and_score(
    #         clone(estimator), X, y, train_idx, valid_idx, sampler, transformer, scorer
    #     )
    #     for train_idx, valid_idx in folds.split(X, y)
    # ]
    return results


def _fit_and_score(
    estimator, X: pd.DataFrame, y, train_idx, valid_idx, sampler, transformer, scorer
) -> dict[str, float]:
    X_train, y_train = X.iloc[train_idx, :], y[train_idx]
    X_valid, y_valid = X.iloc[valid_idx, :], y[valid_idx]
    # balance
    Xs_train, ys_train = sampler.fit_resample(X_train, y_train)
    # preprocess
    X_train = transformer.fit_transform(X_train)
    X_valid = transformer.fit_transform(X_valid)
    Xs_train = transformer.fit_transform(Xs_train)
    # fit
    estimator.fit(Xs_train, ys_train)
    # record scores
    results = {
        "sampled_train": scorer(estimator, Xs_train, ys_train),
        "oiginal_train": scorer(estimator, X_train, y_train),
        "valid": scorer(estimator, X_valid, y_valid),
    }

    return results


def keras_AUC(y_true, y_pred) -> np.float64:
    """Scoring function implementing keras.metrics.AUC."""
    auc = AUC(name="auc")
    auc.update_state(y_true, y_pred)
    return auc.result().numpy()


class GridSearcher:
    """GridSearch with CV."""

    def __init__(self) -> None:
        """Init."""
        pass


if __name__ == "__main__":
    print("!!!")
