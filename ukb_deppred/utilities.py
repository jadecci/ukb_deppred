from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier)
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd


def feature_cols(feature_type: str) -> tuple[dict, list]:
    x_cols = {
        "inflamm": [
            "30000-0.0", "30080-0.0", "30120-0.0", "30130-0.0", "30140-0.0", "30180-0.0",
            "30190-0.0", "30200-0.0", "30710-0.0"],
        "blood_chem": [
            "30690-0.0", "30740-0.0", "30750-0.0", "30760-0.0", "30860-0.0", "30870-0.0"],
        "nmr": (
                ["23400-0.0", "23403-0.0"] + [f"23{i}-0.0" for i in range(405, 431)] + ["23437-0.0"]
                + [f"23{i}-0.0" for i in range(442, 447)] + [f"23{i}-0.0" for i in range(449, 454)]
                + [f"23{i}-0.0" for i in range(456, 460)] + [f"23{i}-0.0" for i in range(464, 468)]
                + ["23470-0.0"] + [f"23{i}-0.0" for i in range(473, 478)]
                + [f"23{i}-0.0" for i in range(488, 579)]
                + [f"23{i}-0.0" for i in range(584, 649)]),
        "cs": [f"27{i}-2.0" for i in range(329, 403)] + [f"27{i}-2.0" for i in range(551, 625)],
        "ct": [f"27{i}-2.0" for i in range(403, 477)] + [f"27{i}-2.0" for i in range(625, 699)],
        "gmv": [f"27{i}-2.0" for i in range(477, 551)] + [f"27{i}-2.0" for i in range(699, 773)],
        "rsfc_full": [f"{i}_full" for i in range(1485)],
        "rsfc_part": [f"{i}_part" for i in range(1485)],
        "covar": [
            "21003-0.0", "31-0.0", "21000-0.0", "25741-2.0", "26521-2.0", "25000-2.0", "54-0.0",
            "25756-2.0", "25757-2.0", "25758-2.0", "25759-2.0", "20116-0.0", "1558-0.0",
            "21001-0.0", "6138-0.0"],
        "cv_split": []}
    cols = {"eid": str, "patient": bool}
    cols.update({col: float for col in x_cols[feature_type]})
    return cols, x_cols[feature_type]


def random_forest(
        train_x: pd.DataFrame, train_y: pd.DataFrame, test_x: pd.DataFrame,
        test_y: pd.DataFrame) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    params = {
        "n_estimators": np.linspace(100, 1000, 5, dtype=int),
        "min_samples_split": np.linspace(0.01, 0.1, 5),
        "max_features": np.linspace(
            np.floor(train_x.shape[1] / 2), train_x.shape[1], train_x.shape[1], dtype=int)}
    rfc_cv = GridSearchCV(
        RandomForestClassifier, param_grid=params, scoring=balanced_accuracy_score)
    rfc_cv.fit(train_x, train_y)
    acc = rfc_cv.score(test_x, test_y)
    ypred = rfc_cv.predict(test_x)
    tscore = rfc_cv.cv_results_["mean_test_score"]
    ftime = rfc_cv.cv_results_["mean_fit_time"]
    return acc, ypred, tscore, ftime


def extra_trees(
        train_x: pd.DataFrame, train_y: pd.DataFrame, test_x: pd.DataFrame,
        test_y: pd.DataFrame) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    # usually lower variance, but slightly higher bias
    params = {
        "n_estimators": np.linspace(100, 1000, 5, dtype=int),
        "min_samples_split": np.linspace(0.01, 0.1, 5),
        "max_features": np.linspace(
            np.floor(train_x.shape[1] / 2), train_x.shape[1], train_x.shape[1], dtype=int)}
    etc_cv = GridSearchCV(ExtraTreesClassifier, param_grid=params, scoring=balanced_accuracy_score)
    acc = etc_cv.score(test_x, test_y)
    ypred = etc_cv.predict(test_x)
    tscore = etc_cv.cv_results_["mean_test_score"]
    ftime = etc_cv.cv_results_["mean_fit_time"]
    return acc, ypred, tscore, ftime


def hist_grad_boost(
        train_x: pd.DataFrame, train_y: pd.DataFrame, test_x: pd.DataFrame,
        test_y: pd.DataFrame) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    # generally faster and more scalable
    params = {
        "max_iter": np.linspace(100, 1000, 5, dtype=int),
        "max_features": np.linspace(
            np.floor(train_x.shape[1] / 2), train_x.shape[1], train_x.shape[1], dtype=int)}
    hgbc_cv = GridSearchCV(
        HistGradientBoostingClassifier, param_grid=params, scoring=balanced_accuracy_score)
    hgbc_cv.fit(train_x, train_y)
    acc = hgbc_cv.score(test_x, test_y)
    ypred = hgbc_cv.predict(test_x)
    tscore = hgbc_cv.cv_results_["mean_test_score"]
    ftime = hgbc_cv.cv_results_["mean_fit_time"]
    return acc, ypred, tscore, ftime
