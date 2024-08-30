from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.metrics import (
    roc_curve, balanced_accuracy_score, roc_auc_score, f1_score, matthews_corrcoef,
    confusion_matrix)
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd


def feature_cols(feature_type: str) -> tuple[dict, list, list]:
    x_cols = {
        "immune": [
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
        "conf": [
            "21003-0.0", "31-0.0", "21000-0.0", "25741-2.0", "26521-2.0", "25000-2.0", "54-0.0",
            "25756-2.0", "25757-2.0", "25758-2.0", "25759-2.0"],
        "smoking": ["20116-0.0"],
        "alcohol": ["1558-0.0"],
        "bmi": ["21001-0.0"],
        "education": ["6138-0.0"],
        "cv_split": []}
    cols = {"eid": str, "patient": bool}
    cols.update({col: float for col in x_cols[feature_type]})
    if feature_type != "cv_split":
        cols.update({col: float for col in x_cols["conf"]})
    return cols, x_cols[feature_type], x_cols["conf"]


def conf_reg(
        train_x: np.ndarray | pd.DataFrame, train_conf: np.ndarray | pd.DataFrame,
        test_x: np.ndarray | pd.DataFrame,
        test_conf: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, ...]:
    reg = LinearRegression()
    reg.fit(train_conf, train_x)
    train_x_resid = train_x - reg.predict(train_conf)
    test_x_resid = test_x - reg.predict(test_conf)
    return train_x_resid, test_x_resid


def acc_youden(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, ...]:
    false_pos_rate, true_pos_rate, thresh = roc_curve(y_true, y_pred, drop_intermediate=False)
    youden_j = true_pos_rate - false_pos_rate
    youden_thresh = thresh[np.argmax(youden_j)]

    bacc = balanced_accuracy_score(y_true, y_pred > youden_thresh)
    f1 = f1_score(y_true, y_pred > youden_thresh)
    mcc = matthews_corrcoef(y_true, y_pred > youden_thresh)

    true_neg, false_pos, false_neg, true_pos = confusion_matrix(
        y_true, y_pred > youden_thresh).ravel()
    ppv = true_pos / (true_pos + false_pos)
    npv = true_neg / (true_neg + false_neg)
    return bacc, f1, mcc, ppv, npv, youden_thresh


def elastic_net(
        train_x: np.ndarray | pd.DataFrame, train_y: np.ndarray | pd.DataFrame,
        test_x: np.ndarray | pd.DataFrame,
        test_y: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, ...]:
    # see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html # noqa: E501
    l1_ratio = [.1, .5, .7, .9, .95, .99, 1]
    en = ElasticNetCV(l1_ratio=l1_ratio, n_alphas=100, max_iter=10000, selection="random")
    en.fit(train_x, train_y)
    y_pred = en.predict(test_x)
    auc = roc_auc_score(test_y, y_pred)
    bacc, f1, mcc, ppv, npv, ythr = acc_youden(test_y.astype(float), y_pred)
    return np.array([auc, bacc, f1, mcc, ppv, npv, ythr]), y_pred, en.l1_ratio_, en.coef_


def kernel_ridge(
        train_x: np.ndarray | pd.DataFrame, train_y: np.ndarray | pd.DataFrame,
        test_x: np.ndarray | pd.DataFrame,
        test_y: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    param_grid = {"alpha": np.linspace(0.1, 1, 10), "kernel": ["linear", "rbf", "cosine"]}
    kr_cv = GridSearchCV(KernelRidge(), param_grid=param_grid, scoring="r2")
    kr_cv.fit(train_x, train_y)
    y_pred = kr_cv.predict(test_x)
    auc = roc_auc_score(test_y, y_pred)
    bacc, f1, mcc, ppv, npv, ythr = acc_youden(test_y.astype(float), y_pred)
    return np.array([auc, bacc, f1, mcc, ppv, npv, ythr]), y_pred, kr_cv.best_params_
