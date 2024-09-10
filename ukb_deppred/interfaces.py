from itertools import product
from pathlib import Path

from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
import numpy as np
import pandas as pd

from ukb_deppred.utilities import feature_cols, conf_reg, elastic_net, feature_covar_groups


class _CrossValSplitInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")


class _CrossValSplitOutputSpec(TraitedSpec):
    cv_split = traits.Dict(dtype=np.ndarray, desc="CV indices for each fold")


class CrossValSplit(SimpleInterface):
    """Generate cross-validation splits"""
    input_spec = _CrossValSplitInputSpec
    output_spec = _CrossValSplitOutputSpec

    def _run_interface(self, runtime):
        cols, _, _ = feature_cols("cv_split")
        data = pd.read_csv(self.inputs.config["in_csv"], usecols=list(cols.keys()), dtype=cols)
        self._results["cv_split"] = {}
        n_repeats = int(self.inputs.config["n_repeats"])
        n_folds = int(self.inputs.config["n_folds"])

        rskf = RepeatedStratifiedKFold(
            n_splits=n_folds, n_repeats=n_repeats, random_state=int(self.inputs.config["cv_seed"]))
        cv_iter = enumerate(rskf.split(data["eid"], data["patient"]))
        for fold, (train_ind, test_ind) in cv_iter:
            key = f"repeat{int(np.floor(fold / n_folds))}_fold{int(fold % n_folds)}"
            self._results["cv_split"][f"{key}_train"] = train_ind
            self._results["cv_split"][f"{key}_test"] = test_ind

            skf = StratifiedKFold(n_splits=5)
            cv_iter_inner = enumerate(skf.split(data["eid"][train_ind], data["patient"][train_ind]))
            for inner, (train_i, test_i) in cv_iter_inner:
                key_inner = f"{key}_inner{inner}"
                self._results["cv_split"][f"{key_inner}_train"] = train_i
                self._results["cv_split"][f"{key_inner}_test"] = test_i
        return runtime


class _FeaturewiseModelInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    feature_type = traits.Str(mandatory=True, desc="Feature type")
    cv_split = traits.Dict(mandatory=True, dtype=list, desc="CV indices for each fold")
    repeat = traits.Int(mandatory=True, desc="Current repeat")
    fold = traits.Int(mandatory=True, desc="Current fold in the repeat")


class _FeaturewiseModelOutputSpec(TraitedSpec):
    results = traits.Dict(desc="Prediction results")
    fw_ypred = traits.Dict(desc="Predicted classes")
    feature_type = traits.Str(desc="Feature type")


class FeaturewiseModel(SimpleInterface):
    """Train and test feature-wise models"""
    input_spec = _FeaturewiseModelInputSpec
    output_spec = _FeaturewiseModelOutputSpec

    def _run_interface(self, runtime):
        self._results["feature_type"] = self.inputs.feature_type
        key = f"repeat{self.inputs.repeat}_fold{self.inputs.fold}"
        key_out = f"{self.inputs.feature_type}_{key}"
        cols, x_cols, conf_cols = feature_cols(self.inputs.feature_type, include_conf=True)
        data = pd.read_csv(
            self.inputs.config["in_csv"], usecols=list(cols.keys()), dtype=cols, index_col="eid")

        train_ind = self.inputs.cv_split[f"{key}_train"]
        test_ind = self.inputs.cv_split[f"{key}_test"]
        train_x, test_x = conf_reg(
            data[x_cols].iloc[train_ind], data[conf_cols].iloc[train_ind],
            data[x_cols].iloc[test_ind], data[conf_cols].iloc[test_ind])

        acc, test_ypred, l1r, coef = elastic_net(
            train_x, data["patient"].iloc[train_ind], test_x, data["patient"].iloc[test_ind])
        self._results["results"] = {
            f"acc_{key_out}": acc, f"ypred_{key_out}": test_ypred, f"l1r_{key_out}": l1r,
            f"coef_{key_out}": coef}

        train_ypred = np.empty(train_ind.shape)
        for inner in range(5):
            train_ind_inner = train_ind[self.inputs.cv_split[f"{key}_inner{inner}_train"]]
            test_ind_inner = train_ind[self.inputs.cv_split[f"{key}_inner{inner}_test"]]
            train_x_inner, test_x_inner = conf_reg(
                data[x_cols].iloc[train_ind_inner], data[conf_cols].iloc[train_ind_inner],
                data[x_cols].iloc[test_ind_inner], data[conf_cols].iloc[test_ind_inner])
            test_i = self.inputs.cv_split[f"{key}_inner{inner}_test"]
            acc, ypred, _, _ = elastic_net(
                train_x_inner, data["patient"].iloc[train_ind_inner], test_x_inner,
                data["patient"].iloc[test_ind_inner])
            train_ypred[test_i] = np.array(ypred > acc[-1]).astype(float)
        self._results["fw_ypred"] = {"train_ypred": train_ypred, "test_ypred": test_ypred}

        return runtime


class _CombinedFeaturesModelInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    cv_split = traits.Dict(mandatory=True, dtype=list, desc="CV indices for each fold")
    repeat = traits.Int(mandatory=True, desc="Current repeat")
    fold = traits.Int(mandatory=True, desc="Current fold in the repeat")


class _CombinedFeaturesModelOutputSpec(TraitedSpec):
    results = traits.Dict(desc="Prediction results")


class CombinedFeaturesModel(SimpleInterface):
    """Train and test combined-features models"""
    input_spec = _CombinedFeaturesModelInputSpec
    output_spec = _CombinedFeaturesModelOutputSpec

    def _run_interface(self, runtime):
        key = f"repeat{self.inputs.repeat}_fold{self.inputs.fold}"
        train_ind = self.inputs.cv_split[f"{key}_train"]
        test_ind = self.inputs.cv_split[f"{key}_test"]
        self._results["results"] = {}

        group_names, _ = feature_covar_groups()
        for group in group_names:
            cols, x_cols, conf_cols = feature_cols(group, include_conf=True, group_feature=True)
            data = pd.read_csv(
                self.inputs.config["in_csv"], usecols=list(cols.keys()), dtype=cols,
                index_col="eid")
            train_x, test_x = conf_reg(
                data[x_cols].iloc[train_ind], data[conf_cols].iloc[train_ind],
                data[x_cols].iloc[test_ind], data[conf_cols].iloc[test_ind])
            acc, ypred, l1r, coef = elastic_net(
                train_x, data["patient"].iloc[train_ind], test_x, data["patient"].iloc[test_ind])
            self._results["results"].update({
                f"acc_{group}_{key}": acc, f"ypred_{group}_{key}": ypred, f"l1r_{group}_{key}": l1r,
                f"coef_{group}_{key}": coef})
        return runtime


class _IntegratedFeaturesModelInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    cv_split = traits.Dict(mandatory=True, dtype=list, desc="CV indices for each fold")
    repeat = traits.Int(mandatory=True, desc="Current repeat")
    fold = traits.Int(mandatory=True, desc="Current fold in the repeat")
    fw_ypred = traits.List(dtype=dict, mandatory=True, desc="Predicted classes")


class _IntegratedFeaturesModelOutputSpec(TraitedSpec):
    results = traits.Dict(desc="Prediction results")


class IntegratedFeaturesModel(SimpleInterface):
    """Train and test integrated-features models"""
    input_spec = _IntegratedFeaturesModelInputSpec
    output_spec = _IntegratedFeaturesModelOutputSpec

    def _extract_data(
            self, sub_ind: list, key: str) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
        cols, x_cols, conf_cols = feature_cols("conf")
        data = pd.read_csv(
            self.inputs.config["in_csv"], usecols=list(cols.keys()), dtype=cols, index_col="eid")
        y = data["patient"].iloc[sub_ind]
        x = np.empty((len(sub_ind), len(self.inputs.fw_ypred)))
        for feature_i, fw_ypred in enumerate(self.inputs.fw_ypred):
            x[:, feature_i] = fw_ypred[key]
        return x, y, data

    def _run_interface(self, runtime):
        key = f"repeat{self.inputs.repeat}_fold{self.inputs.fold}"
        train_ind = self.inputs.cv_split[f"{key}_train"]
        test_ind = self.inputs.cv_split[f"{key}_test"]
        train_x, train_y, train_conf = self._extract_data(train_ind, "train_ypred")
        test_x, test_y, test_conf = self._extract_data(test_ind, "test_ypred")

        train_x_resid, test_x_resid = conf_reg(train_x, train_conf, test_x, test_conf)
        acc, ypred, l1r, coef = elastic_net(train_x_resid, train_y, test_x_resid, test_y)
        self._results["results"] = {
            f"acc_all_{key}": acc, f"ypred_all_{key}": ypred, f"l1r_all_{key}": l1r,
            f"coef_all_{key}": coef}

        group_names, group_inds = feature_covar_groups()
        for group, group_ind in zip(group_names, group_inds):
            train_x_group = train_x[:, group_ind]
            test_x_group = test_x[:, group_ind]
            train_x_resid, test_x_resid = conf_reg(
                train_x_group, train_conf, test_x_group, test_conf)
            acc, ypred, l1r, coef = elastic_net(train_x_resid, train_y, test_x_resid, test_y)
            self._results["results"].update({
                f"acc_{group}_{key}": acc, f"ypred_{group}_{key}": ypred, f"l1r_{group}_{key}": l1r,
                f"coef_{group}_{key}": coef})
        return runtime


class _PredictionCombineInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    results = traits.List(dtype=dict, desc="accuracy results")


class _PredictionCombineOutputSpec(TraitedSpec):
    results = traits.Dict(desc="accuracy results")


class PredictionCombine(SimpleInterface):
    """Combine prediction results across features"""
    input_spec = _PredictionCombineInputSpec
    output_spec = _PredictionCombineOutputSpec

    def _run_interface(self, runtime):
        self._results["results"] = {}
        for results in self.inputs.results:
            for key, val in results.items():
                self._results["results"].update({key: val})

        return runtime


class _PredictionSaveInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    results = traits.List(mandatory=True, dtype=dict, desc="Prediction results")
    model_type = traits.Str(mandatory=True, desc="Type of model")


class PredictionSave(SimpleInterface):
    """Save prediction results"""
    input_spec = _PredictionSaveInputSpec

    def _run_interface(self, runtime):
        out_file = Path(self.inputs.config["out_dir"], f"ukb_deppred_{self.inputs.model_type}.h5")
        for results in self.inputs.results:
            for key, val in results.items():
                if np.array(val).ndim == 0:
                    pd.DataFrame({key: val}, index=[0]).to_hdf(out_file, key)
                else:
                    pd.DataFrame({key: val}).to_hdf(out_file, key)
        return runtime
