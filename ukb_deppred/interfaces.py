from pathlib import Path

from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
import numpy as np
import pandas as pd

from ukb_deppred.utilities import feature_cols, random_forest, extra_trees, hist_grad_boost


class _CrossValSplitInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")


class _CrossValSplitOutputSpec(TraitedSpec):
    cv_split = traits.Dict(dtype=np.ndarray, desc="CV indices for each fold")


class CrossValSplit(SimpleInterface):
    """Generate cross-validation splits"""
    input_spec = _CrossValSplitInputSpec
    output_spec = _CrossValSplitOutputSpec

    def _run_interface(self, runtime):
        cols, _ = feature_cols("cv_split")
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
    feature_type = traits.Str(mandatory=True, desc="Feature type")


class FeaturewiseModel(SimpleInterface):
    """Train and test feature-wise models"""
    input_spec = _FeaturewiseModelInputSpec
    output_spec = _FeaturewiseModelOutputSpec

    def _run_interface(self, runtime):
        self._results["feature_type"] = self.inputs.feature_type
        key = f"repeat{self.inputs.repeat}_fold{self.inputs.fold}"
        key_out = f"{self.inputs.feature_type}_{key}"
        cols, x_cols = feature_cols(self.inputs.feature_type)
        data = pd.read_csv(
            self.inputs.config["in_csv"], usecols=list(cols.keys()), dtype=cols, index_col="eid")

        train_ind = self.inputs.cv_split[f"{key}_train"]
        test_ind = self.inputs.cv_split[f"{key}_test"]
        acc, ypred, tscore, ftime = random_forest(
            data[x_cols].iloc[train_ind], data["patient"].iloc[train_ind],
            data[x_cols].iloc[test_ind], data["patient"].iloc[test_ind])
        self._results["results"] = {
            f"rf_acc_{key_out}": acc, f"rf_test_ypred_{key_out}": ypred,
            f"rf_test_score_{key_out}": tscore, f"rf_fit_time_{key_out}": ftime}

        acc, ypred, tscore, ftime = extra_trees(
            data[x_cols].iloc[train_ind], data["patient"].iloc[train_ind],
            data[x_cols].iloc[test_ind], data["patient"].iloc[test_ind])
        self._results["results"] = {
            f"et_acc_{key_out}": acc, f"et_test_ypred_{key_out}": ypred,
            f"et_test_score_{key_out}": tscore, f"et_fit_time_{key_out}": ftime}

        acc, ypred, tscore, ftime = hist_grad_boost(
            data[x_cols].iloc[train_ind], data["patient"].iloc[train_ind],
            data[x_cols].iloc[test_ind], data["patient"].iloc[test_ind])
        self._results["results"] = {
            f"hgb_acc_{key_out}": acc, f"hgb_test_ypred_{key_out}": ypred,
            f"hgb_test_score_{key_out}": tscore, f"hgb_fit_time_{key_out}": ftime}

        train_ypred = np.empty(train_ind.shape)
        #for inner in range(5):
        #    train_ind_inner = train_ind[self.inputs.cv_split[f"{key}_inner{inner}_train"]]
        #    test_ind_inner = train_ind[self.inputs.cv_split[f"{key}_inner{inner}_test"]]
        #    test_i = self.inputs.cv_split[f"{key}_inner{inner}_test"]
        #    _, train_ypred[test_i], _, _ = random_forest(
        #        data[x_cols].iloc[train_ind_inner], data["patient"].iloc[train_ind_inner],
        #        data[x_cols].iloc[test_ind_inner], data["patient"].iloc[train_ind_inner])
        self._results["fw_ypred"] = {"train_ypred": train_ypred, "test_ypred": ypred}

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
