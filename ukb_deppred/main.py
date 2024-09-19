import configparser
from importlib.resources import files
from pathlib import Path
import argparse

from nipype.interfaces.utility import IdentityInterface
import nipype.pipeline as pe

from ukb_deppred.interfaces import (
    CrossValSplit, FeaturewiseModel, IntegratedFeaturesModel, PredictionCombine, PredictionSave)
import ukb_deppred


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Depression prediction in UK Biobank",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
    required = parser.add_argument_group("Required arguments")
    required.add_argument(
        "--in_csv", type=Path, dest="in_csv", required=True, help="Absolute path to input data")
    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument(
        "--out_dir", type=Path, dest="out_dir", default=Path.cwd(), help="Output directory")
    optional.add_argument(
        "--work_dir", type=Path, dest="work_dir", default=Path.cwd(), help="Work directory")
    optional.add_argument(
        "--config", type=Path, dest="config", default=files(ukb_deppred)/"default.config",
        help="Configuration file for cross-validation")
    optional.add_argument(
        "--condordag", dest="condordag", action="store_true", help="Submit graph to HTCondor")
    optional.add_argument(
        "--multiproc", dest="multiproc", action="store_true", help="Run with multiprocessing")
    config = vars(parser.parse_args())

    # Set-up
    config_parse = configparser.ConfigParser()
    config_parse.read(config["config"])
    config.update({option: config_parse["USER"][option] for option in config_parse["USER"]})
    cv_iter = [
        ("repeat", list(range(int(config["n_repeats"])))),
        ("fold", list(range(int(config["n_folds"]))))]
    fw_iter = [
        ("feature_type", [
            "immune", "blood_chem", "nmr", "cs", "ct", "gmv", "rsfc_full", "rsfc_part",
            "smoking", "alcohol", "bmi", "educ"])]
    config["out_dir"].mkdir(parents=True, exist_ok=True)
    config["work_dir"].mkdir(parents=True, exist_ok=True)

    # Workflow
    deppred_wf = pe.Workflow("ukb_deppred_wf", base_dir=config["work_dir"])
    deppred_wf.config["execution"]["try_hard_link_datasink"] = "false"
    deppred_wf.config["execution"]["crashfile_format"] = "txt"
    deppred_wf.config["execution"]["stop_on_first_crash"] = "true"

    cv_split = pe.Node(CrossValSplit(config=config), "cv_split")
    cv = pe.Node(IdentityInterface(fields=["repeat", "fold"]), "cv", iterables=cv_iter)
    fw_model = pe.Node(FeaturewiseModel(config=config), "fw_model", iterables=fw_iter)
    fw_combine = pe.JoinNode(
        PredictionCombine(config=config), "fw_combine", joinsource="fw_model",
        joinfield=["results"])
    fw_save = pe.JoinNode(
        PredictionSave(config=config, model_type="featurewise"), "fw_save", joinsource="cv",
        joinfield=["results"])
    if_model = pe.JoinNode(
        IntegratedFeaturesModel(config=config), "if_model", joinsource="fw_model",
        joinfield=["fw_ypred"])
    ifs_save = pe.JoinNode(
        PredictionSave(config=config, model_type="integratedfeaturesset"), "ifs_save",
        joinsource="cv", joinfield=["results"])

    deppred_wf.connect([
        (cv_split, fw_model, [("cv_split", "cv_split")]),
        (cv, fw_model, [("repeat", "repeat"), ("fold", "fold")]),
        (fw_model, fw_combine, [("results", "results")]),
        (fw_combine, fw_save, [("results", "results")]),
        (cv_split, if_model, [("cv_split", "cv_split")]),
        (cv, if_model, [("repeat", "repeat"), ("fold", "fold")]),
        (fw_model, if_model, [("fw_ypred", "fw_ypred")]),
        (if_model, ifs_save, [("results", "results")])])

    deppred_wf.write_graph()
    if config["condordag"]:
        deppred_wf.run(
            plugin="CondorDAGMan",
            plugin_args={
                "dagman_args": f"-outfile_dir {config['work_dir']} -import_env",
                "wrapper_cmd": files(ukb_deppred) / "venv_wrapper.sh",
                "override_specs": "request_cpus = 1"})
    elif config["multiproc"]:
        deppred_wf.run(plugin="MultiProc")
    else:
        deppred_wf.run()


if __name__ == "__main__":
    main()
