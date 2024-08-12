import configparser
from importlib.resources import files
from pathlib import Path
import argparse

import nipype.pipeline as pe

from ukb_deppred.interfaces import CrossValSplit, FeaturewiseModel, PredictionSave
import ukb_deppred


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Depression prediction in UK Biobank",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
    required = parser.add_argument_group("Required arguments")
    required.add_argument(
        "--in_csv", type=Path, desc="in_csv", required=True, help="Absolute path to input data")
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
    config = vars(parser.parse_args())

    # Set-up
    config_parse = configparser.ConfigParser()
    config_parse.read(config["config"])
    config.update({option: config_parse["USER"][option] for option in config_parse["USER"]})
    repeat_iter = ("repeat", list(range(int(config["n_repeats"]))))
    fold_iter = ("fold", list(range(int(config["n_folds"]))))
    fw_iter = ("feature_type", [
        "inflamm", "blood_chem", "nmr", "cs", "ct", "gmv", "rsfc_full", "rsfc_part", "covar"])
    config["out_dir"].mkdir(parents=True, exist_ok=True)
    config["work_dir"].mkdir(parents=True, exist_ok=True)

    # Workflow
    deppred_wf = pe.Workflow("ukb_deppred_wf", base_dir=config["work_dir"])
    deppred_wf.config["execution"]["try_hard_link_datasink"] = "false"
    deppred_wf.config["execution"]["crashfile_format"] = "txt"
    deppred_wf.config["execution"]["stop_on_first_crash"] = "true"

    cv_split = pe.Node(CrossValSplit(config=config), "cv_split")
    fw = pe.Node(FeaturewiseModel(config=config), "fw", iterables=[repeat_iter, fold_iter, fw_iter])
    fw_save = pe.JoinNode(
        PredictionSave(config=config, model_type="featurewise"), "fw_save", joinsource="fw",
        joinfield=["results"])

    deppred_wf.connect([
        (cv_split, fw, [("cv_split", "cv_split")]),
        (fw, fw_save, [("results", "results")])])

    deppred_wf.write_graph()
    if config["condordag"]:
        deppred_wf.run(
            plugin="CondorDAGMan",
            plugin_args={
                "dagman_args": f"-outfile_dir {config['work_dir']} -import_env",
                "wrapper_cmd": files(ukb_deppred) / "venv_wrapper.sh",
                "override_specs": "request_memory = 10 GB\nrequest_cpus = 1"})
    else:
        deppred_wf.run()


if __name__ == "__main__":
    main()
