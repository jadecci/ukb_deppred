from os import remove
from pathlib import Path
import argparse

import datalad.api as dl
import pandas as pd


def phy_frailty(data: pd.DataFrame) -> pd.DataFrame:
    weight_loss = (data["2306-0.0"] == 3).astype("Int64")
    exhaustion = ((data["120107-0.0"] == -524) | (data["120107-0.0"] == -523)).astype("Int64")
    walk_speed = (data["924-0.0"] == 1).astype("Int64")

    weakness = pd.Series(index=data.index, dtype="Int64")
    phy_act = pd.Series(index=data.index, dtype="Int64")
    for data_i in data.index:
        bmi_thresh = bmi_threshs[data.loc[data_i, "31-0.0"]]
        bmi = data.loc[data_i, "21001-0.0"]
        weak_thresh = weak_threshs[data.loc[data_i, "31-0.0"]]
        grip_str = (data.loc[data_i, "46-0.0"] + data.loc[data_i, "47-0.0"])
        for thresh_i in range(len(bmi_thresh) - 1):
            if bmi_thresh[thresh_i] < bmi <= bmi_thresh[thresh_i + 1]:
                weakness.loc[data_i] = int((grip_str / 2) <= weak_thresh[thresh_i])
        activity = data.loc[data_i, "6164-0.0"]
        light_act_freq = data.loc[data_i, "1011-0.0"]
        if activity == -7 or (activity == 4 and light_act_freq in [1, 2, 3]):
            phy_act.loc[data_i] = 1

    frailty = (
            weight_loss.fillna(0) + exhaustion.fillna(0) + walk_speed.fillna(0) + weakness.fillna(0)
            + phy_act.fillna(0))
    return data.join(frailty.rename("frailty"))


def fill_pd_row(
        in_pd: pd.DataFrame, sub: str | int, rsfc_type: str, subject_dir: Path) -> pd.DataFrame:
    rsfc_dir = Path(subject_dir, "ses-2", "non-bids", "fMRI")
    rsfc_file = Path(rsfc_dir, f"sub-{sub}_ses-2_task-rest_{rsfc_type}_correlation_matrix_d100.txt")
    if rsfc_file.is_symlink():
        dl.get(rsfc_file, dataset=subject_dir)
        rsfc_curr = pd.read_table(rsfc_file, delimiter="\s+", header=None).squeeze()
        in_pd.loc[sub] = rsfc_curr
        dl.drop(rsfc_file, reckless="kill", dataset=subject_dir)
    return in_pd


parser = argparse.ArgumentParser(
    description="extract UKB data for depression prediction",
    formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("raw_tsv", type=Path, help="Absolute path to UKB raw data tsv")
parser.add_argument("genetic_tsv", type=Path, help="Absolute path to UKB genetic data tsv")
parser.add_argument("ukb_raw_url", type=str, help="Datalad URL to UKB raw dataset")
parser.add_argument("parser_dir", type=Path, help="Absolute path to ukbb_parser directory")
parser.add_argument("antidep_csv", type=Path, help="Absolute path to antidepressant code csv")
parser.add_argument("work_dir", type=Path, help="Absolute path to working directory")
parser.add_argument("out_csv", type=Path, help="Absolute path to output csv file")
args = parser.parse_args()

# Set-ups
encoding = "ISO-8859-1"
chunksize = 1000
if args.out_csv.exists():
    remove(args.out_csv)
antidep_code = pd.read_csv(args.antidep_csv, index_col="Drug name")["Code in UKB"].tolist()
bmi_threshs = {0: [0, 23, 26, 29, 100], 1: [0, 24, 28, 100]}
weak_threshs = {0: [17, 17.3, 18, 21], 1: [29, 30, 32]}

# ICD-10 set-up
coding19 = pd.read_csv(
    Path(args.parser_dir, "ukbb_parser", "scripts", "data", "icd10_level_map.csv"),
    index_col="Coding")
icd_selectable = coding19.loc[coding19["Selectable"] == "Yes"].index.tolist()
icd_include = [
    "F32", "F320", "F321", "F322", "F323", "F328", "F329", "F33", "F330", "F331", "F332", "F333",
    "F334", "F338", "F339", "F34", "F340", "F341", "F348", "F349", "F38", "F380", "F381", "F388",
    "F39"]
icd_exclude = [icd for icd in icd_selectable if "F" in icd or "G" in icd]
icd_exclude.extend([f"I{i}" for i in range(600, 700) if f"I{i}" in icd_selectable])
icd_exclude = [icd for icd in icd_exclude if icd not in icd_include]

# Data fields to read and/or write
sympt_cols = [
    "20536-0.0", "20446-0.0", "20441-0.0", "20449-0.0", "20532-0.0", "20435-0.0", "20450-0.0",
    "20437-0.0"]
diagn_cols = ["20436-0.0", "20439-0.0", "20440-0.0", "20126-0.0", "2090-0.0", "2100-0.0"]
immune_cols = [
    "30000-0.0", "30080-0.0", "30120-0.0", "30130-0.0", "30140-0.0", "30180-0.0", "30190-0.0",
    "30200-0.0", "30710-0.0"]
metabol_cols = (
        ["23400-0.0", "23403-0.0"] + [f"23{i}-0.0" for i in range(405, 431)] + ["23437-0.0"]
        + [f"23{i}-0.0" for i in range(442, 447)] + [f"23{i}-0.0" for i in range(449, 454)]
        + [f"23{i}-0.0" for i in range(456, 460)] + [f"23{i}-0.0" for i in range(464, 468)]
        + ["23470-0.0"] + [f"23{i}-0.0" for i in range(473, 478)]
        + [f"23{i}-0.0" for i in range(488, 579)] + [f"23{i}-0.0" for i in range(584, 649)]
        + ["30690-0.0", "30740-0.0", "30750-0.0", "30760-0.0", "30860-0.0", "30870-0.0"])
morpho_cols = [f"27{i}-2.0" for i in range(329, 773)]
frailty_cols = ["2306-0.0", "120107-0.0", "924-0.0", "46-0.0", "47-0.0", "6164-0.0", "1011-0.0"]
covar_cols = [
    "21003-0.0", "31-0.0", "21000-0.0", "25741-2.0", "26521-2.0", "25000-2.0", "54-0.0",
    "25756-2.0", "25757-2.0", "25758-2.0", "25759-2.0", "20116-0.0", "1558-0.0", "21001-0.0",
    "6138-0.0"]
diagn_out_cols = ["icd10_diagn", "s2023_diagn", "mhq_diagn", "report_diagn", "patient", "control"]
out_cols = immune_cols + metabol_cols + morpho_cols + covar_cols + ["frailty"] + diagn_out_cols

# Data types
col_types = {"eid": str}
col_types.update({col: "Int64" for col in sympt_cols + diagn_cols + frailty_cols})
col_types.update({col: float for col in immune_cols+metabol_cols+morpho_cols+covar_cols})

# Get all columns for multi-column data fields
diagn_multi_cols = {"41270": [], "20002": [], "20544": [], "20003": []}
data_head = pd.read_table(args.raw_tsv, delimiter="\t", nrows=2, encoding=encoding, index_col="eid")
for col in data_head.columns:
    if col.split("-")[0] in diagn_multi_cols.keys():
        diagn_multi_cols[col.split("-")[0]].append(col)
        if col.split("-")[0] == "41270":
            col_types[col] = str
        else:
            col_types[col] = "Int64"

# read raw data by chunks
iterator = pd.read_table(
    args.raw_tsv, delimiter="\t", encoding=encoding, chunksize=chunksize,
    usecols=list(col_types.keys()), dtype=col_types, index_col="eid")
for data_df in iterator:

    # Exclusion Criteria 1: ICD-10: all F/G categories except those in icd_include, I60-69
    data_include = data_df.loc[
        ~data_df[diagn_multi_cols["41270"]].isin(icd_exclude).any(axis="columns")]
    if not data_include.empty:

        # Patients Criteria 1: ICD-10 diagnosis
        icd_diagn = data_include[diagn_multi_cols["41270"]].isin(icd_include).any(axis="columns")
        data_include = data_include.join(icd_diagn.rename("icd10_diagn"))

        # Patients Criteria 2: Smith et al. 2023 diagnosis
        s2023_diagn = data_include["20126-0.0"].isin([3, 4, 5])
        s2023_diagn = s2023_diagn.fillna(False)
        data_include = data_include.join(s2023_diagn.rename("s2023_diagn"))

        # Patients Criteria 3: MHQ diagnosis (based on CIDI-SF)
        mhq_sympt = (data_include["20536-0.0"] >= 1).astype("Int64")
        for sympt_col in sympt_cols[1:]:
            mhq_sympt = mhq_sympt + (data_include[sympt_col] == 1).astype("Int64")
        mhq_diagn = (data_include["20446-0.0"] == 1) | (data_include["20441-0.0"] == 1)
        mhq_diagn = mhq_diagn & (data_include["20436-0.0"] >= 3) & (data_include["20439-0.0"] >= 2)
        mhq_diagn = mhq_diagn & (data_include["20440-0.0"] >= 2) & (mhq_sympt >= 5)
        mhq_diagn = mhq_diagn.fillna(False)
        data_include = data_include.join(mhq_diagn.rename("mhq_diagn"))

        # Patients Criteria 4: Self-reported
        report_cond = data_include[diagn_multi_cols["20002"]].isin([1286]).any(axis="columns")
        report_mhq = data_include[diagn_multi_cols["20544"]].isin([11]).any(axis="columns")
        report_diagn = report_cond | report_mhq
        report_diagn = report_diagn.fillna(False)
        data_include = data_include.join(report_diagn.rename("report_diagn"))

        # patients: meeting 1 of 4 patients criteria
        patient = icd_diagn | s2023_diagn | mhq_diagn | report_diagn
        data_include = data_include.join(patient.rename("patient"))

        # Control Criteria 1: meeting none of 4 patients criteria
        none_diagn = ~icd_diagn & ~s2023_diagn & ~mhq_diagn & ~report_diagn
        # Control Criteria 2: no help seeking
        none_seek = (data_include["2090-0.0"] != 1) & (data_include["2100-0.0"] != 1)
        none_seek = none_seek.fillna(True)
        # Control Criteria 3: no main symptoms
        none_main_sympt = (data_include["20441-0.0"] != 1) & (data_include["20446-0.0"] != 1)
        none_main_sympt = none_main_sympt.fillna(True)
        # Control Criteria 4: no antidepressant use
        none_antidep = ~data_include[
            diagn_multi_cols["20003"]].isin(antidep_code).any(axis="columns")
        none_antidep = none_antidep.fillna(True)

        # controls: meeting all of 4 control criteria
        control = none_diagn & none_seek & none_main_sympt & none_antidep
        data_include = data_include.join(control.rename("control"))

        # Physical frailty (Jiang et al. 2024)
        data_include = phy_frailty(data_include)

        # Exclusion Criteria 1: patient or control
        data_out = data_include.loc[data_include["patient"] | data_include["control"]]
        # Exclusion Criteria 2: no missing data
        data_out = data_out[out_cols].dropna(axis="index", how="any")

        if not data_out.empty:
            if args.out_csv.exists():
                data_out.to_csv(args.out_csv, mode="a", header=False)
            else:
                data_out.to_csv(args.out_csv)

# Exclusion Criteria 3: unrelated
col_types = {"eid": str, "frailty": "Int64"}
col_types.update({col: float for col in immune_cols+metabol_cols+morpho_cols+covar_cols})
col_types.update({col: bool for col in diagn_out_cols})
data_out = pd.read_csv(
    args.out_csv, usecols=list(col_types.keys()), dtype=col_types, index_col="eid")

sub_out = []
iterator = pd.read_table(
    args.genetic_tsv, delimiter="\t", encoding=encoding, chunksize=chunksize,
    usecols=["eid", "22021-0.0"], dtype={"eid": str, "22021-0.0": "Int64"}, index_col="eid")
for data_df in iterator:
    data_unrelated = data_df.loc[(data_df.index.isin(data_out.index)) & (data_df["22021-0.0"] == 0)]
    sub_out.extend(data_unrelated.index)
data_out = data_out.loc[data_out.index.isin(sub_out)]

# Exclusion Criteria 4: no missing RSFC data
sub_list = data_out.index.tolist()
rsfc_full = pd.DataFrame(index=sub_list, columns=range(1485), dtype=float)
rsfc_part = pd.DataFrame(index=sub_list, columns=range(1485), dtype=float)

root_data_dir = Path(args.work_dir, "ukb_raw")
dl.install(root_data_dir, source=args.ukb_raw_url)
for subject in sub_list:
    sub_dir = Path(root_data_dir, f"sub-{subject}")
    if sub_dir.exists():
        dl.get(sub_dir, dataset=root_data_dir, get_data=False)
        rsfc_full = fill_pd_row(rsfc_full, subject, "full", sub_dir)
        rsfc_part = fill_pd_row(rsfc_part, subject, "partial", sub_dir)
dl.remove(dataset=root_data_dir, reckless="kill")

rsfc_all = rsfc_full.join(rsfc_part, lsuffix="_full", rsuffix="_part")
data_out = data_out.join(rsfc_all).dropna(axis="index", how="any")
data_out.to_csv(args.out_csv)
