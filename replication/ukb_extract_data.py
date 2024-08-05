from os import remove
from pathlib import Path
import argparse

import pandas as pd

parser = argparse.ArgumentParser(
    description="extract UKB data for depression prediction",
    formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("raw_tsv", type=Path, help="Absolute path to UKB raw data tsv")
parser.add_argument("genetic_tsv", type=Path, help="Absolute path to UKB genetic data tsv")
parser.add_argument("parser_dir", type=Path, help="Absolute path to ukbb_parser directory")
parser.add_argument("antidep_csv", type=Path, help="Absolute path to antidepressant code csv")
parser.add_argument("out_csv", type=Path, help="Absolute path to output csv file")
args = parser.parse_args()

# Set-ups
encoding = "ISO-8859-1"
chunksize = 1000
if args.out_csv.exists():
    remove(args.out_csv)
antidep_code = pd.read_csv(args.antidep_csv, index_col="Drug name")["Code in UKB"].tolist()

# ICD-10 set-up
coding19 = pd.read_csv(
    Path(args.parser_dir, "ukbb_parser", "scripts", "data", "icd10_level_map.csv"),
    index_col="Coding")
icd_selectable = coding19.loc[coding19["Selectable"] == "Yes"].index.tolist()
icd_include = ["F32", "F33", "F34", "F38", "F39"]
icd_exclude = [icd for icd in icd_selectable if "F" in icd or "G" in icd]
icd_exclude.extend([f"I0{i}" for i in range(60, 70) if f"I0{i}" in icd_selectable])
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
        + [f"23{i}-0.0" for i in range(456, 460)] + [f"23{i}-0.0" for i in range(456, 460)]
        + [f"23{i}-0.0" for i in range(464, 468)] + ["23470-0.0", "23473-0.0"]
        + [f"23{i}-0.0" for i in range(475, 478)] + [f"23{i}-0.0" for i in range(488, 579)]
        + [f"23{i}-0.0" for i in range(584, 649)]
        + ["30690-0.0", "30740-0.0", "30750-0.0", "30760-0.0", "30860-0.0", "30870-0.0"])
morpho_cols = [f"27{i}-2.0" for i in range(329, 773)]
rsfc_cols = ["25751-2.0", "25753-2.0"]
covar_cols = [
    "21003-0.0", "31-0.0", "21000-0.0", "25741-2.0", "26521-2.0", "25000-2.0", "54-0.0",
    "25756-2.0", "25757-2.0", "25758-2.0", "25759-2.0"]
diagn_out_cols = ["icd10_diagn", "s2023_diagn", "mhq_diagn", "report_diagn", "patient", "control"]
out_cols = immune_cols + metabol_cols + morpho_cols + rsfc_cols + covar_cols + diagn_out_cols

# Data types
col_types = {"eid": str}
col_types.update({col: "Int64" for col in sympt_cols+diagn_cols})
col_types.update({col: float for col in immune_cols+metabol_cols+morpho_cols+covar_cols})
col_types.update({col: str for col in rsfc_cols})

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
        icds = data_include[diagn_multi_cols["41270"]].values.tolist()
        icd_diagn = pd.Series(icds, index=data_include.index).isin(icd_include)
        icd_diagn = icd_diagn.fillna(False)
        data_include = data_include.join(icd_diagn.rename("icd10_diagn"))

        # Patients Criteria 2: Smith et al. 2023 diagnosis
        s2023_diagn = (data_include["20126-0.0"] == 1) & (data_include["20126-0.0"].notna())
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
        report_cond = data_include[diagn_multi_cols["20002"]].values.tolist()
        report_mhq = data_include[diagn_multi_cols["20544"]].values.tolist()
        report_diagn = pd.Series(report_cond, index=data_include.index).isin([1286])
        report_diagn = report_diagn | pd.Series(report_mhq, index=data_include.index).isin([11])
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
        antidep = data_include[diagn_multi_cols["20003"]].values.tolist()
        none_antidep = ~pd.Series(antidep, index=data_include.index).isin(antidep_code)
        none_antidep = none_antidep.fillna(True)

        # controls: meeting all of 4 control criteria
        control = none_diagn & none_seek & none_main_sympt & none_antidep
        data_include = data_include.join(control.rename("control"))

        # Exclusion Criteria 1: patient or control
        data_out = data_include.loc[data_include["patient"] | data_include["control"]]
        # Exclusion Criteria 3: no missing data
        data_out = data_out[out_cols].dropna(axis="index", how="any")

        if not data_out.empty:
            if args.out_csv.exists():
                data_out.to_csv(args.out_csv, mode="a", header=False)
            else:
                data_out.to_csv(args.out_csv)

# Exclusion Criteria 2: unrelated
col_types = {"eid": str}
col_types.update({col: float for col in immune_cols+metabol_cols+morpho_cols+covar_cols})
col_types.update({col: str for col in rsfc_cols})
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
data_out.to_csv(args.out_csv)
