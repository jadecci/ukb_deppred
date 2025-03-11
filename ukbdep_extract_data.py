from os import remove
from pathlib import Path
import argparse

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(
        description="Multimodal brain-based psychometric prediction",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("--raw_tsv", type=Path, help="Absolute path to UK Biobank raw data tsv file")
parser.add_argument("--sel_csv", type=Path, help="Absolute path to table of selected fields")
parser.add_argument("--wd_csv", type=Path, help="Absolute path to list of withdrawn subjects")
parser.add_argument("--out_dir", type=Path, help="Absolute path to output directory")
args = parser.parse_args()

encoding = "ISO-8859-1"
chunksize = 1000
field_dict = {"Diagn ICD10": [], "Death record": []}
col_dtypes = {"eid": str}
excludes = {}

# Selected field information
field_cols = {
    "Field ID": str, "Type": str, "Instance": "Int64", "To Exclude 1": float, "To Exclude 2": float}
fields = pd.read_csv(args.sel_csv, usecols=list(field_cols.keys()), dtype=field_cols)
for _, field_row in fields.iterrows():
    col_id = f"{field_row['Field ID']}-{field_row['Instance']}.0"
    if field_row["Type"] in field_dict.keys():
        field_dict[field_row["Type"]].append(col_id)
    else:
        field_dict[field_row["Type"]] = [col_id]
    col_dtypes[col_id] = float
    if not np.isnan(field_row["To Exclude 1"]):
        excludes[col_id] = [field_row["To Exclude 1"]]
        if not np.isnan(field_row["To Exclude 2"]):
            excludes[col_id].append(field_row["To Exclude 2"])

# ICD-10 code, death record
data_head = pd.read_table(args.raw_tsv, delimiter="\t", encoding=encoding, nrows=2)
for col in data_head.columns:
    if col.split("-")[0] in "41270":
        field_dict["Diagn ICD10"].append(col)
        col_dtypes[col] = str
    if col.split("-")[0] == "40023":
        field_dict["Death record"].append(col)
        col_dtypes[col] = float

# Extract all data
all_data_file = Path(args.out_dir, "extracted_data", "ukb_data_all.csv")
if all_data_file.exists():
    remove(all_data_file)
data_iter = pd.read_table(
    args.raw_tsv, delimiter="\t", encoding=encoding, chunksize=chunksize,
    usecols=list(col_dtypes.keys()), dtype=col_dtypes, index_col="eid")
for data_curr in data_iter:
    data_out = data_curr.dropna(axis="index", how="any", subset=field_dict["Dep sympt"])
    if not data_out.empty:
        if all_data_file.exists():
            data_out.to_csv(all_data_file, mode="a", header=False)
        else:
            data_out.to_csv(all_data_file)

# Remove withdrawn subjects
all_data = pd.read_csv(
    all_data_file, usecols=list(col_dtypes.keys()), dtype=col_dtypes, index_col="eid")
print(f"Subjects with all depressive symptom items: N = {all_data.shape[0]}")
wd_sub = pd.read_csv(args.wd_csv, header=None).squeeze()
all_data = all_data.drop(wd_sub, errors="ignore")
all_data.to_csv(all_data_file)
print(f"Subjects with non-retracted consent: N = {all_data.shape[0]}")

# Define test set for each phenotype category, and the remaining as training set
field_incl = (
        field_dict["Brain GMV"] + field_dict["Brain WM"] + field_dict["Diagn ICD10"]
        + field_dict["Death record"] + ["31-0.0"])
col_type_test = [
    "Abdom comp", "Blood biochem", "Blood count", "NMR metabol", "Brain GMV", "Brain WM"]
data_train = all_data.copy()
for col_type in col_type_test:
    field_req = field_dict[col_type] + field_dict["Dep sympt"] + field_dict["Sociodemo"]
    data_curr = all_data[field_req+field_incl].copy()
    data_test = data_curr.dropna(axis="index", how="any", subset=field_req)
    for col in field_dict[col_type]:
        for exclude in excludes[col]:
            data_test = data_test.loc[data_test[col] != exclude]
    pheno_name = col_type.replace(" ", "-")
    data_test.to_csv(Path(args.out_dir, "extracted_data", f"ukb_data_{pheno_name}.csv"))
    print(f"Subjects with all phenotypes for {col_type}: N = {data_test.shape[0]}")
    data_train = data_train.drop(data_test.index, errors="ignore")
data_train.to_csv(Path(args.out_dir, "extracted_data", "ukb_data_train.csv"))
print(f"Training set: N = {data_train.shape[0]}")
