from os import remove
from pathlib import Path
import argparse

import datalad.api as dl
import pandas as pd


def get_fields(field_file: Path) -> tuple[dict, dict]:
    field_file_cols = {
        "Field ID": str, "To Exclude 1": float, "To Exclude 2": float, "To Exclude 3": float,
        "Type": str, "Instance": str}
    fields = pd.read_csv(field_file, usecols=list(field_file_cols.keys()), dtype=field_file_cols)

    field_cols = {}
    excludes = {}
    for _, field in fields.iterrows():
        col_curr = f"{field['Field ID']}-{field['Instance']}.0"
        if field["Type"] in field_cols.keys():
            field_cols[field["Type"]].append(col_curr)
        else:
            field_cols[field["Type"]] = [col_curr]
        excludes[col_curr] = [field[f"To Exclude {i+1}"] for i in range(3)]
    return field_cols, excludes


parser = argparse.ArgumentParser(
    description="extract UKB data for depression prediction",
    formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("raw_tsv", type=Path, help="Absolute path to UKB raw data tsv")
parser.add_argument("genetic_tsv", type=Path, help="Absolute path to UKB genetic data tsv")
parser.add_argument("field_pheno", type=Path, help="Absolute path to selected phenotype fields csv")
parser.add_argument("field_comp", type=Path, help="Absolute path to selected composite fields csv")
parser.add_argument("field_dep", type=Path, help="Absolute path to selected depression fields csv")
parser.add_argument("work_dir", type=Path, help="Absolute path to working directory")
parser.add_argument("out_dir", type=Path, help="Absolute path to output directory")
parser.add_argument(
    "--seed", type=int, dest="seed", default=42, help="Random state seed for train-test split")
args = parser.parse_args()

# Set-ups
encoding = "ISO-8859-1"
chunksize = 1000

# Data fields to read and/or write
pheno_cols, pheno_excludes = get_fields(args.field_pheno)
comp_cols, _ = get_fields(args.field_comp)
dep_cols, dep_excludes = get_fields(args.field_dep)
in_excludes = pheno_excludes | dep_excludes
in_dtypes = {"eid": str, "31-0.0": float}
in_dtypes.update({col: float for col_list in pheno_cols.values() for col in col_list})
in_dtypes.update({col: "Int64" for col_list in dep_cols.values() for col in col_list})
out_dtypes = in_dtypes.copy()
out_dtypes.update({col: float for col_list in comp_cols.values() for col in col_list})

# Read raw data by chunks
all_data_file = Path(args.work_dir, "ukb_extracted_data_all.csv")
if all_data_file.exists():
    remove(all_data_file)
iterator = pd.read_table(
    args.raw_tsv, delimiter="\t", encoding=encoding, chunksize=chunksize,
    usecols=list(in_dtypes.keys()), dtype=in_dtypes, index_col="eid")
for data_df in iterator:
    data_out = data_df.dropna(axis="index", how="all")
    data_out = data_out.dropna(axis="index", how="any", subset=list(dep_cols.keys()))
    data_out = data_out.loc[~(data_out[list(dep_cols.keys())] == 1).all(axis=1)]
    for col in list(pheno_cols.keys())+list(dep_cols.keys()):
        for exclude in in_excludes[col]:
            data_out = data_out.loc[data_out[col] != exclude]
    if not data_out.empty:
        data_out["c001-0.0"] = data_out["48-2.0"] / data_out["49-2.0"]
        data_out["c002-0.0"] = data_out["30140-0.0"] * data_out["30080-0.0"] / data_out["30120-0.0"]
        data_out["c003-0.0"] = data_out["30140-0.0"] / data_out["30120-0.0"]
        data_out["c004-0.0"] = data_out["30080-0.0"] / data_out["30120-0.0"]
        data_out["c005-0.0"] = data_out["30120-0.0"] / data_out["30190-0.0"]
        if all_data_file.exists():
            data_out.to_csv(all_data_file, mode="a", header=False)
        else:
            data_out.to_csv(all_data_file)

# Only use unrelated subjects
data_out = pd.read_csv(
    all_data_file, usecols=list(out_dtypes.keys()), dtype=out_dtypes, index_col="eid")
sub_out = []
iterator = pd.read_table(
    args.genetic_tsv, delimiter="\t", encoding=encoding, chunksize=chunksize,
    usecols=["eid", "22021-0.0"], dtype={"eid": str, "22021-0.0": "Int64"}, index_col="eid")
for data_df in iterator:
    data_unrelated = data_df.loc[(data_df.index.isin(data_out.index)) & (data_df["22021-0.0"] == 0)]
    sub_out.extend(data_unrelated.index)
data_unrelated = data_out.loc[data_out.index.isin(sub_out)]

# Split into train and test set
data_train = data_unrelated.copy().sample(frac=0.5, random_state=args.seed)
data_train.to_csv(Path(args.out_dir, "ukb_extracted_data_train.csv"))
for pheno_type, pheno_col_list in pheno_cols.items():
    data_pheno = data_unrelated[pheno_col_list+["31-0.0"]+list(dep_cols.keys())].copy()
    data_pheno = data_pheno.drop(data_train.index).dropna()
    pheno_name = pheno_type.replace(" ", "-")
    pheno_name = pheno_name.replace("/", "-")
    data_pheno.to_csv(Path(args.out_dir, f"ukb_extracted_data_{pheno_name}.csv"))
