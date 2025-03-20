from pathlib import Path
import argparse

from scipy.stats import pearsonr, ttest_ind
import pandas as pd
import seaborn as sns


parser = argparse.ArgumentParser(
        description="Plot association between depressive scores and sociodemographic phenotypes",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument(
    "--data_csv", type=Path, help="Absolute path to the test data csv file to use")
parser.add_argument("--img_dir", type=Path, help="Absolute path to output plots directory")
args = parser.parse_args()

field_dict = {"Death record": [], "Dep score": [], "Sociodemo": []}
col_dtypes = {"eid": str}
args.img_dir.mkdir(parents=True, exist_ok=True)

# Phenotype field information
field_cols = {"Field ID": str, "Type": str, "Instance": "Int64"}
fields = pd.read_csv(args.sel_csv, usecols=list(field_cols.keys()), dtype=field_cols)
for _, field_row in fields.loc[fields["Type"] == "Sociodemo"].iterrows():
    col_id = f"{field_row['Field ID']}-{field_row['Instance']}.0"
    field_dict["Sociodemo"].append(col_id)
    col_dtypes[col_id] = float

# Depressive sum scores of interest
for cluster in [1, 2]:
    col_id = f"Sum score (cluster {cluster})"
    field_dict["Dep score"].append(col_id)
    col_dtypes[col_id] = float
dep_desc = {
    "Sum score (cluster 1)": f"Depressive mood\nsymptoms",
    "Sum score (cluster 2)": f"Depressive somatic\nsymptoms"}
