from pathlib import Path
import argparse

from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def pcorr(data: pd.DataFrame, x: str, y: str) -> tuple[float, float]:
    covar = ["21003-2.0", "age2"]
    x_resid = data[x] - np.dot(data[covar], np.linalg.lstsq(data[covar], data[x], rcond=-1)[0])
    r, p = pearsonr(x_resid, data[y])
    return r, p


def corr_analysis(
        data: dict, pheno_dict: dict, fields_pd: pd.DataFrame, gender: str,
        out_dir: Path) -> pd.DataFrame:
    genders = {0: "female", 1: "male"}

    # Correlation for all phenotype
    data_corr = {}
    for pheno, data_curr in data.items():
        for pheno_col in field_dict[pheno]:
            fields_curr = fields_pd.loc[fields_pd["Field ID"] == pheno_col.split("-")[0]]
            field_desc = fields_curr["Field Description"].values[0]
            for dep_col in field_dict["Dep score"]:
                if gender == "mixed":
                    ind = f"{pheno_col}-{dep_col}-{gender}"
                    r, p = pcorr(data_curr, pheno_col, dep_col)
                    data_corr[ind] = {
                        "Type": pheno, "Data field": pheno_col, "r": r, "p": p, "Gender": gender,
                        "Depressive score field": dep_col, "Absolute r": np.abs(r),
                        "Field description": field_desc, "Depressive score": dep_desc[dep_col]}
                elif gender == "split":
                    for gender_i in [0, 1]:
                        ind = f"{pheno_col}-{dep_col}-{genders[gender_i]}"
                        data_gender_curr = data_curr.loc[data_curr["31-0.0"] == gender_i]
                        r, p = pcorr(data_gender_curr, pheno_col, dep_col)
                        data_corr[ind] = {
                            "Type": pheno, "Data field": pheno_col, "r": r, "p": p,
                            "Gender": genders[gender_i], "Depressive score field": dep_col,
                            "Absolute r": np.abs(r), "Field description": field_desc,
                            "Depressive score": dep_desc[dep_col]}
    data_corr = pd.DataFrame(data_corr).T
    data_corr.to_csv(Path(out_dir, f"ukb_dep_corr_pheno_{gender}.csv"))

    # Correct for multiple comparisons
    fdr = multipletests(data_corr["p"], method="fdr_bh")
    data_corr_fdr = data_corr.loc[fdr[0]]
    data_corr_fdr.to_csv(Path(out_dir, f"ukb_dep_corr_pheno_fdr_{gender}.csv"))
    return data_corr_fdr


parser = argparse.ArgumentParser(
        description="Compute correlations between depressive sum scores and brain/body biomarkers",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("--data_dir", type=Path, help="Absolute path to extracted data")
parser.add_argument("--sel_csv", type=Path, help="Absolute path to table of selected fields")
parser.add_argument("--img_dir", type=Path, help="Absolute path to output plots directory")
parser.add_argument("--out_dir", type=Path, help="Absolute path to output directory")
args = parser.parse_args()

field_dict = {"Diagn ICD10": [], "Death record": [], "Dep score": []}
col_dtypes = {"eid": str, "31-0.0": float, "21003-2.0": float}
args.img_dir.mkdir(parents=True, exist_ok=True)
args.out_dir.mkdir(parents=True, exist_ok=True)

# Phenotype field information
field_cols = {
    "Field ID": str, "Field Description": str, "Type": str, "Instance": "Int64", "Notes": str}
fields = pd.read_csv(args.sel_csv, usecols=list(field_cols.keys()), dtype=field_cols)
for _, field_row in fields.iterrows():
    col_id = f"{field_row['Field ID']}-{field_row['Instance']}.0"
    if field_row["Type"] in field_dict.keys():
        field_dict[field_row["Type"]].append(col_id)
    else:
        field_dict[field_row["Type"]] = [col_id]
    col_dtypes[col_id] = float

# Depressive sum scores of interest
for cluster in [1, 2]:
    col_id = f"Sum score (cluster {cluster})"
    field_dict["Dep score"].append(col_id)
    col_dtypes[col_id] = float
dep_desc = {
    "Sum score (cluster 1)": f"Depressive mood\nsymptoms",
    "Sum score (cluster 2)": f"Depressive somatic\nsymptoms"}

# Test data
col_type_test = [
    "Abdom comp", "Blood biochem", "Blood biochem 1", "Blood count", "Blood count 1",
    "Blood count 2", "NMR metabol", "NMR metabol 1"]
data_test = dict.fromkeys(col_type_test)
for col_type in col_type_test:
    col_list_curr = (
            field_dict[col_type] + field_dict["Brain GMV"] + field_dict["Brain WM"]
            + field_dict["Dep score"] + ["31-0.0", "21003-2.0", "eid"])
    col_dtype_curr = {key: col_dtypes[key] for key in col_list_curr}
    pheno_name = col_type.replace(" ", "-")
    data_test_curr = pd.read_csv(
        Path(args.data_dir, f"ukb_data_{pheno_name}_clusters.csv"), usecols=col_list_curr,
        dtype=col_dtype_curr, index_col="eid")
    data_test_curr = data_test_curr.assign(age2=np.power(data_test_curr["21003-2.0"], 2))
    data_test[col_type] = data_test_curr
data_test["Brain GMV"] = data_test["Blood count"]
data_test["Brain WM"] = data_test["Blood count"]

# Add derived measures for blood count
for ses, col_type in enumerate(["Blood count", "Blood count 1", "Blood count 2"]):
    data_test[col_type].loc[:, "sii"] = (
            data_test[col_type][f"30140-{ses}.0"] * data_test[col_type][f"30080-{ses}.0"]
            / data_test[col_type][f"30120-{ses}.0"])
    data_test[col_type].loc[:, "nlr"] = (
            data_test[col_type][f"30140-{ses}.0"] / data_test[col_type][f"30120-{ses}.0"])
    data_test[col_type].loc[:, "plr"] = (
            data_test[col_type][f"30080-{ses}.0"] / data_test[col_type][f"30120-{ses}.0"])
    data_test[col_type].loc[:, "lmr"] = (
            data_test[col_type][f"30120-{ses}.0"] / data_test[col_type][f"30190-{ses}.0"])
fields = pd.concat([fields, pd.DataFrame({
    "Field ID": ["sii", "nlr", "plr", "lmr"], "Field Description": [
        "Systemic immune-inflammation index (SII)", "Neutrophil-to-lymphocyte ratio (NLR)",
        "Platelet-to-lymphocyte ratio (PLR)", "Lymphocyte-to-monocyte ratio (LMR)"]})])

# Correlation analysis for mixed cohort
data_corr_mixed = corr_analysis(data_test, field_dict, fields, "mixed", args.out_dir)
with sns.plotting_context(context="paper", font_scale=1.5):
    g = sns.catplot(
        kind="strip", data=data_corr_mixed, x="r", y="Type", hue="Depressive score",
        hue_order=[f"Depressive mood\nsymptoms", f"Depressive somatic\nsymptoms"],
        palette=["darkseagreen", "pink"], jitter=False, dodge=True, height=8, aspect=1, size=10,
        linewidth=1)
    for ax in g.axes.flat:
        ax.axvline(color="lightgray", linestyle="--")
plt.savefig(Path(args.img_dir, "ukb_dep_corr_pheno_mixed.png", bbox_inches="tight", dpi=500))

# Save top 3 biomarkers for somatic symptoms in each body category
# Use the latest visit with sufficient number of subjects
for col_type in ["Abdom comp", "Blood biochem", "Blood count 2", "NMR metabol 1"]:
    pheno_name = col_type.replace(" ", "-")
    data_marker = data_corr_mixed.loc[
        (data_corr_mixed["Type"] == col_type)
        & (data_corr_mixed["Depressive score field"] == "Sum score (cluster 2)")]
    data_sort = data_marker.sort_values(by="Absolute r", axis="index", ascending=False)
    data_top = data_sort.head(3)
    if len(np.unique(data_top["Data field"])) < 3:
        data_top = data_sort.head(4)
    data_top.to_csv(Path(args.out_dir, f"ukb_dep_corr_somatic_{pheno_name}_fdr_mixed.csv"))

# Correlation analysis with separated genders
data_corr_split = corr_analysis(data_test, field_dict, fields, "split", args.out_dir)
with sns.plotting_context(context="paper", font_scale=1.5):
    g = sns.catplot(
        kind="strip", data=data_corr_split, x="r", y="Type", hue="Depressive score", col="Gender",
        hue_order=[f"Depressive mood\nsymptoms", f"Depressive somatic\nsymptoms"],
        palette=["darkseagreen", "pink"], jitter=False, dodge=True, height=8, aspect=0.7, size=10,
        linewidth=1)
    for ax in g.axes.flat:
        ax.axvline(color="lightgray", linestyle="--")
plt.savefig(Path(args.img_dir, "ukb_dep_corr_pheno_split.png", bbox_inches="tight", dpi=500))
