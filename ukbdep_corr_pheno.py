from pathlib import Path
from textwrap import wrap
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
        data: pd.DataFrame, fields_pd: pd.DataFrame, pheno: str, out_dir: Path) -> pd.DataFrame:
    out_name = col_type.replace(" ", "-")
    genders = {0: "female", 1: "male"}
    data_corr = {}
    for gender_i in [0, 1]:
        for diagn in [0, 1]:
            data_curr = data.loc[(data["31-0.0"] == gender_i) & (data["MDD diagnosis"] == diagn)]
            for pheno_col in field_dict[pheno]:
                fields_curr = fields_pd.loc[fields_pd["Field ID"] == pheno_col.split("-")[0]]
                field_desc = fields_curr["Field Description"].values[0]
                for dep_col in field_dict["Dep score"]:
                    ind = f"{pheno_col}-{dep_col}-{genders[gender_i]}-{diagn}"
                    r, p = pcorr(data_curr, pheno_col, dep_col)
                    data_corr[ind] = {
                        "Type": pheno, "Data field": pheno_col, "r": r, "p": p,
                        "Absolute r": np.abs(r), "Gender": genders[gender_i],
                        "MDD diagnosis": diagn, "Depressive score field": dep_col,
                        "Field description": field_desc, "Depressive score": dep_desc[dep_col]}

    data_corr = pd.DataFrame(data_corr).T
    data_corr.to_csv(Path(out_dir, f"ukb_dep_corr_pheno_{out_name}.csv"))
    return data_corr


parser = argparse.ArgumentParser(
        description="Compute correlations between depressive sum scores and brain/body biomarkers",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("--data_dir", type=Path, help="Absolute path to extracted data")
parser.add_argument("--sel_csv", type=Path, help="Absolute path to table of selected fields")
parser.add_argument("--img_dir", type=Path, help="Absolute path to output plots directory")
parser.add_argument("--out_dir", type=Path, help="Absolute path to output directory")
args = parser.parse_args()

dep_desc = {
    "Sum score (cluster 6)": f"Depressive mood\nsymptoms",
    "Sum score (cluster 5)": f"Depressive somatic\nsymptoms"}
field_dict = {"Diagn ICD10": [], "Dep score": ["Sum score (cluster 5)", "Sum score (cluster 6)"]}
col_dtypes = {
    "eid": str, "31-0.0": float, "21003-2.0": float, "Sum score (cluster 5)": float,
    "Sum score (cluster 6)": float, "MDD diagnosis": float}
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

# Correlation analysis
col_list_req = field_dict["Dep score"] + ["31-0.0", "21003-2.0", "eid", "MDD diagnosis"]
col_type_pheno = [
    "Abdom comp", "Brain GMV", "Brain WM", "Blood biochem", "Blood count", "NMR metabol"]
data_corrs = []
for col_type in col_type_pheno:
    # Association sample
    col_list_curr = field_dict[col_type] + col_list_req
    col_dtype_curr = {key: col_dtypes[key] for key in col_list_curr}
    pheno_name = col_type.replace(" ", "-")
    data_pheno = pd.read_csv(
        Path(args.data_dir, f"ukb_data_{pheno_name}_clusters.csv"), usecols=col_list_curr,
        dtype=col_dtype_curr, index_col="eid")
    data_pheno = data_pheno.assign(age2=np.power(data_pheno["21003-2.0"], 2))

    # Add derived measures for blood count
    if col_type == "Blood count":
        data_pheno.loc[:, "sii"] = (
                data_pheno["30140-2.0"] * data_pheno["30080-2.0"] / data_pheno["30120-2.0"])
        data_pheno.loc[:, "nlr"] = data_pheno["30140-2.0"] / data_pheno["30120-2.0"]
        data_pheno.loc[:, "plr"] = data_pheno["30080-2.0"] / data_pheno["30120-2.0"]
        data_pheno.loc[:, "lmr"] = data_pheno["30120-2.0"] / data_pheno["30190-2.0"]
        fields = pd.concat([fields, pd.DataFrame({
            "Field ID": ["sii", "nlr", "plr", "lmr"], "Field Description": [
                "Systemic immune-inflammation index (SII)", "Neutrophil-to-lymphocyte ratio (NLR)",
                "Platelet-to-lymphocyte ratio (PLR)", "Lymphocyte-to-monocyte ratio (LMR)"]})])

    data_corr_curr = corr_analysis(data_pheno, fields, col_type, args.out_dir)
    data_corrs.append(data_corr_curr)
data_corr_all = pd.concat(data_corrs, axis="index", join="inner")

# Correct for multiple comparisons
fdr = multipletests(data_corr_all["p"], method="fdr_bh")
data_corr_fdr = data_corr_all.loc[fdr[0]]
data_corr_fdr.to_csv(Path(args.out_dir, f"ukb_dep_corr_pheno_fdr.csv"))

# Plot correlation values
plot_type = ["Abdom comp", "Blood biochem", "Blood count", "NMR metabol"]
data_plot = data_corr_fdr.loc[data_corr_fdr["Type"].isin(plot_type)]
for diagn in [0, 1]:
    data_plot_curr = data_plot.loc[data_plot["MDD diagnosis"] == diagn]
    if data_plot_curr.shape[0]:
        data_plot_curr = data_plot_curr.sort_values(by="Absolute r", ascending=False)
        with sns.plotting_context(context="paper", font_scale=1.5):
            g = sns.catplot(
                kind="strip", data=data_plot_curr, x="r", y="Field description",
                hue="Depressive score", col="Gender", palette=["darkseagreen", "pink"],
                jitter=False, dodge=False, height=10, aspect=1, size=15, linewidth=1,
                hue_order=[f"Depressive mood\nsymptoms", f"Depressive somatic\nsymptoms"],
                edgecolor="black")
            for ax in g.axes.flat:
                ax.axvline(color="lightgray", linestyle="--")
                ax.yaxis.grid(True)
        plt.savefig(
            Path(args.img_dir, f"ukb_dep_corr_pheno_{diagn}.png"), bbox_inches="tight", dpi=500)
        plt.close()
