from pathlib import Path
import argparse

from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_fields(field_file: Path) -> tuple[dict, dict]:
    field_file_cols = {"Field ID": str, "Field Description": str, "Type": str, "Instance": str}
    fields = pd.read_csv(field_file, usecols=list(field_file_cols.keys()), dtype=field_file_cols)

    field_cols = {}
    field_desc = {}
    for _, field in fields.iterrows():
        col_curr = f"{field['Field ID']}-{field['Instance']}.0"
        if field["Type"] in field_cols.keys():
            field_cols[field["Type"]].append(col_curr)
        else:
            field_cols[field["Type"]] = [col_curr]
        field_desc[col_curr] = field["Field Description"]
    return field_cols, field_desc


def plot_corr(data: pd.DataFrame, outfile: Path):
    hue_order = [f"Sum score (cluster {i})" for i in [1, 2, 3, 4]]
    sns.catplot(
        data, kind="swarm", x="r", y="Type", hue="Depression score", col="Gender",
        dodge=True, palette="Set2", size=5, height=10, aspect=0.5, hue_order=hue_order)
    plt.savefig(outfile)
    plt.close()
    return


parser = argparse.ArgumentParser(
    description="Correlation analysis between depression scores and phenotypes in UKB",
    formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("data_dir", type=Path, help="Absolute path to input data directory")
parser.add_argument("field_pheno", type=Path, help="Absolute path to selected phenotype fields csv")
parser.add_argument("field_comp", type=Path, help="Absolute path to selected composite fields csv")
parser.add_argument("out_dir", type=Path, help="Absolute path to output data directory")
parser.add_argument("img_dir", type=Path, help="Absolute path to output images directory")
args = parser.parse_args()

args.out_dir.mkdir(parents=True, exist_ok=True)
args.img_dir.mkdir(parents=True, exist_ok=True)

# Data fields to read and/or write
pheno_cols, pheno_desc = get_fields(args.field_pheno)
comp_cols, comp_desc = get_fields(args.field_comp)
pheno_cols.update(comp_cols)
pheno_desc.update(comp_desc)
dep_col_list = [f"Sum score (cluster {i})" for i in [1, 2, 3, 4]]
dtypes = {"eid": str, "31-0.0": float}
dtypes.update({col: float for col in dep_col_list})

# Compute correlations for each gender separately
corr_ind = [
    f"{col}_{i}_{j}" for col in list(pheno_desc.keys()) for i in range(2) for j in range(4)]
corr_all = pd.DataFrame(
    columns=["Type", "Field ID", "Phenotype description", "Depression score", "Gender", "r", "p"],
    index=corr_ind)
for pheno_type, pheno_col_list in pheno_cols.items():
    for i, gender in enumerate(["female", "male"]):
        pheno_name = pheno_type.replace(" ", "-")
        pheno_name = pheno_name.replace("/", "-")
        dtypes_pheno = dtypes.copy()
        dtypes_pheno.update({col: float for col in pheno_col_list})
        data_curr = pd.read_csv(
            Path(args.data_dir, f"ukb_extracted_data_{pheno_name}_{gender}_clusters.csv"),
            usecols=list(dtypes_pheno.keys()), dtype=dtypes_pheno, index_col="eid")
        for col in pheno_col_list:
            for j in range(4):
                r, p = pearsonr(data_curr[col], data_curr[dep_col_list[j]])
                corr_all.loc[f"{col}_{i}_{j}"] = {
                    "Type": pheno_type, "Field ID": col, "Phenotype description": pheno_desc[col],
                    "Depression score": dep_col_list[j], "Gender": gender, "r": r, "p": p}

# Only keep significant correlations (correcting for multiple comparisons)
fdr_h = multipletests(corr_all["p"], method="bonferroni")
corr_sig = corr_all.loc[[h for h in fdr_h[0]]]
corr_sig.to_csv(Path(args.out_dir, "ukb_dep-pheno_corr_sig.csv"))
corr_sig["Absolute r"] = abs(corr_sig["r"])
plot_corr(corr_sig, Path(args.img_dir, "ukb_corr.eps"))
