from pathlib import Path
import argparse

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


parser = argparse.ArgumentParser(
        description="Plot association between depressive scores and sociodemographic phenotypes",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("--data_dir", type=Path, help="Absolute path to extracted data")
parser.add_argument("--icd_code_csv", type=Path, help="Absolute path to ICD10 code mapping csv")
parser.add_argument("--out_dir", type=Path, help="Absolute path to output directory")
parser.add_argument("--img_dir", type=Path, help="Absolute path to output plots directory")
args = parser.parse_args()

args.out_dir.mkdir(parents=True, exist_ok=True)
args.img_dir.mkdir(parents=True, exist_ok=True)
code_map = pd.read_csv(args.icd_code_csv)
col_dtypes = {"eid": str, "Sum score (cluster 1)": float, "Sum score (cluster 2)": float}
dep_clusters = ["Sum score (cluster 1)", "Sum score (cluster 2)"]
dep_desc = {
    "Sum score (cluster 1)": f"Depressive mood\nsymptoms",
    "Sum score (cluster 2)": f"Depressive somatic\nsymptoms"}

# ICD-10 code
icd_list = []
data_head = pd.read_csv(Path(args.data_dir, "ukb_data_Abdom-comp_clusters.csv"), nrows=2)
for col in data_head.columns:
    if col.split("-")[0] == "41270":
        icd_list.append(col)
        col_dtypes[col] = str

# Test data
col_type_test = ["Abdom comp", "Blood biochem", "Blood count 2", "NMR metabol 1"]
data_test = []
for col_type in col_type_test:
    pheno_name = col_type.replace(" ", "-")
    data_test_curr = pd.read_csv(
        Path(args.data_dir, f"ukb_data_{pheno_name}_clusters.csv"), usecols=list(col_dtypes.keys()),
        dtype=col_dtypes, index_col="eid")
    data_test.append(data_test_curr)
data_test = pd.concat(data_test).drop_duplicates()
print(f"Number of test subjects: {data_test.shape[0]}")

# ICD code for metabolic diseases and musculoskeletal diseases
mm_code_dict = {}
code_list = []
for _, block_row in code_map.loc[code_map["Level"] == 1].iterrows():
    if block_row["Parent"] in ["4", "13"]:
        code_desc = block_row["Meaning"][8:]
        mm_code_dict[code_desc] = {}
        blocks_next = code_map["Node"].loc[code_map["Parent"] == str(block_row["Node"])]
        for block in blocks_next:
            code_next = code_map.loc[code_map["Parent"] == str(block)]
            for _, code_row in code_next.iterrows():
                mm_code_dict[code_desc][code_row["Coding"]] = code_row["Meaning"]
                code_list.append(code_row["Coding"])

# ICD code for depression
dep_code_dict = {}
for _, block_row in code_map.loc[code_map["Level"] == 2].iterrows():
    if block_row["Coding"] in ["F32", "F33", "F34", "F38", "F39"]:
        block_desc = block_row["Meaning"][4:]
        dep_code_dict[block_desc] = {}
        if block_row["Selectable"] == "Yes":
            dep_code_dict[block_desc][block_row["Coding"]] = block_row["Meaning"]
            code_list.append(block_row["Coding"])
        else:
            code_next = code_map.loc[code_map["Parent"] == str(block_row["Node"])]
            for _, code_row in code_next.iterrows():
                dep_code_dict[block_desc][code_row["Coding"]] = code_row["Meaning"]
                code_list.append(code_row["Coding"])

# Add individual diagnosis columns
mm_diagn = {key: {} for key in mm_code_dict.keys()}
dep_diagn = {key: {} for key in dep_code_dict.keys()}
for data_i, data_row in data_test.iterrows():
    code_list_curr = []
    for col in icd_list:
        if data_row[col] != "" and data_row[col] in code_list:
            code_list_curr.append(data_row[col])
    for diagn_type, diagn_code_list in mm_code_dict.items():
        intersect = list(set(diagn_code_list).intersection(code_list_curr))
        mm_diagn[diagn_type][data_i] = 1 if intersect else 0
    for diagn_type, diagn_code_list in dep_code_dict.items():
        intersect = list(set(diagn_code_list).intersection(code_list_curr))
        dep_diagn[diagn_type][data_i] = 1 if intersect else 0
data_diagn = pd.concat([
    data_test, pd.DataFrame(mm_diagn),
    pd.DataFrame(dep_diagn).any(axis="columns").astype(float).rename("Depression")], axis="columns")

# t test for depression patients and control separately
data_t = {}
for dep_diagn in [0, 1]:
    data_curr = data_diagn.loc[data_diagn["Depression"] == dep_diagn]
    for code_i, code_type in enumerate(mm_code_dict.keys()):
        if data_curr.loc[data_curr[code_type] == 1].shape[0] > 1:
            for cluster_i, cluster in enumerate(dep_clusters):
                results = ttest_ind(
                    a=data_curr[cluster].loc[data_curr[code_type] == 0],
                    b=data_curr[cluster].loc[data_curr[code_type] == 1], equal_var=False)
                data_t[f"{dep_diagn}_{code_i}_{cluster_i}"] = {
                    "t": results.statistic, "p": results.pvalue, "Disease type": code_type,
                    "Depressive score field": cluster, "Depressive score": dep_desc[cluster],
                    "N": data_curr.loc[data_curr[code_type] == 1].shape,
                    "Depression diagnosis": bool(dep_diagn)}

# Correct for multiple comparisons
fdr = multipletests([val["p"] for val in data_t.values()], method="fdr_bh")
for key_i, key in enumerate(data_t.keys()):
    data_t[key]["p (corrected)"] = fdr[1][key_i]
    data_t[key]["Rejected"] = fdr[0][key_i]
data_t = pd.DataFrame(data_t).T
data_t.to_csv(Path(args.out_dir, "ukb_dep_comorbid_t.csv"))

# Plot significant t statistics
data_t = data_t.loc[data_t["Rejected"] == True]
hue_order = [dep_desc[cluster] for cluster in dep_clusters]
with sns.plotting_context(context="paper", font_scale=1.5):
    g = sns.catplot(
        kind="strip", data=data_t, x="t", y="Disease type", hue="Depressive score",
        col="Depression diagnosis", palette=["darkseagreen", "pink"], hue_order=hue_order,
        orient="h", jitter=False, size=15, height=8, aspect=1, linewidth=1, edgecolor="black")
    for ax in g.axes.flat:
        ax.yaxis.grid(True)
plt.savefig(Path(args.img_dir, "ukb_dep_comorbid_t.png"), bbox_inches="tight", dpi=500)

# Prevalence and prevalence ratio
data_pr = {}
for code_i, code_type in enumerate(mm_code_dict.keys()):
    n_comorbid = data_diagn.loc[
        (data_diagn[code_type] == 1) & (data_diagn["Depression"] == 1)].shape[0]
    p_patient = n_comorbid / data_diagn.loc[data_diagn["Depression"] == 1].shape[0]
    p_control = (
        data_diagn.loc[(data_diagn[code_type] == 1) &(data_diagn["Depression"] == 0)].shape[0]
        / data_diagn.loc[data_diagn["Depression"] == 0].shape[0])
    p_ratio = 0 if p_control == 0 else (p_patient / p_control)
    if p_patient != 0 and n_comorbid > 20:
        data_pr[code_i] = {
            "Prevalence": p_patient, "Prevalence ratio": p_ratio, "Disease type": code_type,
            "N (comorbid)": n_comorbid}
data_pr = pd.DataFrame(data_pr).T
data_pr.to_csv(Path(args.out_dir, "ukb_dep_comorbid_pr.csv"))

# Plot prevalence and prevalence ratio
with sns.plotting_context(context="paper", font_scale=1.5):
    g = sns.PairGrid(
        data_pr, x_vars=["Prevalence", "Prevalence ratio"], y_vars=["Disease type"], height=8,
        aspect=0.7)
    g.map(
        sns.stripplot, orient="h", jitter=False, size=10, linewidth=1, color="lightgray",
        edgecolor="black")
    for ax in g.axes.flat:
        ax.yaxis.grid(True)
plt.savefig(Path(args.img_dir, "ukb_dep_comorbid_pr.png"), bbox_inches="tight", dpi=500)
