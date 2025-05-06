from pathlib import Path
import argparse

from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot(
        data: pd.DataFrame, y_col: str, ytick: list | None, yticklabel: list, x_cols: list,
        out_name: str, plot_type: str):
    dep_desc = [f"Depressive mood symptoms", f"Depressive somatic symptoms"]
    f, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=(10, 3))
    with sns.plotting_context(context="paper", font_scale=1.5):
        for x_i, x in enumerate(x_cols):
            if plot_type == "box":
                sns.boxplot(
                    data, y=y_col, x=x, orient="h", ax=ax[x_i], color="w", linecolor="k",
                    dodge=False, order=yticklabel)
            elif plot_type == "reg":
                sns.regplot(
                    data, y=y_col, x=x, ax=ax[x_i], marker="x", scatter_kws={"color": ".2"},
                    line_kws={"color": "red"})
                r, p = pearsonr(data[y_col].astype(float), data[x].astype(float))
                ax[x_i].set_title(f"R = {r:.4f}, p = {p:.4f}")
                ax[x_i].set_yticks(ytick, labels=yticklabel)
            ax[x_i].set(xlabel=dep_desc[x_i], ylabel="")
    f.savefig(Path(args.img_dir, f"ukb_dep_{out_name}.png"), bbox_inches="tight", dpi=500)
    plt.close(f)


parser = argparse.ArgumentParser(
        description="Plot association between depressive scores and sociodemographic phenotypes",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("--data_csv", type=Path, help="Absolute path to the test data csv file to use")
parser.add_argument("--sel_csv", type=Path, help="Absolute path to table of selected fields")
parser.add_argument("--img_dir", type=Path, help="Absolute path to output plots directory")
args = parser.parse_args()

col_dtypes = {"eid": str, "40023-0.0": float}
excludes = {}
args.img_dir.mkdir(parents=True, exist_ok=True)

# Phenotype field information
field_cols = {
    "Field ID": str, "Type": str, "To Exclude 1": float, "To Exclude 2": float, "Instance": "Int64"}
fields = pd.read_csv(args.sel_csv, usecols=list(field_cols.keys()), dtype=field_cols)
for _, field_row in fields.loc[fields["Type"] == "Sociodemo"].iterrows():
    col_id = f"{field_row['Field ID']}-{field_row['Instance']}.0"
    col_dtypes[col_id] = float
    if not np.isnan(field_row["To Exclude 1"]):
        excludes[col_id] = [field_row["To Exclude 1"]]
        if not np.isnan(field_row["To Exclude 2"]):
            excludes[col_id].append(field_row["To Exclude 2"])

# Depressive sum scores of interest
dep_score = []
for cluster in [1, 2]:
    col_id = f"Sum score (cluster {cluster})"
    dep_score.append(col_id)
    col_dtypes[col_id] = float

data_test = pd.read_csv(
    args.data_csv, usecols=list(col_dtypes.keys()), dtype=col_dtypes, index_col="eid")
for col_id, exclude_list in excludes.items():
    for exclude in exclude_list:
        data_test = data_test.loc[data_test[col_id] != exclude]

# boxplot: gender
gender_dict = {0: "Female", 1: "Male"}
data_test = data_test.assign(gender=data_test["31-0.0"].apply(lambda x: gender_dict.get(x)))
plot(data_test, "gender", None, ["Female", "Male"], dep_score, "gender", "box")

# boxplot: education
educ_dict = {
    -7: "None of the above", 1: "College or University degree",
    2: "A levels/AS levels or equivalent", 3: "O levels/GCSEs or equivalent",
    4: "CSEs or equivalent", 5: "NVQ or NHD or HNC or equivalent",
    6: "Other professional qualifications"}
data_test = data_test.assign(educ=data_test["6138-2.0"].apply(lambda x: educ_dict.get(x)))
yticks_l = [educ_dict[i] for i in [1, 2, 3, 4, 5, 6, -7]]
plot(data_test, "educ", None, yticks_l, dep_score, "educ", "box")

# regplot: household income
income_dict = {
    1: "Less than 18,000", 2: "18,000 to 30,999", 3: "31,000 to 51,999", 4: "52,000 to 100,000",
    5: "Greater than 100,000"}
data_test = data_test.assign(income=data_test["738-2.0"].apply(lambda x: income_dict.get(x)))
yticks_l = [income_dict[i] for i in [1, 2, 3, 4, 5]]
plot(data_test, "income", None, yticks_l, dep_score, "income", "box")

# regplot: age
yticks_l = ["50", "60", "70", "80"]
data_alive = data_test.loc[np.isnan(data_test["40023-0.0"])]
plot(data_alive, "21003-2.0", [50, 60, 70, 80], yticks_l, dep_score, "age", "reg")

# regplot: age by age groups
yticks_l = ["45", "50", "55", "60"]
data_age_low = data_alive.loc[data_alive["21003-2.0"] < 60]
plot(data_age_low, "21003-2.0", [45, 50, 55, 60], yticks_l, dep_score, "age_low", "reg")
yticks_l = ["60", "65", "70", "75", "80"]
data_age_high = data_alive.loc[data_alive["21003-2.0"] >= 60]
plot(data_age_high, "21003-2.0", [60, 65, 70, 75, 80], yticks_l, dep_score, "age_high", "reg")
