from pathlib import Path
import argparse

from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_fields(field_file: Path) -> tuple[list, dict, list]:
    field_file_cols = {"Field ID": str, "Field Description": str, "Type": str, "Instance": str}
    fields = pd.read_csv(field_file, usecols=list(field_file_cols.keys()), dtype=field_file_cols)

    field_col_list = []
    field_cols = {}
    field_desc = []
    for _, field in fields.iterrows():
        col_curr = f"{field['Field ID']}-{field['Instance']}.0"
        if field["Type"] in field_cols.keys():
            field_cols[field["Type"]].append(col_curr)
        else:
            field_cols[field["Type"]] = [col_curr]
        field_col_list.append(col_curr)
        field_desc.append(field["Field Description"])
    return field_col_list, field_cols, field_desc


def plot_dendrogram(model: FeatureAgglomeration, labels: np.ndarray, outfile: Path) -> dict:
    counts = np.zeros(model.children_.shape[0])
    for i, merge in enumerate(model.children_):
        for child_ind in merge:
            if child_ind < len(model.labels_):
                counts[i] = counts[i] + 1 # leaf node
            else:
                counts[i] = counts[i] + counts[child_ind - len(model.labels_)]
    linkage_mat = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    dendro_res = dendrogram(
        linkage_mat, orientation="left", labels=labels, leaf_font_size=8)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    return dendro_res


parser = argparse.ArgumentParser(
    description="Clustering analysis for depressive symptoms in UKB",
    formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("data_dir", type=Path, help="Absolute path to input data directory")
parser.add_argument("field_pheno", type=Path, help="Absolute path to selected phenotype fields csv")
parser.add_argument("field_comp", type=Path, help="Absolute path to selected composite fields csv")
parser.add_argument(
    "field_demo", type=Path, help="Absolute path to selected sociodemographics fields csv")
parser.add_argument("field_dep", type=Path, help="Absolute path to selected depression fields csv")
parser.add_argument("out_dir", type=Path, help="Absolute path to output data directory")
parser.add_argument("res_dir", type=Path, help="Absolute path to output cluster results directory")
parser.add_argument("img_dir", type=Path, help="Absolute path to output images directory")
args = parser.parse_args()

args.out_dir.mkdir(parents=True, exist_ok=True)
args.res_dir.mkdir(parents=True, exist_ok=True)
args.img_dir.mkdir(parents=True, exist_ok=True)
train_file = Path(args.data_dir, "ukb_extracted_data_train.csv")

# Data fields to read and/or write
pheno_col_list, pheno_cols, _ = get_fields(args.field_pheno)
comp_col_list, comp_cols, _ = get_fields(args.field_comp)
demo_col_list, demo_cols, _ = get_fields(args.field_demo)
dep_col_list, _, dep_desc = get_fields(args.field_dep)
train_dtypes = {"eid": str}
train_dtypes.update({col: float for col in pheno_col_list+comp_col_list+demo_col_list+dep_col_list})
test_dtypes = {"eid": str}
test_dtypes.update({col: float for col in demo_col_list+dep_col_list})

# ICD-10 code columns
icd10_col_list = []
data_head = pd.read_csv(train_file, nrows=2, index_col="eid")
for col in data_head.columns:
    if col.split("-")[0] == "41270":
        icd10_col_list.append(col)
        train_dtypes[col] = str
        test_dtypes[col] = str

# Hierarchical clustering in training set
clusters = {"female": {}, "male": {}}
data_train = pd.read_csv(train_file, usecols=list(train_dtypes.keys()), dtype=train_dtypes)
args.img_dir.mkdir(parents=True, exist_ok=True)

# Data fields to read and/or write
_, pheno_cols, _ = get_fields(args.field_pheno)
_, comp_cols, _ = get_fields(args.field_comp)
dep_col_list, _, dep_desc = get_fields(args.field_dep)
pheno_cols.update(comp_cols)
dtypes = {"eid": str, "31-0.0": float}
dtypes.update({col: "Int64" for col in dep_col_list})

# Hierarchical clustering in training set
data_train = pd.read_csv(Path(args.data_dir, "ukb_extracted_data_train.csv"))
clusters = {"female": {}, "male": {}}
for gender_ind, gender in enumerate(["female", "male"]): # female, male
    data_train_curr = data_train.loc[data_train["31-0.0"] == gender_ind].copy()
    data_train_curr_std = StandardScaler().fit_transform(data_train_curr[dep_col_list])
    model_curr = FeatureAgglomeration(n_clusters=2, compute_distances=True)
    model_curr.fit(data_train_curr_std)
    outfile_curr = Path(args.img_dir, f"ukb_cluster_{gender}.eps")
    dendro_res_curr = plot_dendrogram(model_curr, np.array(dep_desc), outfile_curr)
    for leaf, color in zip(dendro_res_curr["leaves"], dendro_res_curr["leaves_color_list"]):
        cluster_ind = int(color[-1])
        if cluster_ind in clusters[gender].keys():
            clusters[gender][cluster_ind].append(dep_col_list[leaf])
        else:
            clusters[gender][cluster_ind] = [dep_col_list[leaf]]
pd.DataFrame(clusters).to_csv(Path(args.res_dir, "ukb_dep_clusters.csv"))

# Compute sum scores in test set
for pheno_type, pheno_col_list in pheno_cols.items():
    test_dtypes_curr = test_dtypes.copy()
    test_dtypes_curr.update({col: float for col in pheno_col_list})
    if pheno_type in comp_cols.keys():
        test_dtypes_curr.update({col:float for col in comp_cols[pheno_type]})
    pheno_name = pheno_type.replace(" ", "-")
    test_file = Path(args.data_dir, f"ukb_extracted_data_{pheno_name}.csv")
    data_test = pd.read_csv(
        test_file, usecols=list(test_dtypes_curr.keys()), dtype=test_dtypes_curr, index_col="eid")
    pheno_name = pheno_type.replace(" ", "-")
    pheno_name = pheno_name.replace("/", "-")
    dtypes_pheno = dtypes.copy()
    dtypes_pheno.update({col: float for col in pheno_col_list})
    data_test = pd.read_csv(
        Path(args.data_dir, f"ukb_extracted_data_{pheno_name}.csv"),
        usecols=list(dtypes_pheno.keys()), dtype=dtypes_pheno, index_col="eid")

    for gender_ind, gender in enumerate(["female", "male"]):
        data_test_gender = data_test.loc[data_test["31-0.0"] == gender_ind].copy()
        for cluster_ind, cluster in clusters[gender].items():
            cluster_name = f"Sum score (cluster {cluster_ind})"
            data_test_gender[cluster_name] = data_test_gender[cluster].sum(axis="columns")
        data_test_gender.to_csv(
            Path(args.out_dir, f"ukb_extracted_data_{pheno_name}_{gender}_clusters.csv"))
