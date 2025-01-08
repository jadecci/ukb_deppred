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
        linkage_mat, orientation="left", labels=labels, leaf_font_size=8,
        color_threshold=0.75*max(linkage_mat[:, 2]))
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
parser.add_argument("field_dep", type=Path, help="Absolute path to selected depression fields csv")
parser.add_argument("out_dir", type=Path, help="Absolute path to output data directory")
parser.add_argument("img_dir", type=Path, help="Absolute path to output images directory")
args = parser.parse_args()

args.out_dir.mkdir(parents=True, exist_ok=True)
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
clusters = {0: {}, 1: {}}
for gender in [0, 1]: # female, male
    data_train_curr = data_train.loc[data_train["31-0.0"] == gender].copy()
    data_train_curr_std = StandardScaler().fit_transform(data_train_curr[dep_col_list])
    model_curr = FeatureAgglomeration(n_clusters=2, compute_distances=True)
    model_curr.fit(data_train_curr_std)
    outfile_curr = Path(args.img_dir, f"ukb_cluster_{gender}.png")
    dendro_res_curr = plot_dendrogram(model_curr, np.array(dep_desc), outfile_curr)
    for leaf, color in zip(dendro_res_curr["leaves"], dendro_res_curr["leaves_color_list"]):
        cluster_ind = int(color[-1])
        if cluster_ind in clusters[gender].keys():
            clusters[gender][cluster_ind].append(dep_col_list[leaf])
        else:
            clusters[gender][cluster_ind] = [dep_col_list[leaf]]

# Compute sum scores in test set
for pheno_type, pheno_col_list in pheno_cols.items():
    pheno_name = pheno_type.replace(" ", "-")
    pheno_name = pheno_name.replace("/", "-")
    dtypes_pheno = dtypes.copy()
    dtypes_pheno.update({col: float for col in pheno_col_list})
    data_test = pd.read_csv(
        Path(args.data_dir, f"ukb_extracted_data_{pheno_name}.csv"),
        usecols=list(dtypes_pheno.keys()), dtype=dtypes_pheno, index_col="eid")

    for gender_ind, gender in enumerate(["female", "male"]):
        data_test_gender = data_test.loc[data_test["31-0.0"] == gender_ind].copy()
        for cluster_ind, cluster in clusters[gender_ind].items():
            cluster_name = f"Sum score (cluster {cluster_ind})"
            data_test_gender[cluster_name] = data_test_gender[cluster].sum(axis="columns")
        data_test_gender.to_csv(
            Path(args.out_dir, f"ukb_extracted_data_{pheno_name}_{gender}_clusters.csv"))
