from pathlib import Path
import argparse

from semopy.efa import explore_cfa_model
import pandas as pd
import numpy as np


def get_fields(field_file: Path, corr_infos: pd.DataFrame) -> tuple[dict, dict]:
    field_file_cols = {"Field ID": str, "Field Description": str, "Type": str, "Instance": str}
    fields = pd.read_csv(field_file, usecols=list(field_file_cols.keys()), dtype=field_file_cols)

    field_cols = {}
    field_desc = {}
    for _, field in fields.iterrows():
        col_curr = f"{field['Field ID']}-{field['Instance']}.0"
        if col_curr in corr_sig["Field ID"].values:
            corr_infos_curr = corr_infos.loc[corr_sig["Field ID"] == col_curr]
            for _, corr_info_curr in corr_infos_curr.iterrows():
                pheno_type = field["Type"].replace(" ", "-")
                pheno_type = pheno_type.replace("/", "-")
                score_name = corr_info_curr["Depression score"].replace(" ", "-")
                key_curr = f"{pheno_type}_{score_name}_{corr_info_curr['Gender']}"
                if key_curr in field_cols.keys():
                    field_cols[key_curr].append(col_curr)
                else:
                    field_cols[key_curr] = [col_curr]
            field_desc[field["Field ID"]] = field["Field Description"]
    return field_cols, field_desc


parser = argparse.ArgumentParser(
    description="CFA for phenotypes in UKB",
    formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("train_file", type=Path, help="Absolute path to input training data")
parser.add_argument("corr_file", type=Path, help="Absolute path to input correlation data")
parser.add_argument("field_pheno", type=Path, help="Absolute path to selected phenotype fields csv")
parser.add_argument("field_comp", type=Path, help="Absolute path to selected composite fields csv")
parser.add_argument("out_dir", type=Path, help="Absolute path to output data directory")
args = parser.parse_args()

args.out_dir.mkdir(parents=True, exist_ok=True)
gender_ind = {"female": 0, "male": 1}
log_file = Path(args.out_dir, "ukb_pheno_efa.log")

# Only use datafields with significant correlations with depressive scores
corr_cols = {"Type": str, "Field ID": str, "Depression score": str, "Gender": str}
corr_sig = pd.read_csv(args.corr_file, usecols=list(corr_cols.keys()), dtype=corr_cols, index_col=0)

# Data fields to read and/or write
pheno_cols, pheno_desc = get_fields(args.field_pheno, corr_sig)
comp_cols, comp_desc = get_fields(args.field_comp, corr_sig)
pheno_cols.update(comp_cols)
pheno_desc.update(comp_desc)
dtypes = {"eid": str, "31-0.0": float}
dtypes.update({col: float for col_list in pheno_cols.values() for col in col_list})

# Run CFA in training set
data_train = pd.read_csv(
    args.train_file, usecols=list(dtypes.keys()), dtype=dtypes, index_col="eid")
for key, col_list in pheno_cols.items():
    gender_curr = key.split("_")[2]
    col_renames = [col.split("-")[0] for col in col_list]
    data_train_curr = data_train.loc[data_train["31-0.0"] == gender_ind[gender_curr]].copy()
    data_pheno_curr = data_train_curr[col_list].dropna()
    data_pheno_curr.columns = col_renames

    if data_pheno_curr.shape[1] <= 2:
        with open(log_file, "a") as f:
            print(key, "Too few phenotypes for EFA", file=f)
    else:
        try:
            latent_desc = explore_cfa_model(data_pheno_curr)
            efa_file = Path(args.out_dir, f"ukb_cfa_{key}.txt")
            with open(efa_file, "w") as f:
                f.writelines(latent_desc)
            with open(log_file, "a") as f:
                print(key, "EFA finished", file=f)

            pheno_desc_curr = {}
            for desc_line in latent_desc.split("\n"):
                if desc_line:
                    eta, phenos = desc_line.split(" =~ ")
                    pheno_desc_curr[eta] = {col: pheno_desc[col] for col in phenos.split(" + ")}
            pheno_file = Path(args.out_dir, f"ukb_pheno_{key}.csv")
            pd.DataFrame(pheno_desc_curr).to_csv(pheno_file)
        except SyntaxError or np.linalg.LinAlgError:
            with open(log_file, "a") as f:
                print(key, "EFA failed", file=f)
