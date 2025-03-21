from pathlib import Path
import argparse

from semopy import Model, semplot
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import pandas as pd

parser = argparse.ArgumentParser(
        description="Plot association between depressive scores and sociodemographic phenotypes",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("--data_dir", type=Path, help="Absolute path to extracted data")
parser.add_argument("--corr_dir", type=Path, help="Absolute path to correlation results")
parser.add_argument("--out_dir", type=Path, help="Absolute path to output directory")
parser.add_argument("--img_dir", type=Path, help="Absolute path to output plots directory")
args = parser.parse_args()

args.out_dir.mkdir(parents=True, exist_ok=True)
args.img_dir.mkdir(parents=True, exist_ok=True)
p_val = []

for col_type in ["Abdom comp", "Blood biochem", "Blood count 2", "NMR metabol 1"]:
    # Data fields of top 3 biomarker
    pheno_name = col_type.replace(" ", "-")
    data_corr = pd.read_csv(
            Path(args.corr_dir, f"ukb_dep_corr_somatic_{pheno_name}_fdr.csv"),
            usecols=["Data field"], dtype=str).squeeze().to_list()
    data_corr = list(set(data_corr))

    # Test data
    col_dtype = {"eid": str, "31-0.0": float, "Sum score (cluster 2)": float}
    col_dtype.update({key: float for key in data_corr})
    data_test_curr = pd.read_csv(
        Path(args.data_dir, f"ukb_data_{pheno_name}_clusters.csv"), usecols=list(col_dtype.keys()),
        dtype=col_dtype)

    # Rename columns
    col_names = {"Sum score (cluster 2)": "DepressiveSomatic"}
    col_names.update({key: key.split("-")[0] for key in data_corr})
    data_test = data_test_curr.rename(columns=col_names)

    # Model description
    desc = ""
    for col_i, col in enumerate(data_corr):
        desc += f"{col_names[col]} ~ DepressiveSomatic\n"
        desc += f"DepressiveSomatic ~ {col_names[col]}\n"
        for col_j in range(col_i+1, len(data_corr)):
            desc += f"{col_names[col]} ~~ {col_names[data_corr[col_j]]}\n"

    # Structural equation modelling (SEM) for each gender separately
    for gender_i, gender in enumerate(["female", "male"]):
        data_curr = StandardScaler().fit_transform(data_test.loc[data_test["31-0.0"] == gender_i])
        data_curr = pd.DataFrame(data_curr, columns=data_test.columns)
        model = Model(desc)
        model.fit(data_curr)
        inspect = model.inspect()
        p_val.extend(inspect["p-value"].to_list())

        pheno_name = col_type.replace(" ", "-")
        inspect.to_csv(Path(args.out_dir, f"sem_inspect_somatic_{pheno_name}_{gender}.csv"))
        semplot(
            model, str(f"{args.img_dir}/sem_model_somatic_{pheno_name}_{gender}.png"),
            plot_covs=True)

# Correct for multiple comparisons
fdr = multipletests(p_val, method="fdr_bh")
pd.DataFrame(fdr[:2]).T.to_csv(Path(args.out_dir, "sem_p_fdr.csv"))
