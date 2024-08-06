from pathlib import Path
import argparse

import datalad.api as dl
import pandas as pd


def fill_pd_row(
        in_pd: pd.DataFrame, sub: str | int, rsfc_type: str, subject_dir: Path) -> pd.DataFrame:
    rsfc_dir = Path(subject_dir, "ses-2", "non-bids", "fMRI")
    rsfc_file = Path(rsfc_dir, f"sub-{sub}_ses-2_task-rest_{rsfc_type}_correlation_matrix_d100.txt")
    dl.get(rsfc_file, dataset=subject_dir)

    rsfc_curr = pd.read_table(rsfc_file, delim_whitespace=True, header=None).squeeze()
    in_pd.loc[sub] = rsfc_curr

    dl.drop(rsfc_file, reckless="kill", dataset=subject_dir)
    return in_pd


parser = argparse.ArgumentParser(
    description="extract UKB RSFC data for depression prediction",
    formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("ukb_raw_url", type=Path, help="Datalad URL to UKB raw dataset")
parser.add_argument("extracted_csv", type=Path, help="Absolute path to extracted data csv")
parser.add_argument("work_dir", type=Path, help="Absolute path to working directory")
parser.add_argument("out_csv", type=Path, help="Absolute path to output csv file")
args = parser.parse_args()

# Set-up
sub_list = pd.read_csv(args.extracted_csv, usecols=["eid"]).squeeze().tolist()
rsfc_full = pd.DataFrame(index=sub_list, columns=range(1485), dtype=float)
rsfc_part = pd.DataFrame(index=sub_list, columns=range(1485), dtype=float)

# Install dataset
root_data_dir = Path(args.work_dir, "ukb_raw")
dl.install(root_data_dir, source=args.url)

for subject in sub_list:
    sub_dir = Path(args.raw_dir, f"sub-{subject}")
    dl.get(sub_dir, dataset=root_data_dir, get_data=False)

    rsfc_full = fill_pd_row(rsfc_full, subject, "full", sub_dir)
    rsfc_part = fill_pd_row(rsfc_part, subject, "partial", sub_dir)

rsfc_all = rsfc_full.join(rsfc_part, lsuffix="_full", rsuffix="_part")
rsfc_all.to_csv(args.out_csv)
