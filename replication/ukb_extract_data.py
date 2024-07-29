from pathlib import Path
import argparse

import pandas as pd

parser = argparse.ArgumentParser(
    description="extract UKB data for depression prediction",
    formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("in_tsv", type=Path, help="Absolute path to nput tsv of UKB raw data")
parser.add_argument("out_dir", type=Path, help="Absolute path to output directory")
args = parser.parse_args()

