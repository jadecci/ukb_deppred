from pathlib import Path
import argparse

import nibabel as nib
from nilearn.datasets import load_fsaverage, load_fsaverage_data
from nilearn.plotting import plot_glass_brain, plot_surf_stat_map
from nilearn.surface import load_surf_data
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(
        description="Plot correlations between depressive sum scores and brain biomarkers",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument(
    "--corr_csv", type=Path, help="Absolute path to csv file containing all correlations")
parser.add_argument("--jhu_atlas", type=Path, help="Absolute path to JHU white matter atlas")
parser.add_argument("--jhu_lut", type=Path, help="Absolute path to JHU atlas LUT")
parser.add_argument("--l_annot", type=Path, help="Absolute path to left a2009s annot")
parser.add_argument("--r_annot", type=Path, help="Absolute path to right a2009s annot")
parser.add_argument("--fs_lut", type=Path, help="Absolute path to Destrieux 2009 atlas LUT")
parser.add_argument("--img_dir", type=Path, help="Absolute path to output plots directory")
args = parser.parse_args()

args.img_dir.mkdir(parents=True, exist_ok=True)
data_corr = pd.read_csv(args.corr_csv)
dep_scores = ["Sum score (cluster 1)", "Sum score (cluster 2)"]
genders = ["female", "male"]

# JHU atlas
jhu_img = nib.load(args.jhu_atlas)
jhu_vol = jhu_img.get_fdata()
jhu_lut = pd.read_csv(args.jhu_lut)

# fsaverage surface and Destrieux 2009 atlas
fs_meshes = load_fsaverage(mesh="fsaverage")
fs_bg = load_fsaverage_data(mesh="fsaverage", mesh_type="inflated", data_type="sulcal")
l_annot = load_surf_data(args.l_annot)
r_annot = load_surf_data(args.r_annot)
fs_lut = pd.read_csv(args.fs_lut)

# White matter biomarkers
wm_markers = ["FA", "ICVF", "ISOVF", "L1", "L2", "L3", "MD", "MO", "OD"]
data_corr_wm = data_corr.loc[data_corr["Type"] == "Brain WM"]
for dep_i, dep in enumerate(dep_scores):
    for gender in genders:
        data_curr = data_corr_wm.loc[
            (data_corr_wm["Depressive score field"] == dep) & (data_corr_wm["Gender"] == gender)]

        for wm in wm_markers:
            data_marker = data_curr.loc[data_curr["Field description"].str.contains(f"Mean {wm}")]
            data_sort = data_marker.sort_values(by="Absolute r", axis="index", ascending=False)
            data_top = data_sort.head(3)
            vol_curr = np.zeros(jhu_vol.shape)
            for _, data_row in data_top.iterrows():
                wm_region = data_row["Field description"].split("in ")[1]
                wm_region_val = jhu_lut["Atlas value"].loc[jhu_lut["Brain region"] == wm_region]
                vol_curr[np.where(jhu_vol == wm_region_val.values[0])] = data_row["r"]

            wm_img = nib.Nifti1Image(vol_curr, jhu_img.affine)
            plt = plot_glass_brain(
                wm_img, colorbar=True, cmap="coolwarm", vmax=0.1, symmetric_cbar=True,
                plot_abs=False, threshold=0.01)
            plt.savefig(Path(args.img_dir, f"corr_dep{dep_i+1}_{wm}_{gender}.png"))
            plt.close()

# Gray matter volume biomarkers
data_corr_gmv = data_corr.loc[data_corr["Type"] == "Brain GMV"]
for dep_i, dep in enumerate(dep_scores):
    for gender in genders:
        data_curr = data_corr_gmv.loc[
            (data_corr_gmv["Depressive score field"] == dep) & (data_corr_gmv["Gender"] == gender)]
        data_sort = data_curr.sort_values(by="Absolute r", axis="index", ascending=False)
        data_top = data_sort.head(3)

        fs_surf_l = np.zeros(l_annot.shape)
        fs_surf_r = np.zeros(r_annot.shape)
        for _, data_row in data_top.iterrows():
            fs_region = data_row["Field description"].split("Volume of ")[1]
            if "left" in fs_region:
                fs_region_val = fs_lut["Atlas value"].loc[fs_lut["Left brain region"] == fs_region]
                fs_surf_l[np.where(l_annot == fs_region_val.values[0])] = data_row["r"]
            if "right" in fs_region:
                fs_region_val = fs_lut["Atlas value"].loc[fs_lut["Right brain region"] == fs_region]
                fs_surf_r[np.where(r_annot == fs_region_val.values[0])] = data_row["r"]

        plt = plot_surf_stat_map(
            surf_mesh=fs_meshes["inflated"], stat_map=fs_surf_l, bg_map=fs_bg, hemi="left",
            cmap="coolwarm", colorbar=True, vmax=0.1, symmetric_cbar=True, threshold=0.01)
        plt.savefig(Path(args.img_dir, f"corr_dep{dep_i+1}_gmv_{gender}_l.png"))
        plt = plot_surf_stat_map(
            surf_mesh=fs_meshes["inflated"], stat_map=fs_surf_r, bg_map=fs_bg, hemi="right",
            cmap="coolwarm", colorbar=True, vmax=0.1, symmetric_cbar=True, threshold=0.01)
        plt.savefig(Path(args.img_dir, f"corr_dep{dep_i+1}_gmv_{gender}_r.png"))
