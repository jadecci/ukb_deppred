from pathlib import Path
import argparse

import nibabel as nib
from nilearn.datasets import load_fsaverage, load_fsaverage_data
from nilearn.plotting import plot_glass_brain, plot_surf_stat_map
from nilearn.surface import load_surf_data
import numpy as np
import pandas as pd


def plot_wm_jhu(
        data_dir: Path, data: pd.DataFrame, wm: str, dep: str, dep_i: int, gender: str,
        img_dir: Path):
    # JHU atlas
    jhu_img = nib.load(Path(data_dir, "JHU-ICBM-labels-1mm.nii.gz"))
    jhu_vol = jhu_img.get_fdata()
    jhu_lut = pd.read_csv(Path(data_dir, "JHU_labels.csv"))

    data_curr = data.loc[
        (data["Field description"].str.contains(f"Mean {wm}"))
        & (data["Depressive score field"] == dep)]
    vol_curr = np.zeros(jhu_vol.shape)
    for _, data_row in data_curr.iterrows():
        wm_region = data_row["Field description"].split("in ")[1]
        wm_region_val = jhu_lut["Atlas value"].loc[jhu_lut["Brain region"] == wm_region]
        vol_curr[np.where(jhu_vol == wm_region_val.values[0])] = data_row["r"]

    wm_img = nib.Nifti1Image(vol_curr, jhu_img.affine)
    plt = plot_glass_brain(
        wm_img, colorbar=True, cmap="coolwarm", vmax=0.1, symmetric_cbar=True,
        plot_abs=False, threshold=0.01)
    plt.savefig(Path(img_dir, f"corr_dep{dep_i + 1}_{wm}_{gender}.png"))
    plt.close()
    return


def plot_gmv_a2009s(
        data_dir: Path, data: pd.DataFrame, dep: str, dep_i: int, gender: str, img_dir: Path):
    # fsaverage surface and Destrieux 2009 atlas
    fs_meshes = load_fsaverage(mesh="fsaverage")
    fs_bg = load_fsaverage_data(mesh="fsaverage", mesh_type="inflated", data_type="sulcal")
    l_annot = load_surf_data(Path(data_dir, "lh.aparc.a2009s.annot"))
    r_annot = load_surf_data(Path(data_dir, "rh.aparc.a2009s.annot"))
    fs_lut = pd.read_csv(Path(data_dir, "FS_a2009s_labels.csv"))

    data_curr = data.loc[
        (data["Type"] == "Brain GMV") & (data["Data field"].str[:2] == "27")
        & (data["Depressive score field"] == dep)]
    fs_surf_l = np.zeros(l_annot.shape)
    fs_surf_r = np.zeros(r_annot.shape)
    for _, data_row in data_curr.iterrows():
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
    plt.savefig(Path(img_dir, f"corr_dep{dep_i + 1}_gmv_{gender}_l.png"))
    plt = plot_surf_stat_map(
        surf_mesh=fs_meshes["inflated"], stat_map=fs_surf_r, bg_map=fs_bg, hemi="right",
        cmap="coolwarm", colorbar=True, vmax=0.1, symmetric_cbar=True, threshold=0.01)
    plt.savefig(Path(img_dir, f"corr_dep{dep_i + 1}_gmv_{gender}_r.png"))
    return


def plot_gmv_aseg(
        data_dir: Path, data: pd.DataFrame, dep: str, dep_i: int, gender: str, img_dir: Path):
    # FreeSurfer aseg segmentation in MNI space
    aseg_img = nib.load(Path(data_dir, "mni_aseg.mgz"))
    aseg_vol = aseg_img.get_fdata()
    aseg_lut = pd.read_csv(Path(data_dir, "FS_aseg_labels.csv"))

    data_curr = data.loc[
        (data["Type"] == "Brain GMV") & (data["Data field"].str[:2] == "26")
        & (data["Depressive score field"] == dep)]
    vol_curr = np.zeros(aseg_vol.shape)
    for _, data_row in data_curr.iterrows():
        aseg_region = data_row["Field description"].split("Volume of ")[1]
        aseg_region_val = aseg_lut["Atlas value"].loc[aseg_lut["Brain region"] == aseg_region]
        vol_curr[np.where(aseg_vol == aseg_region_val.values[0])] = data_row["r"]

    gmv_img = nib.Nifti1Image(vol_curr, aseg_img.affine)
    plt = plot_glass_brain(
        gmv_img, colorbar=True, cmap="coolwarm", vmax=0.1, symmetric_cbar=True,
        plot_abs=False, threshold=0.01)
    plt.savefig(Path(img_dir, f"corr_dep{dep_i + 1}_gmv_subcort_{gender}.png"))
    plt.close()
    return


parser = argparse.ArgumentParser(
        description="Plot correlations between depressive sum scores and brain biomarkers",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("--results_dir", type=Path, help="Absolute path to results directory")
parser.add_argument("--data_dir", type=Path, help="Absolute path to data directory")
parser.add_argument("--img_dir", type=Path, help="Absolute path to output plots directory")
args = parser.parse_args()

args.img_dir.mkdir(parents=True, exist_ok=True)
data_corr_mixed = pd.read_csv(Path(args.results_dir, "ukb_dep_corr_pheno_fdr_mixed.csv"))
data_corr_split = pd.read_csv(Path(args.results_dir, "ukb_dep_corr_pheno_fdr_split.csv"))
dep_scores = ["Sum score (cluster 1)", "Sum score (cluster 2)"]

# White matter biomarkers
wm_markers = ["FA", "ICVF", "ISOVF", "MD", "MO", "OD"]
for dep_ind, dep_score in enumerate(dep_scores):
    for wm_marker in wm_markers:
        plot_wm_jhu(
            args.data_dir, data_corr_mixed, wm_marker, dep_score, dep_ind, "mixed", args.img_dir)
        plot_wm_jhu(
            args.data_dir, data_corr_split.loc[data_corr_split["Gender"] == "female"], wm_marker,
            dep_score, dep_ind, "female", args.img_dir)
        plot_wm_jhu(
            args.data_dir, data_corr_split.loc[data_corr_split["Gender"] == "male"], wm_marker,
            dep_score, dep_ind, "male", args.img_dir)

# Gray matter volume biomarkers
for dep_ind, dep_score in enumerate(dep_scores):
    plot_gmv_a2009s(args.data_dir, data_corr_mixed, dep_score, dep_ind, "mixed", args.img_dir)
    plot_gmv_a2009s(
        args.data_dir, data_corr_split.loc[data_corr_split["Gender"] == "female"], dep_score,
        dep_ind, "female", args.img_dir)
    plot_gmv_a2009s(
        args.data_dir, data_corr_split.loc[data_corr_split["Gender"] == "male"], dep_score,
        dep_ind, "male", args.img_dir)

    plot_gmv_aseg(args.data_dir, data_corr_mixed, dep_score, dep_ind, "mixed", args.img_dir)
    plot_gmv_aseg(
        args.data_dir, data_corr_split.loc[data_corr_split["Gender"] == "female"], dep_score,
        dep_ind, "female", args.img_dir)
    plot_gmv_aseg(
        args.data_dir, data_corr_split.loc[data_corr_split["Gender"] == "male"], dep_score,
        dep_ind, "male", args.img_dir)
