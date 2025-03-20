# Mood and somatic dimensions of depressive symptoms in UK Biobank relate differently to body and brain biomarkers

## Reference

t.b.a.

## Usage & Replication
### 1. Preparation

Download/copy the aggregated raw data tsv files to `ukb_raw.tsv`. 
Then set up the project virtual environment:

```bash
python3 -m venv ~/.venvs/ukb_deppred && source ~/.venvs/ukb_deppred/bin/activate
datalad clone git@github.com:jadecci/ukb_deppred.git ukb_deppred
python3 -m pip install -r ukb_deppred/requirements.txt
```

### 2. Data Extraction

Extract training and test data:

```bash
python3 ukb_deppred/ukbdep_extract_data.py --raw_tsv ukb_raw_tsv \
  --sel_csv ukb_deppred/replication_data/UKB_selected_fields_pheno.csv \
  --wd_csv ukb_deppred/replication_data/w41655_20241217.csv \
  --out_dir results/extracted_data \
```

Cluster depressive scores and compute sum scores by cluster:

```bash
python3 ukb_deppred/ukbdep_cluster.py --data_dir results/extracted_data \
  --sel_csv ukb_deppred/replication_data/UKB_selected_fields_pheno.csv \
  --img_dir results/plots
```

### 3. Association analyses

Correlation between all depressive sum scores and all body/brain biomarkers:

```bash
python3 ukb_deppred/ukbdep_corr_pheno.py --data_dir results/extracted_data \
  --sel_csv ukb_deppred/replication_data/UKB_selected_fields_pheno.csv \
  --img_dir results/plots --out_dir results/corr_dep
```

Also plot the brain correlations:

```bash
python3 ukb_deppred/plot_scripts/ukbdep_corr_brain.py \
  --cor_csv results/corr_dep/ukb_dep_corr_pheno_fdr.csv \
  --jhu_atlas ukb_deppred/replication_data/JHU-ICBM-labels-1mm.nii.gz \
  --jhu_lut ukb_deppred/replication_data/JHU_labels.csv \
  --l_annot ukb_deppred/replication_data/lh.aparc.a2009s.annot \
  --r_annot ukb_deppred/replication_data/rh.aparc.a2009s.annot \
  --fs_lut ukb_deppred/replication_data/FS_a2009s_labels.csv \
  --img_dir results/raw_plots
```

Association between sociodemographic factors and depressive sum scores:

```bash
python3 ukb_deppred/plot_scripts/ukbdep_sociodemo.py
```
