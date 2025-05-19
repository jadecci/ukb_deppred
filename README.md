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

Association between sociodemographic factors and depressive sum scores:

```bash
python3 ukb_deppred/ukbdep_sociodemo.py \
  --data_csv results/extracted_data/ukb_data_Blood-count_clusters.csv \
  --sel_csv ukb_deppred/replication_data/UKB_selected_fields_pheno.csv \
  --img_dir results/plots
```

Correlation between body/brain biomarkers and depressive sum scores:

```bash
python3 ukb_deppred/ukbdep_corr_pheno.py --data_dir results/extracted_data \
  --sel_csv ukb_deppred/replication_data/UKB_selected_fields_pheno.csv \
  --img_dir results/plots --out_dir results/corr_dep
```

Also plot the brain correlations:

```bash
python3 ukb_deppred/ukbdep_corr_brain.py --results_dir results/corr_dep \
  --data_dir ukb_deppred/replication_data \
  --img_dir results/raw_plots
```
