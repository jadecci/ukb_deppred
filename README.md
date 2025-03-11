# Mood and somatic dimensions of depressive symptoms in UK Biobank relate differently to body and brain biomarkers

## Reference

t.b.a.

## Usage & Replication
### 1. Preparation

Download/copy the aggregated raw data tsv files to `${project_dir}/data/ukb_raw.tsv`. 
Then set up the project virtual environment:

```bash
python3 -m venv ~/.venvs/ukb_deppred
source ~/.venvs/ukb_deppred/bin/activate
datalad clone git@github.com:jadecci/ukb_deppred.git ${project_dir}/ukb_deppred
python3 -m pip install -r ${project_dir}/ukb_deppred/requirements.txt
```

### 2. Data Extraction

```bash
python3 ${project_dir}/ukb_deppred/ukbdep_extract_data.py 
```
