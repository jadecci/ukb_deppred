## 1. Preparation
### 1.1. Install dataset

```bash
python3 -m venv ~/.venvs/ukb_deppred
source ~/.venvs/ukb_deppred/bin/activate
datalad clone git@gin.g-node.org:/jadecci/ukb_deppred.git ${project_dir}/ukb_deppred
cd ${project_dir}/ukb_deppred && datalad get -r . && cd ${project_dir}
```

### 1.2. Prepare data
Download the aggregated tsv data file to `${project_dir}/ukb_raw.tsv`.

## 2. Data Extraction

