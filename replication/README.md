## 1. Preparation
### 1.1. Install dataset

```bash
python3 -m venv ~/.venvs/ukb_deppred
source ~/.venvs/ukb_deppred/bin/activate
datalad clone git@gin.g-node.org:/jadecci/ukb_deppred.git ${project_dir}/ukb_deppred
cd ${project_dir}/ukb_deppred && datalad get -r . && cd ${project_dir}
datalad clone git@github.com:USC-IGC/ukbb_parser.git ${project_dir}/ukbb_parser
```

### 1.2. Prepare data
Download the aggregated raw data tsv files to `${project_dir}/data/ukb_raw.tsv` and 
`${project_dir}/data/ukb_genetic.tsv`.
Prepare the list of antidepressant code in `${project_dir}/data/UKB_antidepressant_code.csv`

## 2. Data Extraction
### 2.1. Extract data from raw tsv files

```bash
python3 ${project_dir}/ukb_deppred/replication/ukb_extract_data.py ${project_dir}/data/ukb_raw.tsv \
  ${project_dir}/data/ukb_genetic.tsv ${project_dir}/ukbb_parser \
  ${project_dir}/data/UKB_antidepressant_code.csv ${project_dir}/ukb_extracted_data.csv
```

### 2.2. Extract RSFC data
