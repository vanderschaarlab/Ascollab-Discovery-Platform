tcga_kirc_neurips_hiseqv2 = """
This dataset is a local copy of the UCSC Xena TCGA KIRC RNA-seq matrix `TCGA.KIRC.sampleMap/HiSeqV2`.

Source metadata (UCSC Xena):
- Cohort / title: TCGA Kidney Clear Cell Carcinoma (KIRC); "gene expression by RNAseq (polyA+ IlluminaHiSeq)".
- Xena wrangling notes: TCGA DCC Level 3 `*.rsem.genes.normalized_results` processed by UCSC Xena with log2(x+1) transform.

Useful details:
- Rows correspond to gene/features and columns correspond to TCGA samples.
- The first column is `sample` (feature identifier), and the remaining columns are TCGA barcodes.
- Barcode suffix convention follows TCGA standards: `01` generally indicates tumor tissue and `11` indicates solid tissue normal.
- Values are the same style as Xena sampleMap HiSeqV2 matrices (continuous normalized expression values).

Location:
- Repo path: "sandbox/datasets/KIRC_NeurIPS/TCGA.KIRC.sampleMap_HiSeqV2.gz"
- Container path: "/datasets/KIRC_NeurIPS/TCGA.KIRC.sampleMap_HiSeqV2.gz"
- Format: tab-delimited, gzip-compressed text.
- Dataset shape (after `index_col='sample'`): (20530, 606)
- Load with: `pd.read_csv(data_path, sep='\\t', compression='gzip', index_col='sample')`

Example data (first 3 rows and subset of columns):
| sample    | TCGA-BP-4162-01 | TCGA-CJ-5677-11 | TCGA-DV-5566-01 | TCGA-BP-5191-01 | TCGA-BP-5200-01 |
| :-------- | --------------: | --------------: | --------------: | --------------: | --------------: |
| ARHGEF10L | 10.503          | 10.8969         | 10.7612         | 10.2063         | 10.0616         |
| HIF3A     | 5.5283          | 6.4943          | 5.7842          | 5.0063          | 5.3326          |
| RNF17     | 3.8036          | 0               | 0               | 0               | 0               |
"""

tcga_kirc_neurips_rppa = """
This dataset is a local copy of the UCSC Xena TCGA KIRC RPPA matrix `TCGA.KIRC.sampleMap/RPPA`.

Source metadata (UCSC Xena):
- Cohort / title: TCGA Kidney Clear Cell Carcinoma (KIRC); "reverse phase protein array".
- Xena wrangling notes: TCGA DCC Level 3 `*.protein_expression.*` files processed into Xena repository.

Useful details:
- Rows are RPPA protein features (total proteins and modified protein epitopes).
- Columns are TCGA sample barcodes.
- The first column is `sample` (feature id), with all remaining columns corresponding to samples.
- RPPA samples partially overlap with RNA-seq/survival cohorts and can be matched by barcode.

Location:
- Repo path: "sandbox/datasets/KIRC_NeurIPS/TCGA.KIRC.sampleMap_RPPA.gz"
- Container path: "/datasets/KIRC_NeurIPS/TCGA.KIRC.sampleMap_RPPA.gz"
- Format: tab-delimited, gzip-compressed text.
- Dataset shape (after `index_col='sample'`): (273, 478)
- Load with: `pd.read_csv(data_path, sep='\\t', compression='gzip', index_col='sample')`

Example data (first 3 rows and subset of columns):
| sample             | TCGA-B8-A54D-01 | TCGA-G6-A8L7-01 | TCGA-B8-A54F-01 | TCGA-B8-A8YJ-01 | TCGA-B8-A54K-01 |
| :----------------- | --------------: | --------------: | --------------: | --------------: | --------------: |
| 14-3-3_beta-R-V    | 0.0650066       | -0.103411       | -0.0717885      | 0.55692         | 0.130937        |
| 14-3-3_epsilon-M-C | -0.175905       | 0.130026        | -0.0800838      | 0.175525        | 0.19844         |
| 14-3-3_zeta-R-V    | -0.195639       | -0.174381       | 0.0645871       | -1.27267        | 0.168871        |
"""

tcga_kirc_neurips_merged_reals = """
This dataset is a local copy of the UCSC Xena pathway-activity matrix `merge_merged_reals/KIRC_merge_merged_reals.txt` for KIRC.

Source metadata (UCSC Xena):
- Data subtype / type: `pathway activity` / `genomicMatrix`.
- Method provenance: PARADIGM integrates pathway, expression, and copy-number data on a SuperPathway network.

Useful details:
- This matrix combines multiple feature types into a single aligned sample-by-feature table.
- Rows correspond to merged feature ids and columns correspond to TCGA samples.
- The first column is `sample` (feature id); the remaining columns are TCGA barcodes.
- Feature names include gene-like symbols and complex/protein-like entries (for example `UBE2Q1`, `GRIK_3_homomer_(complex)`).

Location:
- Repo path: "sandbox/datasets/KIRC_NeurIPS/merge_merged_reals_KIRC_merge_merged_reals.txt.gz"
- Container path: "/datasets/KIRC_NeurIPS/merge_merged_reals_KIRC_merge_merged_reals.txt.gz"
- Format: tab-delimited, gzip-compressed text.
- Dataset shape (after `index_col='sample'`): (19503, 507)
- Load with: `pd.read_csv(data_path, sep='\\t', compression='gzip', index_col='sample')`

Example data (first 3 rows and subset of columns):
| sample                   | TCGA-BP-4799-01 | TCGA-CZ-4856-01 | TCGA-CZ-5464-01 | TCGA-CJ-4878-01 | TCGA-A3-3352-01 |
| :----------------------- | --------------: | --------------: | --------------: | --------------: | --------------: |
| UBE2Q1                   | 0               | 0               | 0               | 0               | 0               |
| GRIK_3_homomer_(complex) | 0.334967        | 0.334967        | 0.334967        | 0.334967        | 0.334967        |
| UBE2Q2                   | 0.192315        | 0               | -1.44315        | -1.44315        | 0               |
"""

tcga_kirc_neurips_survival = """
This dataset is a local copy of the UCSC Xena clinical endpoint table `survival/KIRC_survival.txt`.

Source metadata (UCSC Xena):
- Label: `Curated survival data`.
- Provenance: TCGA-CDR Pan-Cancer clinical resource paper ("An Integrated TCGA Pan-Cancer Clinical Data Resource...").

Useful details:
- One row corresponds to one sample barcode.
- Primary endpoint columns include `OS`, `OS.time`, `DSS`, `DSS.time`, `DFI`, `DFI.time`, `PFI`, and `PFI.time`.
- `sample` is the key used to join with molecular matrices.
- Some endpoint columns can be empty for subsets of patients (missing follow-up values).

Location:
- Repo path: "sandbox/datasets/KIRC_NeurIPS/survival_FKIRC_survival.txt"
- Container path: "/datasets/KIRC_NeurIPS/survival_FKIRC_survival.txt"
- Format: tab-delimited text.
- Dataset shape (after `index_col='sample'`): (944, 11)
- Load with: `pd.read_csv(data_path, sep='\\t', index_col='sample')`

Example data (first 3 rows and subset of columns):
| sample          | _PATIENT     | OS  | OS.time | DSS | DSS.time |
| :-------------- | :----------- | --: | ------: | --: | -------: |
| TCGA-3Z-A93Z-01 | TCGA-3Z-A93Z | 0   | 385     | 0   | 385      |
| TCGA-6D-AA2E-01 | TCGA-6D-AA2E | 0   | 362     | 0   | 362      |
| TCGA-A3-3306-01 | TCGA-A3-3306 | 0   | 1120    | 0   | 1120     |
"""

tcga_luad_hiseqv2 = """
This dataset is a local copy of the UCSC Xena TCGA LUAD RNA-seq matrix `TCGA.LUAD.sampleMap/HiSeqV2`.

Source metadata (UCSC Xena):
- Cohort / title: TCGA Lung Adenocarcinoma (LUAD); "gene expression by RNAseq (polyA+ IlluminaHiSeq)".
- Xena wrangling notes: TCGA DCC Level 3 `*.rsem.genes.normalized_results` processed by UCSC Xena with log2(x+1) transform.

Useful details:
- Rows correspond to genes/features and columns correspond to LUAD TCGA samples.
- The first column is `sample` (feature id), and remaining columns are sample barcodes.
- TCGA barcode suffix convention applies (`01` tumor, `11` solid tissue normal where present).
- Values are continuous normalized RNA expression measurements in the sampleMap HiSeqV2 format.

Location:
- Repo path: "sandbox/datasets/lung_cancer/TCGA.LUAD.sampleMap_HiSeqV2.gz"
- Container path: "/datasets/lung_cancer/TCGA.LUAD.sampleMap_HiSeqV2.gz"
- Format: tab-delimited, gzip-compressed text.
- Dataset shape (after `index_col='sample'`): (20530, 576)
- Load with: `pd.read_csv(data_path, sep='\\t', compression='gzip', index_col='sample')`

Example data (first 3 rows and subset of columns):
| sample    | TCGA-69-7978-01 | TCGA-62-8399-01 | TCGA-78-7539-01 | TCGA-50-5931-11 | TCGA-73-4658-01 |
| :-------- | --------------: | --------------: | --------------: | --------------: | --------------: |
| ARHGEF10L | 9.9898          | 10.4257         | 9.6264          | 8.6835          | 9.2078          |
| HIF3A     | 4.2598          | 11.6239         | 9.1362          | 9.4824          | 5.0288          |
| RNF17     | 0.4181          | 0               | 1.1231          | 0.8221          | 0               |
"""

tcga_luad_rppa = """
This dataset is a local copy of the UCSC Xena TCGA LUAD RPPA matrix `TCGA.LUAD.sampleMap/RPPA`.

Source metadata (UCSC Xena):
- Cohort / title: TCGA Lung Adenocarcinoma (LUAD); "reverse phase protein array".
- Xena wrangling notes: TCGA DCC Level 3 `*.protein_expression.*` files processed into Xena repository.

Useful details:
- Rows are protein features (including phospho-sites and total proteins).
- Columns are TCGA LUAD sample barcodes.
- The first column is `sample` (feature id), with remaining columns as per-sample values.
- RPPA and RNA cohorts overlap only partially, so sample matching should be barcode-based.

Location:
- Repo path: "sandbox/datasets/lung_cancer/TCGA.LUAD.sampleMap_RPPA.gz"
- Container path: "/datasets/lung_cancer/TCGA.LUAD.sampleMap_RPPA.gz"
- Format: tab-delimited, gzip-compressed text.
- Dataset shape (after `index_col='sample'`): (276, 365)
- Load with: `pd.read_csv(data_path, sep='\\t', compression='gzip', index_col='sample')`

Example data (first 3 rows and subset of columns):
| sample             | TCGA-44-7670-01 | TCGA-44-5644-01 | TCGA-55-6979-01 | TCGA-73-7498-01 | TCGA-78-7148-01 |
| :----------------- | --------------: | --------------: | --------------: | --------------: | --------------: |
| 14-3-3_epsilon-M-C | 0.195039        | 0.00326726      | -0.102205       | 0.257452        | 0.09475         |
| 4E-BP1-R-V         | 0.859388        | 0.46694         | 0.172731        | -0.71101        | -0.152258       |
| 4E-BP1_pS65-R-V    | 0.420423        | 0.620014        | 0.196888        | 0.137856        | 0.00608641      |
"""

tcga_luad_merged_reals = """
This dataset is a local copy of the UCSC Xena pathway-activity matrix `merge_merged_reals/LUAD_merge_merged_reals.txt`.

Source metadata (UCSC Xena):
- Label and unit: `Paradigm IPLs`, unit `Paradigm IPL score`.
- Method provenance: PARADIGM integrates pathway, expression, and copy-number data on a SuperPathway network.

Useful details:
- Rows are merged feature ids across modalities and columns are samples.
- The first column is `sample` (feature id), and remaining columns are TCGA barcodes.
- Feature names include both gene-like and complex/protein-like entities.
- This matrix is useful for methods that require a single integrated feature table.

Location:
- Repo path: "sandbox/datasets/lung_cancer/merge_merged_reals_LUAD_merge_merged_reals.txt.gz"
- Container path: "/datasets/lung_cancer/merge_merged_reals_LUAD_merge_merged_reals.txt.gz"
- Format: tab-delimited, gzip-compressed text.
- Dataset shape (after `index_col='sample'`): (19503, 507)
- Load with: `pd.read_csv(data_path, sep='\\t', compression='gzip', index_col='sample')`

Example data (first 3 rows and subset of columns):
| sample                   | TCGA-05-4430-01 | TCGA-55-A57B-01 | TCGA-86-8281-01 | TCGA-55-7910-01 | TCGA-91-6847-01 |
| :----------------------- | --------------: | --------------: | --------------: | --------------: | --------------: |
| UBE2Q1                   | 0.288492        | 0               | 0.288492        | 0.288492        | 0               |
| GRIK_3_homomer_(complex) | -1.07048        | -1.00429        | -1.07048        | -1.07048        | -1.00429        |
| UBE2Q2                   | 0.192315        | -1.44315        | -1.44315        | 0.313955        | 0               |
"""

tcga_luad_survival = """
This dataset is a local copy (renamed as `survival_FLUAD_survival.txt`) of the UCSC Xena clinical endpoint table `survival/LUAD_survival.txt`.

Source metadata (UCSC Xena):
- Label: `Curated survival data`.
- Provenance: TCGA-CDR Pan-Cancer clinical resource paper ("An Integrated TCGA Pan-Cancer Clinical Data Resource...").

Useful details:
- One row per TCGA sample barcode.
- Clinical endpoints include `OS`, `DSS`, `DFI`, and `PFI` with paired time columns.
- `sample` is the primary join key for connecting to RNA, RPPA, and merged molecular matrices.
- Missing entries are present for some endpoints and should be handled as NaN during analysis.

Location:
- Repo path: "sandbox/datasets/lung_cancer/survival_FLUAD_survival.txt"
- Container path: "/datasets/lung_cancer/survival_FLUAD_survival.txt"
- Format: tab-delimited text.
- Dataset shape (after `index_col='sample'`): (641, 11)
- Load with: `pd.read_csv(data_path, sep='\\t', index_col='sample')`

Example data (first 3 rows and subset of columns):
| sample          | _PATIENT     | OS  | OS.time | DSS | DSS.time |
| :-------------- | :----------- | --: | ------: | --: | -------: |
| TCGA-05-4244-01 | TCGA-05-4244 | 0   | 0       | 0   | 0        |
| TCGA-05-4249-01 | TCGA-05-4249 | 0   | 1523    | 0   | 1523     |
| TCGA-05-4250-01 | TCGA-05-4250 | 1   | 121     |     | 121      |
"""

tcga_paad_hiseqv2 = """
This dataset is a local copy of the UCSC Xena TCGA PAAD RNA-seq matrix `TCGA.PAAD.sampleMap/HiSeqV2`.

Source metadata (UCSC Xena):
- Cohort / title: TCGA Pancreatic Cancer (PAAD); "gene expression by RNAseq (polyA+ IlluminaHiSeq)".
- Xena wrangling notes: TCGA DCC Level 3 `*.rsem.genes.normalized_results` processed by UCSC Xena with log2(x+1) transform.

Useful details:
- Rows correspond to gene/features and columns correspond to TCGA PAAD samples.
- `sample` is the feature identifier column; remaining columns are TCGA barcodes.
- Values are continuous expression values in sampleMap HiSeqV2 format.
- Can be aligned to PAAD RPPA and PAAD survival tables by TCGA sample barcode.

Location:
- Repo path: "sandbox/datasets/PAAD/TCGA.PAAD.sampleMap_HiSeqV2.gz"
- Container path: "/datasets/PAAD/TCGA.PAAD.sampleMap_HiSeqV2.gz"
- Format: tab-delimited, gzip-compressed text.
- Dataset shape (after `index_col='sample'`): (20530, 183)
- Load with: `pd.read_csv(data_path, sep='\\t', compression='gzip', index_col='sample')`

Example data (first 3 rows and subset of columns):
| sample    | TCGA-2L-AAQL-01 | TCGA-2J-AABI-01 | TCGA-3A-A9J0-01 | TCGA-3A-A9I7-01 | TCGA-2J-AABO-01 |
| :-------- | --------------: | --------------: | --------------: | --------------: | --------------: |
| ARHGEF10L | 10.243          | 9.5734          | 10.3872         | 10.1431         | 9.8022          |
| HIF3A     | 6.6983          | 6.4428          | 5.5034          | 4.6491          | 7.2137          |
| RNF17     | 0               | 0               | 0               | 1.1814          | 0               |
"""

tcga_paad_rppa = """
This dataset is a local copy of the UCSC Xena TCGA PAAD RPPA matrix `TCGA.PAAD.sampleMap/RPPA`.

Source metadata (UCSC Xena):
- Cohort / title: TCGA Pancreatic Cancer (PAAD); "reverse phase protein array".
- Xena wrangling notes: TCGA DCC Level 3 `*.protein_expression.*` files processed into Xena repository.

Useful details:
- Rows are RPPA feature ids and columns are TCGA sample barcodes.
- The first column is `sample` (feature identifier), followed by one column per patient sample.
- Includes total and modified protein markers commonly used in pathway-level analyses.
- Barcode overlap with PAAD RNA/survival is partial and should be matched explicitly.

Location:
- Repo path: "sandbox/datasets/PAAD/TCGA.PAAD.sampleMap_RPPA.gz"
- Container path: "/datasets/PAAD/TCGA.PAAD.sampleMap_RPPA.gz"
- Format: tab-delimited, gzip-compressed text.
- Dataset shape (after `index_col='sample'`): (229, 123)
- Load with: `pd.read_csv(data_path, sep='\\t', compression='gzip', index_col='sample')`

Example data (first 3 rows and subset of columns):
| sample             | TCGA-2J-AABA-01 | TCGA-FB-AAQ6-01 | TCGA-Z5-AAPL-01 | TCGA-XD-AAUI-01 | TCGA-2J-AABR-01 |
| :----------------- | --------------: | --------------: | --------------: | --------------: | --------------: |
| 14-3-3_beta-R-V    | 0.109961        | 0.77485         | 0.241729        | -0.0229336      | 0.137529        |
| 14-3-3_epsilon-M-C | 0.0701853       | 0.107562        | -0.210982       | 0.100178        | 0.230904        |
| 14-3-3_zeta-R-V    | 0.0390999       | -0.234284       | 0.588326        | -0.237403       | 0.481003        |
"""

tcga_paad_merged_reals = """
This dataset is a local copy of the UCSC Xena pathway-activity matrix `merge_merged_reals/PAAD_merge_merged_reals.txt`.

Source metadata (UCSC Xena):
- Label and unit: `Paradigm IPLs`, unit `Paradigm IPL score`.
- Method provenance: PARADIGM integrates pathway, expression, and copy-number data on a SuperPathway network.

Useful details:
- Rows are merged feature ids across modalities and columns are TCGA PAAD samples.
- The first column is `sample` (feature id); subsequent columns are patient/sample barcodes.
- Feature identifiers include gene-like and complex/protein-like names.
- Useful when you want one integrated matrix rather than modality-specific tables.

Location:
- Repo path: "sandbox/datasets/PAAD/merge_merged_reals_PAAD_merge_merged_reals.txt.gz"
- Container path: "/datasets/PAAD/merge_merged_reals_PAAD_merge_merged_reals.txt.gz"
- Format: tab-delimited, gzip-compressed text.
- Dataset shape (after `index_col='sample'`): (19503, 176)
- Load with: `pd.read_csv(data_path, sep='\\t', compression='gzip', index_col='sample')`

Example data (first 3 rows and subset of columns):
| sample                   | TCGA-2L-AAQL-01 | TCGA-3A-A9IS-01 | TCGA-FB-A545-01 | TCGA-Q3-AA2A-01 | TCGA-IB-AAUM-01 |
| :----------------------- | --------------: | --------------: | --------------: | --------------: | --------------: |
| UBE2Q1                   | 0.288492        | -2.16255        | 0               | 0               | 0               |
| GRIK_3_homomer_(complex) | 0.239549        | -1.22353        | 0.334967        | -1.07048        | 0.364151        |
| UBE2Q2                   | 0               | -1.8081         | 0               | 0               | -1.44315        |
"""

tcga_paad_survival = """
This dataset is a local copy (renamed as `survival_FPAAD_survival.txt`) of the UCSC Xena clinical endpoint table `survival/PAAD_survival.txt`.

Source metadata (UCSC Xena):
- Label: `Curated survival data`.
- Provenance: TCGA-CDR Pan-Cancer clinical resource paper ("An Integrated TCGA Pan-Cancer Clinical Data Resource...").

Useful details:
- Each row is a TCGA sample/patient entry.
- Endpoint columns include overall and disease-specific outcomes (`OS`, `DSS`) and progression-related fields (`DFI`, `PFI`) with matching time columns.
- `sample` is the primary key for joining with PAAD RNA, RPPA, or merged matrices.
- Missing values occur in some endpoint columns depending on available follow-up.

Location:
- Repo path: "sandbox/datasets/PAAD/survival_FPAAD_survival.txt"
- Container path: "/datasets/PAAD/survival_FPAAD_survival.txt"
- Format: tab-delimited text.
- Dataset shape (after `index_col='sample'`): (196, 11)
- Load with: `pd.read_csv(data_path, sep='\\t', index_col='sample')`

Example data (first 3 rows and subset of columns):
| sample          | _PATIENT     | OS  | OS.time | DSS | DSS.time |
| :-------------- | :----------- | --: | ------: | --: | -------: |
| TCGA-2J-AAB1-01 | TCGA-2J-AAB1 | 1   | 66      | 1   | 66       |
| TCGA-2J-AAB4-01 | TCGA-2J-AAB4 | 0   | 729     | 0   | 729      |
| TCGA-2J-AAB6-01 | TCGA-2J-AAB6 | 1   | 293     | 1   | 293      |
"""


nhanes = """
Title:
- NHANES processed cohort (51 labeled features + outcome)

Context:
- Harmonized features and outcomes assembled into a single DataFrame; feature codes mapped to human-readable labels.

Object:
- Path: /datasets/NHANES/nhanes_dataset.pkl
- Format: pandas DataFrame (pickle); rows = participants (ID), columns = 51 labeled features + outcome


Load:
```python
import pickle, pandas as pd
with open("/datasets/NHANES/nhanes_dataset.pkl", "rb") as f:
    nhanes = pickle.load(f)
nhanes.head()
```

"""

mi_gwas_genes = """
Title:
- MI GWAS gene-level associations (FinnGen I9_MI)

Context:
- Aggregates significant variants to genes (nearest TSS, ±50 kb); lead variant per gene (min p), plus count of significant variants.
- Source meta (OpenGWAS: finn-b-I9_MI): GRCh37; European; year 2021; ncase 12801; ncontrol 187840; nsnp 16380433.

Object:
- Path: /datasets/NHANES/mi_genes.csv.gz
- Columns: gene_symbol, n_sig_variants, pval, lp, es, or, chrom, pos, rsid, distance

Load:
```python
import pandas as pd
mi_genes = pd.read_csv("/datasets/NHANES/mi_genes.csv.gz")
mi_genes.head()
```

Example (first row):
| gene_symbol | n_sig_variants | pval         | lp      | es     | or       | chrom | pos       | rsid        | distance |
| ----------- | -------------- | ------------ | ------- | ------ | -------- | ----- | --------- | ----------- | -------- |
| ARHGAP10    | 1              | 4.181961e-09 | 8.37862 | 0.3045 | 1.355947 | 4     | 147711494 | rs534388792 | 20569    |
"""


DATASETS = {
    # TCGA KIRC (UCSC Xena)
    "tcga_kirc_neurips_hiseqv2": tcga_kirc_neurips_hiseqv2,
    "tcga_kirc_neurips_rppa": tcga_kirc_neurips_rppa,
    "tcga_kirc_neurips_merged_reals": tcga_kirc_neurips_merged_reals,
    "tcga_kirc_neurips_survival": tcga_kirc_neurips_survival,
    # TCGA LUAD (UCSC Xena)
    "tcga_luad_hiseqv2": tcga_luad_hiseqv2,
    "tcga_luad_rppa": tcga_luad_rppa,
    "tcga_luad_merged_reals": tcga_luad_merged_reals,
    "tcga_luad_survival": tcga_luad_survival,
    # TCGA PAAD (UCSC Xena)
    "tcga_paad_hiseqv2": tcga_paad_hiseqv2,
    "tcga_paad_rppa": tcga_paad_rppa,
    "tcga_paad_merged_reals": tcga_paad_merged_reals,
    "tcga_paad_survival": tcga_paad_survival,
    # NHANES and MI GWAS
    "nhanes": nhanes,
    "mi_gwas_genes": mi_gwas_genes,
}
