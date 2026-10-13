# Data

The repository contains no data. This page explains where to get the public datasets the
experiments use, where to put them, and how to describe your own data to the agents.

## Where the data go

Each TCGA description in `env_setup/datasets.py` gives two paths (the NHANES and MI GWAS
descriptions give only the container path, which the preflight maps to `sandbox/datasets/`):

- **Repo path** (`sandbox/datasets/<folder>/<file>`): where the files live on your machine. The
  dataset preflight checks them there before a run.
- **Container path** (`/datasets/<folder>/<file>`): where the agents' code sees them. The sandbox
  server mounts `SANDBOX_DATASETS_DIR` read-only at `/datasets`.

So keep the files under `sandbox/datasets/` (or symlink that folder to where they are) and start
the sandbox server with `SANDBOX_DATASETS_DIR=$PWD/sandbox/datasets`.

| Bundle (`dataset=`) | Folder | Files |
|---|---|---|
| `kirc` | `KIRC_NeurIPS/` | RNA-seq, RPPA, PARADIGM pathway activity, survival |
| `luad` | `lung_cancer/` | the same four for LUAD |
| `paad` | `PAAD/` | the same four for PAAD |
| `nhanes` | `NHANES/` | `nhanes_dataset.pkl`, `mi_genes.csv.gz` |

## TCGA (KIRC, LUAD, PAAD)

Source: the [UCSC Xena](https://xenabrowser.net) TCGA hub (`tcga.xenahubs.net`), which
redistributes open-access data from The Cancer Genome Atlas. Download all three cohorts with:

```bash
python -m scripts.data.download_tcga                    # or: --cohorts kirc
```

The script saves each file under the name its description expects and checks its SHA-256 against
`sys_config/dataset/checksums.yaml`. A mismatch means Xena has changed the file since the paper's
runs, so the data may differ. The dataset preflight checks the same checksums before every run:
with `data.preflight_strict: true` (the default) a mismatch stops the run; with `false` it is
reported and the run continues.

Terms: TCGA open-access data follow the
[GDC data use policies](https://gdc.cancer.gov/access-data/data-access-processes-and-tools) and the
[TCGA publication guidelines](https://www.cancer.gov/ccg/research/genome-sequencing/tcga/using-tcga-data/citing);
cite UCSC Xena (Goldman et al., *Nature Biotechnology*, 2020) when you use it.

## NHANES and MI GWAS (the `nhanes` bundle)

Get the raw data from:

- **NHANES**: the [CDC NHANES portal](https://wwwn.cdc.gov/nchs/nhanes/). Public-use files are
  free to use under the NCHS data use restrictions shown on the portal (in particular, no attempt
  to identify participants).
- **Myocardial infarction GWAS**: FinnGen summary statistics for `I9_MI`, available from
  [OpenGWAS](https://gwas.mrcieu.ac.uk/datasets/finn-b-I9_MI/) as `finn-b-I9_MI`. Check FinnGen's
  terms for the summary-statistics release you download.

## You need to get the data and describe it

Agents only learn about a dataset through its description in `env_setup/datasets.py`: what it is,
its path and format, how to load it, and (except for `nhanes`) its columns and a few example rows.
For the `nhanes` bundle you therefore have two options once you have downloaded the raw data:

1. **Match our description.** Prepare files that match it exactly (path, format, columns): a pandas
   DataFrame pickle at `NHANES/nhanes_dataset.pkl` with one row per participant and 51 labelled
   features plus the outcome, and `NHANES/mi_genes.csv.gz` with the gene-level columns the
   `mi_gwas_genes` description lists. Then use our descriptions as they are.
2. **Describe your own files.** Prepare the files the way you want and write your own
   description for them, copying ours as a template.

The TCGA files from the download script already match their descriptions.

## Adding or describing a dataset

1. Put the files under `sandbox/datasets/<folder>/`.
2. Add a description string to `env_setup/datasets.py`. Copy an existing one and keep its parts:
   a title and source, what rows and columns mean, the repo path and container path, the format,
   a load snippet that runs in the sandbox, the shape, and a few example rows.
3. Register it in the `DATASETS` dictionary at the bottom of that file under a new key.
4. Use the key in a run, either in a bundle (`sys_config/dataset/<name>.yaml`, selected with
   `dataset=<name>`) or directly with `discovery_dataset.dataset_keys=[<key>,...]`.
5. Check it: `python -m env_setup.preflight_data --config sys_config/dataset/<name>.yaml` reads
   each file the descriptions point to, checks recorded checksums, prints whether the data are
   available and any errors, and writes each file's shape and a sample to
   `artifacts/data_manifest.json` (`--manifest`). Runs do the same at start-up (`data.preflight`).
