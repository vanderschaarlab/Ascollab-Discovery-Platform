# Changelog

## 1.0.0

Initial public release, with the code for the paper's experiments:

- the multi-agent pipeline (`system.main`) with the paper's default settings in
  `sys_config/ascollab.yaml`;
- Azure OpenAI and OpenAI-compatible model endpoints, with any mix of research-agent models;
- one `run.sh` per experiment and a `rank.sh` for each except 06 (`scripts/experiments/`), the
  AutoRanker and the statistics (`scripts.analysis.stat_tests`);
- download and checksums for the public TCGA data, and instructions for NHANES and the MI GWAS
  data;
- the Docker sandbox server;
- tests that run every pipeline condition of the experiment scripts with scripted models (no API
  calls).
