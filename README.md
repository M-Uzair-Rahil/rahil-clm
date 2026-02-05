# Author: Mohammad Uzair Rahil
# Michigan State University (MSU)
# Date: 01/01/2026

# rahil-clm

`rahil-clm` is a Python package to generate Latin Hypercube Sampling (LHS)–based
crop parameter ensembles for **CLM/CTSM crop yield optimization**.

The package is designed so users **do not need to provide any input file paths**.
Required input files (Excel parameter bounds and base CLM NetCDF parameters) are
automatically downloaded from GitHub Releases and cached locally on first use.

---

## Installation

```bash
pip install rahil-clm



## Quick start

import rahil

out = rahil.generate_lhs(
    Ninit=150,
    seed=42,
    output_dir="outputs"
)

print(out.param_output_dir)
print(out.main_run_file)
print(out.param_list_file)
print(out.psets_df.head())



This will:

generate LHS samples for crop parameters
write per-case CLM NetCDF parameter files
create workflow text files for running CTSM/CLM ensembles





