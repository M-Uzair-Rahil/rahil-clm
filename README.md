# Author: Mohammad Uzair Rahil
# Michigan State University (MSU)
# Date: 01/01/2026

# rahil-clm

**rahil-clm** is a lightweight Python package for generating **Latin Hypercube Sampling (LHS) ensembles** of crop parameters for the **Community Land Model (CLM / CTSM)** and for **diagnosing the sampled parameter distributions** against prescribed bounds.

The package is designed so users can simply install it, import `rahil`, and generate ready-to-run CLM parameter NetCDF files **without manually handling input files or paths**.

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


## What this does
Running generate_lhs() will automatically:

1. Generate Latin Hypercube samples for crop parameters
(rainfed corn, soybean, and spring wheat)

2. Write per-case CLM NetCDF parameter files

3. Create workflow text files for running CTSM/CLM ensembles

4. Automatically fetch required base files from GitHub releases

5. No manual input file paths are required.

## Output structure
After running the example above, your directory will look like:

outputs/
├── paramfile/
│   └── pe_crops/
│       ├── pe_crops_0_0000.nc
│       ├── pe_crops_0_0001.nc
│       └── ...
├── workflow/
│   ├── pe_crops_0.main_run.txt
│   └── pe_crops_0.param_list.txt


## Key outputs
1. paramfile/pe_crops/*.nc
CLM-compatible NetCDF parameter files (one per ensemble member)

2. workflow/*.param_list.txt
Table of all sampled parameters (used for diagnostics and plotting)

3. workflow/*.main_run.txt
Case IDs for batch CTSM/CLM execution

### Checking sampled parameter distributions
You can quickly verify that sampled parameters lie within prescribed bounds using:

rahil.check_distribution(
    output_dir="outputs",
    show=False
)

This will:

1. Read the sampled parameter list from outputs/workflow/
2. Load parameter bounds automatically from the package release
3. Generate article-style distribution plots (one figure per crop)
4. Save figures to:

bash
Copy code
outputs/figs_distributions_by_crop/
├── LHS_distributions_corn.png
├── LHS_distributions_soybean.png
└── LHS_distributions_wheat.png
Each figure shows:
Histograms of sampled values
Dashed red lines indicating minimum and maximum bounds
Separate panels for each parameter

### Example: generate + diagnose in one script

import rahil

out = rahil.generate_lhs(
    Ninit=50,
    seed=10,
    output_dir="outputs_test"
)

rahil.check_distribution(
    output_dir="outputs_test",
    show=False
)


