# IV-PCMCI Causal Discovery Project

A Python research project for comparing baseline PCMCI-style causal discovery with an instrumental-variable (IV) extension, including synthetic time-series generation, conditional-independence testing, Tigramite integration, sensitivity analysis, metrics, and saved result artifacts.

> **Status:** Research / experimental code. Causal-discovery outputs depend on the validity of the data-generating assumptions, model choices, and instrumental-variable assumptions; they are not automatic evidence of real-world causality.

## Overview

The project generates synthetic data with a known edge structure, runs baseline and IV-informed causal-discovery workflows, compares discovered edges against the reference graph, and writes JSON and CSV outputs. It also includes a sensitivity experiment and a saved plot.

The work is contained in `iv_pcmci_project/`.

## Components

| File or area | Purpose |
| --- | --- |
| `generate_data.py` | Generates the synthetic time-series dataset. |
| `config.py` | Centralizes experiment configuration. |
| `baseline_pcmci.py` | Runs the baseline causal-discovery workflow. |
| `iv_pcmci.py` | Implements the IV-aware PCMCI-related workflow. |
| `iv_cond_ind_test.py` | Implements the IV conditional-independence test logic. |
| `run_experiment.py` | Orchestrates the project experiment. |
| `tigramite_integration.py` | Supports integration with Tigramite-based processing. |
| `run_tigramite_ivpcmci.py` | Runs the Tigramite IV-PCMCI workflow. |
| `sensitivity.py`, `sensitivity_experiment.py` | Defines and runs sensitivity analysis. |
| `utils.py` | Shared utility functions. |
| `run.sh`, `run_tigramite.sh` | Shell shortcuts for the base and Tigramite workflows. |

## Inputs and outputs

| Artifact | Description |
| --- | --- |
| `synthetic_data.csv` | Generated time-series data used by the experiments. |
| `true_edges.json` | Reference causal-edge structure used for evaluation. |
| `baseline_edges.json`, `ivpcmci_edges.json` | Edges produced by the baseline and IV-aware workflows. |
| `baseline_summary.csv`, `ivpcmci_summary.csv` | Summary outputs from the two primary workflows. |
| `metrics.csv` | Comparison metrics for the primary implementation. |
| `tigramite_*` JSON and CSV files | Outputs from Tigramite-based baseline and IV workflows. |
| `sensitivity_results.csv`, `sensitivity_diagnostics.csv` | Outputs from sensitivity experiments. |
| `sensitivity_plot.png` | Visualization generated from sensitivity results. |

## Requirements

- Python 3.
- Numerical and data-analysis packages required by the project source, typically including NumPy and pandas.
- Tigramite for the Tigramite integration/workflow.
- Matplotlib or other plotting dependencies required by the sensitivity analysis.

This repository does not currently include a pinned dependency file. Inspect imports in the Python scripts and create a virtual environment before running the project.

## Setup

```bash
git clone https://github.com/Verlopat/012.git
cd 012/iv_pcmci_project

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the libraries required by the implementation. A starting point, subject to the actual imports and compatible versions, is:

```bash
pip install numpy pandas matplotlib tigramite
```

## Run the experiment

Use the provided shell wrapper or invoke the Python scripts directly:

```bash
# Review the wrapper first
cat run.sh

# Run the base workflow
bash run.sh

# Or execute stages explicitly
python3 generate_data.py
python3 run_experiment.py
```

For the Tigramite-based workflow:

```bash
cat run_tigramite.sh
bash run_tigramite.sh

# Or run the corresponding Python entry point directly
python3 run_tigramite_ivpcmci.py
```

Run the sensitivity analysis after the main pipeline has produced its expected inputs:

```bash
python3 sensitivity_experiment.py
```

## Workflow

1. Configure the synthetic process and statistical settings in `config.py`.
2. Generate `synthetic_data.csv` and the reference edge structure.
3. Run the baseline PCMCI workflow.
4. Run the IV-aware workflow and conditional-independence tests.
5. Optionally run the Tigramite baseline and IV integrations.
6. Compare discovered edges with `true_edges.json` and write metrics.
7. Run sensitivity experiments and inspect the resulting diagnostics and plot.

## Interpreting results

Compare baseline and IV-aware outputs under the same generated dataset and configuration. Relevant questions include whether the approach recovers expected edges, avoids false positives, remains stable under changed parameters, and changes its behavior when IV assumptions are violated.

An instrumental-variable approach requires substantive assumptions—such as relevance, exclusion, and appropriate independence conditions—that cannot be established solely by running this code. A synthetic benchmark can test implementation behavior, but it cannot prove that an approach will identify causal effects in a real dataset.

## Repository layout

```text
.
├── README.md
└── iv_pcmci_project/
    ├── config.py
    ├── generate_data.py
    ├── baseline_pcmci.py
    ├── iv_pcmci.py
    ├── iv_cond_ind_test.py
    ├── run_experiment.py
    ├── tigramite_integration.py
    ├── run_tigramite_ivpcmci.py
    ├── sensitivity.py
    ├── sensitivity_experiment.py
    ├── utils.py
    ├── run.sh, run_tigramite.sh
    ├── synthetic_data.csv
    ├── true_edges.json
    ├── *_edges.json, *_summary.csv, *_metrics.csv
    └── sensitivity_plot.png
```

## Reproducibility

- Save the source revision, Python version, package versions, configuration values, and random seeds.
- Preserve the generated `synthetic_data.csv` and `true_edges.json` used for every result comparison.
- Keep baseline, IV, and Tigramite runs on exactly the same dataset when comparing outputs.
- Do not treat committed CSVs or PNGs as current results after source or dependency changes; regenerate them.

## Suggested improvements

1. Add a `requirements.txt` or `pyproject.toml` with tested, pinned versions.
2. Document the synthetic data-generating process, every variable, lag convention, and ground-truth edge.
3. State the IV assumptions explicitly and add diagnostics or failure-mode experiments for each.
4. Add unit tests for conditional-independence tests, edge scoring, data generation, and metrics.
5. Add a command-line interface for setting seeds, sample size, lags, significance thresholds, and output paths.
6. Add tables and plots that compare precision, recall, false-discovery behavior, runtime, and sensitivity outcomes.
7. Remove local editor lock files and `__pycache__/` from source control; add appropriate `.gitignore` rules.
8. Add a license and contribution guidance.

## License

No license file is currently included. Add an explicit license before distributing or accepting external contributions.
