# ML Data Streams Project

This repository experiments with online learning on data streams, with a focus on concept drift and the Active Expert Repository (AER) approach.

The main entry point is `test.py`. It evaluates several stream-learning wrappers on synthetic recurring streams real River datasets, and a synthetic known-drift stream. Results are written as CSV files and sliding-window accuracy plots.

## What Is Included

- `aer.py`: Active Expert Repository using one base model type.
- `aer_multi_model.py`: AER variant whose warning-phase challengers can use multiple model types.
- `shadow_expert_model.py`: Baseline model that resets to a fresh expert when drift is detected.
- `recurring_stream.py`: Helper for cycling through labeled synthetic concepts.
- `plot_utils.py`: Sliding-window accuracy plotting.
- `test.py`: CLI and reusable evaluation functions.

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv my_env
source my_env/bin/activate
pip install -r requirements.txt
```

## Quick Start

Run the default synthetic recurring experiment:

```bash
./my_env/bin/python test.py
```

Run on the Phishing dataset:

```bash
python test.py --dataset phishing
```

Run only one model type for a short smoke test:

```bash
python test.py --model-types HT --stream-length 100 --chunk-size 50
```

Run a real dataset and write outputs to a custom directory:

```bash
python test.py --dataset elec2 --output-dir real_data_results
```

## Using `test.py`

`test.py` evaluates each requested model type with three wrappers:

- `AER_<model_type>`: Active Expert Repository with that base model.
- `AER_MultiModel`: Active Expert Repository with challengers from all selected model types.
- `ShadowExpertModel_<model_type>`: Reset-on-drift baseline using that model type.

Supported model types are:

- `HT`: River `HoeffdingTreeClassifier`
- `ARF`: River `ARFClassifier`
- `SRP`: River `SRPClassifier`
- `NaiveBayes`: River `GaussianNB`

The default model list is:

```text
HT ARF SRP NaiveBayes
```

### CLI Options

```bash
python test.py --help
```

Available options:

| Option | Default | Description |
| --- | --- | --- |
| `--dataset` | `synthetic_recurring` | Stream to evaluate. |
| `--model-types` | `HT ARF SRP NaiveBayes` | One or more model types to evaluate. |
| `--stream-length` | Dataset-specific | Number of stream instances to process. |
| `--chunk-size` | Dataset-specific | Number of instances between CSV summary rows. |
| `--output-dir` | Dataset-specific | Directory for CSV and plot outputs. |
| `--river-data-dir` | `.river_data` | Cache directory for remote River datasets. |
| `--plot-window` | `100` | Sliding-window size used in accuracy plots. |
| `--show-plot-inline` | `False` | Show plots interactively in addition to saving them. |

### Dataset Notes

- `synthetic_recurring` cycles between two Agrawal concepts:
  - `Agrawal_Function_0`
  - `Agrawal_Function_6`
- `phishing` uses `river.datasets.Phishing`.
- `elec2` uses `river.datasets.Elec2`.
- `synthetic_sea_drift` uses `river.datasets.synth.ConceptDriftStream` with SEA variant `0` drifting to SEA variant `1`.

River datasets that need local cached files use the `RIVER_DATA` environment variable. `test.py` sets it automatically from `--river-data-dir` if it is not already defined.

## Output Files

Each model type produces two files:

- Accuracy CSV: chunk-level period and cumulative accuracy.
- PNG plot: sliding-window accuracy over time with detected events and known drift markers when available.

The CSV columns are:

| Column | Meaning |
| --- | --- |
| `evaluated_model_type` | Model type being evaluated, such as `HT` or `ARF`. |
| `wrapper_model_name` | Wrapper being evaluated, such as `AER_ARF` or `AER_MultiModel`. |
| `chunk_index` | 1-based chunk number. |
| `stream_label` | Label for the current concept/dataset stream. |
| `active_model` | Current active expert label at the end of the chunk. |
| `period_accuracy` | Accuracy percentage within the current chunk. |
| `cumulative_accuracy` | Accuracy percentage over all processed instances so far. |
