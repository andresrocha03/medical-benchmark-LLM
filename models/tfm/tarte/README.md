# TARTE in the medical model benchmark

This experiment uses the authors' official [`tarte-ai`](https://github.com/soda-inria/tarte-ai)
package as a frozen table featurizer. It does not train or maintain a local
TARTE backbone.

For each fixed medical train/test split, the pipeline is:

1. `TARTE_TablePreprocessor` prepares the pandas table.
2. `TARTE_TableEncoder(layer_index=2)` extracts frozen pretrained features.
3. `XGBClassifier` fits the binary target using the frozen TARTE features.

The preprocessor and encoder are fitted only on the training split. The fitted
pipeline then transforms the test split, so test information is not used while
fitting preprocessing or XGBoost parameters.

## Run

`tarte-ai` requires Python 3.11 or newer. Install this experiment's dependencies:

```bash
python -m pip install -r models/tfm/tarte/requirements.txt
```

Then run all three datasets from any working directory:

```bash
python models/tfm/tarte/run_tests.py
```

With the default library configuration, TARTE obtains its pretrained model and
FastText resources on first use. The first run therefore needs network access;
later runs can use the local cache.

Results are written under `results/tarte`: one metrics CSV, one confusion
matrix per dataset, and a JSON file recording the package version, encoder
layer, device, seed, datasets, and XGBoost parameters.

## Files

- `tarte_featurizer_wrapper.py`: the official TARTE preprocessing/encoding
  pipeline and XGBoost classifier.
- `run_tests.py`: dataset loading, evaluation, metrics, plots, and metadata.
- `requirements.txt`: the direct Python dependencies for this experiment.

Dataset loading, metrics, plotting, and reproducibility helpers are shared by
all models in `models/utils.py`.
