# TARTE in the medical model benchmark

This folder contains the TARTE-based experiment used by the broader medical
benchmark. It is one model pipeline alongside the other tabular foundation
models and traditional methods in `models/`.

The experiment first pretrains a small TARTE encoder on YAGO. The frozen encoder
then converts rows from the fixed diabetes, hepatitis, and heart splits into
features. XGBoost uses those features for binary classification. The resulting
metrics and confusion matrices can therefore be compared with the outputs of
the other benchmarked models.

The architecture and training configuration are fixed in the code to keep runs
comparable: 15,000 YAGO entities, 3,000 optimizer steps, a 64-dimensional hidden
space, four attention heads, and two transformer layers.

## Files

- `common.py`: shared random-seed handling, device selection, and the projection
  layer used by TARTE.
- `pretrained.py`: FastText and row encoding, the fixed TARTE architecture,
  batching, checkpoint validation, and frozen feature extraction. Despite its
  name, it does not contain model weights.
- `yago_data.py`: reads supported YAGO exports, samples entities reproducibly,
  and groups triple records into entity-level training examples.
- `pretrain_tarte.py`: pretrains the TARTE encoder on YAGO and writes its fixed
  checkpoint and pretraining metadata.
- `tarte_featurizer_wrapper.py`: fits XGBoost on frozen TARTE features and
  produces prediction probabilities.
- `run_tests.py`: runs the three medical benchmark evaluations, computes the
  shared classification metrics, and saves confusion matrices and run metadata.
- `requirements.txt`: records the Python packages used by this model pipeline.
- `.gitignore`: excludes generated TARTE checkpoints from version control.

## Inputs and outputs

The medical train/test splits are shared benchmark inputs from
`data/pre-processed`. TARTE pretraining additionally requires an external YAGO
export and an English FastText `.bin` model; these large assets are not stored in
this repository. The same FastText model must be used when pretraining and
evaluating the checkpoint.

YAGO may be represented as an entity table, JSONL objects, or
`head`/`relation`/`tail` triples. `yago_data.py` converts any of these supported
representations into the entity rows expected by the pretraining stage.

Pretraining artifacts are written to `models/tfm/tarte/checkpoints`. Medical
benchmark results are written to `results/tarte`, including the metrics CSV,
confusion matrices, and run metadata.

This implementation is a small TARTE adaptation for comparison inside this
repository. It is not an exact reproduction of the original paper or its
reported results.
