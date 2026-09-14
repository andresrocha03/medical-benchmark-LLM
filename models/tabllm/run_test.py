import argparse
import os
from pathlib import Path
from time import perf_counter

import pandas as pd

try:
    from .config.setup_config import DATASETS, MODELS, RESULTS_DIR
    from .prediction import (
        SERIALIZATION_COLUMNS,
        clear_model_resources,
        load_model,
        run_dataset,
    )
    from .compute_metrics import evaluate_and_save_run
except ImportError:  # Allow direct execution from models/tabllm.
    from models.tabllm.config.setup_config import DATASETS, MODELS, RESULTS_DIR
    from models.tabllm.prediction import (
        SERIALIZATION_COLUMNS,
        clear_model_resources,
        load_model,
        run_dataset,
    )
    from models.tabllm.compute_metrics import evaluate_and_save_run


RESULT_KEY = ["dataset", "model", "serialization"]


def upsert_result(results, result):
    """Replace an existing trio result or append it when it is new."""
    if results.empty:
        return pd.DataFrame([result])

    matching_row = pd.Series(True, index=results.index)
    for column in RESULT_KEY:
        matching_row &= results[column] == result[column]

    return pd.concat(
        [results.loc[~matching_row], pd.DataFrame([result])],
        ignore_index=True,
    )


def run_model_tests(
    models=None,
    datasets=None,
    serializations=None,
    batch=1,
):
    """
    Run all configured models on all configured datasets.

    Saves one raw-response CSV and confusion matrix per trio, plus one results
    CSV containing the standard benchmark metrics for every trio.
    """
    if batch < 1:
        raise ValueError("--batch must be at least 1.")

    if models is None:
        models = list(MODELS.keys())

    if datasets is None:
        datasets = list(DATASETS.keys())

    if serializations is None:
        serializations = list(SERIALIZATION_COLUMNS.keys())


    selected_models = models
    selected_datasets = datasets
    selected_serializations = serializations
    max_examples = None

    dataset_serializations = {
        dataset_name: selected_serializations
        for dataset_name in selected_datasets
    }

    if not any(dataset_serializations.values()):
        print("No requested serialization columns were found. Nothing to run.")
        return pd.DataFrame()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results_path = Path(RESULTS_DIR) / (
        "tabllm_results.csv"
    )

    if results_path.exists():
        all_results = pd.read_csv(results_path)
    else:
        all_results = pd.DataFrame()

    for model_key in selected_models:
        model_name = MODELS[model_key]

        print(f"\nLoading model: {model_key} -> {model_name}")

        tokenizer, model = load_model(model_name)

        for dataset_name in selected_datasets:
            for serialization_style in dataset_serializations[dataset_name]:
                prediction_start = perf_counter()
                dataset_config = DATASETS[dataset_name]

                run_metadata = run_dataset(
                    dataset_name=dataset_name,
                    dataset_config=dataset_config,
                    model_key=model_key,
                    tokenizer=tokenizer,
                    model=model,
                    serialization_style=serialization_style,
                    max_examples=max_examples,
                    batch_size=batch,
                )

                prediction_time_seconds = perf_counter() - prediction_start
                result = evaluate_and_save_run(
                    run_metadata["output_file"],
                    dataset=dataset_name,
                    serialization=serialization_style,
                    model=model_key,
                    prediction_time_seconds=prediction_time_seconds,
                    total_examples=run_metadata["total_examples"],
                    batch_size=run_metadata["batch_size"],
                )
                all_results = upsert_result(all_results, result)
                all_results = all_results.sort_values(RESULT_KEY).reset_index(
                    drop=True
                )
                # Persist after every trio so completed runs survive an
                # interruption during a later model evaluation.
                all_results.to_csv(results_path, index=False)

                print(
                    f"\nFinished {dataset_name} / {serialization_style} with {model_key}. "
                    f"Saved metrics and confusion matrix to {RESULTS_DIR}."
                )

        clear_model_resources(model, tokenizer)
    print(f"\nAll results saved to {results_path}.")
    print(all_results)

    return all_results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
        help="Models to run."
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
        help="Datasets to run."
    )

    parser.add_argument(
        "--serializations",
        nargs="+",
        default=list(SERIALIZATION_COLUMNS.keys()),
        choices=list(SERIALIZATION_COLUMNS.keys()),
        help="Serialization styles to run."
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size for generation. Default is 1, meaning no batching.",
    )


    args = parser.parse_args()

    run_model_tests(
        models=args.models,
        datasets=args.datasets,
        serializations=args.serializations,
        batch=args.batch,
    )


if __name__ == "__main__":
    main()
