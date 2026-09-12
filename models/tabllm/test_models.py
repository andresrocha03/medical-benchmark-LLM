import argparse
import os
import time
from pathlib import Path

import pandas as pd

try:
    from .setup_config import DATASETS, MODELS, RESULTS_DIR
    from .model_testing import (
        SERIALIZATION_COLUMNS,
        clear_model_resources,
        load_model,
        run_dataset,
    )
except ImportError:  # Allow direct execution from models/tabllm.
    from setup_config import DATASETS, MODELS, RESULTS_DIR
    from model_testing import (
        SERIALIZATION_COLUMNS,
        clear_model_resources,
        load_model,
        run_dataset,
    )


def run_model_tests(
    models=None,
    datasets=None,
    serializations=None,
    batch=1,
    test=False,
):
    """
    Run all configured models on all configured datasets.

    Saves:
    - One raw-response CSV per dataset/model pair.
    - One metrics JSON per dataset/model pair.
    - One global CSV summary with all results.
    """
    if batch < 1:
        raise ValueError("--batch must be at least 1.")

    if models is None:
        models = list(MODELS.keys())

    if datasets is None:
        datasets = list(DATASETS.keys())

    if serializations is None:
        serializations = ["text_template"]

    if test:
        selected_models = list(MODELS.keys())
        selected_datasets = list(DATASETS.keys())
        selected_serializations = list(SERIALIZATION_COLUMNS.keys())
        max_examples = 3
        run_tag = "test"
        print("\nRunning smoke test: all models, all datasets, all serializations, first 3 rows.")
    else:
        selected_models = models
        selected_datasets = datasets
        selected_serializations = serializations
        max_examples = None
        run_tag = None

    dataset_serializations = {}

    if test:
        for dataset_name in selected_datasets:
            dataset_config = DATASETS[dataset_name]
            columns = pd.read_csv(dataset_config["test_path"], nrows=0).columns
            found_serializations = [
                serialization_style
                for serialization_style in selected_serializations
                if SERIALIZATION_COLUMNS[serialization_style] in columns
            ]

            missing_serializations = [
                serialization_style
                for serialization_style in selected_serializations
                if SERIALIZATION_COLUMNS[serialization_style] not in columns
            ]

            for serialization_style in missing_serializations:
                expected_column = SERIALIZATION_COLUMNS[serialization_style]
                print(
                    f"Skipping {dataset_name} / {serialization_style}: "
                    f"missing column '{expected_column}'."
                )

            dataset_serializations[dataset_name] = found_serializations
    else:
        dataset_serializations = {
            dataset_name: selected_serializations
            for dataset_name in selected_datasets
        }

    if not any(dataset_serializations.values()):
        print("No requested serialization columns were found. Nothing to run.")
        return pd.DataFrame()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary_path = Path(RESULTS_DIR) / (
        "all_results_test_summary.csv" if test else "all_results_summary.csv"
    )

    if summary_path.exists():
        all_results = pd.read_csv(summary_path)
    else:
        all_results = pd.DataFrame()

    for model_key in selected_models:
        model_name = MODELS[model_key]

        print(f"\nLoading model: {model_key} -> {model_name}")

        tokenizer, model = load_model(model_name)

        for dataset_name in selected_datasets:
            for serialization_style in dataset_serializations[dataset_name]:
                start_time = time.time()
                dataset_config = DATASETS[dataset_name]

                result = run_dataset(
                    dataset_name=dataset_name,
                    dataset_config=dataset_config,
                    model_key=model_key,
                    tokenizer=tokenizer,
                    model=model,
                    serialization_style=serialization_style,
                    max_examples=max_examples,
                    run_tag=run_tag,
                    batch_size=batch,
                )

                end_time = time.time()
                result["time_sec"] = end_time - start_time
                all_results = pd.concat([all_results, pd.DataFrame([result])], ignore_index=True)

                print(
                    f"\nFinished {dataset_name} / {serialization_style} with {model_key}. "
                    f"Saved raw responses to {result['output_file']}"
                )

        clear_model_resources(model, tokenizer)



    results_df = pd.DataFrame(all_results)

    results_df.to_csv(summary_path, index=False)

    print("\nAll results saved.")
    print(results_df)

    return results_df


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
        default=["text_template"],
        choices=list(SERIALIZATION_COLUMNS.keys()),
        help="Serialization styles to run."
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size for generation. Default is 1, meaning no batching.",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Smoke test all models, datasets, and serializations on the first 3 rows.",
    )

    args = parser.parse_args()

    run_model_tests(
        models=args.models,
        datasets=args.datasets,
        serializations=args.serializations,
        batch=args.batch,
        test=args.test,
    )


if __name__ == "__main__":
    main()
