from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SERIALIZED_DATA_DIR = REPOSITORY_ROOT / "data" / "serialized"
CONFIG_DIR = Path(__file__).resolve().parent


MODELS = {
    "llama3": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "meditron": "epfl-llm/meditron-7b",
    "biomistral": "BioMistral/BioMistral-7B",
}


DATASETS = {
    "hepatitis": {
        "train_path": SERIALIZED_DATA_DIR / "hepatitis_train_serialized.csv",
        "test_path": SERIALIZED_DATA_DIR / "hepatitis_test_serialized.csv",
        "config_path": CONFIG_DIR / "hepatitis_config.json",
        "task": "Predict the hepatitis outcome.",
        "choices": ["live", "die"],
        "positive_label": "die",
    },
    "heart": {
        "train_path": SERIALIZED_DATA_DIR / "heart_train_serialized.csv",
        "test_path": SERIALIZED_DATA_DIR / "heart_test_serialized.csv",
        "config_path": CONFIG_DIR / "heart_config.json",
        "task": "Predict whether the patient has heart disease.",
        "choices": ["no heart disease", "heart disease"],
        "positive_label": "heart disease",
    },
    "diabetes": {
        "train_path": SERIALIZED_DATA_DIR / "diabetes_train_serialized.csv",
        "test_path": SERIALIZED_DATA_DIR / "diabetes_test_serialized.csv",
        "config_path": CONFIG_DIR / "diabetes_config.json",
        "task": "Predict whether the patient will have a hypoglycemic event tomorrow.",
        "choices": [
            "no hypoglycemic event tomorrow",
            "hypoglycemic event tomorrow",
        ],
        "positive_label": "hypoglycemic event tomorrow",
    },
}


RESULTS_DIR = REPOSITORY_ROOT / "results" / "tabllm"
