from datasets import load_dataset, Dataset
import json
import os
import random

def load_json(file_path):
    """Load a JSON file and return the data."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def format_example(example):
    """Format the example for model input/output."""
    if 'table' not in example or 'header' not in example['table']:
        print(f"Skipping example due to missing 'table' or 'header': {example}")
        return None

    schema = ", ".join(example['table']['header'])

    if 'question' not in example or 'sql' not in example or 'human_readable' not in example['sql']:
        print(f"Skipping example due to missing 'question' or 'sql': {example}")
        return None

    return {
        'input_text': f"translate natural language to SQL: {example['question']} | Schema: {schema}",
        'target_text': example['sql']['human_readable']
    }

class Text2SQLDataset:
    def __init__(self, data_dir, phase=None, fraction=0.3):
        self.data_dir = data_dir
        self.phase = phase
        self.fraction = fraction

    def load_data(self):
        """Load and prepare the dataset."""
        # Define file paths based on phase
        if self.phase:
            train_path = os.path.join(self.data_dir, f"train_{self.phase}.json")
            val_path = os.path.join(self.data_dir, f"validation_{self.phase}.json")
            test_path = os.path.join(self.data_dir, f"test_{self.phase}.json")
        else:
            train_path = os.path.join(self.data_dir, "train.json")
            val_path = os.path.join(self.data_dir, "validation.json")
            test_path = os.path.join(self.data_dir, "test.json")

        # Load JSON files
        train_data = load_json(train_path)
        val_data = load_json(val_path)
        test_data = load_json(test_path)

        # Sample data if fraction < 1.0
        if self.fraction < 1.0:
            train_data = random.sample(train_data, int(len(train_data) * self.fraction))
            test_data = random.sample(test_data, int(len(test_data) * self.fraction))
            val_data = random.sample(val_data, int(len(val_data) * self.fraction))

        # Format datasets
        formatted_train = [format_example(entry) for entry in train_data if entry]
        formatted_test = [format_example(entry) for entry in test_data if entry]
        formatted_val = [format_example(entry) for entry in val_data if entry]

        # Convert to HuggingFace datasets
        return (
            Dataset.from_list(formatted_train),
            Dataset.from_list(formatted_test),
            Dataset.from_list(formatted_val)
        )
