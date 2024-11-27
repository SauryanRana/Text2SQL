from datasets import load_dataset
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
import os, re, torch, random, wandb
from functools import partial


import json

def load_json(file_path):
    """
    Load a JSON file and return the data.
    Args:
        file_path: Path to the JSON file.
    Returns:
        JSON data as a Python object.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def format_example(example):
    """
    Format the example to include input and target text.
    Args:
        example: A single data example.
    Returns:
        Formatted example with input_text and target_text.
    """
    # Check if 'table' key exists
    if 'table' not in example or 'header' not in example['table']:
        print(f"Skipping example due to missing 'table' or 'header': {example}")
        return None  # Return None for problematic entries

    schema = ", ".join(example['table']['header'])

    # Check if 'question' and 'sql' keys exist
    if 'question' not in example or 'sql' not in example or 'human_readable' not in example['sql']:
        print(f"Skipping example due to missing 'question' or 'sql': {example}")
        return None

    return {
        'input_text': f"translate natural language to SQL: {example['question']} | Schema: {schema}",
        'target_text': example['sql']['human_readable']
    }

from datasets import Dataset

def data_preparation(data_dir):
    """
    Load preprocessed JSON files, combine train and validation datasets,
    and format the data for model input/output.
    Args:
        data_dir: Directory containing the JSON files (train.json, val.json, test.json).
    Returns:
        Combined train and validation dataset, and test dataset as Hugging Face Dataset objects.
    """
    train_path = os.path.join(data_dir, "train.json")
    val_path = os.path.join(data_dir, "validation.json")
    test_path = os.path.join(data_dir, "test.json")

    # Load JSON files
    train_data = load_json(train_path)
    val_data = load_json(val_path)
    test_data = load_json(test_path)

    # Combine train and validation datasets
    combined_train_data = train_data + val_data


    # Format datasets
    formatted_train_data = [format_example(entry) for entry in combined_train_data if entry]
    formatted_test_data = [format_example(entry) for entry in test_data if entry]

    # Convert to Hugging Face Dataset objects
    train_dataset = Dataset.from_list(formatted_train_data)
    test_dataset = Dataset.from_list(formatted_test_data)

    return train_dataset, test_dataset




def load_and_prepare_tokenizer(model_name='t5-small', train_data=None):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    # Add special tokens from the schema headers in the training data
    if train_data is not None:
        special_tokens = set()
        for i, row in enumerate(train_data):
            try:
                # Extract schema from the input_text
                input_text = row['input_text']
                schema_start = input_text.find("Schema: ")
                if schema_start != -1:
                    schema_part = input_text[schema_start + len("Schema: "):]
                    schema_headers = [col.strip() for col in schema_part.split(",")]
                    special_tokens.update([f"<{col}>" for col in schema_headers])
                else:
                    print(f"Skipping entry at index {i} due to missing schema: {row}")
            except Exception as e:
                print(f"Error processing entry at index {i}: {e}")
        
        # Add special tokens to the tokenizer
        if special_tokens:
            tokenizer.add_special_tokens({"additional_special_tokens": list(special_tokens)})
        else:
            print("No special tokens were added from the training data.")
    
    return tokenizer




def tokenize_function(batch, tokenizer, max_input_length=512, max_target_length=128):
    """
    Tokenize input and target text in the batch.
    """
    inputs = tokenizer(
        batch['input_text'], max_length=max_input_length, padding='max_length', truncation=True
    )
    targets = tokenizer(
        batch['target_text'], max_length=max_target_length, padding='max_length', truncation=True
    )
    # Replace padding token id in labels with -100 for loss calculation
    targets['input_ids'] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in target]
        for target in targets['input_ids']
    ]

    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': targets['input_ids']
    }


def tokenize_data(dataset, tokenizer, max_input_length=512, max_target_length=128):
    tokenized_dataset = dataset.map(
        lambda batch: tokenize_function(batch, tokenizer, max_input_length, max_target_length),
        batched=True
    )
    return tokenized_dataset



### 3. Check for GPU
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

### 4. Model Training
from transformers import DataCollatorForSeq2Seq

def train_model(tokenized_train, tokenized_eval, tokenizer, output_dir='t5-wikisql-scratch'):
    wandb.init(project="wikisql-t5-scratch", name="T5-small-WikiSQL-scratch", sync_tensorboard=True)

    device = get_device()  # Check if GPU is available
    
    # Initialize T5 model from scratch
    config = T5Config.from_pretrained("t5-small")
    model = T5ForConditionalGeneration(config).to(device)

    # Resize token embeddings after adding special tokens to the tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # Define a data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        per_device_train_batch_size=8,  # Smaller batch size
        per_device_eval_batch_size=2,  # Smaller evaluation batch size
        gradient_accumulation_steps=4,  # Effective batch size = 4 * 4 = 16
        learning_rate=5e-5,
        weight_decay=0.01,
        save_total_limit=5,
        num_train_epochs=10,
        predict_with_generate=False,  # Disable generation during training
        logging_steps=500,  # Reduce logging frequency
        report_to="wandb"
    )

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer)
    )

    # Start training
    trainer.train()
    trainer.evaluate()
    wandb.finish()
    
    return trainer



### 5. Compute Metrics
def compute_metrics(pred, tokenizer):
    def parse_sql(query, index):
        """
        Parses the SQL query into canonical form, correctly identifying aggregate functions
        and handling special characters in conditions.
        """
        canonical_form = {
            "sel": None,
            "agg": 0,  # Default to 0 (no aggregation)
            "conds": {
                "column_index": [],
                "operator_index": [],
                "condition": []
            }
        }

        # Define mappings for aggregate functions
        agg_mapping = {
            "MIN": 1,
            "MAX": 2,
            "COUNT": 3,
            "SUM": 4,
            "AVG": 5
        }

        try:
            # Extract SELECT part
            select_match = re.search(r'SELECT\s+(.*?)(?:\s+FROM|$)', query, re.IGNORECASE)
            if select_match:
                sel_part = select_match.group(1).strip()
                # Check for aggregate functions
                agg_match = re.match(r'(MIN|MAX|COUNT|SUM|AVG)\s+(.*)', sel_part, re.IGNORECASE)
                if agg_match:
                    agg_func = agg_match.group(1).upper()  # Extract the aggregation function
                    canonical_form['agg'] = agg_mapping.get(agg_func, 0)  # Map it to its ID
                    canonical_form['sel'] = agg_match.group(2).strip()  # Extract the column name
                else:
                    canonical_form['sel'] = sel_part  # No aggregation, use as-is

            # Extract WHERE part
            where_match = re.search(r'WHERE\s+(.*)', query, re.IGNORECASE)
            if where_match:
                conditions = where_match.group(1).strip().split('AND')
                for cond in conditions:
                    try:
                        # Match column, operator, and value dynamically
                        match = re.match(r'^(.*?)\s*([=<>]+)\s*(.*)$', cond.strip())
                        if match:
                            col = match.group(1).strip()
                            op = match.group(2).strip()
                            value = match.group(3).strip()

                            # Retain full condition values, including parentheses, percentages, etc.
                            canonical_form['conds']['column_index'].append(col)
                            canonical_form['conds']['operator_index'].append(op)
                            canonical_form['conds']['condition'].append(value)
                        else:
                            continue  # Skip invalid condition
                    except Exception:
                        continue  # Skip on exception
        except Exception:
            pass  # Silently handle outer parsing errors

        return canonical_form

    # Decode predictions and labels
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, batch_size=32)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True, batch_size=32)

    sel_correct = 0
    agg_correct = 0
    conds_correct = 0
    total_correct = 0
    total = 0

    for idx, (pred_query, ref_query) in enumerate(zip(pred_str, label_str)):

        if idx % 100 == 0:
            print(f"Processing index {idx}/{len(pred_str)}")
    

        if not pred_query or not ref_query:
            continue  # Skip empty predictions or labels

        pred_canonical = parse_sql(pred_query, idx)
        ref_canonical = parse_sql(ref_query, idx)

        if not pred_canonical or not ref_canonical:  # Skip if parsing failed
            continue

        # Check selection column accuracy
        if pred_canonical['sel'] == ref_canonical['sel']:
            sel_correct += 1

        # Check aggregation accuracy
        if pred_canonical['agg'] == ref_canonical['agg']:
            agg_correct += 1

        # Check condition accuracy
        if (
            pred_canonical['conds']['column_index'] == ref_canonical['conds']['column_index'] and
            pred_canonical['conds']['operator_index'] == ref_canonical['conds']['operator_index'] and
            pred_canonical['conds']['condition'] == ref_canonical['conds']['condition']
        ):
            conds_correct += 1

        # Check overall accuracy
        if pred_canonical == ref_canonical:
            total_correct += 1
        total += 1

    overall_accuracy = total_correct / total if total > 0 else 0
    sel_accuracy = sel_correct / total if total > 0 else 0
    agg_accuracy = agg_correct / total if total > 0 else 0
    conds_accuracy = conds_correct / total if total > 0 else 0

     # Log metrics to wandb
    metrics = {
        "overall_accuracy": round(overall_accuracy, 4),
        "sel_accuracy": round(sel_accuracy, 4),
        "agg_accuracy": round(agg_accuracy, 4),
        "conds_accuracy": round(conds_accuracy, 4),
    }
    wandb.log(metrics)

    return metrics


### Test Metrics
def test_compute_metrics():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    
    # Mock predictions and labels
    mock_predictions = [
        "SELECT Notes FROM table WHERE Current slogan = 'SOUTH AUSTRALIA'",
        "SELECT MIN Mass FROM table WHERE Radius = 10"
    ]
    
    mock_labels = [
        "SELECT Notes FROM table WHERE Current slogan = 'SOUTH AUSTRALIA'",
        "SELECT MIN Mass FROM table WHERE Radius = 10"
    ]
    
    # Simulate predictions and label IDs
    mock_pred_ids = tokenizer(mock_predictions, padding=True, truncation=True, return_tensors="pt").input_ids
    mock_label_ids = tokenizer(mock_labels, padding=True, truncation=True, return_tensors="pt").input_ids

    # Pack them into the mock pred object
    class MockPred:
        predictions = mock_pred_ids
        label_ids = mock_label_ids

    pred = MockPred()

    # Call the metric computation
    metrics = compute_metrics(pred, tokenizer)

    print(f"Metrics: {metrics}")




### 6. Save Model
def save_model(trainer, tokenizer, output_dir='t5-wikisql-scratch'):
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Create a simple model card
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write("# T5-small (Scratch) Fine-tuned on WikiSQL\n\nThis model generates SQL queries from natural language using the WikiSQL dataset.")


### 7. Test Model
import pandas as pd

def test_model(test_data, tokenizer, model_dir='t5-wikisql-scratch', output_file="test_results.csv"):
    """
    Test the model, print results, and store them in a CSV file.
    """
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = T5Tokenizer.from_pretrained(model_dir)  # Ensure same tokenizer is loaded with special tokens

    results = []
    random_samples = random.sample(list(test_data), 10)  # Choose 10 random examples

    for sample in random_samples:
        input_text = sample['input_text']
        actual_sql = sample['target_text']

        inputs = tokenizer(
            input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512
        )

        outputs = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        predicted_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Store in results list
        results.append({
            "Input": input_text,
            "Actual SQL": actual_sql,
            "Predicted SQL": predicted_sql
        })

        print(f"Input: {input_text}")
        print(f"Actual SQL: {actual_sql}")
        print(f"Predicted SQL: {predicted_sql}")
        print("-" * 10)

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")



### 8. Main Execution
def main():

    data_directory = "preprocessed_wikisql"  # Directory containing train.json, val.json, and test.json

    # Load and prepare data
    train_data, test_data = data_preparation(data_directory)

    # Prepare tokenizer with special tokens from the original train dataset
    tokenizer = load_and_prepare_tokenizer('t5-small', train_data)
    
    # Tokenize data
    tokenized_train = tokenize_data(train_data, tokenizer)
    tokenized_test = tokenize_data(test_data, tokenizer)
    
    # Train the model
    trainer = train_model(tokenized_train, tokenized_test, tokenizer)
    
    # Save model and tokenizer
    save_model(trainer, tokenizer, 't5-wikisql-scratch')

    test_model(test_data, tokenizer, 't5-wikisql-scratch')



if __name__ == "__main__":
    main()
