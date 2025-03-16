import argparse
import random
import re
import wandb
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer


class Evaluator:
    def __init__(self, model_dir, tokenizer, test_data, wandb_project="sql-model-evaluation"):
        self.model_dir = model_dir
        self.tokenizer = tokenizer
        self.test_data = test_data
        self.wandb_project = wandb_project

    def evaluate(self, output_file="test_results.csv"):
        """Evaluate model on test data."""
        wandb.init(project=self.wandb_project, name=output_file)
        
        model = T5ForConditionalGeneration.from_pretrained(self.model_dir)
        results = []
        
        # Sample random examples for evaluation
        random_samples = random.sample(list(self.test_data), 50)
        
        for sample in random_samples:
            input_text = sample['input_text']
            actual_sql = sample['target_text']
            
            # Generate prediction
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=180
            )
            
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=180
            )
            
            predicted_sql = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            results.append({
                "Input": input_text,
                "Actual SQL": actual_sql,
                "Predicted SQL": predicted_sql
            })
        
        # Save results
        df = pd.DataFrame(results)
        csv_path = f"{self.model_dir}/{output_file}"
        df.to_csv(csv_path, index=False)
        
        # Log to wandb
        wandb_table = wandb.Table(dataframe=df)
        wandb.log({"test_results_table": wandb_table})
        wandb.save(csv_path)
        wandb.finish()
        
        return results


def parse_sql(query):
    """Parse SQL query into canonical form for evaluation."""
    canonical_form = {
        "sel": None,
        "agg": 0,
        "conds": {
            "column_index": [],
            "operator_index": [],
            "condition": []
        }
    }

    agg_mapping = {
        "MIN": 1, "MAX": 2, "COUNT": 3,
        "SUM": 4, "AVG": 5
    }

    try:
        # Extract SELECT part
        select_match = re.search(r'SELECT\s+(.*?)(?:\s+FROM|$)', query, re.IGNORECASE)
        if select_match:
            sel_part = select_match.group(1).strip()
            agg_match = re.match(r'(MIN|MAX|COUNT|SUM|AVG)\s+(.*)', sel_part, re.IGNORECASE)
            if agg_match:
                canonical_form['agg'] = agg_mapping.get(agg_match.group(1).upper(), 0)
                canonical_form['sel'] = agg_match.group(2).strip()
            else:
                canonical_form['sel'] = sel_part

        # Extract WHERE part
        where_match = re.search(r'WHERE\s+(.*)', query, re.IGNORECASE)
        if where_match:
            conditions = where_match.group(1).strip().split('AND')
            for cond in conditions:
                match = re.match(r'^(.*?)\s*([=<>]+)\s*(.*)$', cond.strip())
                if match:
                    canonical_form['conds']['column_index'].append(match.group(1).strip())
                    canonical_form['conds']['operator_index'].append(match.group(2).strip())
                    canonical_form['conds']['condition'].append(match.group(3).strip())

    except Exception:
        pass

    return canonical_form

def compute_metrics(pred, tokenizer):
    """Compute evaluation metrics."""
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # Sample subset for evaluation
    max_samples = 20
    indices = random.sample(range(len(pred_ids)), max_samples)
    pred_ids = pred_ids[indices]
    labels_ids = labels_ids[indices]

    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, batch_size=32)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True, batch_size=32)

    print('label',label_str[0])

    print('pred',pred_str[0])

    # Initialize counters
    metrics_count = {
        'sel_correct': 0, 'agg_correct': 0,
        'column_index': 0, 'operator_index': 0,
        'condition': 0, 'total_correct': 0, 'total': 0
    }

    # Evaluate each prediction
    for pred_query, ref_query in zip(pred_str, label_str):
        if not pred_query or not ref_query:
            continue

        pred_canonical = parse_sql(pred_query)
        ref_canonical = parse_sql(ref_query)

        if not pred_canonical or not ref_canonical:
            continue

        # Update counters
        metrics_count['total'] += 1
        if pred_canonical['sel'] == ref_canonical['sel']:
            metrics_count['sel_correct'] += 1
        if pred_canonical['agg'] == ref_canonical['agg']:
            metrics_count['agg_correct'] += 1
        if pred_canonical['conds']['column_index'] == ref_canonical['conds']['column_index']:
            metrics_count['column_index'] += 1
        if pred_canonical['conds']['operator_index'] == ref_canonical['conds']['operator_index']:
            metrics_count['operator_index'] += 1
        if pred_canonical['conds']['condition'] == ref_canonical['conds']['condition']:
            metrics_count['condition'] += 1
        if pred_canonical == ref_canonical:
            metrics_count['total_correct'] += 1

    # Calculate metrics
    total = metrics_count['total']
    metrics = {
        "overall_accuracy": metrics_count['total_correct'] / total if total > 0 else 0,
        "sel_accuracy": metrics_count['sel_correct'] / total if total > 0 else 0,
        "agg_accuracy": metrics_count['agg_correct'] / total if total > 0 else 0,
        "column_index_accuracy": metrics_count['column_index'] / total if total > 0 else 0,
        "operator_index_accuracy": metrics_count['operator_index'] / total if total > 0 else 0,
        "condition_accuracy": metrics_count['condition'] / total if total > 0 else 0,
    }

    wandb.log(metrics)
    return metrics


if __name__ == "__main__":
    evaluator = Evaluator()
    pass
