import os
import re
from random import randint
from typing import Dict, List

import numpy as np
import Levenshtein

from lib.dbengine import DBEngine
from lib.query import Query


def lcs_length(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def lcs_similarity(s1, s2):
    lcs_len = lcs_length(s1, s2)
    return lcs_len / (len(s1) + len(s2) - lcs_len)  # Normalization 2


# Find the index of the closest string in the list using Levenshtein distance
def closest_string(search_str, str_list):
    # distances = [Levenshtein.distance(search_str, s) for s in str_list]
    distances = [1 - lcs_similarity(search_str, s) for s in str_list]
    closest_index = min(range(len(distances)), key=distances.__getitem__)
    return closest_index


def filter_function(sample, tokenizer, input_max_length=96, output_max_length=48):
    # Tokenize without truncation
    full_input = f"{sample['question']}|{'|'.join(sample['table']['header'])}"
    target_sql = sample['sql']['human_readable']

    tokenized_input = tokenizer(full_input, truncation=False)["input_ids"]
    tokenized_target = tokenizer(target_sql, truncation=False)["input_ids"]

    # Keep samples that meet length requirements
    return len(tokenized_input) <= input_max_length and len(tokenized_target) <= output_max_length


def preprocess_function(batch):
    batch["input_text"] = []
    batch["label_text"] = []

    for question, table, sql in zip(batch["question"], batch["table"], batch["sql"]):
        table_headers_text = "[SEP]".join(table["header"])
        # Combine question and headers to create input text
        full_input = f"{question}[SEP]{table_headers_text}"

        batch["input_text"].append(full_input)
        batch["label_text"].append(sql["human_readable"])

    return batch


def encode_rare_chars(batch, mapping):
    """
    Replace characters in the text field with their respective mappings.

    :param batch: A batch of examples with a 'input_text' and 'label_text' field.
    :param mapping: A mapping from rare characters to a sequence of special token and index.
    :return: Processed examples.
    """
    batch['input_text'] = [''.join(mapping.get(char, char) for char in text) for text in batch['input_text']]
    batch['label_text'] = [''.join(mapping.get(char, char) for char in text) for text in batch['label_text']]
    return batch


def tokenize(batch, tokenizer, input_max_length=128, output_max_length=64, padding="max_length"):
    inputs = tokenizer(batch["input_text"], max_length=input_max_length, truncation=True, padding=padding)
    labels = tokenizer(batch["label_text"], max_length=output_max_length, truncation=True, padding=padding)["input_ids"]

    inputs["labels"] = labels

    return inputs


def decode_text(encoded_text, reverse_mapping):
    """
    Decode a string by replacing [MAP]x[/MAP] instances with the corresponding characters.

    :param encoded_text: The text to decode.
    :param reverse_mapping: The reverse mapping dictionary.
    :return: Decoded text.
    """
    def replace_match(match):
        token = match.group(0)
        return reverse_mapping.get(token, token)

    return re.sub(r"\[MAP\]\d+\[/MAP\]", replace_match, encoded_text)


# this function transforms the string output of the model into the canonical representation wikiSQL uses
# It allows more detailed analytics of which part of a query our model struggles with
# Query.from_sequence does something similar but it uses the tokenized output of the model which would require us
# to use a specific tokenizer (Stanza I think? Which is deprecated now but still runnable in a docker container?) so
# that each SQL keyword is represented by its specific token. Maybe in the future.
def parse_sql_to_canonical(query, table_header, mapping=None):
    # Initialize the canonical form
    canonical_form = {
        "sel": None,
        "agg": 0,  # Default to 0 (no aggregation)
        "conds": set()
    }

    # Define mappings for aggregate functions
    agg_mapping = {
        "": 0,
        "MAX": 1,
        "MIN": 2,
        "COUNT": 3,
        "SUM": 4,
        "AVG": 5
    }

    cond_mapping = {'=': 0, '>': 1, '<': 2}

    header_mapping = {column_name: i for i, column_name in enumerate(table_header)}

    if mapping:
        query = decode_text(query, mapping)

    # Extract SELECT part
    select_match = re.search(r'SELECT\s+(.*?)(?:\s+FROM|$)', query, re.IGNORECASE)
    if select_match:
        sel_part = select_match.group(1).strip()
        # Check for aggregate functions
        agg_match = re.match(r'(MIN|MAX|COUNT|SUM|AVG)\s+(.*)', sel_part, re.IGNORECASE)
        if agg_match:
            agg_func = agg_match.group(1).upper()  # Extract the aggregation function
            canonical_form['agg'] = agg_mapping[agg_func]  # Map it to its ID
            canonical_form['sel'] = closest_string(agg_match.group(2).strip(), table_header) #  header_mapping.get(agg_match.group(2).strip(), randint(0, len(header_mapping)-1))  # Extract the column name
        else:
            canonical_form['sel'] = closest_string(sel_part, table_header)  # header_mapping.get(sel_part, randint(0, len(header_mapping)-1))  # No aggregation, use as-is

    # Extract WHERE part
    where_match = re.search(r'WHERE\s+(.*)', query, re.IGNORECASE)
    if where_match:
        conditions = where_match.group(1).strip().split('AND')
        for cond in conditions:
            # Match column, operator, and value dynamically
            match = re.match(r'^(.*?)\s*([=<>]+)\s*(.*)$', cond.strip())
            if match:
                col = closest_string(match.group(1).strip(), table_header)  # header_mapping.get(match.group(1).strip(), randint(0, len(header_mapping)-1))
                op = cond_mapping.get(match.group(2).strip())
                value = match.group(3).strip()

                canonical_form['conds'].add((col, op, value))  # append((col, op, value))
            else:
                continue  # Skip invalid condition

    return canonical_form


def sel_accuracy(predictions, labels):
    return sum(1 for pred, label in zip(predictions, labels) if pred["sel"] == label["sel"]) / len(predictions)


def agg_accuracy(predictions, labels):
    return sum(1 for pred, label in zip(predictions, labels) if pred["agg"] == label["agg"]) / len(predictions)


def conds_accuracy(predictions, labels):
    return sum(1 for pred, label in zip(predictions, labels) if pred["conds"] == label["conds"]) / len(predictions)


def lf_accuracy(predictions, labels):
    return sum(1 for pred, label in zip(predictions, labels) if pred == label) / len(predictions)


def compute_execution_accuracy(predictions_sql, labels_sql, val_dataset, db_engine):
    execution_correct = 0
    for pred_sql, label_sql, example in zip(predictions_sql, labels_sql, val_dataset):
        table_id = example["table"]["id"]
        try:
            pred_result = db_engine.execute_query(table_id, Query.from_dict(pred_sql))
            gold_result = db_engine.execute_query(table_id, Query.from_dict(label_sql))
            if pred_result == gold_result:
                execution_correct += 1
        except Exception as e:
            # Handle cases where the SQL is not executable
            # print(f"Execution error: {e}")
            pass

    return execution_correct / len(predictions_sql)


def create_metrics_computer(val_dataset, tokenizer, db_path, mapping=None):
    """
    Creates a compute_metrics function with preloaded resources.
    """
    # Preload the database engine
    # Resolve absolute database path
    db_path = os.path.abspath(db_path)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")
    db_engine = DBEngine(db_path)

    val_data_by_text = {hash(tuple(label)): {"sql": sql, "table": table} for label, sql, table in zip(val_dataset["labels"], val_dataset["sql"], val_dataset["table"])}

    def compute_metrics(eval_preds):
        """
        Metrics computation function that reuses preloaded resources.
        """
        predictions, labels = eval_preds
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)

        # Decode predictions and labels
        predictions_text = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels_text = tokenizer.batch_decode(labels, skip_special_tokens=True)
        labels_data = [val_data_by_text[hash(tuple(label))] for label in labels]

        # Parse predictions to canonical form
        predictions_canonical = [
            parse_sql_to_canonical(pred_text, example["table"]["header"], mapping)
            for pred_text, example in zip(predictions_text, labels_data)
        ]

        # Calculate detailed metrics
        labels_canonical = [
            parse_sql_to_canonical(label_text, example["table"]["header"], mapping)
            for label_text, example in zip(labels_text, labels_data)
        ]
        sel_acc = sel_accuracy(predictions_canonical, labels_canonical)
        agg_acc = agg_accuracy(predictions_canonical, labels_canonical)
        conds_acc = conds_accuracy(predictions_canonical, labels_canonical)
        lf_acc = lf_accuracy(predictions_canonical, labels_canonical)

        # Calculate execution accuracy using the preloaded db_engine
        exec_acc = compute_execution_accuracy(predictions_canonical, labels_canonical, labels_data, db_engine)

        return {
            "overall_accuracy": lf_acc,
            "sel_accuracy": sel_acc,
            "agg_accuracy": agg_acc,
            "conds_accuracy": conds_acc,
            "execution_accuracy": exec_acc,
        }

    return compute_metrics
