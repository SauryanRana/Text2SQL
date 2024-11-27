from typing import Dict, List
import sqlparse
from lib.dbengine import DBEngine
from lib.query import Query


def preprocess_function(batch: Dict[str, List], tokenizer) -> Dict[str, List]:
    inputs = []
    targets = []

    for question, table, sql in zip(batch["question"], batch["table"], batch["sql"]):
        table_headers_text = "|".join(table["header"])
        target_sql = sql["human_readable"]  # Use the human-readable SQL field

        # Combine question and headers to create input text
        full_input = f"{question}|{table_headers_text}"
        inputs.append(full_input)
        targets.append(target_sql)

    model_inputs = tokenizer(inputs, max_length=64, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=64, truncation=True, padding="max_length")["input_ids"]
    model_inputs["labels"] = labels

    return model_inputs


# this function transforms the string output of the model into the canonical representation wikiSQL uses
# It allows more detailed analytics of which part of a query our model struggles with
# Query.from_sequence does something similar but it uses the tokenized output of the model which would require us
# to use a specific tokenizer (Stanza I think? Which is deprecated now but still runnable in a docker container?) so
# that each SQL keyword is represented by its specific token. Maybe in the future.
def parse_sql_to_canonical(sql_query, table_headers):
    parsed = sqlparse.parse(sql_query)[0]
    tokens = [token for token in parsed.tokens if not token.is_whitespace]
    canonical_form = {
        "sel": None,
        "agg": 0,
        "conds": {
            "column_index": [],
            "operator_index": [],
            "condition": []
        }
    }

    if tokens[0].ttype is sqlparse.tokens.DML and tokens[0].value.upper() == "SELECT":
        select_clause = tokens[1]
        selected_column = select_clause.get_real_name()
        canonical_form["sel"] = table_headers.index(selected_column)

        if select_clause.get_name() != selected_column:
            aggregation_function = select_clause.get_name().upper()
            canonical_form["agg"] = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"].index(aggregation_function)

    if "WHERE" in sql_query.upper():
        where_index = [i for i, t in enumerate(tokens) if t.value.upper() == "WHERE"][0]
        where_clause = tokens[where_index + 1]
        condition_column = where_clause.left.get_real_name()
        condition_operator = where_clause.token_next_match(1, sqlparse.tokens.Operator)
        condition_value = where_clause.right.get_real_name().strip("'")

        canonical_form["conds"]["column_index"].append(table_headers.index(condition_column))
        canonical_form["conds"]["operator_index"].append(["=", ">", "<", ">=", "<=", "!="].index(condition_operator.value))
        canonical_form["conds"]["condition"].append(condition_value)

    return canonical_form


def sel_accuracy(predictions, labels):
    return sum(1 for pred, label in zip(predictions, labels) if pred["sel"] == label["sel"]) / len(predictions)


def agg_accuracy(predictions, labels):
    return sum(1 for pred, label in zip(predictions, labels) if pred["agg"] == label["agg"]) / len(predictions)


def conds_accuracy(predictions, labels):
    correct = 0
    for pred, label in zip(predictions, labels):
        if (
            pred["conds"]["column_index"] == label["conds"]["column_index"]
            and pred["conds"]["operator_index"] == label["conds"]["operator_index"]
            and pred["conds"]["condition"] == label["conds"]["condition"]
        ):
            correct += 1
    return correct / len(predictions)


def compute_execution_accuracy(predictions_text, labels_text, val_dataset, db_engine):
    execution_correct = 0
    for pred_sql, label_sql, example in zip(predictions_text, labels_text, val_dataset):
        table_id = example["table_id"]
        try:
            pred_result = db_engine.execute_query(table_id, Query.from_dict(
                parse_sql_to_canonical(pred_sql, example["table"]["header"])))
            gold_result = db_engine.execute_query(table_id, Query.from_dict(
                parse_sql_to_canonical(label_sql, example["table"]["header"])))
            if pred_result == gold_result:
                execution_correct += 1
        except Exception as e:
            # Handle cases where the SQL is not executable
            print(f"Execution error: {e}")

    return execution_correct / len(predictions_text)


def create_metrics_computer(val_dataset, tokenizer, db_path):
    """
    Creates a compute_metrics function with preloaded resources.
    """
    # Preload the database engine
    db_engine = DBEngine(db_path)

    val_data_by_text = {example["sql"]["human_readable"]: example for example in val_dataset}

    def compute_metrics(eval_preds):
        """
        Metrics computation function that reuses preloaded resources.
        """
        predictions, labels = eval_preds

        # Decode predictions and labels
        predictions_text = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
        labels_text = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
        labels_data = [example[labels_text] for example in val_data_by_text]

        # Parse predictions to canonical form
        predictions_canonical = [
            parse_sql_to_canonical(pred_text, example["table"]["header"])
            for pred_text, example in zip(predictions_text, labels_data)
        ]

        # Calculate detailed metrics
        labels_canonical = [label["sql"] for label in labels_data]
        sel_acc = sel_accuracy(predictions_canonical, labels_canonical)
        agg_acc = agg_accuracy(predictions_canonical, labels_canonical)
        conds_acc = conds_accuracy(predictions_canonical, labels_canonical)
        overall_accuracy = sum(
            1 for pred, label in zip(predictions_canonical, labels_canonical) if pred == label) / len(predictions)

        # Calculate execution accuracy using the preloaded db_engine
        exec_acc = compute_execution_accuracy(predictions_text, labels_text, val_dataset, db_engine)

        return {
            "overall_accuracy": overall_accuracy,
            "sel_accuracy": sel_acc,
            "agg_accuracy": agg_acc,
            "conds_accuracy": conds_acc,
            "execution_accuracy": exec_acc,
        }

    return compute_metrics
