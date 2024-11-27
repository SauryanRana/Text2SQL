import os
import json

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

def read_json(file_path):
    """
    Read JSON data from a file.
    Args:
        file_path: Path to the JSON file.
    Returns:
        JSON data as a Python object.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(data, file_path):
    """
    Write JSON data to a file.
    Args:
        data: Python object to save as JSON.
        file_path: Path to the JSON file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def preprocess_and_verify(input_dir, output_dir):
    """
    Read JSON files, verify SQL queries using parse_sql, remove problematic entries,
    and add canonical_form for successfully parsed entries.
    Args:
        input_dir: Directory containing the input JSON files.
        output_dir: Directory to save the processed JSON files.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".json"):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            print(f"Processing file: {file_name}")
            data = read_json(input_path)
            processed_data = []
            not_parsed_data = []

            for idx, entry in enumerate(data):
                sql_query = entry["sql"]['human_readable']

                # Attempt to parse the SQL query
                try:
                    parsed_result = parse_sql(sql_query, idx)

                    # If parsing succeeds, add the canonical form
                    entry['canonical_form'] = parsed_result
                    processed_data.append(entry)
                except Exception as e:
                    # If parsing fails, log the entry as unparseable
                    print(f"Error parsing SQL at index {idx} in file {file_name}: {e}")
                    not_parsed_data.append(entry)

            # Save processed data back to the original file
            write_json(processed_data, input_path)
            print(f"Updated {len(processed_data)} entries in {file_name}.")

            # Save unparseable entries to the output directory
            if not_parsed_data:
                print(f"Found {len(not_parsed_data)} unparseable entries in {file_name}.")
                write_json(not_parsed_data, output_path)
            else:
                print(f"All entries parsed successfully in {file_name}.")

            print(f"Processed file saved to: {output_path}")

if __name__ == "__main__":
    input_directory = "preprocessed_wikisql"  # Directory containing input JSON files
    output_directory = "not_proper_wikisql"  # Directory to save unparseable JSON files
    preprocess_and_verify(input_directory, output_directory)
