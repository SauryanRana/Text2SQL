import json

def classify_sql_complexity(entry):
    """
    Classifies SQL complexity based on defined criteria.
    """
    agg = entry['sql']['agg']
    conditions = entry['sql']['conds']['condition']
    num_conditions = len(conditions)

    if  num_conditions > 1:
        return "Moderate"
    else:
        return "Simple"

def divide_dataset_by_complexity(file_path):
    """
    Divides dataset into simple and moderate subsets.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    simple_queries = []
    moderate_queries = []

    for entry in data:
        complexity = classify_sql_complexity(entry)
        if complexity == "Simple":
            simple_queries.append(entry)
        elif complexity == "Moderate":
            moderate_queries.append(entry)

    return simple_queries, moderate_queries

def process_datasets(dataset_files):
    """
    Processes multiple dataset files and divides them into simple and moderate subsets.
    """
    for file_name in dataset_files:
        print(f"Processing {file_name}...")
        simple, moderate = divide_dataset_by_complexity(file_name)
        
        # Generate output file names based on input file name
        simple_output = file_name.replace(".json", "_simple.json")
        moderate_output = file_name.replace(".json", "_moderate.json")
        
        # Save results to separate files
        with open(simple_output, "w") as f:
            json.dump(simple, f, indent=4)
        with open(moderate_output, "w") as f:
            json.dump(moderate, f, indent=4)
        
        print(f"Saved {simple_output} and {moderate_output}")

# List of dataset files to process
dataset_files = ["train.json", "validation.json", "test.json"]
process_datasets(dataset_files)
