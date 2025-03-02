from transformers import AutoModelForSeq2SeqLM, T5Tokenizer, PreTrainedTokenizerFast, convert_slow_tokenizer
from datasets import load_dataset
from utils import parse_sql_to_canonical, tokenize, encode_rare_chars
import torch
import json
from lib.dbengine import DBEngine
from lib.query import Query
import os

# Load model and tokenizer
checkpoint_path = "models/4_heads_2e-4_lr_constant_512MappingTokenizer_128_bs_64_dff_32_kv_128d_1"
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
tokenizer = PreTrainedTokenizerFast(tokenizer_object=convert_slow_tokenizer.convert_slow_tokenizer(T5Tokenizer("tokenizers/sp_512_bpe_encoded.model", legacy=False, load_from_cache_file=False)))
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load dataset
path = 'datasets/wikisql'
dataset = load_dataset(path+'/data')

# Load mappings
with open('mapping.json', 'r', encoding='utf-8') as mapping_file:
    mapping = json.load(mapping_file)

with open('reverse_mapping.json', 'r', encoding='utf-8') as reverse_mapping_file:
    reverse_mapping = json.load(reverse_mapping_file)

import random

def get_random_sample():    

    index = random.randint(1, 1000)  # Choose a random number between 1 and 1000

    sample = dataset["test"].select(range(index, index+1))
    return {
        "question": sample["question"][0],
        "table": sample["table"][0],
        "table_id": sample["table"][0]["id"]
    }

def preprocess_function(sample):
    """Preprocess a single sample for inference"""
    # Create empty lists for batch processing
    sample["input_text"] = []
    sample["label_text"] = []
    
    # Get the table headers - table is a dictionary with 'header' key
    table_headers_text = "[SEP]".join(sample["table"][0]["header"])
    
    # Clean the question text (remove special unicode spaces)
    cleaned_question = sample["question"][0].replace("\xa0", " ")
    
    # Combine question and headers to create input text
    full_input = f"{cleaned_question}[SEP]{table_headers_text}"
    
    sample["input_text"].append(full_input)
    sample["label_text"].append("")  # Empty string for inference
    
    return sample

def preprocess_sample(question, table, sql=None):
    # Create the input format expected by the model
    sample = {
        "question": [question], 
        "table": [table],  # table is already in the correct format
        "sql": [sql] if sql else None
    }
    
    # Apply preprocessing directly
    preprocessed_sample = preprocess_function(sample)
    encoded_preprocessed_sample = encode_rare_chars(preprocessed_sample, mapping)
    tokenized_sample = tokenize(encoded_preprocessed_sample, tokenizer)
    
    return tokenized_sample["input_ids"]

def inference(input_ids):
    model.eval()
    with torch.no_grad():
        outputs = model.generate(input_ids=torch.tensor(input_ids), num_beams=10, max_length=128)
    output = tokenizer.decode(token_ids=outputs[0][1:], skip_special_tokens=True)
    return output

def post_processing(query, table_header):
    cleaned_canonical = parse_sql_to_canonical(query, table_header, reverse_mapping)
    
    # If cleaned_canonical is a dictionary with sets, convert them to lists
    if isinstance(cleaned_canonical, dict):
        if 'conds' in cleaned_canonical and isinstance(cleaned_canonical['conds'], set):
            cleaned_canonical['conds'] = list(cleaned_canonical['conds'])
    
    return cleaned_canonical, None

def execute_query(table_id, query):
    db_path = os.path.abspath(path+'/tables/test/test.db')
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")
    db_engine = DBEngine(db_path)
    
    # Ensure query is a dictionary
    if isinstance(query, str):
        try:
            query = json.loads(query)
        except json.JSONDecodeError:
            raise ValueError("Invalid query format")
    
    # Convert query conditions from list back to set if necessary
    if isinstance(query, dict) and 'conds' in query:
        if isinstance(query['conds'], list):
            query['conds'] = set(tuple(cond) if isinstance(cond, list) else cond 
                               for cond in query['conds'])
    
    try:
        query_obj = Query.from_dict(query)
        result = db_engine.execute_query(table_id, query_obj)
    except Exception as e:
        result = f"Execution error: {str(e)}"
    
    return result