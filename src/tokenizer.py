from transformers import T5Tokenizer
import os

class SQLTokenizer:
    def __init__(self, model_name='t5-small', tokenizer_dir='saved_tk'):
        self.model_name = model_name
        self.tokenizer_dir = tokenizer_dir
        
    def load_tokenizer(self):
        """Load or create tokenizer."""
        if os.path.exists(self.tokenizer_dir):
            print(f"Loading tokenizer from {self.tokenizer_dir}")
            tokenizer = T5Tokenizer.from_pretrained(self.tokenizer_dir)
        else:
            print("Preparing tokenizer...")
            tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        return tokenizer
    
    def tokenize_function(self, batch, tokenizer, max_input_length=240, max_target_length=180):
        """Tokenize input and target text."""
        inputs = tokenizer(
            batch['input_text'], 
            max_length=max_input_length, 
            padding='max_length', 
            truncation=True
        )
        targets = tokenizer(
            batch['target_text'], 
            max_length=max_target_length, 
            padding='max_length', 
            truncation=True
        )
        
        targets['input_ids'] = [
            [(label if label != tokenizer.pad_token_id else -100) for label in target]
            for target in targets['input_ids']
        ]
        
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': targets['input_ids']
        }

    def tokenize_data(self, dataset, max_input_length=256, max_target_length=256):
        """Tokenize an entire dataset."""
        tokenizer = self.load_tokenizer()
        tokenized_dataset = dataset.map(
            lambda batch: self.tokenize_function(batch, tokenizer, max_input_length, max_target_length),
            batched=True,
            remove_columns=['input_text', 'target_text']
        )
        return tokenized_dataset 