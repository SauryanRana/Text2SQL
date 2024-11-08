# TODO: Allow parameterization via argparse
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from utils import preprocess_function, compute_metrics

model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-efficient-tiny")
tokenizer = AutoTokenizer.from_pretrained("google/t5-efficient-tiny")

path = '../../datasets/wikisql'
dataset = load_dataset("wikisql", data_dir=path + '/data')
train_data = dataset["train"]
val_data = dataset["validation"]
tokenized_train_data = train_data.map(lambda batch: preprocess_function(batch, tokenizer), batched=True, batch_size=2048)
tokenized_val_data = val_data.map(lambda batch: preprocess_function(batch, tokenizer), batched=True, batch_size=2048)

# TODO: activate tqdm
# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_val_data,
    eval_dataset=tokenized_val_data,
    compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenized_val_data, tokenizer, path+'/tables/validation/dev.db')
)


# TODO: Evaluation is extremely slow, profile code and figure out the issue. I assume that it is the 'parse_sql_to_canonical' calls. If we can't optimize that we may have to go with the different tokenizer approach described in the functions comment
# Train
trainer.train()
