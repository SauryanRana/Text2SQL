import argparse
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import wandb
import torch
from functools import partial
from .eval import compute_metrics


class Text2SQLTrainer:
    def __init__(
        self,
        model_name,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        output_dir='custom-t5-wikisql',
        project='',
        phase="Simple",
        training_args=None
    ):
        self.model_name = model_name
        self.model = model
        self.processing_class = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.project = project
        self.phase = phase
        self.training_args = training_args or {}
        
    def get_training_args(self):
        """Define training arguments with defaults that can be overridden."""
        default_args = {
            'output_dir': self.output_dir,
            'evaluation_strategy': "steps",
            'eval_steps': 50,
            'save_strategy': "epoch",
            'logging_dir': f"{self.output_dir}/logs",
            'per_device_train_batch_size': 8,
            'per_device_eval_batch_size': 8,
            'eval_accumulation_steps': 2,
            'learning_rate': 0.0001,
            'weight_decay': 0.01,
            'save_total_limit': 5,
            'num_train_epochs': 40 if self.phase == "simple" else 10,
            'predict_with_generate': True,
            'logging_steps': 500,
            'report_to': "wandb",
            'fp16': False,
            'dataloader_num_workers': 4,
            'dataloader_prefetch_factor': 1,
        }
        
        # Update default args with any custom args provided
        default_args.update(self.training_args)
        
        return Seq2SeqTrainingArguments(**default_args)

    def train(self):
        """Train the model."""
        # Initialize wandb
        wandb.init(
            project=self.project,
            name=f"{self.phase}_{self.output_dir}",
            sync_tensorboard=True
        )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.processing_class,
            model=self.model,
            padding=True
        )

        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.get_training_args(),
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.processing_class,
            data_collator=data_collator,
            compute_metrics=partial(compute_metrics, tokenizer=self.processing_class)
        )

        # Train and evaluate
        trainer.train()
        trainer.evaluate()
        wandb.finish()

        return trainer

    def save_model(self, trainer):
        """Save model and create model card."""
        trainer.save_model(self.output_dir)
        self.processing_class.save_pretrained(self.output_dir)
        
        # Create model card
        with open(f"{self.output_dir}/README.md", 'w') as f:
            f.write("# T5 Fine-tuned on WikiSQL\n\nThis model generates SQL queries from natural language using the WikiSQL dataset.")


if __name__ == "__main__":
    trainer = Text2SQLTrainer()
    trainer.train()
    