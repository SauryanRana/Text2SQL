from .dataset import Text2SQLDataset
from .model import CustomT5
from .tokenizer import SQLTokenizer
from .train import Text2SQLTrainer
from .eval import Evaluator
from .utils import get_device
import torch
import argparse
import os

def get_config(model_choice):
    """Get configuration based on model choice."""
    base_config = {
        'data_directory': "data/preprocessed_wikisql",
        'wandb_project': 't5-tiny-Input-QT-S',
        'output_dir': 'outputs/simple-data-0.0001',
        'phase': 'simple',
        'training_args': {
            'learning_rate': 0.0001,
            'num_train_epochs': 20,
            'per_device_train_batch_size': 16,
            'per_device_eval_batch_size': 16,
            'gradient_accumulation_steps': 2,
            'eval_steps': 200,
            'warmup_steps': 500,
            'dataloader_num_workers': 4,
        }
    }
    
    model_configs = {
        'base': {
            'model_name': 't5-small',
            'model_type': 'base',
            'output_suffix': 'base'
        },
        'efficient': {
            'model_name': 'google/t5-efficient-tiny',
            'model_type': 'efficient',
            'output_suffix': 'efficient'
        }
    }
    
    # Update base config with model-specific settings
    config = {**base_config, **model_configs[model_choice]}
    # Update output directory with model type
    config['output_dir'] = f"{config['output_dir']}-{config['output_suffix']}"
    
    return config

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Text2SQL model')
    parser.add_argument('--model', 
                       choices=['base', 'efficient'],
                       default='base',
                       help='Choose model type: base (custom T5) or efficient (T5-efficient-tiny)')
    parser.add_argument('--data_dir',
                       default="data/preprocessed_wikisql",
                       help='Directory containing the dataset')
    parser.add_argument('--phase',
                       default="simple",
                       help='Training phase (e.g., simple, moderate)')
    args = parser.parse_args()

    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Get configuration based on model choice
    config = get_config(args.model)
    
    # Update config with command line arguments
    config['data_directory'] = args.data_dir
    config['phase'] = args.phase
    
    print(f"Using model configuration: {config}")
    
    # Load and prepare data
    dataset = Text2SQLDataset(config['data_directory'], phase=config['phase'])
    train_data, test_data, val_data = dataset.load_data()
    
    # Initialize tokenizer and tokenize datasets
    tokenizer_handler = SQLTokenizer(config['model_name'])
    tokenizer = tokenizer_handler.load_tokenizer()
    
    print("Tokenizing datasets...")
    train_data = tokenizer_handler.tokenize_data(train_data)
    val_data = tokenizer_handler.tokenize_data(val_data)
    #stest_data = tokenizer_handler.tokenize_data(test_data)
    print("Tokenization complete.")
    
    # Create model
    model_handler = CustomT5(vocab_size=len(tokenizer), model_type=config['model_type'])
    model = model_handler.create_model()
    model.to(get_device())
    
    # Use training args from config instead of defining them here
    trainer = Text2SQLTrainer(
        model_name=config['model_name'],
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        output_dir=config['output_dir'],
        project=config['wandb_project'],
        phase=config['phase'],
        training_args=config['training_args']  # Use training args from config
    )
    
    trained_trainer = trainer.train()
    trainer.save_model(trained_trainer)
    
    # Evaluate model
    evaluator = Evaluator(
        model_dir=config['output_dir'],
        tokenizer=tokenizer,
        test_data=test_data,
        wandb_project=config['wandb_project']
    )
    evaluator.evaluate()

if __name__ == "__main__":
    main()


"""
# For base custom T5 model
python -m src.main --model base

# For efficient-tiny T5 model
python -m src.main --model efficient

"""