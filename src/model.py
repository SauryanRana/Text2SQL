from transformers import T5Config, T5ForConditionalGeneration
import torch

class CustomT5:
    def __init__(self, vocab_size=32128, model_type="base"):
        self.model_type = model_type
        if model_type == "base":
            self.config = self._get_base_config(vocab_size)
        elif model_type == "efficient":
            self.config = self._get_efficient_config(vocab_size)
        else:
            raise ValueError("model_type must be either 'base' or 'efficient'")

    def _get_base_config(self, vocab_size):
        """Configuration for base T5 model"""
        return T5Config(
            vocab_size=vocab_size,
            d_model=256,
            d_ff=1024,
            num_layers=6,
            num_decoder_layers=6,
            num_heads=4,
            dropout_rate=0.1,
            is_encoder_decoder=True,
            pad_token_id=0,
            eos_token_id=1,
            decoder_start_token_id=0,
        )

    def _get_efficient_config(self, vocab_size):
        """Configuration for efficient-tiny T5 model"""
        return T5Config(
            vocab_size=vocab_size,
            d_model=128,  # Smaller dimension
            d_ff=512,     # Smaller feed-forward dimension
            num_layers=4,  # Fewer layers
            num_decoder_layers=4,
            num_heads=4,
            dropout_rate=0.1,
            is_encoder_decoder=True,
            pad_token_id=0,
            eos_token_id=1,
            decoder_start_token_id=0,
            use_cache=True,
            tie_word_embeddings=True,
        )
        
    def create_model(self):
        """Create and return a custom T5 model."""
        model = T5ForConditionalGeneration(self.config)
        model.gradient_checkpointing_enable()
        
        # Print model statistics
        total_params = sum(p.numel() for p in model.parameters())
        
        # Corrected calculation for embedding parameters
        # Input embeddings + output embeddings (if not tied)
        embedding_params = self.config.vocab_size * self.config.d_model
        if not getattr(self.config, 'tie_word_embeddings', True):
            embedding_params *= 2  # Count input and output embeddings separately
        
        # Calculate non-embedding parameters
        params_excluding_vocab = total_params - embedding_params
        
        print(f"Model Type: {self.model_type}")
        """print(f"Total Parameters: {total_params / 1e6:.2f}M")
        print(f"Parameters Excluding Vocabulary: {params_excluding_vocab / 1e6:.2f}M")
        print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")"""
        
        return model

