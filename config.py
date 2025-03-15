"""
Configuration for the Bio-ChemTransformer model.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class ModelConfig:
    # Embedding configurations
    bio_clinical_bert_model: str = "emilyalsentzer/Bio_ClinicalBERT"
    chembert_model: str = "seyonec/ChemBERTa-zinc-base-v1"  # Can be replaced with other ChemBERT variants
    
    # Dimension alignment
    bio_clinical_bert_dim: int = 768
    chembert_dim: int = 768
    projection_dim: int = 768  # Final dimension after alignment
    
    # Embedding combination strategy: "concat" or "sum"
    embedding_combination: str = "concat"
    
    # Encoder configurations
    num_encoder_layers: int = 6
    encoder_attention_heads: int = 12
    encoder_ffn_dim: int = 3072
    encoder_layerdrop: float = 0.0  # Reduced from 0.1 to improve information flow
    
    # Decoder configurations
    num_decoder_layers: int = 6
    decoder_attention_heads: int = 12
    decoder_ffn_dim: int = 3072
    decoder_layerdrop: float = 0.0  # Reduced from 0.1 to improve information flow
    
    # General transformer settings
    dropout: float = 0.2  # Increased to prevent overfitting
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    activation_function: str = "gelu"
    max_position_embeddings: int = 512
    
    # Tokenizer settings
    pad_token_id: int = 0
    eos_token_id: int = 2
    
    # DMA settings
    use_dma: bool = True  # Diagonal Masked Attention
    dma_probability: float = 0.15  # Reduced from 0.25 to make learning easier initially


@dataclass
class TrainingConfig:
    output_dir: str = "./checkpoints"
    overwrite_output_dir: bool = True
    
    # Training hyperparameters
    learning_rate: float = 5e-5  # Increased from 3e-5 for better initial learning
    weight_decay: float = 0.005  # Reduced from 0.01 to allow more flexibility in learning
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Training settings
    num_train_epochs: int = 20  # Increased from 10 to allow more time to learn
    max_steps: int = -1  # Overrides num_train_epochs if > 0
    per_device_train_batch_size: int = 2  # Kept at 2 for memory constraints
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    
    # Learning rate schedule
    lr_scheduler_type: str = "cosine"  # Changed from linear to cosine for better convergence
    warmup_ratio: float = 0.2  # Increased from 0.1 for more gradual warmup
    
    # Evaluation and logging
    evaluation_strategy: str = "epoch"
    logging_dir: str = "./logs"
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 1000
    save_total_limit: int = 3
    
    # Early stopping
    early_stopping_patience: int = 5  # Increased from 3 to give more room for improvement
    early_stopping_threshold: float = 0.005  # Reduced from 0.01 to be more sensitive to small improvements
    
    # Mixed precision
    fp16: bool = True
    fp16_opt_level: str = "O1"


@dataclass
class DataConfig:
    # Data paths
    train_data_path: str = "data/processed/train.json"
    validation_data_path: str = "data/processed/val.json"
    test_data_path: str = "data/processed/test.json"
    
    # Data processing
    max_adr_length: int = 256
    max_smiles_length: int = 256
    mask_probability: float = 0.15  # Reduced from 0.25 to make the task easier initially
    
    # Data loading
    shuffle: bool = True
    seed: int = 42
    num_workers: int = 4
    cache_dir: str = "./cache"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # General settings
    seed: int = 42
    debug: bool = False
    verbose: bool = True
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create a Config object from a dictionary."""
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        data_config = DataConfig(**config_dict.get("data", {}))
        
        return cls(
            model=model_config, 
            training=training_config,
            data=data_config,
            seed=config_dict.get("seed", 42),
            debug=config_dict.get("debug", False),
            verbose=config_dict.get("verbose", True)
        )


def get_default_config():
    """Return the default configuration."""
    return Config() 