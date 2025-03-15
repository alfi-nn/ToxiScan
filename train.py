"""
Main training script for Bio-ChemTransformer.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import random
import numpy as np
import logging
from transformers import get_linear_schedule_with_warmup

from config import Config, get_default_config
from models.transformer import BioChemTransformer
from models.embedding import BioChemEmbedding
from utils.data_utils import create_data_loaders
from training.trainer import Trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    """Main training function."""
    # Get configuration
    config = get_default_config()
    
    # Set random seed
    set_seed(config.seed)
    
    # Enable mixed precision training if available
    if config.training.fp16 and torch.cuda.is_available():
        try:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            use_amp = True
            logger.info("Using mixed precision training")
        except ImportError:
            logger.warning("Mixed precision training not available, using full precision")
            use_amp = False
            scaler = None
    else:
        use_amp = False
        scaler = None
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        bio_clinical_bert_tokenizer=config.model.bio_clinical_bert_model,
        chembert_tokenizer=config.model.chembert_model,
        max_adr_length=config.data.max_adr_length,
        max_smiles_length=config.data.max_smiles_length,
        mask_probability=config.data.mask_probability,
        batch_size=config.training.per_device_train_batch_size,
        num_workers=config.data.num_workers,
        max_samples=args.max_samples
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = BioChemTransformer(
        bio_clinical_bert_model=config.model.bio_clinical_bert_model,
        chembert_model=config.model.chembert_model,
        bio_clinical_bert_dim=config.model.bio_clinical_bert_dim,
        chembert_dim=config.model.chembert_dim,
        projection_dim=config.model.projection_dim,
        embedding_combination=config.model.embedding_combination,
        num_encoder_layers=config.model.num_encoder_layers,
        num_decoder_layers=config.model.num_decoder_layers,
        encoder_attention_heads=config.model.encoder_attention_heads,
        decoder_attention_heads=config.model.decoder_attention_heads,
        encoder_ffn_dim=config.model.encoder_ffn_dim,
        decoder_ffn_dim=config.model.decoder_ffn_dim,
        dropout=config.model.dropout,
        attention_dropout=config.model.attention_dropout,
        activation_dropout=config.model.activation_dropout,
        activation_function=config.model.activation_function,
        max_position_embeddings=config.model.max_position_embeddings,
        pad_token_id=config.model.pad_token_id,
        eos_token_id=config.model.eos_token_id,
        use_dma=config.model.use_dma,
        dma_probability=config.model.dma_probability,
        freeze_pretrained=True  # Start with frozen pretrained models
    )
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        eps=config.training.adam_epsilon
    )
    
    # Initialize scheduler
    total_steps = len(train_loader) * config.training.num_train_epochs
    warmup_steps = int(total_steps * config.training.warmup_ratio)
    
    if config.training.lr_scheduler_type == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        max_grad_norm=config.training.max_grad_norm,
        checkpoint_dir=config.training.output_dir,
        use_wandb=args.use_wandb,
        wandb_project="bio-chemtransformer",
        wandb_name=args.run_name or "bio-chemtransformer-run",
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        scaler=scaler,
        use_amp=use_amp
    )
    
    # Log model architecture summary
    logger.info("Model Architecture:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Log training configuration
    logger.info(f"Training for {config.training.num_train_epochs} epochs")
    logger.info(f"Batch size: {config.training.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation steps: {config.training.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    logger.info(f"Scheduler: {config.training.lr_scheduler_type} with {warmup_steps} warmup steps")
    
    # Load checkpoint if specified
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        # If a relative path is provided, prepend the checkpoint directory
        if not os.path.isabs(checkpoint_path) and not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(config.training.output_dir, checkpoint_path)
        
        if os.path.exists(checkpoint_path):
            logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
            trainer.load_checkpoint(os.path.basename(checkpoint_path))
        else:
            logger.warning(f"Checkpoint file {checkpoint_path} not found. Starting training from scratch.")
    
    # Train model
    logger.info("Starting training...")
    trainer.train(
        num_epochs=config.training.num_train_epochs,
        eval_frequency=1,
        save_frequency=config.training.save_steps // len(train_loader) + 1,
        early_stopping_patience=config.training.early_stopping_patience
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    trainer.val_loader = test_loader  # Temporarily replace val_loader with test_loader
    test_metrics = trainer.evaluate()
    trainer.val_loader = val_loader  # Restore val_loader
    
    logger.info(f"Test metrics: {test_metrics}")
    
    # Clean up
    trainer.close()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bio-ChemTransformer")
    
    # Data arguments
    parser.add_argument("--train_file", type=str, default="data/processed/train.json",
                        help="Path to training data file")
    parser.add_argument("--val_file", type=str, default="data/processed/val.json",
                        help="Path to validation data file")
    parser.add_argument("--test_file", type=str, default="data/processed/test.json",
                        help="Path to test data file")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use (set to None to use full dataset)")
    
    # Training arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for the training run")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a checkpoint file to resume training from")
    
    args = parser.parse_args()
    main(args) 