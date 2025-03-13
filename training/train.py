"""
Training script for Bio-ChemTransformer model.
"""

import os
import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
from typing import Dict, List, Optional, Tuple

from models.transformer import BioChemTransformer
from utils.data_utils import ADRDataset
from utils.tokenizers import BioChemTokenizer
from config import ModelConfig, TrainingConfig, DataConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        data_config: DataConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the trainer.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
            data_config: Data configuration
            device: Device to use for training
        """
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        self.device = device
        
        # Initialize model
        self.model = BioChemTransformer(
            bio_clinical_bert_model=model_config.bio_clinical_bert_model,
            chembert_model=model_config.chembert_model,
            bio_clinical_bert_dim=model_config.bio_clinical_bert_dim,
            chembert_dim=model_config.chembert_dim,
            projection_dim=model_config.projection_dim,
            num_encoder_layers=model_config.num_encoder_layers,
            num_decoder_layers=model_config.num_decoder_layers,
            num_attention_heads=model_config.num_attention_heads,
            dropout=model_config.dropout,
            max_position_embeddings=model_config.max_position_embeddings,
            pad_token_id=model_config.pad_token_id,
            eos_token_id=model_config.eos_token_id,
            vocab_size=model_config.vocab_size
        ).to(device)
        
        # Initialize tokenizer
        self.tokenizer = BioChemTokenizer(
            bio_clinical_bert_model=model_config.bio_clinical_bert_model,
            chembert_model=model_config.chembert_model
        )
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=training_config.num_epochs,
            eta_min=training_config.min_learning_rate
        )
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=model_config.pad_token_id)
        
        # Initialize best metrics
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Initialize wandb if enabled
        if training_config.use_wandb:
            wandb.init(
                project=training_config.wandb_project,
                name=training_config.wandb_run_name,
                config={
                    "model_config": model_config.__dict__,
                    "training_config": training_config.__dict__,
                    "data_config": data_config.__dict__
                }
            )
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load and prepare the datasets.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Load datasets
        train_dataset = ADRDataset(
            data_path=self.data_config.train_data_path,
            tokenizer=self.tokenizer,
            max_length=self.model_config.max_position_embeddings
        )
        
        val_dataset = ADRDataset(
            data_path=self.data_config.val_data_path,
            tokenizer=self.tokenizer,
            max_length=self.model_config.max_position_embeddings
        )
        
        test_dataset = ADRDataset(
            data_path=self.data_config.test_data_path,
            tokenizer=self.tokenizer,
            max_length=self.model_config.max_position_embeddings
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=self.training_config.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            num_workers=self.training_config.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            num_workers=self.training_config.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def save_checkpoint(self, epoch: int, val_loss: float):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'model_config': self.model_config.__dict__,
            'training_config': self.training_config.__dict__,
            'data_config': self.data_config.__dict__
        }
        
        checkpoint_path = os.path.join(
            self.training_config.checkpoint_dir,
            f'checkpoint_epoch_{epoch}.pt'
        )
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            best_model_path = os.path.join(
                self.training_config.checkpoint_dir,
                'best_model.pt'
            )
            torch.save(checkpoint, best_model_path)
            logger.info(f"Saved best model to {best_model_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint['val_loss']
        self.best_epoch = checkpoint['epoch']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        with tqdm(train_loader, desc="Training", leave=False) as pbar:
            for batch in pbar:
                # Move batch to device
                smiles_input_ids = batch['smiles_input_ids'].to(self.device)
                smiles_attention_mask = batch['smiles_attention_mask'].to(self.device)
                adr_input_ids = batch['adr_input_ids'].to(self.device)
                adr_attention_mask = batch['adr_attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    smiles_input_ids=smiles_input_ids,
                    smiles_attention_mask=smiles_attention_mask,
                    adr_input_ids=adr_input_ids,
                    adr_attention_mask=adr_attention_mask
                )
                
                # Calculate loss
                loss = self.criterion(
                    outputs.logits.view(-1, self.model_config.vocab_size),
                    adr_input_ids.view(-1)
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                if self.training_config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config.max_grad_norm
                    )
                
                # Update weights
                self.optimizer.step()
                
                # Update progress bar
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Evaluate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = len(val_loader)
        
        with tqdm(val_loader, desc="Evaluating", leave=False) as pbar:
            for batch in pbar:
                # Move batch to device
                smiles_input_ids = batch['smiles_input_ids'].to(self.device)
                smiles_attention_mask = batch['smiles_attention_mask'].to(self.device)
                adr_input_ids = batch['adr_input_ids'].to(self.device)
                adr_attention_mask = batch['adr_attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    smiles_input_ids=smiles_input_ids,
                    smiles_attention_mask=smiles_attention_mask,
                    adr_input_ids=adr_input_ids,
                    adr_attention_mask=adr_attention_mask
                )
                
                # Calculate loss
                loss = self.criterion(
                    outputs.logits.view(-1, self.model_config.vocab_size),
                    adr_input_ids.view(-1)
                )
                
                # Update progress bar
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches
    
    def train(self):
        """Train the model."""
        # Create checkpoint directory
        os.makedirs(self.training_config.checkpoint_dir, exist_ok=True)
        
        # Load data
        train_loader, val_loader, test_loader = self.load_data()
        
        # Training loop
        for epoch in range(self.training_config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.training_config.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            logger.info(f"Training loss: {train_loss:.4f}")
            
            # Evaluate
            val_loss = self.evaluate(val_loader)
            logger.info(f"Validation loss: {val_loss:.4f}")
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            if self.training_config.use_wandb:
                wandb.log({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'epoch': epoch + 1
                })
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, val_loss)
        
        logger.info("Training complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
        
        # Load best model for final evaluation
        self.load_checkpoint(os.path.join(self.training_config.checkpoint_dir, 'best_model.pt'))
        
        # Final evaluation
        test_loss = self.evaluate(test_loader)
        logger.info(f"Test loss: {test_loss:.4f}")
        
        if self.training_config.use_wandb:
            wandb.log({'test_loss': test_loss})
            wandb.finish()


def main():
    """Main function."""
    # Load configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    data_config = DataConfig()
    
    # Initialize trainer
    trainer = Trainer(model_config, training_config, data_config)
    
    # Train model
    trainer.train()


if __name__ == "__main__":
    main() 