"""
Training utilities for Bio-ChemTransformer.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
import logging
import wandb
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for Bio-ChemTransformer."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "./checkpoints",
        use_wandb: bool = False,
        wandb_project: str = "bio-chemtransformer",
        wandb_name: Optional[str] = None,
        gradient_accumulation_steps: int = 1,
        scaler: Optional[Any] = None,
        use_amp: bool = False
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler (optional)
            device: Device to use for training
            max_grad_norm: Maximum gradient norm for gradient clipping
            checkpoint_dir: Directory to save checkpoints
            use_wandb: Whether to use Weights & Biases for logging
            wandb_project: Weights & Biases project name
            wandb_name: Weights & Biases run name
            gradient_accumulation_steps: Number of steps to accumulate gradients before updating weights
            scaler: Gradient scaler for mixed precision training
            use_amp: Whether to use automatic mixed precision
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.scaler = scaler
        self.use_amp = use_amp
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize Weights & Biases
        if use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_name,
                config={
                    "model_type": type(model).__name__,
                    "optimizer": type(optimizer).__name__,
                    "scheduler": type(scheduler).__name__ if scheduler is not None else None,
                    "batch_size": train_loader.batch_size,
                    "max_grad_norm": max_grad_norm
                }
            )
            wandb.watch(model)
        
        # Move model to device
        self.model.to(device)
    
    def train_epoch(self) -> float:
        """
        Train the model for one epoch.
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        # For monitoring gradients
        gradient_norms = []
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                adr_input_ids = batch["adr_input_ids"].to(self.device)
                adr_attention_mask = batch["adr_attention_mask"].to(self.device)
                smiles_input_ids = batch["smiles_input_ids"].to(self.device)
                smiles_attention_mask = batch["smiles_attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass with mixed precision if enabled
                if self.use_amp and self.scaler is not None:
                    from torch.cuda.amp import autocast
                    with autocast():
                        outputs = self.model(
                            adr_input_ids=adr_input_ids,
                            adr_attention_mask=adr_attention_mask,
                            smiles_input_ids=smiles_input_ids,
                            smiles_attention_mask=smiles_attention_mask
                        )
                        
                        # Calculate loss
                        logits = outputs["logits"]
                        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                        
                        # Ensure labels have the correct shape
                        labels = labels.squeeze(0) if labels.dim() == 3 else labels
                        labels = labels.view(-1)  # Flatten to [batch_size * seq_length]
                        
                        loss = loss_fct(logits, labels)
                        
                        # Scale loss for gradient accumulation if needed
                        if self.gradient_accumulation_steps > 1:
                            loss = loss / self.gradient_accumulation_steps
                else:
                    # Regular forward pass without mixed precision
                    outputs = self.model(
                        adr_input_ids=adr_input_ids,
                        adr_attention_mask=adr_attention_mask,
                        smiles_input_ids=smiles_input_ids,
                        smiles_attention_mask=smiles_attention_mask
                    )
                    
                    # Calculate loss
                    logits = outputs["logits"]
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                    
                    # Ensure labels have the correct shape
                    labels = labels.squeeze(0) if labels.dim() == 3 else labels
                    labels = labels.view(-1)  # Flatten to [batch_size * seq_length]
                    
                    loss = loss_fct(logits, labels)
                    
                    # Scale loss for gradient accumulation if needed
                    if self.gradient_accumulation_steps > 1:
                        loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with mixed precision if enabled
                if self.use_amp and self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Track gradient norms for monitoring training health
                if batch_idx % 10 == 0:  # Check every 10 batches
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    gradient_norms.append(total_norm)
                
                # Gradient accumulation - only update every n steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or batch_idx == len(self.train_loader) - 1:
                    # Gradient clipping
                    if self.max_grad_norm > 0:
                        if self.use_amp and self.scaler is not None:
                            # Use scaler for gradient clipping
                            self.scaler.unscale_(self.optimizer)
                            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        else:
                            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    # Update weights
                    if self.use_amp and self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    # Update learning rate
                    if self.scheduler is not None:
                        self.scheduler.step()
                    
                    # Zero the gradients
                    self.optimizer.zero_grad()
                
                # Update progress bar
                current_loss = loss.item() * (self.gradient_accumulation_steps if self.gradient_accumulation_steps > 1 else 1)
                total_loss += current_loss
                progress_bar.set_postfix({
                    "loss": current_loss,
                    "avg_loss": total_loss / (batch_idx + 1)
                })
                
                # Log to Weights & Biases
                if self.use_wandb:
                    wandb.log({
                        "train_batch_loss": current_loss,
                        "learning_rate": self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]['lr']
                    })
                    
                    # Log gradient norms periodically
                    if batch_idx % 10 == 0 and gradient_norms:
                        wandb.log({"gradient_norm": gradient_norms[-1]})
            
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                continue  # Skip problematic batch and continue training
        
        # Calculate average loss
        avg_loss = total_loss / num_batches
        
        # Log average gradient norm
        avg_gradient_norm = sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0
        logger.info(f"Average gradient norm for epoch: {avg_gradient_norm:.4f}")
        
        # Log to Weights & Biases
        if self.use_wandb:
            wandb.log({
                "train_epoch_loss": avg_loss,
                "epoch_gradient_norm": avg_gradient_norm
            })
        
        return avg_loss
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the validation set.
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                try:
                    # Move batch to device
                    adr_input_ids = batch["adr_input_ids"].to(self.device)
                    adr_attention_mask = batch["adr_attention_mask"].to(self.device)
                    smiles_input_ids = batch["smiles_input_ids"].to(self.device)
                    smiles_attention_mask = batch["smiles_attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        adr_input_ids=adr_input_ids,
                        adr_attention_mask=adr_attention_mask,
                        smiles_input_ids=smiles_input_ids,
                        smiles_attention_mask=smiles_attention_mask
                    )
                    
                    # Calculate loss
                    logits = outputs["logits"]
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                    
                    # Ensure labels have the correct shape
                    labels = labels.squeeze(0) if labels.dim() == 3 else labels
                    labels = labels.view(-1)  # Flatten to [batch_size * seq_length]
                    
                    loss = loss_fct(logits, labels)
                    
                    # Update total loss
                    total_loss += loss.item()
                    
                    # Get predictions
                    preds = torch.argmax(logits, dim=-1)
                    
                    # Only consider non-ignored tokens (-100)
                    valid_mask = labels != -100
                    if valid_mask.sum() > 0:
                        all_preds.extend(preds[valid_mask].cpu().numpy())
                        all_labels.extend(labels[valid_mask].cpu().numpy())
                
                except Exception as e:
                    logger.error(f"Error during evaluation: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue  # Skip problematic batch and continue evaluation
        
        # Calculate average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        # Calculate additional metrics
        metrics = {
            "val_loss": avg_loss
        }
        
        # Calculate accuracy if predictions are available
        if all_preds and all_labels:
            accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
            metrics["accuracy"] = accuracy
            
            # Additional metrics - per class accuracy if needed
            # (add more metrics here if needed)
        
        # Log to Weights & Biases
        if self.use_wandb:
            wandb.log(metrics)
        
        return metrics
    
    def train(
        self,
        num_epochs: int,
        eval_frequency: int = 1,
        save_frequency: int = 1,
        early_stopping_patience: int = -1
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            eval_frequency: Frequency of evaluation (in epochs)
            save_frequency: Frequency of saving checkpoints (in epochs)
            early_stopping_patience: Patience for early stopping (-1 to disable)
            
        Returns:
            Dictionary with training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": []
        }
        
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"Epoch {epoch}/{num_epochs}")
            
            # Train for one epoch
            train_loss = self.train_epoch()
            history["train_loss"].append(train_loss)
            
            # Evaluate if necessary
            if epoch % eval_frequency == 0:
                metrics = self.evaluate()
                history["val_loss"].append(metrics["val_loss"])
                
                if "accuracy" in metrics:
                    history["val_accuracy"].append(metrics["accuracy"])
                
                logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {metrics['val_loss']:.4f}")
                
                # Check for best model
                if metrics["val_loss"] < best_val_loss:
                    best_val_loss = metrics["val_loss"]
                    patience_counter = 0
                    
                    # Save best model
                    self.save_checkpoint(f"best_model.pt")
                    logger.info(f"Saved best model with val_loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                
                # Early stopping
                if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping after {epoch} epochs")
                    break
            
            # Save checkpoint if necessary
            if epoch % save_frequency == 0:
                self.save_checkpoint(f"epoch_{epoch}.pt")
        
        logger.info("Training completed")
        return history
    
    def save_checkpoint(self, filename: str) -> None:
        """
        Save a model checkpoint.
        
        Args:
            filename: Name of the checkpoint file
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
            
        # Save mixed precision training state if used
        if self.use_amp and self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, filename: str) -> None:
        """
        Load a model checkpoint.
        
        Args:
            filename: Name of the checkpoint file
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file {checkpoint_path} not found")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        # Load mixed precision training state if available
        if self.use_amp and self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def close(self) -> None:
        """Clean up resources."""
        if self.use_wandb:
            wandb.finish() 