"""
Data utilities for the Bio-ChemTransformer.
"""

import os
import torch
import numpy as np
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Union
import logging
from transformers import AutoTokenizer
from utils.smiles_tokenization import SMILESTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BioChemDataset(Dataset):
    """
    Dataset for Bio-ChemTransformer.
    
    This dataset processes ADR text and SMILES strings for training
    and evaluation of the Bio-ChemTransformer model.
    """
    
    def __init__(
        self,
        data_file: str,
        bio_clinical_bert_tokenizer: str = "emilyalsentzer/Bio_ClinicalBERT",
        chembert_tokenizer: str = "seyonec/ChemBERTa-zinc-base-v1",
        max_adr_length: int = 256,
        max_smiles_length: int = 256,
        mask_probability: float = 0.25,
        is_training: bool = True,
        max_samples: Optional[int] = None  # New parameter for limiting dataset size
    ):
        """
        Initialize the dataset.
        
        Args:
            data_file: Path to the data file (JSON lines format)
            bio_clinical_bert_tokenizer: Bio_ClinicalBERT tokenizer name or path
            chembert_tokenizer: ChemBERT tokenizer name or path
            max_adr_length: Maximum length for ADR text tokens
            max_smiles_length: Maximum length for SMILES tokens
            mask_probability: Probability of masking tokens in ADR text
            is_training: Whether the dataset is for training
            max_samples: Maximum number of samples to load (for testing)
        """
        self.data_file = data_file
        self.max_adr_length = max_adr_length
        self.max_smiles_length = max_smiles_length
        self.mask_probability = mask_probability
        self.is_training = is_training
        self.max_samples = max_samples
        
        # Load tokenizers
        self.adr_tokenizer = AutoTokenizer.from_pretrained(bio_clinical_bert_tokenizer)
        self.smiles_tokenizer = SMILESTokenizer(
            chembert_model=chembert_tokenizer,
            max_length=max_smiles_length,
            canonicalize=True
        )
        
        # Load data
        self.data = self.load_data(data_file)
        logger.info(f"Loaded {len(self.data)} samples from {data_file}")
    
    def load_data(self, data_file: str) -> List[Dict]:
        """
        Load data from file.
        
        Args:
            data_file: Path to the data file
            
        Returns:
            List of data samples
        """
        data = []
        
        # Choose the appropriate loading method based on file extension
        if data_file.endswith('.json') or data_file.endswith('.jsonl'):
            # Load JSON Lines format
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
                    if self.max_samples and len(data) >= self.max_samples:
                        break
                    
        elif data_file.endswith('.csv'):
            # Load CSV format
            df = pd.read_csv(data_file)
            if self.max_samples:
                df = df.head(self.max_samples)
            data = df.to_dict('records')
            
        else:
            raise ValueError(f"Unsupported file format: {data_file}")
        
        # Filter out samples with missing data
        filtered_data = []
        for item in data:
            if item.get('smiles') and item.get('adr_text'):
                filtered_data.append(item)
        
        return filtered_data
    
    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.data)
    
    def mask_adr_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly mask tokens in the ADR text.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Tuple of (masked_input_ids, labels)
        """
        if not self.is_training:
            # No masking during evaluation
            return input_ids, input_ids.clone()
        
        # Create label tensor (will be -100 for non-masked tokens)
        labels = torch.full_like(input_ids, -100)
        
        # Create a mask for tokens that can be masked
        special_tokens = [
            self.adr_tokenizer.cls_token_id,
            self.adr_tokenizer.sep_token_id,
            self.adr_tokenizer.pad_token_id
        ]
        can_mask = torch.ones_like(input_ids, dtype=torch.bool)
        
        for token_id in special_tokens:
            can_mask = can_mask & (input_ids != token_id)
        
        # Also respect attention mask (don't mask padding)
        can_mask = can_mask & (attention_mask == 1)
        
        # Get indices of tokens that can be masked
        maskable_indices = can_mask.nonzero(as_tuple=True)[0]
        
        # Randomly select tokens to mask
        num_tokens = len(maskable_indices)
        num_to_mask = max(1, int(num_tokens * self.mask_probability))
        mask_indices = maskable_indices[torch.randperm(num_tokens)[:num_to_mask]]
        
        # Create masked input IDs (copy of original)
        masked_input_ids = input_ids.clone()
        
        # Apply masking
        masked_input_ids[mask_indices] = self.adr_tokenizer.mask_token_id
        
        # Set labels for masked tokens
        labels[mask_indices] = input_ids[mask_indices]
        
        return masked_input_ids, labels
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dict with tokenized inputs and labels
        """
        item = self.data[idx]
        
        # Get ADR text and SMILES
        adr_text = item['adr_text']
        smiles = item['smiles']
        
        # Tokenize ADR text
        adr_tokens = self.adr_tokenizer(
            adr_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_adr_length,
            return_tensors='pt'
        )
        
        # Tokenize SMILES
        smiles_tokens = self.smiles_tokenizer.tokenize(
            [smiles],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare inputs
        adr_input_ids = adr_tokens['input_ids'].squeeze(0)
        adr_attention_mask = adr_tokens['attention_mask'].squeeze(0)
        smiles_input_ids = smiles_tokens['input_ids'].squeeze(0)
        smiles_attention_mask = smiles_tokens['attention_mask'].squeeze(0)
        
        # Apply masking to ADR tokens during training
        if self.is_training:
            masked_adr_input_ids, labels = self.mask_adr_tokens(
                adr_input_ids,
                adr_attention_mask
            )
        else:
            masked_adr_input_ids = adr_input_ids
            labels = adr_input_ids.clone()
        
        # Ensure labels have the correct shape [batch_size, seq_length]
        labels = labels.unsqueeze(0)  # Add batch dimension if needed
        
        # Create sample dict
        sample = {
            'adr_input_ids': masked_adr_input_ids,
            'adr_attention_mask': adr_attention_mask,
            'smiles_input_ids': smiles_input_ids,
            'smiles_attention_mask': smiles_attention_mask,
            'labels': labels,
            'original_adr_input_ids': adr_input_ids,
            'drug_name': item.get('drug_name', ''),
            'source': item.get('source', '')
        }
        
        return sample


def create_data_loaders(
    train_file: str,
    val_file: str,
    test_file: str,
    bio_clinical_bert_tokenizer: str = "emilyalsentzer/Bio_ClinicalBERT",
    chembert_tokenizer: str = "seyonec/ChemBERTa-zinc-base-v1",
    max_adr_length: int = 256,
    max_smiles_length: int = 256,
    mask_probability: float = 0.25,
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples: Optional[int] = None  # New parameter for limiting dataset size
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_file: Path to training data file
        val_file: Path to validation data file
        test_file: Path to test data file
        bio_clinical_bert_tokenizer: Bio_ClinicalBERT tokenizer name or path
        chembert_tokenizer: ChemBERT tokenizer name or path
        max_adr_length: Maximum length for ADR text tokens
        max_smiles_length: Maximum length for SMILES tokens
        mask_probability: Probability of masking tokens in ADR text
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        max_samples: Maximum number of samples to load (for testing)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = BioChemDataset(
        data_file=train_file,
        bio_clinical_bert_tokenizer=bio_clinical_bert_tokenizer,
        chembert_tokenizer=chembert_tokenizer,
        max_adr_length=max_adr_length,
        max_smiles_length=max_smiles_length,
        mask_probability=mask_probability,
        is_training=True,
        max_samples=max_samples
    )
    
    val_dataset = BioChemDataset(
        data_file=val_file,
        bio_clinical_bert_tokenizer=bio_clinical_bert_tokenizer,
        chembert_tokenizer=chembert_tokenizer,
        max_adr_length=max_adr_length,
        max_smiles_length=max_smiles_length,
        mask_probability=mask_probability,
        is_training=False,
        max_samples=max_samples
    )
    
    test_dataset = BioChemDataset(
        data_file=test_file,
        bio_clinical_bert_tokenizer=bio_clinical_bert_tokenizer,
        chembert_tokenizer=chembert_tokenizer,
        max_adr_length=max_adr_length,
        max_smiles_length=max_smiles_length,
        mask_probability=mask_probability,
        is_training=False,
        max_samples=max_samples
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader 