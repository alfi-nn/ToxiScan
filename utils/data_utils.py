"""
Data utilities for the Bio-ChemTransformer model.
"""

import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from torch.utils.data import Dataset, DataLoader
from .tokenizers import BioChemTokenizer


class ADRDataset(Dataset):
    """
    Dataset for Adverse Drug Reaction (ADR) prediction.
    """
    
    def __init__(
        self,
        file_path: str,
        tokenizer: BioChemTokenizer,
        mask_probability: float = 0.0,
        max_adr_length: int = 256,
        max_smiles_length: int = 256
    ):
        """
        Initialize the ADR dataset.
        
        Args:
            file_path: Path to the data file (JSON or CSV)
            tokenizer: The BioChemTokenizer instance
            mask_probability: Probability of masking tokens in ADR text (for training)
            max_adr_length: Maximum length of ADR text tokens
            max_smiles_length: Maximum length of SMILES tokens
        """
        self.tokenizer = tokenizer
        self.mask_probability = mask_probability
        self.max_adr_length = max_adr_length
        self.max_smiles_length = max_smiles_length
        
        # Load the data
        self.data = self._load_data(file_path)
    
    def _load_data(self, file_path: str) -> List[Dict]:
        """
        Load data from a file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            List of dictionaries containing data samples
        """
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            data = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Ensure data has the required fields
        for item in data:
            if 'drug_name' not in item:
                item['drug_name'] = ""
            if 'smiles' not in item:
                raise ValueError("Data must contain 'smiles' field")
            if 'adr_text' not in item:
                raise ValueError("Data must contain 'adr_text' field")
        
        return data
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a data sample.
        
        Args:
            idx: The index of the sample
            
        Returns:
            Dictionary containing the tokenized inputs and targets
        """
        item = self.data[idx]
        
        # Extract text and SMILES
        drug_name = item.get('drug_name', "")
        smiles = item['smiles']
        adr_text = item['adr_text']
        
        # Tokenize inputs
        tokenized = self.tokenizer.tokenize_batch(
            adr_texts=[adr_text],
            smiles_strings=[smiles],
            mask_probability=self.mask_probability
        )
        
        # For training, the target is the original (unmasked) ADR text
        # For this, we tokenize the ADR text again without masking
        target_encodings = self.tokenizer.tokenize_adr(
            adr_text,
            padding="max_length",
            truncation=True
        )
        
        return {
            "drug_name": drug_name,
            "adr_input_ids": tokenized["adr_input_ids"][0],
            "adr_attention_mask": tokenized["adr_attention_mask"][0],
            "smiles_input_ids": tokenized["smiles_input_ids"][0],
            "smiles_attention_mask": tokenized["smiles_attention_mask"][0],
            "labels": target_encodings.input_ids[0]
        }


def create_data_loaders(
    train_file: str,
    val_file: str,
    tokenizer: BioChemTokenizer,
    batch_size: int = 32,
    mask_probability: float = 0.25,
    max_adr_length: int = 256,
    max_smiles_length: int = 256,
    num_workers: int = 4,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and validation.
    
    Args:
        train_file: Path to the training data file
        val_file: Path to the validation data file
        tokenizer: The BioChemTokenizer instance
        batch_size: Batch size for data loaders
        mask_probability: Probability of masking tokens in ADR text
        max_adr_length: Maximum length of ADR text tokens
        max_smiles_length: Maximum length of SMILES tokens
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the training data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = ADRDataset(
        train_file,
        tokenizer,
        mask_probability=mask_probability,
        max_adr_length=max_adr_length,
        max_smiles_length=max_smiles_length
    )
    
    val_dataset = ADRDataset(
        val_file,
        tokenizer,
        mask_probability=0.0,  # No masking for validation
        max_adr_length=max_adr_length,
        max_smiles_length=max_smiles_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
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
    
    return train_loader, val_loader


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching samples.
    
    Args:
        batch: List of dictionaries containing data samples
        
    Returns:
        Dictionary containing the batched tensors
    """
    drug_names = [item["drug_name"] for item in batch]
    
    adr_input_ids = torch.stack([item["adr_input_ids"] for item in batch])
    adr_attention_mask = torch.stack([item["adr_attention_mask"] for item in batch])
    
    smiles_input_ids = torch.stack([item["smiles_input_ids"] for item in batch])
    smiles_attention_mask = torch.stack([item["smiles_attention_mask"] for item in batch])
    
    labels = torch.stack([item["labels"] for item in batch])
    
    return {
        "drug_names": drug_names,
        "adr_input_ids": adr_input_ids,
        "adr_attention_mask": adr_attention_mask,
        "smiles_input_ids": smiles_input_ids,
        "smiles_attention_mask": smiles_attention_mask,
        "labels": labels
    } 