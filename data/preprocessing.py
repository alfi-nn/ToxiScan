"""
Data preprocessing utilities for Bio-ChemTransformer.
"""

import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Union, Optional
import warnings
import logging
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')


def canonicalize_smiles(smiles: str) -> str:
    """
    Convert SMILES string to canonical form.
    
    Args:
        smiles: SMILES string to canonicalize
        
    Returns:
        Canonicalized SMILES string, or original string if conversion fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        else:
            warnings.warn(f"Failed to parse SMILES: {smiles}")
            return smiles
    except Exception as e:
        warnings.warn(f"Error canonicalizing SMILES {smiles}: {str(e)}")
        return smiles


def clean_adr_text(text: str) -> str:
    """
    Clean ADR text by removing special characters and normalizing.
    
    Args:
        text: ADR text to clean
        
    Returns:
        Cleaned ADR text
    """
    if not text:
        return ""
    
    # Replace multiple spaces with single space
    text = ' '.join(text.split())
    
    # Remove special characters that might interfere with tokenization
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    # Convert to lowercase
    text = text.lower()
    
    return text.strip()


def process_sider_data(file_path: str, output_path: str):
    """
    Process SIDER database data.
    
    Args:
        file_path: Path to SIDER data file
        output_path: Path to save processed data
    """
    logger.info(f"Processing SIDER data from {file_path}")
    
    # Load SIDER data (assuming it's a CSV with drug_name, smiles, and adr_text columns)
    df = pd.read_csv(file_path)
    
    # Process each entry
    processed_data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing SIDER data"):
        # Extract data
        drug_name = row['drug_name']
        smiles = row['smiles']
        adr_text = row['adr_text']
        
        # Clean data
        canonicalized_smiles = canonicalize_smiles(smiles)
        cleaned_adr_text = clean_adr_text(adr_text)
        
        # Skip empty entries
        if not canonicalized_smiles or not cleaned_adr_text:
            continue
            
        # Create entry
        entry = {
            'drug_name': drug_name,
            'smiles': canonicalized_smiles,
            'adr_text': cleaned_adr_text,
            'source': 'SIDER'
        }
        
        processed_data.append(entry)
    
    # Save processed data
    with open(output_path, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    logger.info(f"Processed {len(processed_data)} SIDER entries. Saved to {output_path}")
    
    return processed_data


def process_dailymed_data(file_path: str, output_path: str):
    """
    Process DailyMed database data.
    
    Args:
        file_path: Path to DailyMed data file
        output_path: Path to save processed data
    """
    logger.info(f"Processing DailyMed data from {file_path}")
    
    # Load DailyMed data (assuming it's a CSV with drug_name, smiles, and adr_text columns)
    df = pd.read_csv(file_path)
    
    # Process each entry
    processed_data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing DailyMed data"):
        # Extract data
        drug_name = row['drug_name']
        smiles = row['smiles']
        adr_text = row['adr_text']
        
        # Clean data
        canonicalized_smiles = canonicalize_smiles(smiles)
        cleaned_adr_text = clean_adr_text(adr_text)
        
        # Skip empty entries
        if not canonicalized_smiles or not cleaned_adr_text:
            continue
            
        # Create entry
        entry = {
            'drug_name': drug_name,
            'smiles': canonicalized_smiles,
            'adr_text': cleaned_adr_text,
            'source': 'DailyMed'
        }
        
        processed_data.append(entry)
    
    # Save processed data
    with open(output_path, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    logger.info(f"Processed {len(processed_data)} DailyMed entries. Saved to {output_path}")
    
    return processed_data


def process_faers_data(file_path: str, output_path: str):
    """
    Process FAERS database data.
    
    Args:
        file_path: Path to FAERS data file
        output_path: Path to save processed data
    """
    logger.info(f"Processing FAERS data from {file_path}")
    
    # Load FAERS data (assuming it's a CSV with drug_name, smiles, and adr_text columns)
    df = pd.read_csv(file_path)
    
    # Process each entry
    processed_data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing FAERS data"):
        # Extract data
        drug_name = row['drug_name']
        smiles = row['smiles']
        adr_text = row['adr_text']
        
        # Clean data
        canonicalized_smiles = canonicalize_smiles(smiles)
        cleaned_adr_text = clean_adr_text(adr_text)
        
        # Skip empty entries
        if not canonicalized_smiles or not cleaned_adr_text:
            continue
            
        # Create entry
        entry = {
            'drug_name': drug_name,
            'smiles': canonicalized_smiles,
            'adr_text': cleaned_adr_text,
            'source': 'FAERS'
        }
        
        processed_data.append(entry)
    
    # Save processed data
    with open(output_path, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    logger.info(f"Processed {len(processed_data)} FAERS entries. Saved to {output_path}")
    
    return processed_data


def split_data(data: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42):
    """
    Split data into train, validation, and test sets.
    
    Args:
        data: List of data entries
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Shuffle data
    indices = np.random.permutation(len(data))
    
    # Calculate split points
    train_end = int(train_ratio * len(data))
    val_end = train_end + int(val_ratio * len(data))
    
    # Split data
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    test_data = [data[i] for i in test_indices]
    
    return train_data, val_data, test_data


def combine_datasets(datasets: List[List[Dict]]) -> List[Dict]:
    """
    Combine multiple datasets into a single dataset.
    
    Args:
        datasets: List of datasets to combine
        
    Returns:
        Combined dataset
    """
    combined_data = []
    for dataset in datasets:
        combined_data.extend(dataset)
    
    return combined_data


def main():
    """Main function to process all datasets."""
    # Set up paths
    os.makedirs("data/processed", exist_ok=True)
    
    # Process SIDER data
    sider_data = []
    if os.path.exists("data/raw/sider.csv"):
        sider_data = process_sider_data(
            "data/raw/sider.csv",
            "data/processed/sider.json"
        )
    
    # Process DailyMed data
    dailymed_data = []
    if os.path.exists("data/raw/dailymed.csv"):
        dailymed_data = process_dailymed_data(
            "data/raw/dailymed.csv",
            "data/processed/dailymed.json"
        )
    
    # Process FAERS data
    faers_data = []
    if os.path.exists("data/raw/faers.csv"):
        faers_data = process_faers_data(
            "data/raw/faers.csv",
            "data/processed/faers.json"
        )
    
    # Combine datasets
    combined_data = combine_datasets([sider_data, dailymed_data, faers_data])
    
    # Split data
    train_data, val_data, test_data = split_data(combined_data)
    
    # Save split data
    with open("data/processed/train.json", 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open("data/processed/val.json", 'w') as f:
        json.dump(val_data, f, indent=2)
    
    with open("data/processed/test.json", 'w') as f:
        json.dump(test_data, f, indent=2)
    
    logger.info(f"Data split complete. Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")


if __name__ == "__main__":
    main() 