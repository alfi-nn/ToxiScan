"""
Script to prepare training data by splitting into train, validation, and test sets.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_training_data(
    input_file: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    Split the processed data into train, validation, and test sets.
    
    Args:
        input_file: Path to the processed data file (JSON Lines format)
        output_dir: Directory to save the split data
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        seed: Random seed for reproducibility
    """
    logger.info(f"Preparing training data from {input_file}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    logger.info(f"Loaded {len(data)} samples from {input_file}")
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(data)
    
    # Check for required columns
    required_columns = ['drug_name', 'smiles', 'adr_text']
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Required column '{col}' not found in data")
            return
    
    # Remove samples with missing values
    df = df.dropna(subset=['smiles', 'adr_text'])
    df = df[df['smiles'] != '']
    df = df[df['adr_text'] != '']
    
    logger.info(f"After cleaning, {len(df)} samples remain")
    
    # Verify ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
    
    # Split data
    train_data, temp_data = train_test_split(
        df,
        train_size=train_ratio,
        random_state=seed,
        stratify=df['source'] if 'source' in df.columns else None
    )
    
    # Adjust val_ratio for the remaining data
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    
    val_data, test_data = train_test_split(
        temp_data,
        train_size=val_ratio_adjusted,
        random_state=seed,
        stratify=temp_data['source'] if 'source' in temp_data.columns else None
    )
    
    logger.info(f"Split data into {len(train_data)} train, {len(val_data)} validation, and {len(test_data)} test samples")
    
    # Save splits
    train_file = os.path.join(output_dir, 'train.json')
    val_file = os.path.join(output_dir, 'val.json')
    test_file = os.path.join(output_dir, 'test.json')
    
    train_data.to_json(train_file, orient='records', lines=True)
    val_data.to_json(val_file, orient='records', lines=True)
    test_data.to_json(test_file, orient='records', lines=True)
    
    logger.info(f"Saved train data to {train_file}")
    logger.info(f"Saved validation data to {val_file}")
    logger.info(f"Saved test data to {test_file}")
    
    # Save sample counts
    with open(os.path.join(output_dir, 'data_stats.txt'), 'w') as f:
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Train samples: {len(train_data)}\n")
        f.write(f"Validation samples: {len(val_data)}\n")
        f.write(f"Test samples: {len(test_data)}\n")
        
        # Source distribution
        if 'source' in df.columns:
            f.write("\nSource distribution:\n")
            f.write(f"Overall: {df['source'].value_counts().to_dict()}\n")
            f.write(f"Train: {train_data['source'].value_counts().to_dict()}\n")
            f.write(f"Validation: {val_data['source'].value_counts().to_dict()}\n")
            f.write(f"Test: {test_data['source'].value_counts().to_dict()}\n")
    
    logger.info("Data preparation completed")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data by splitting into train, validation, and test sets")
    
    parser.add_argument("--input_file", type=str, default="data/processed/all_data.json",
                        help="Path to the processed data file")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Directory to save the split data")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Ratio of data to use for training")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Ratio of data to use for validation")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Ratio of data to use for testing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    prepare_training_data(
        input_file=args.input_file,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main() 