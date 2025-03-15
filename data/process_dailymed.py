"""
Process DailyMed adverse reactions data.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')


def clean_text(text):
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Replace XML tags and entities
    import re
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    
    # Replace multiple spaces with single space
    text = ' '.join(text.split())
    
    return text.strip()


def match_drug_with_smiles(drug_name, structures_df):
    """Match a drug name with its SMILES structure."""
    # Try exact match
    exact_match = structures_df[structures_df['drug_name'].str.lower() == drug_name.lower()]
    if not exact_match.empty:
        return exact_match.iloc[0]['smiles']
    
    # Try partial match
    partial_matches = structures_df[structures_df['drug_name'].str.lower().str.contains(drug_name.lower())]
    if not partial_matches.empty:
        return partial_matches.iloc[0]['smiles']
    
    # Try matching drug name in parts (e.g., "Metformin Hydrochloride" -> "Metformin")
    parts = drug_name.split()
    if len(parts) > 1:
        first_part = parts[0].lower()
        first_part_matches = structures_df[structures_df['drug_name'].str.lower().str.contains(first_part)]
        if not first_part_matches.empty:
            return first_part_matches.iloc[0]['smiles']
    
    return ""


def process_dailymed_data(input_file, structures_file, output_file):
    """
    Process DailyMed adverse reactions data and combine with drug structures.
    
    Args:
        input_file: Path to DailyMed adverse reactions CSV file
        structures_file: Path to drug structures file with SMILES
        output_file: Path to save processed data
    """
    logger.info(f"Processing DailyMed adverse reactions from {input_file}")
    
    # Load adverse reactions data
    try:
        dailymed_df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(dailymed_df)} records from DailyMed")
    except Exception as e:
        logger.error(f"Error loading DailyMed data: {str(e)}")
        return pd.DataFrame()
    
    # Load drug structures
    try:
        structures_df = pd.read_csv(structures_file, sep='\t', header=None, names=['drug_id', 'smiles'])
        
        # If drug_id is a SIDER ID, load drug_names.tsv to get names
        drug_names_file = os.path.join(os.path.dirname(structures_file), 'drug_names.tsv')
        if os.path.exists(drug_names_file):
            drug_names_df = pd.read_csv(drug_names_file, sep='\t', header=None, names=['drug_id', 'drug_name'])
            structures_df = pd.merge(structures_df, drug_names_df, on='drug_id', how='left')
        else:
            # If no drug names file, assume drug_id is the drug name
            structures_df['drug_name'] = structures_df['drug_id']
        
        logger.info(f"Loaded {len(structures_df)} drug structures")
    except Exception as e:
        logger.error(f"Error loading structures data: {str(e)}")
        structures_df = pd.DataFrame(columns=['drug_id', 'drug_name', 'smiles'])
    
    # Process and clean data
    processed_data = []
    
    for _, row in tqdm(dailymed_df.iterrows(), total=len(dailymed_df), desc="Processing DailyMed data"):
        try:
            # Extract and clean data
            drug_name = row.get('product_name', '').strip()
            adverse_text = clean_text(row.get('adverse_reactions', ''))
            boxed_warnings = clean_text(row.get('boxed_warnings', ''))
            
            # Skip if missing adverse reactions
            if not adverse_text:
                continue
            
            # Get SMILES structure
            smiles = match_drug_with_smiles(drug_name, structures_df)
            
            # Combine adverse reactions and boxed warnings
            combined_text = adverse_text
            if boxed_warnings:
                combined_text = f"{boxed_warnings}. {combined_text}"
            
            # Store processed data
            processed_data.append({
                'drug_name': drug_name,
                'smiles': smiles,
                'adr_text': combined_text,
                'source': 'dailymed'
            })
        
        except Exception as e:
            logger.warning(f"Error processing row: {str(e)}")
    
    # Create processed dataframe
    processed_df = pd.DataFrame(processed_data)
    logger.info(f"Processed {len(processed_df)} valid records from DailyMed")
    
    # Save to output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    processed_df.to_json(output_file, orient='records', lines=True)
    logger.info(f"Saved processed data to {output_file}")
    
    return processed_df


def main():
    # Set paths
    input_file = 'data/raw/dailymed/adverse_reactions.csv'
    structures_file = 'data/raw/sider/structures.tsv'
    output_file = 'data/processed/dailymed_processed.json'
    
    # Process data
    process_dailymed_data(input_file, structures_file, output_file)


if __name__ == "__main__":
    main() 