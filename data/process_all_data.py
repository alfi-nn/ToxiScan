"""
Script to process all downloaded datasets and combine them into a single dataset.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import requests
from rdkit import Chem
import gzip
import json

# Import our DailyMed processor
from process_dailymed import process_dailymed_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, raw_data_dir: str, output_dir: str):
        """
        Initialize the data processor.
        
        Args:
            raw_data_dir: Directory containing raw data
            output_dir: Directory to save processed data
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_sider_data(self):
        """Process SIDER data files."""
        logger.info("Processing SIDER data...")
        sider_dir = self.raw_data_dir / 'sider'
        
        try:
            # Load drug names
            drug_names = pd.read_csv(
                sider_dir / 'drug_names.tsv',
                sep='\t',
                header=None,
                names=['compound_id', 'drug_name']
            )
            
            # Load structures with SMILES
            structures = pd.read_csv(
                sider_dir / 'structures.tsv',
                sep='\t',
                header=None,
                names=['compound_id', 'smiles']
            )
            
            # Load side effects
            try:
                # Try gzipped file first
                with gzip.open(sider_dir / 'meddra_all_se.tsv.gz', 'rt') as f:
                    side_effects = pd.read_csv(
                        f,
                        sep='\t',
                        header=None,
                        names=['compound_id', 'side_effect_name', 'umls_concept_id', 'concept_name']
                    )
            except (FileNotFoundError, IOError):
                # If not found, try uncompressed file
                side_effects = pd.read_csv(
                    sider_dir / 'meddra_all_se.tsv',
                    sep='\t',
                    header=None,
                    names=['compound_id', 'side_effect_name', 'umls_concept_id', 'concept_name']
                )
            
            # Check if structures.tsv exists and has data
            structures_file = sider_dir / 'structures.tsv'
            if not structures_file.exists() or structures_file.stat().st_size == 0:
                logger.warning("structures.tsv is empty or missing. Please run download_structures.py first.")
                return pd.DataFrame()
            
            # Merge data
            merged_data = (
                drug_names
                .merge(structures, on='compound_id', how='inner')
                .merge(side_effects, on='compound_id', how='inner')
            )
            
            # Group side effects by drug
            grouped_data = merged_data.groupby(['drug_name', 'smiles'])['concept_name'].agg(list).reset_index()
            grouped_data['adr_text'] = grouped_data['concept_name'].apply(lambda x: '. '.join(set(x)))
            grouped_data = grouped_data.drop('concept_name', axis=1)
            
            # Validate SMILES strings
            grouped_data['valid_smiles'] = grouped_data['smiles'].apply(lambda x: Chem.MolFromSmiles(x) is not None)
            grouped_data = grouped_data[grouped_data['valid_smiles']].drop('valid_smiles', axis=1)
            
            return grouped_data
            
        except Exception as e:
            logger.error(f"Error processing SIDER data: {str(e)}")
            return pd.DataFrame()
    
    def process_dailymed_data(self):
        """Process DailyMed data files."""
        logger.info("Processing DailyMed data...")
        dailymed_dir = self.raw_data_dir / 'dailymed'
        structures_file = self.raw_data_dir / 'sider' / 'structures.tsv'
        output_file = self.output_dir / 'dailymed_processed.json'
        
        # Check if input file exists
        input_file = dailymed_dir / 'adverse_reactions.csv'
        if not input_file.exists():
            logger.warning(f"DailyMed adverse reactions file not found: {input_file}")
            logger.info("Please run 'python data/download_dailymed.py' first")
            return pd.DataFrame()
        
        # Process DailyMed data
        processed_data = process_dailymed_data(
            input_file=str(input_file),
            structures_file=str(structures_file),
            output_file=str(output_file)
        )
        
        logger.info(f"Processed {len(processed_data)} DailyMed records")
        return processed_data
    
    def process_faers_data(self):
        """Process FAERS data files."""
        logger.info("Processing FAERS data...")
        faers_dir = self.raw_data_dir / 'faers'
        
        try:
            # Check for OpenFDA processed data first
            openfda_file = faers_dir / 'faers_raw.csv'
            if openfda_file.exists():
                df = pd.read_csv(openfda_file)
                
                # Get SMILES using PubChem API
                def get_smiles_from_pubchem(drug_name: str) -> str:
                    try:
                        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/IsomericSMILES/JSON"
                        response = requests.get(url)
                        data = response.json()
                        smiles = data['PropertyTable']['Properties'][0]['IsomericSMILES']
                        # Validate SMILES
                        if Chem.MolFromSmiles(smiles):
                            return smiles
                        return None
                    except:
                        return None
                
                if 'smiles' not in df.columns:
                    logger.info("Fetching SMILES strings from PubChem...")
                    tqdm.pandas()
                    df['smiles'] = df['drug_name'].progress_apply(get_smiles_from_pubchem)
                    df = df.dropna(subset=['smiles'])
                    df.to_csv(openfda_file, index=False)
                
                return df
            
            logger.warning("No processed FAERS data found")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error processing FAERS data: {str(e)}")
            return pd.DataFrame()
    
    def combine_datasets(self):
        """Combine all processed datasets."""
        logger.info("Combining all datasets...")
        
        # Process individual datasets
        sider_data = self.process_sider_data()
        dailymed_data = self.process_dailymed_data()
        faers_data = self.process_faers_data()
        
        # Combine all datasets
        all_data = pd.concat([sider_data, dailymed_data, faers_data], ignore_index=True)
        
        # Drop duplicates based on drug_name and adr_text
        all_data = all_data.drop_duplicates(subset=['drug_name', 'adr_text'])
        
        # Remove rows with missing SMILES or ADR text
        all_data = all_data.dropna(subset=['smiles', 'adr_text'])
        all_data = all_data[all_data['smiles'] != '']
        all_data = all_data[all_data['adr_text'] != '']
        
        # Save combined dataset
        output_file = self.output_dir / 'all_data.json'
        all_data.to_json(output_file, orient='records', lines=True)
        
        logger.info(f"Combined dataset has {len(all_data)} records")
        logger.info(f"Saved combined dataset to {output_file}")
        
        return all_data


def main():
    """Main function."""
    processor = DataProcessor(
        raw_data_dir='data/raw',
        output_dir='data/processed'
    )
    
    processor.combine_datasets()


if __name__ == "__main__":
    main() 