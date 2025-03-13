"""
Script to download SMILES structures for drug compounds in SIDER.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import requests
import time
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StructuresDownloader:
    def __init__(self, output_dir: str):
        """
        Initialize the structures downloader.
        
        Args:
            output_dir: Directory to save structures.tsv file
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_from_pubchem(self, drug_names_file: str):
        """
        Download SMILES structures from PubChem API.
        
        Args:
            drug_names_file: Path to drug_names.tsv file
        """
        logger.info("Downloading SMILES structures from PubChem...")
        
        # Load drug names
        drug_names = pd.read_csv(
            drug_names_file,
            sep='\t',
            header=None,
            names=['compound_id', 'drug_name']
        )
        
        # Create a function to get SMILES from PubChem
        def get_smiles_from_pubchem(drug_name: str) -> str:
            try:
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/IsomericSMILES/JSON"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    return data['PropertyTable']['Properties'][0]['IsomericSMILES']
                return None
            except Exception as e:
                logger.warning(f"Error fetching SMILES for {drug_name}: {str(e)}")
                return None
        
        # Download SMILES for each drug
        structures = []
        
        for _, row in tqdm(drug_names.iterrows(), total=len(drug_names), desc="Downloading SMILES"):
            compound_id = row['compound_id']
            drug_name = row['drug_name']
            
            # Get SMILES
            smiles = get_smiles_from_pubchem(drug_name)
            
            if smiles:
                structures.append([compound_id, smiles])
            
            # Rate limiting
            time.sleep(0.2)  # Respect PubChem API limits
        
        # Create structures DataFrame
        structures_df = pd.DataFrame(structures, columns=['compound_id', 'smiles'])
        
        # Save to structures.tsv
        output_file = self.output_dir / 'structures.tsv'
        structures_df.to_csv(output_file, sep='\t', header=False, index=False)
        
        logger.info(f"Successfully downloaded {len(structures_df)} SMILES structures")
        logger.info(f"Saved to {output_file}")


def main():
    """Main function."""
    downloader = StructuresDownloader(
        output_dir='data/raw/sider'
    )
    
    # Replace with the path to your drug_names.tsv file
    drug_names_file = 'data/raw/sider/drug_names.tsv'
    
    downloader.download_from_pubchem(drug_names_file)


if __name__ == "__main__":
    main() 