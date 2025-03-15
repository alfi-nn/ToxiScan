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
from rdkit import Chem
from rdkit import RDLogger

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

class StructuresDownloader:
    def __init__(self, output_dir: str):
        """
        Initialize the structures downloader.
        
        Args:
            output_dir: Directory to save structures.tsv file
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def canonicalize_smiles(self, smiles: str) -> str:
        """Canonicalize SMILES string."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            return None
        except:
            return None
    
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
                
                # Try alternative API if first attempt fails
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/CanonicalSMILES/JSON"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    return data['PropertyTable']['Properties'][0]['CanonicalSMILES']
                
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
            
            # Try with variations of the drug name if not found
            if not smiles and ' ' in drug_name:
                first_word = drug_name.split(' ')[0]
                smiles = get_smiles_from_pubchem(first_word)
            
            # Canonicalize SMILES if found
            if smiles:
                canonical_smiles = self.canonicalize_smiles(smiles)
                if canonical_smiles:
                    structures.append([compound_id, canonical_smiles])
                    logger.debug(f"Found SMILES for {drug_name}: {canonical_smiles}")
                else:
                    logger.warning(f"Could not canonicalize SMILES for {drug_name}: {smiles}")
            else:
                logger.warning(f"No SMILES found for drug: {drug_name}")
            
            # Rate limiting
            time.sleep(0.3)  # Respect PubChem API limits
        
        # Create structures DataFrame
        structures_df = pd.DataFrame(structures, columns=['compound_id', 'smiles'])
        
        # Save to structures.tsv
        output_file = self.output_dir / 'structures.tsv'
        structures_df.to_csv(output_file, sep='\t', header=False, index=False)
        
        logger.info(f"Successfully downloaded {len(structures_df)} SMILES structures")
        logger.info(f"Saved to {output_file}")
        
        # Display statistics
        coverage = (len(structures_df) / len(drug_names)) * 100
        logger.info(f"Coverage: {coverage:.1f}% of drugs have SMILES")
        
        return structures_df
    
    def download_from_chembl(self, drug_names_file: str):
        """
        Download SMILES structures from ChEMBL API as a backup.
        
        Args:
            drug_names_file: Path to drug_names.tsv file
        """
        logger.info("Downloading SMILES structures from ChEMBL API...")
        
        # Load drug names
        drug_names = pd.read_csv(
            drug_names_file,
            sep='\t',
            header=None,
            names=['compound_id', 'drug_name']
        )
        
        # Function to get SMILES from ChEMBL
        def get_smiles_from_chembl(drug_name: str) -> str:
            try:
                url = f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?molecule_synonyms__molecule_synonym__iexact={drug_name}"
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    if data["molecules"] and len(data["molecules"]) > 0:
                        return data["molecules"][0]["molecule_structures"]["canonical_smiles"]
                
                return None
            except Exception as e:
                logger.warning(f"Error fetching SMILES from ChEMBL for {drug_name}: {str(e)}")
                return None
        
        # Try to get SMILES for drugs that don't have them yet
        existing_file = self.output_dir / 'structures.tsv'
        existing_structures = pd.DataFrame()
        
        if existing_file.exists():
            try:
                existing_structures = pd.read_csv(
                    existing_file, 
                    sep='\t', 
                    header=None, 
                    names=['compound_id', 'smiles']
                )
                logger.info(f"Loaded {len(existing_structures)} existing structures")
            except:
                logger.warning(f"Could not load existing structures file: {existing_file}")
        
        # Create set of compounds that already have structures
        existing_compounds = set(existing_structures['compound_id'].values) if not existing_structures.empty else set()
        
        # Download SMILES for each drug that doesn't have one yet
        new_structures = []
        
        for _, row in tqdm(drug_names.iterrows(), total=len(drug_names), desc="Downloading from ChEMBL"):
            compound_id = row['compound_id']
            drug_name = row['drug_name']
            
            # Skip if we already have this compound
            if compound_id in existing_compounds:
                continue
            
            # Get SMILES
            smiles = get_smiles_from_chembl(drug_name)
            
            if smiles:
                canonical_smiles = self.canonicalize_smiles(smiles)
                if canonical_smiles:
                    new_structures.append([compound_id, canonical_smiles])
            
            # Rate limiting
            time.sleep(0.3)
        
        # Combine with existing structures
        if not existing_structures.empty:
            new_structures_df = pd.DataFrame(new_structures, columns=['compound_id', 'smiles'])
            combined_df = pd.concat([existing_structures, new_structures_df])
            combined_df = combined_df.drop_duplicates(subset=['compound_id'])
            
            # Save combined structures
            output_file = self.output_dir / 'structures.tsv'
            combined_df.to_csv(output_file, sep='\t', header=False, index=False)
            
            logger.info(f"Added {len(new_structures)} new structures from ChEMBL")
            logger.info(f"Total structures: {len(combined_df)}")
            
            return combined_df
        
        else:
            # If no existing structures, save new ones
            new_structures_df = pd.DataFrame(new_structures, columns=['compound_id', 'smiles'])
            output_file = self.output_dir / 'structures.tsv'
            new_structures_df.to_csv(output_file, sep='\t', header=False, index=False)
            
            logger.info(f"Downloaded {len(new_structures_df)} structures from ChEMBL")
            
            return new_structures_df


def main():
    """Main function."""
    downloader = StructuresDownloader(
        output_dir='data/raw/sider'
    )
    
    # Path to drug_names.tsv file
    drug_names_file = 'data/raw/sider/drug_names.tsv'
    
    # First try PubChem
    structures_df = downloader.download_from_pubchem(drug_names_file)
    
    # Then try ChEMBL for any missing structures
    if len(structures_df) < pd.read_csv(drug_names_file, sep='\t', header=None).shape[0]:
        downloader.download_from_chembl(drug_names_file)


if __name__ == "__main__":
    main() 