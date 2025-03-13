"""
Script to convert downloaded raw data into the required format for Bio-ChemTransformer.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Set
import xml.etree.ElementTree as ET
import gzip
import json
import logging
from tqdm import tqdm
import requests
from rdkit import Chem
from rdkit.Chem import PandasTools

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataConverter:
    def __init__(self, raw_data_dir: str, output_dir: str):
        """
        Initialize the data converter.
        
        Args:
            raw_data_dir: Directory containing downloaded raw data
            output_dir: Directory to save converted CSV files
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_sider_data(self):
        """Process SIDER data files."""
        logger.info("Processing SIDER data...")
        
        # Load drug names
        drug_names = pd.read_csv(
            self.raw_data_dir / 'drug_names.tsv',
            sep='\t',
            header=None,
            names=['compound_id', 'drug_name']
        )
        
        # Load structures with SMILES
        structures = pd.read_csv(
            self.raw_data_dir / 'structures.tsv',
            sep='\t',
            header=None,
            names=['compound_id', 'smiles']
        )
        
        # Load side effects
        with gzip.open(self.raw_data_dir / 'meddra_all_se.tsv.gz', 'rt') as f:
            side_effects = pd.read_csv(
                f,
                sep='\t',
                header=None,
                names=['compound_id', 'side_effect_name', 'umls_concept_id', 'concept_name']
            )
        
        # Merge data
        merged_data = (
            drug_names
            .merge(structures, on='compound_id', how='inner')
            .merge(side_effects, on='compound_id', how='inner')
        )
        
        # Group side effects by drug
        grouped_data = merged_data.groupby(['drug_name', 'smiles'])['concept_name'].agg(list).reset_index()
        
        # Convert side effects list to text
        grouped_data['adr_text'] = grouped_data['concept_name'].apply(lambda x: '. '.join(set(x)))
        
        # Save processed data
        output_file = self.output_dir / 'sider.csv'
        grouped_data[['drug_name', 'smiles', 'adr_text']].to_csv(output_file, index=False)
        logger.info(f"Saved processed SIDER data to {output_file}")
    
    def process_dailymed_data(self):
        """Process DailyMed data files."""
        logger.info("Processing DailyMed data...")
        
        # Function to extract info from SPL XML
        def extract_spl_info(xml_file: Path) -> Dict:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Extract drug name
                drug_name = root.find(".//name").text
                
                # Extract adverse reactions
                adr_sections = root.findall(".//section[code/@code='34084-4']")
                adr_text = []
                for section in adr_sections:
                    text_elements = section.findall(".//text")
                    adr_text.extend([elem.text for elem in text_elements if elem.text])
                
                return {
                    'drug_name': drug_name,
                    'adr_text': ' '.join(adr_text)
                }
            except Exception as e:
                logger.warning(f"Error processing {xml_file}: {str(e)}")
                return None
        
        # Process all SPL files
        spl_dir = self.raw_data_dir / 'dm_spl_release_human_rx'
        results = []
        
        for xml_file in tqdm(list(spl_dir.glob('**/*.xml')), desc="Processing SPL files"):
            info = extract_spl_info(xml_file)
            if info:
                results.append(info)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Get SMILES using PubChem API
        def get_smiles_from_pubchem(drug_name: str) -> str:
            try:
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/IsomericSMILES/JSON"
                response = requests.get(url)
                data = response.json()
                return data['PropertyTable']['Properties'][0]['IsomericSMILES']
            except:
                return None
        
        df['smiles'] = df['drug_name'].progress_map(get_smiles_from_pubchem)
        
        # Remove entries without SMILES
        df = df.dropna(subset=['smiles'])
        
        # Save processed data
        output_file = self.output_dir / 'dailymed.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"Saved processed DailyMed data to {output_file}")
    
    def process_faers_data(self):
        """Process FAERS data files."""
        logger.info("Processing FAERS data...")
        
        # Function to load FAERS file
        def load_faers_file(filename: str) -> pd.DataFrame:
            return pd.read_csv(
                filename,
                delimiter='$',
                dtype=str,
                encoding='latin1'
            )
        
        # Process each quarter
        all_results = []
        
        for quarter_dir in self.raw_data_dir.glob('faers_*'):
            try:
                # Load quarterly files
                demo = load_faers_file(quarter_dir / 'DEMO*.txt')
                drug = load_faers_file(quarter_dir / 'DRUG*.txt')
                reac = load_faers_file(quarter_dir / 'REAC*.txt')
                
                # Merge data
                merged = (
                    drug[['primaryid', 'drugname']]
                    .merge(reac[['primaryid', 'pt']], on='primaryid', how='inner')
                    .groupby('drugname')['pt']
                    .agg(list)
                    .reset_index()
                )
                
                all_results.append(merged)
            
            except Exception as e:
                logger.warning(f"Error processing {quarter_dir}: {str(e)}")
        
        # Combine all quarters
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Group reactions by drug
        grouped = combined_df.groupby('drugname')['pt'].agg(lambda x: list(set(sum(x, [])))).reset_index()
        grouped.columns = ['drug_name', 'adr_text']
        
        # Convert reaction lists to text
        grouped['adr_text'] = grouped['adr_text'].apply(lambda x: '. '.join(x))
        
        # Get SMILES using PubChem API (same function as above)
        def get_smiles_from_pubchem(drug_name: str) -> str:
            try:
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/IsomericSMILES/JSON"
                response = requests.get(url)
                data = response.json()
                return data['PropertyTable']['Properties'][0]['IsomericSMILES']
            except:
                return None
        
        grouped['smiles'] = grouped['drug_name'].progress_map(get_smiles_from_pubchem)
        
        # Remove entries without SMILES
        grouped = grouped.dropna(subset=['smiles'])
        
        # Save processed data
        output_file = self.output_dir / 'faers.csv'
        grouped.to_csv(output_file, index=False)
        logger.info(f"Saved processed FAERS data to {output_file}")
    
    def convert_all(self):
        """Convert all datasets."""
        self.process_sider_data()
        self.process_dailymed_data()
        self.process_faers_data()
        logger.info("All data conversion complete!")


def main():
    """Main function."""
    converter = DataConverter(
        raw_data_dir='data/raw',
        output_dir='data/raw'
    )
    converter.convert_all()


if __name__ == "__main__":
    main() 