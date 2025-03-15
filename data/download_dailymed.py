"""
Download DailyMed adverse reactions using the DailyMed API.
"""

import os
import requests
import pandas as pd
import time
import xml.etree.ElementTree as ET
from tqdm import tqdm
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DailyMedDownloader:
    def __init__(self, output_dir='data/raw/dailymed'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://dailymed.nlm.nih.gov/dailymed/services"
        
    def download_drug_list(self, limit=500):
        """Download list of drug products from DailyMed."""
        logger.info(f"Downloading list of drug products (limit: {limit})...")
        
        # API endpoint for drug listings
        url = f"{self.base_url}/v2/spls.json?limit={limit}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            # Extract drug information
            drug_list = []
            for item in data.get('data', []):
                drug_list.append({
                    'setid': item.get('setid', ''),
                    'product_name': item.get('product_name', ''),
                    'spl_version': item.get('spl_version', '')
                })
            
            # Save to file
            drug_df = pd.DataFrame(drug_list)
            output_file = self.output_dir / 'drug_list.csv'
            drug_df.to_csv(output_file, index=False)
            
            logger.info(f"Downloaded {len(drug_list)} drug products to {output_file}")
            return drug_df
            
        except Exception as e:
            logger.error(f"Error downloading drug list: {str(e)}")
            return pd.DataFrame()
    
    def extract_adverse_reactions(self, setid):
        """Extract adverse reactions for a specific drug."""
        # API endpoint for drug SPL
        url = f"{self.base_url}/v2/spls/{setid}.xml"
        
        try:
            response = requests.get(url)
            root = ET.fromstring(response.content)
            
            # Define XML namespaces
            namespaces = {
                'v3': 'urn:hl7-org:v3'
            }
            
            # Extract product name
            product_elem = root.find(".//v3:manufacturedProduct/v3:manufacturedMedicine/v3:name", namespaces)
            product_name = ""
            if product_elem is not None and product_elem.text:
                product_name = product_elem.text
            
            # Extract adverse reactions section (LOINC code 34084-4 is for Adverse Reactions)
            adverse_section = root.find(".//v3:section[v3:code[@code='34084-4']]", namespaces)
            adverse_text = ""
            
            if adverse_section is not None:
                # Try to get text from text elements
                text_elements = adverse_section.findall(".//v3:text", namespaces)
                text_parts = []
                for elem in text_elements:
                    if elem.text and elem.text.strip():
                        text_parts.append(elem.text.strip())
                
                adverse_text = ' '.join(text_parts)
                
                # If no text found, try to get all content as string
                if not adverse_text:
                    adverse_text = ET.tostring(adverse_section, encoding='unicode')
            
            # Also check for BOXED WARNING section (LOINC code 34066-1)
            boxed_section = root.find(".//v3:section[v3:code[@code='34066-1']]", namespaces)
            boxed_text = ""
            
            if boxed_section is not None:
                text_elements = boxed_section.findall(".//v3:text", namespaces)
                text_parts = []
                for elem in text_elements:
                    if elem.text and elem.text.strip():
                        text_parts.append(elem.text.strip())
                
                boxed_text = ' '.join(text_parts)
                
                # If no text found, try to get all content as string
                if not boxed_text:
                    boxed_text = ET.tostring(boxed_section, encoding='unicode')
            
            return product_name, adverse_text, boxed_text
            
        except Exception as e:
            logger.warning(f"Error extracting information for {setid}: {str(e)}")
            return "", "", ""
    
    def download_adverse_reactions(self, drug_list=None, limit=200):
        """Download adverse reactions for drugs."""
        logger.info("Downloading adverse reactions...")
        
        # Load or download drug list
        if drug_list is None:
            drug_list_file = self.output_dir / 'drug_list.csv'
            if drug_list_file.exists():
                drug_list = pd.read_csv(drug_list_file)
                logger.info(f"Loaded existing drug list with {len(drug_list)} entries")
            else:
                drug_list = self.download_drug_list(limit=limit)
        
        # Extract adverse reactions for each drug
        adverse_reactions = []
        
        for _, row in tqdm(drug_list.iterrows(), total=min(len(drug_list), limit), desc="Downloading adverse reactions"):
            setid = row['setid']
            
            # Extract product information and adverse reactions
            product_name, adverse_text, boxed_text = self.extract_adverse_reactions(setid)
            
            # Add to list if information found
            if product_name or adverse_text or boxed_text:
                adverse_reactions.append({
                    'setid': setid,
                    'product_name': product_name or row['product_name'],  # Fallback to the name from drug list
                    'adverse_reactions': adverse_text,
                    'boxed_warnings': boxed_text
                })
            
            # Rate limiting to be respectful to the API
            time.sleep(0.5)
            
            # Stop after reaching limit
            if len(adverse_reactions) >= limit:
                break
        
        # Save to file
        adverse_df = pd.DataFrame(adverse_reactions)
        output_file = self.output_dir / 'adverse_reactions.csv'
        adverse_df.to_csv(output_file, index=False)
        
        logger.info(f"Downloaded {len(adverse_reactions)} adverse reactions to {output_file}")
        return adverse_df

def main():
    downloader = DailyMedDownloader()
    # Step 1: Get drug list (will only download if not already present)
    drug_list = downloader.download_drug_list(limit=500)
    # Step 2: Download adverse reactions for 200 drugs
    downloader.download_adverse_reactions(drug_list=drug_list, limit=200)

if __name__ == "__main__":
    main() 