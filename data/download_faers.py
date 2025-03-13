"""
Script to download FAERS data using OpenFDA API.
"""

import os
import requests
import json
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FAERSDownloader:
    def __init__(self, output_dir: str, api_key: str = None):
        """
        Initialize the FAERS downloader.
        
        Args:
            output_dir: Directory to save downloaded data
            api_key: OpenFDA API key (optional but recommended)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key
        self.base_url = "https://api.fda.gov/drug/event.json"
    
    def download_from_openfda(self, limit: int = 1000):
        """
        Download FAERS data using OpenFDA API.
        
        Args:
            limit: Number of records to download (max 1000 per request)
        """
        logger.info("Downloading FAERS data from OpenFDA API...")
        
        params = {
            'limit': min(limit, 1000),
            'api_key': self.api_key
        }
        
        if not self.api_key:
            logger.warning("No API key provided. Requests will be rate-limited.")
        
        results = []
        total_downloaded = 0
        
        try:
            while total_downloaded < limit:
                params['skip'] = total_downloaded
                response = requests.get(self.base_url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    results.extend(data['results'])
                    total_downloaded += len(data['results'])
                    logger.info(f"Downloaded {total_downloaded} records")
                    
                    # Rate limiting
                    time.sleep(0.1)  # Respect API limits
                else:
                    logger.error(f"Error downloading data: {response.status_code}")
                    break
            
            # Process downloaded data
            processed_data = []
            for record in tqdm(results, desc="Processing records"):
                try:
                    # Extract drug information
                    for drug in record.get('patient', {}).get('drug', []):
                        if drug.get('medicinalproduct'):
                            # Extract reactions
                            reactions = [
                                r.get('reactionmeddrapt', '')
                                for r in record.get('patient', {}).get('reaction', [])
                            ]
                            
                            processed_data.append({
                                'drug_name': drug['medicinalproduct'],
                                'adr_text': '. '.join(filter(None, reactions))
                            })
                except Exception as e:
                    logger.warning(f"Error processing record: {str(e)}")
            
            # Convert to DataFrame and save
            df = pd.DataFrame(processed_data)
            df = df.groupby('drug_name')['adr_text'].agg(lambda x: '. '.join(set(x.str.split('. ').sum()))).reset_index()
            
            output_file = self.output_dir / 'faers_raw.csv'
            df.to_csv(output_file, index=False)
            logger.info(f"Saved raw FAERS data to {output_file}")
            
        except Exception as e:
            logger.error(f"Error downloading FAERS data: {str(e)}")
    
    def download_quarterly_files(self, year: int, quarter: int):
        """
        Download quarterly FAERS files from FDA website.
        
        Args:
            year: Year (e.g., 2023)
            quarter: Quarter (1-4)
        """
        logger.info(f"Downloading FAERS quarterly files for {year}Q{quarter}...")
        
        base_url = "https://fis.fda.gov/content/Exports/faers_ascii_"
        year_short = str(year)[-2:]
        files = ['DEMO', 'DRUG', 'REAC']
        
        for file in files:
            filename = f"{file}{year_short}Q{quarter}.txt"
            url = f"{base_url}{filename}"
            output_file = self.output_dir / filename
            
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    total_size = int(response.headers.get('content-length', 0))
                    
                    with open(output_file, 'wb') as f, tqdm(
                        desc=filename,
                        total=total_size,
                        unit='iB',
                        unit_scale=True
                    ) as pbar:
                        for data in response.iter_content(chunk_size=1024):
                            size = f.write(data)
                            pbar.update(size)
                    
                    logger.info(f"Downloaded {filename}")
                else:
                    logger.error(f"Error downloading {filename}: {response.status_code}")
            
            except Exception as e:
                logger.error(f"Error downloading {filename}: {str(e)}")


def main():
    """Main function."""
    # You can get an API key from https://open.fda.gov/apis/authentication/
    api_key = os.getenv('OPENFDA_API_KEY')
    
    downloader = FAERSDownloader(
        output_dir='data/raw/faers',
        api_key=api_key
    )
    
    # Method 1: Download using OpenFDA API
    downloader.download_from_openfda(limit=5000)  # Adjust limit as needed
    
    # Method 2: Download quarterly files
    # Uncomment and modify as needed
    # downloader.download_quarterly_files(year=2023, quarter=4)


if __name__ == "__main__":
    main() 