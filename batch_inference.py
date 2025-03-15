"""
Batch inference script for Bio-ChemTransformer model.
"""

import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from inference import load_model, prepare_input, generate_adr_text
from transformers import AutoTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def batch_inference(
    checkpoint_path: str,
    input_file: str,
    output_file: str,
    bio_clinical_bert_model: str = "emilyalsentzer/Bio_ClinicalBERT",
    chembert_model: str = "seyonec/ChemBERTa-zinc-base-v1",
    max_adr_length: int = 256,
    max_smiles_length: int = 256,
    max_generation_length: int = 128,
    batch_size: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Run batch inference on a dataset.
    
    Args:
        checkpoint_path: Path to model checkpoint
        input_file: Path to input CSV/JSON file with 'adr_text' and 'smiles' columns
        output_file: Path to save output predictions
        bio_clinical_bert_model: Bio_ClinicalBERT model name or path
        chembert_model: ChemBERT model name or path
        max_adr_length: Maximum ADR text length
        max_smiles_length: Maximum SMILES length
        max_generation_length: Maximum generation length
        batch_size: Batch size for processing
        device: Device to run inference on
    """
    # Load model
    model = load_model(
        checkpoint_path=checkpoint_path,
        bio_clinical_bert_model=bio_clinical_bert_model,
        chembert_model=chembert_model,
        device=device
    )
    
    # Initialize tokenizer for decoding
    adr_tokenizer = AutoTokenizer.from_pretrained(bio_clinical_bert_model)
    
    # Load input data
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    elif input_file.endswith('.json') or input_file.endswith('.jsonl'):
        df = pd.read_json(input_file, lines=input_file.endswith('.jsonl'))
    else:
        raise ValueError(f"Unsupported file format: {input_file}")
    
    # Check if required columns exist
    if 'adr_text' not in df.columns or 'smiles' not in df.columns:
        raise ValueError("Input file must contain 'adr_text' and 'smiles' columns")
    
    # Initialize results storage
    results = []
    
    # Process in batches
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch_df = df.iloc[i:i+batch_size]
        
        for _, row in batch_df.iterrows():
            try:
                # Prepare input
                inputs = prepare_input(
                    adr_text=row['adr_text'],
                    smiles=row['smiles'],
                    bio_clinical_bert_model=bio_clinical_bert_model,
                    chembert_model=chembert_model,
                    max_adr_length=max_adr_length,
                    max_smiles_length=max_smiles_length,
                    device=device
                )
                
                # Generate predictions
                generated_texts = generate_adr_text(
                    model=model,
                    inputs=inputs,
                    adr_tokenizer=adr_tokenizer,
                    max_length=max_generation_length
                )
                
                # Store results
                result = row.to_dict()
                result['generated_adr'] = generated_texts[0]  # Take first prediction
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing row {row.name}: {str(e)}")
                # Add error information to results
                result = row.to_dict()
                result['generated_adr'] = f"ERROR: {str(e)}"
                results.append(result)
    
    # Save results
    results_df = pd.DataFrame(results)
    
    # Determine output format based on extension
    if output_file.endswith('.csv'):
        results_df.to_csv(output_file, index=False)
    elif output_file.endswith('.json'):
        results_df.to_json(output_file, orient='records')
    elif output_file.endswith('.jsonl'):
        results_df.to_json(output_file, orient='records', lines=True)
    else:
        # Default to CSV
        results_df.to_csv(output_file, index=False)
    
    logger.info(f"Results saved to {output_file}")
    return results_df

def main():
    """Main function for batch inference."""
    parser = argparse.ArgumentParser(description='Run batch inference with Bio-ChemTransformer')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV/JSON file')
    parser.add_argument('--output', type=str, required=True, help='Path to output file')
    parser.add_argument('--bio_bert', type=str, default="emilyalsentzer/Bio_ClinicalBERT", 
                        help='Bio_ClinicalBERT model name or path')
    parser.add_argument('--chem_bert', type=str, default="seyonec/ChemBERTa-zinc-base-v1", 
                        help='ChemBERT model name or path')
    parser.add_argument('--max_adr_length', type=int, default=256, help='Maximum ADR text length')
    parser.add_argument('--max_smiles_length', type=int, default=256, help='Maximum SMILES length')
    parser.add_argument('--max_generation_length', type=int, default=128, 
                        help='Maximum generation length')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help='Device to run inference on')
    args = parser.parse_args()
    
    # Run batch inference
    batch_inference(
        checkpoint_path=args.checkpoint,
        input_file=args.input,
        output_file=args.output,
        bio_clinical_bert_model=args.bio_bert,
        chembert_model=args.chem_bert,
        max_adr_length=args.max_adr_length,
        max_smiles_length=args.max_smiles_length,
        max_generation_length=args.max_generation_length,
        batch_size=args.batch_size,
        device=args.device
    )

if __name__ == "__main__":
    main() 