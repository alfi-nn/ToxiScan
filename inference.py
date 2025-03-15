"""
Inference script for Bio-ChemTransformer model.
"""

import os
import torch
import argparse
from models.transformer import BioChemTransformer
from utils.data_utils import BioChemDataset
from transformers import AutoTokenizer
from utils.smiles_tokenization import SMILESTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(
    checkpoint_path: str,
    bio_clinical_bert_model: str = "emilyalsentzer/Bio_ClinicalBERT",
    chembert_model: str = "seyonec/ChemBERTa-zinc-base-v1",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> BioChemTransformer:
    """
    Load the trained Bio-ChemTransformer model.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        bio_clinical_bert_model: Bio_ClinicalBERT model name or path
        chembert_model: ChemBERT model name or path
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")
    
    # Initialize the tokenizer to get vocab size
    tokenizer = AutoTokenizer.from_pretrained(bio_clinical_bert_model)
    vocab_size = len(tokenizer)
    
    # Initialize the model
    model = BioChemTransformer(
        bio_clinical_bert_model=bio_clinical_bert_model,
        chembert_model=chembert_model,
        vocab_size=vocab_size
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Move model to device
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    return model

def prepare_input(
    adr_text: str,
    smiles: str,
    bio_clinical_bert_model: str = "emilyalsentzer/Bio_ClinicalBERT",
    chembert_model: str = "seyonec/ChemBERTa-zinc-base-v1",
    max_adr_length: int = 256,
    max_smiles_length: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> dict:
    """
    Prepare input for the model from ADR text and SMILES string.
    
    Args:
        adr_text: ADR text
        smiles: SMILES string
        bio_clinical_bert_model: Bio_ClinicalBERT model name or path
        chembert_model: ChemBERT model name or path
        max_adr_length: Maximum length for ADR text tokens
        max_smiles_length: Maximum length for SMILES tokens
        device: Device to load tensors on
        
    Returns:
        Dictionary with input tensors
    """
    # Initialize tokenizers
    adr_tokenizer = AutoTokenizer.from_pretrained(bio_clinical_bert_model)
    smiles_tokenizer = SMILESTokenizer(
        chembert_model=chembert_model,
        max_length=max_smiles_length,
        canonicalize=True
    )
    
    # Tokenize ADR text
    adr_tokens = adr_tokenizer(
        adr_text,
        padding='max_length',
        truncation=True,
        max_length=max_adr_length,
        return_tensors='pt'
    )
    
    # Tokenize SMILES
    smiles_tokens = smiles_tokenizer.tokenize(
        [smiles],
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Prepare inputs
    inputs = {
        'adr_input_ids': adr_tokens['input_ids'].to(device),
        'adr_attention_mask': adr_tokens['attention_mask'].to(device),
        'smiles_input_ids': smiles_tokens['input_ids'].to(device),
        'smiles_attention_mask': smiles_tokens['attention_mask'].to(device)
    }
    
    return inputs

def generate_adr_text(
    model: BioChemTransformer,
    inputs: dict,
    adr_tokenizer: AutoTokenizer,
    max_length: int = 128,
    min_length: int = 5,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    num_return_sequences: int = 1
) -> list:
    """
    Generate ADR text using the model.
    
    Args:
        model: Bio-ChemTransformer model
        inputs: Input tensors
        adr_tokenizer: Tokenizer for ADR text
        max_length: Maximum length of generated text
        min_length: Minimum length of generated text
        temperature: Temperature for sampling
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        repetition_penalty: Penalty for repetition
        num_return_sequences: Number of sequences to generate
        
    Returns:
        List of generated texts
    """
    with torch.no_grad():
        # Generate token IDs
        generated_ids = model.generate(
            adr_input_ids=inputs['adr_input_ids'],
            adr_attention_mask=inputs['adr_attention_mask'],
            smiles_input_ids=inputs['smiles_input_ids'],
            smiles_attention_mask=inputs['smiles_attention_mask'],
            max_length=max_length,
            min_length=min_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences
        )
    
    # Convert token IDs to text
    generated_texts = []
    for ids in generated_ids:
        # Remove padding and special tokens
        ids = ids[ids != model.pad_token_id]
        if model.eos_token_id in ids:
            # Truncate after EOS
            ids = ids[:torch.where(ids == model.eos_token_id)[0][0]]
        
        # Decode to text
        text = adr_tokenizer.decode(ids, skip_special_tokens=True)
        generated_texts.append(text)
    
    return generated_texts

def run_inference(
    checkpoint_path: str,
    adr_text: str,
    smiles: str,
    bio_clinical_bert_model: str = "emilyalsentzer/Bio_ClinicalBERT",
    chembert_model: str = "seyonec/ChemBERTa-zinc-base-v1",
    max_adr_length: int = 256,
    max_smiles_length: int = 256,
    max_generation_length: int = 128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> list:
    """
    Run inference using the Bio-ChemTransformer model.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        adr_text: ADR text input
        smiles: SMILES string input
        bio_clinical_bert_model: Bio_ClinicalBERT model name or path
        chembert_model: ChemBERT model name or path
        max_adr_length: Maximum length for ADR text tokens
        max_smiles_length: Maximum length for SMILES tokens
        max_generation_length: Maximum length for generated text
        device: Device to run inference on
        
    Returns:
        List of generated texts
    """
    # Load model
    model = load_model(
        checkpoint_path=checkpoint_path,
        bio_clinical_bert_model=bio_clinical_bert_model,
        chembert_model=chembert_model,
        device=device
    )
    
    # Prepare input
    inputs = prepare_input(
        adr_text=adr_text,
        smiles=smiles,
        bio_clinical_bert_model=bio_clinical_bert_model,
        chembert_model=chembert_model,
        max_adr_length=max_adr_length,
        max_smiles_length=max_smiles_length,
        device=device
    )
    
    # Initialize tokenizer for decoding
    adr_tokenizer = AutoTokenizer.from_pretrained(bio_clinical_bert_model)
    
    # Generate text
    generated_texts = generate_adr_text(
        model=model,
        inputs=inputs,
        adr_tokenizer=adr_tokenizer,
        max_length=max_generation_length
    )
    
    return generated_texts

def main():
    """Main function for inference."""
    parser = argparse.ArgumentParser(description='Run inference with Bio-ChemTransformer')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--adr_text', type=str, required=True, help='ADR text input')
    parser.add_argument('--smiles', type=str, required=True, help='SMILES string input')
    parser.add_argument('--bio_bert', type=str, default="emilyalsentzer/Bio_ClinicalBERT", 
                        help='Bio_ClinicalBERT model name or path')
    parser.add_argument('--chem_bert', type=str, default="seyonec/ChemBERTa-zinc-base-v1", 
                        help='ChemBERT model name or path')
    parser.add_argument('--max_adr_length', type=int, default=256, help='Maximum ADR text length')
    parser.add_argument('--max_smiles_length', type=int, default=256, help='Maximum SMILES length')
    parser.add_argument('--max_generation_length', type=int, default=128, 
                        help='Maximum generation length')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help='Device to run inference on')
    args = parser.parse_args()
    
    # Run inference
    generated_texts = run_inference(
        checkpoint_path=args.checkpoint,
        adr_text=args.adr_text,
        smiles=args.smiles,
        bio_clinical_bert_model=args.bio_bert,
        chembert_model=args.chem_bert,
        max_adr_length=args.max_adr_length,
        max_smiles_length=args.max_smiles_length,
        max_generation_length=args.max_generation_length,
        device=args.device
    )
    
    # Print results
    print("\nGenerated ADR Text:")
    for i, text in enumerate(generated_texts):
        print(f"{i+1}. {text}")

if __name__ == "__main__":
    main() 