"""
Test script for Bio-ChemTransformer model.
"""

import torch
import argparse
import logging
from typing import List, Dict, Optional
from transformers import AutoTokenizer

from models.transformer import BioChemTransformer
from utils.smiles_tokenization import SMILESTokenizer
from utils.data_utils import BioChemDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> BioChemTransformer:
    """
    Load a trained Bio-ChemTransformer model.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    # Initialize the model architecture
    model = BioChemTransformer()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    return model


def prepare_input(
    adr_text: str,
    smiles: str,
    bio_clinical_bert_tokenizer: str = "emilyalsentzer/Bio_ClinicalBERT",
    chembert_tokenizer: str = "seyonec/ChemBERTa-zinc-base-v1",
    max_adr_length: int = 256,
    max_smiles_length: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Prepare input data for the model.
    
    Args:
        adr_text: ADR text input
        smiles: SMILES string input
        bio_clinical_bert_tokenizer: Bio_ClinicalBERT tokenizer name or path
        chembert_tokenizer: ChemBERT tokenizer name or path
        max_adr_length: Maximum length for ADR text tokens
        max_smiles_length: Maximum length for SMILES tokens
        device: Device to put tensors on
        
    Returns:
        Dictionary with input tensors
    """
    # Load tokenizers
    adr_tokenizer = AutoTokenizer.from_pretrained(bio_clinical_bert_tokenizer)
    smiles_tokenizer = SMILESTokenizer(
        chembert_model=chembert_tokenizer,
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
    adr_input_ids = adr_tokens['input_ids'].to(device)
    adr_attention_mask = adr_tokens['attention_mask'].to(device)
    smiles_input_ids = smiles_tokens['input_ids'].to(device)
    smiles_attention_mask = smiles_tokens['attention_mask'].to(device)
    
    return {
        'adr_input_ids': adr_input_ids,
        'adr_attention_mask': adr_attention_mask,
        'smiles_input_ids': smiles_input_ids,
        'smiles_attention_mask': smiles_attention_mask
    }


def generate_prediction(
    model: BioChemTransformer,
    adr_input_ids: torch.Tensor,
    adr_attention_mask: torch.Tensor,
    smiles_input_ids: torch.Tensor,
    smiles_attention_mask: torch.Tensor,
    adr_tokenizer: AutoTokenizer,
    max_length: int = 128,
    num_return_sequences: int = 1
) -> List[str]:
    """
    Generate predictions using the model.
    
    Args:
        model: Bio-ChemTransformer model
        adr_input_ids: Input IDs for ADR text
        adr_attention_mask: Attention mask for ADR text
        smiles_input_ids: Input IDs for SMILES
        smiles_attention_mask: Attention mask for SMILES
        adr_tokenizer: Bio_ClinicalBERT tokenizer
        max_length: Maximum length for generated sequence
        num_return_sequences: Number of sequences to return
        
    Returns:
        List of generated text
    """
    # Generate output IDs
    with torch.no_grad():
        output_ids = model.generate(
            adr_input_ids=adr_input_ids,
            adr_attention_mask=adr_attention_mask,
            smiles_input_ids=smiles_input_ids,
            smiles_attention_mask=smiles_attention_mask,
            max_length=max_length,
            num_return_sequences=num_return_sequences
        )
    
    # Convert output IDs to text
    if num_return_sequences > 1:
        # Handle multiple sequences
        batch_size, num_sequences, seq_length = output_ids.shape
        output_ids = output_ids.view(batch_size * num_sequences, seq_length)
    
    # Convert IDs to tokens
    predictions = []
    for ids in output_ids:
        # Skip padding and special tokens
        valid_ids = ids[ids != model.pad_token_id]
        if model.eos_token_id in valid_ids:
            valid_ids = valid_ids[:valid_ids.tolist().index(model.eos_token_id)]
        
        # Decode tokens to text
        text = adr_tokenizer.decode(valid_ids, skip_special_tokens=True)
        predictions.append(text)
    
    return predictions


def run_inference(
    model: BioChemTransformer,
    adr_text: str,
    smiles: str,
    bio_clinical_bert_tokenizer: str = "emilyalsentzer/Bio_ClinicalBERT",
    chembert_tokenizer: str = "seyonec/ChemBERTa-zinc-base-v1",
    max_length: int = 128,
    num_return_sequences: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> List[str]:
    """
    Run inference on a single input.
    
    Args:
        model: Bio-ChemTransformer model
        adr_text: ADR text input
        smiles: SMILES string input
        bio_clinical_bert_tokenizer: Bio_ClinicalBERT tokenizer name or path
        chembert_tokenizer: ChemBERT tokenizer name or path
        max_length: Maximum length for generated sequence
        num_return_sequences: Number of sequences to return
        device: Device to run inference on
        
    Returns:
        List of generated texts
    """
    # Prepare input
    inputs = prepare_input(
        adr_text=adr_text,
        smiles=smiles,
        bio_clinical_bert_tokenizer=bio_clinical_bert_tokenizer,
        chembert_tokenizer=chembert_tokenizer,
        device=device
    )
    
    # Load tokenizer for decoding
    adr_tokenizer = AutoTokenizer.from_pretrained(bio_clinical_bert_tokenizer)
    
    # Generate prediction
    predictions = generate_prediction(
        model=model,
        adr_input_ids=inputs['adr_input_ids'],
        adr_attention_mask=inputs['adr_attention_mask'],
        smiles_input_ids=inputs['smiles_input_ids'],
        smiles_attention_mask=inputs['smiles_attention_mask'],
        adr_tokenizer=adr_tokenizer,
        max_length=max_length,
        num_return_sequences=num_return_sequences
    )
    
    return predictions


def batch_inference(
    model: BioChemTransformer,
    data_file: str,
    bio_clinical_bert_tokenizer: str = "emilyalsentzer/Bio_ClinicalBERT",
    chembert_tokenizer: str = "seyonec/ChemBERTa-zinc-base-v1",
    max_adr_length: int = 256,
    max_smiles_length: int = 256,
    batch_size: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_file: Optional[str] = None
) -> List[Dict]:
    """
    Run inference on a batch of inputs from a file.
    
    Args:
        model: Bio-ChemTransformer model
        data_file: Path to the data file
        bio_clinical_bert_tokenizer: Bio_ClinicalBERT tokenizer name or path
        chembert_tokenizer: ChemBERT tokenizer name or path
        max_adr_length: Maximum length for ADR text tokens
        max_smiles_length: Maximum length for SMILES tokens
        batch_size: Batch size for inference
        device: Device to run inference on
        output_file: Path to save results (optional)
        
    Returns:
        List of result dictionaries
    """
    from torch.utils.data import DataLoader
    
    # Create dataset
    dataset = BioChemDataset(
        data_file=data_file,
        bio_clinical_bert_tokenizer=bio_clinical_bert_tokenizer,
        chembert_tokenizer=chembert_tokenizer,
        max_adr_length=max_adr_length,
        max_smiles_length=max_smiles_length,
        is_training=False
    )
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load tokenizer for decoding
    adr_tokenizer = AutoTokenizer.from_pretrained(bio_clinical_bert_tokenizer)
    
    results = []
    for batch in data_loader:
        # Move batch to device
        adr_input_ids = batch["adr_input_ids"].to(device)
        adr_attention_mask = batch["adr_attention_mask"].to(device)
        smiles_input_ids = batch["smiles_input_ids"].to(device)
        smiles_attention_mask = batch["smiles_attention_mask"].to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                adr_input_ids=adr_input_ids,
                adr_attention_mask=adr_attention_mask,
                smiles_input_ids=smiles_input_ids,
                smiles_attention_mask=smiles_attention_mask
            )
        
        # Get predictions
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=-1)
        
        # Process predictions
        for i, pred in enumerate(preds):
            # Skip padding and special tokens
            valid_ids = pred[pred != model.pad_token_id]
            if model.eos_token_id in valid_ids:
                valid_ids = valid_ids[:valid_ids.tolist().index(model.eos_token_id)]
            
            # Decode tokens to text
            text = adr_tokenizer.decode(valid_ids, skip_special_tokens=True)
            
            # Create result dictionary
            result = {
                "drug_name": batch["drug_name"][i],
                "source": batch["source"][i],
                "predicted_adr": text,
                "original_adr": adr_tokenizer.decode(
                    batch["original_adr_input_ids"][i],
                    skip_special_tokens=True
                ),
                "smiles": batch["smiles"][i] if "smiles" in batch else None
            }
            results.append(result)
    
    # Save results if output file is provided
    if output_file:
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved results to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test Bio-ChemTransformer model")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True, 
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["single", "batch"], 
        default="single", 
        help="Inference mode: single or batch"
    )
    parser.add_argument(
        "--adr_text", 
        type=str, 
        default=None,
        help="ADR text input (for single mode)"
    )
    parser.add_argument(
        "--smiles", 
        type=str, 
        default=None, 
        help="SMILES string input (for single mode)"
    )
    parser.add_argument(
        "--data_file", 
        type=str, 
        default=None, 
        help="Path to the data file (for batch mode)"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default=None, 
        help="Path to save results (for batch mode)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8, 
        help="Batch size for inference (for batch mode)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu", 
        help="Device to run inference on"
    )
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    
    if args.mode == "single":
        if not args.adr_text or not args.smiles:
            parser.error("--adr_text and --smiles are required for single mode")
        
        # Run single inference
        predictions = run_inference(
            model=model,
            adr_text=args.adr_text,
            smiles=args.smiles,
            device=args.device
        )
        
        # Print predictions
        print("Predictions:")
        for i, pred in enumerate(predictions):
            print(f"{i+1}. {pred}")
    
    elif args.mode == "batch":
        if not args.data_file:
            parser.error("--data_file is required for batch mode")
        
        # Run batch inference
        results = batch_inference(
            model=model,
            data_file=args.data_file,
            batch_size=args.batch_size,
            device=args.device,
            output_file=args.output_file
        )
        
        # Print summary
        print(f"Processed {len(results)} samples")
        if args.output_file:
            print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main() 