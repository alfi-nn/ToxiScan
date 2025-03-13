"""
Inference script for Bio-ChemTransformer model.
"""

import os
import json
import logging
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Union
from rdkit import Chem
from transformers import PreTrainedTokenizer

from models.transformer import BioChemTransformer
from utils.tokenizers import BioChemTokenizer
from config import ModelConfig, DataConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Predictor:
    def __init__(
        self,
        model_path: str,
        model_config: ModelConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model checkpoint
            model_config: Model configuration
            device: Device to use for inference
        """
        self.model_config = model_config
        self.device = device
        
        # Initialize model
        self.model = BioChemTransformer(
            bio_clinical_bert_model=model_config.bio_clinical_bert_model,
            chembert_model=model_config.chembert_model,
            bio_clinical_bert_dim=model_config.bio_clinical_bert_dim,
            chembert_dim=model_config.chembert_dim,
            projection_dim=model_config.projection_dim,
            num_encoder_layers=model_config.num_encoder_layers,
            num_decoder_layers=model_config.num_decoder_layers,
            num_attention_heads=model_config.num_attention_heads,
            dropout=model_config.dropout,
            max_position_embeddings=model_config.max_position_embeddings,
            pad_token_id=model_config.pad_token_id,
            eos_token_id=model_config.eos_token_id,
            vocab_size=model_config.vocab_size
        ).to(device)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize tokenizer
        self.tokenizer = BioChemTokenizer(
            bio_clinical_bert_model=model_config.bio_clinical_bert_model,
            chembert_model=model_config.chembert_model
        )
    
    def canonicalize_smiles(self, smiles: str) -> str:
        """
        Convert SMILES string to canonical form.
        
        Args:
            smiles: SMILES string to canonicalize
            
        Returns:
            Canonicalized SMILES string
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            else:
                raise ValueError(f"Failed to parse SMILES: {smiles}")
        except Exception as e:
            raise ValueError(f"Error canonicalizing SMILES {smiles}: {str(e)}")
    
    def preprocess_smiles(self, smiles: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess SMILES string for model input.
        
        Args:
            smiles: SMILES string to preprocess
            
        Returns:
            Dictionary containing preprocessed tensors
        """
        # Canonicalize SMILES
        canonical_smiles = self.canonicalize_smiles(smiles)
        
        # Tokenize SMILES
        tokenized = self.tokenizer.tokenize_smiles(
            canonical_smiles,
            max_length=self.model_config.max_position_embeddings,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'smiles_input_ids': tokenized['input_ids'].to(self.device),
            'smiles_attention_mask': tokenized['attention_mask'].to(self.device)
        }
    
    @torch.no_grad()
    def predict_adr(
        self,
        smiles: str,
        max_length: int = 128,
        num_beams: int = 4,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate ADR predictions for a given SMILES string.
        
        Args:
            smiles: SMILES string to predict ADRs for
            max_length: Maximum length of generated sequence
            num_beams: Number of beams for beam search
            temperature: Temperature for sampling
            top_k: Top-k filtering value
            top_p: Top-p filtering value
            repetition_penalty: Repetition penalty value
            length_penalty: Length penalty value
            num_return_sequences: Number of sequences to return
            
        Returns:
            List of predicted ADR texts
        """
        try:
            # Preprocess input
            model_inputs = self.preprocess_smiles(smiles)
            
            # Generate predictions
            outputs = self.model.generate(
                smiles_input_ids=model_inputs['smiles_input_ids'],
                smiles_attention_mask=model_inputs['smiles_attention_mask'],
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.model_config.pad_token_id,
                eos_token_id=self.model_config.eos_token_id
            )
            
            # Decode predictions
            predictions = []
            for output in outputs:
                decoded = self.tokenizer.decode_adr(
                    output,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                predictions.append(decoded)
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error generating prediction for SMILES {smiles}: {str(e)}")
            return []
    
    def batch_predict(
        self,
        smiles_list: List[str],
        batch_size: int = 32,
        **kwargs
    ) -> List[List[str]]:
        """
        Generate ADR predictions for a list of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size for processing
            **kwargs: Additional arguments for predict_adr
            
        Returns:
            List of lists containing predicted ADR texts for each input SMILES
        """
        all_predictions = []
        
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="Generating predictions"):
            batch_smiles = smiles_list[i:i + batch_size]
            batch_predictions = []
            
            for smiles in batch_smiles:
                try:
                    predictions = self.predict_adr(smiles, **kwargs)
                    batch_predictions.append(predictions)
                except Exception as e:
                    logger.error(f"Error processing SMILES {smiles}: {str(e)}")
                    batch_predictions.append([])
            
            all_predictions.extend(batch_predictions)
        
        return all_predictions
    
    def evaluate_predictions(
        self,
        predictions: List[str],
        ground_truth: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate predictions against ground truth.
        
        Args:
            predictions: List of predicted ADR texts
            ground_truth: List of ground truth ADR texts
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # TODO: Implement more sophisticated evaluation metrics
        # For now, we'll just compute exact match accuracy
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        accuracy = correct / len(predictions) if predictions else 0
        
        return {
            'accuracy': accuracy,
            'num_samples': len(predictions)
        }


def main():
    """Main function."""
    # Load configurations
    model_config = ModelConfig()
    data_config = DataConfig()
    
    # Initialize predictor
    predictor = Predictor(
        model_path="checkpoints/best_model.pt",
        model_config=model_config
    )
    
    # Example usage
    smiles = "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"  # Example SMILES
    predictions = predictor.predict_adr(
        smiles,
        num_beams=4,
        temperature=0.7,
        num_return_sequences=3
    )
    
    logger.info(f"Input SMILES: {smiles}")
    logger.info("Predicted ADRs:")
    for i, pred in enumerate(predictions, 1):
        logger.info(f"{i}. {pred}")


if __name__ == "__main__":
    main() 