"""
Evaluation script for Bio-ChemTransformer model.
"""

import os
import json
import logging
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from models.transformer import BioChemTransformer
from utils.tokenizers import BioChemTokenizer
from utils.data_utils import ADRDataset
from config import ModelConfig, DataConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self,
        model_path: str,
        model_config: ModelConfig,
        data_config: DataConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained model checkpoint
            model_config: Model configuration
            data_config: Data configuration
            device: Device to use for evaluation
        """
        self.model_config = model_config
        self.data_config = data_config
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
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def load_test_data(self) -> ADRDataset:
        """
        Load test dataset.
        
        Returns:
            Test dataset
        """
        return ADRDataset(
            data_path=self.data_config.test_data_path,
            tokenizer=self.tokenizer,
            max_length=self.model_config.max_position_embeddings
        )
    
    @torch.no_grad()
    def generate_predictions(
        self,
        test_dataset: ADRDataset,
        batch_size: int = 32,
        num_beams: int = 4,
        max_length: int = 128
    ) -> Tuple[List[str], List[str]]:
        """
        Generate predictions for test dataset.
        
        Args:
            test_dataset: Test dataset
            batch_size: Batch size for processing
            num_beams: Number of beams for beam search
            max_length: Maximum length of generated sequence
            
        Returns:
            Tuple of (predictions, ground_truth)
        """
        predictions = []
        ground_truth = []
        
        data_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        for batch in tqdm(data_loader, desc="Generating predictions"):
            # Move batch to device
            smiles_input_ids = batch['smiles_input_ids'].to(self.device)
            smiles_attention_mask = batch['smiles_attention_mask'].to(self.device)
            adr_input_ids = batch['adr_input_ids']  # Keep on CPU for ground truth
            
            # Generate predictions
            outputs = self.model.generate(
                smiles_input_ids=smiles_input_ids,
                smiles_attention_mask=smiles_attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                pad_token_id=self.model_config.pad_token_id,
                eos_token_id=self.model_config.eos_token_id
            )
            
            # Decode predictions and ground truth
            for pred, truth in zip(outputs, adr_input_ids):
                pred_text = self.tokenizer.decode_adr(
                    pred,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                truth_text = self.tokenizer.decode_adr(
                    truth,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                predictions.append(pred_text)
                ground_truth.append(truth_text)
        
        return predictions, ground_truth
    
    def compute_bleu_score(
        self,
        predictions: List[str],
        ground_truth: List[str]
    ) -> float:
        """
        Compute BLEU score.
        
        Args:
            predictions: List of predicted texts
            ground_truth: List of ground truth texts
            
        Returns:
            BLEU score
        """
        # Tokenize predictions and ground truth
        pred_tokens = [[p.split()] for p in predictions]
        truth_tokens = [[g.split()] for g in ground_truth]
        
        # Compute BLEU score with smoothing
        smoothing = SmoothingFunction().method1
        return corpus_bleu(truth_tokens, pred_tokens, smoothing_function=smoothing)
    
    def compute_rouge_scores(
        self,
        predictions: List[str],
        ground_truth: List[str]
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores.
        
        Args:
            predictions: List of predicted texts
            ground_truth: List of ground truth texts
            
        Returns:
            Dictionary containing ROUGE scores
        """
        rouge_scores = {
            'rouge1_f': [],
            'rouge2_f': [],
            'rougeL_f': []
        }
        
        for pred, truth in zip(predictions, ground_truth):
            scores = self.rouge_scorer.score(truth, pred)
            rouge_scores['rouge1_f'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2_f'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL_f'].append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1_f': np.mean(rouge_scores['rouge1_f']),
            'rouge2_f': np.mean(rouge_scores['rouge2_f']),
            'rougeL_f': np.mean(rouge_scores['rougeL_f'])
        }
    
    def compute_exact_match(
        self,
        predictions: List[str],
        ground_truth: List[str]
    ) -> float:
        """
        Compute exact match accuracy.
        
        Args:
            predictions: List of predicted texts
            ground_truth: List of ground truth texts
            
        Returns:
            Exact match accuracy
        """
        return accuracy_score(ground_truth, predictions)
    
    def evaluate(
        self,
        batch_size: int = 32,
        num_beams: int = 4,
        max_length: int = 128,
        output_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on test dataset.
        
        Args:
            batch_size: Batch size for processing
            num_beams: Number of beams for beam search
            max_length: Maximum length of generated sequence
            output_path: Path to save evaluation results
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Load test data
        test_dataset = self.load_test_data()
        
        # Generate predictions
        predictions, ground_truth = self.generate_predictions(
            test_dataset,
            batch_size=batch_size,
            num_beams=num_beams,
            max_length=max_length
        )
        
        # Compute metrics
        metrics = {
            'bleu': self.compute_bleu_score(predictions, ground_truth),
            'exact_match': self.compute_exact_match(predictions, ground_truth),
            **self.compute_rouge_scores(predictions, ground_truth)
        }
        
        # Log results
        logger.info("Evaluation Results:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Save results if output path is provided
        if output_path:
            results = {
                'metrics': metrics,
                'predictions': predictions,
                'ground_truth': ground_truth
            }
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Saved evaluation results to {output_path}")
        
        return metrics


def main():
    """Main function."""
    # Load configurations
    model_config = ModelConfig()
    data_config = DataConfig()
    
    # Initialize evaluator
    evaluator = Evaluator(
        model_path="checkpoints/best_model.pt",
        model_config=model_config,
        data_config=data_config
    )
    
    # Run evaluation
    metrics = evaluator.evaluate(
        batch_size=32,
        num_beams=4,
        max_length=128,
        output_path="results/evaluation_results.json"
    )


if __name__ == "__main__":
    main() 