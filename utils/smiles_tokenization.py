"""
Utilities for SMILES tokenization with ChemBERT.
"""

import torch
from transformers import AutoTokenizer
from rdkit import Chem
from rdkit.Chem import AllChem
import logging
from typing import List, Dict, Union, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def canonicalize_smiles(smiles: str) -> str:
    """
    Convert a SMILES string to canonical form.
    
    Args:
        smiles: SMILES string to canonicalize
        
    Returns:
        Canonical SMILES string or original if conversion fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        else:
            return smiles
    except:
        return smiles


class SMILESTokenizer:
    """Wrapper for ChemBERT tokenization of SMILES strings."""
    
    def __init__(
        self,
        chembert_model: str = "seyonec/ChemBERTa-zinc-base-v1",
        max_length: int = 256,
        canonicalize: bool = True
    ):
        """
        Initialize the SMILES tokenizer.
        
        Args:
            chembert_model: ChemBERT model name or path for tokenization
            max_length: Maximum token sequence length
            canonicalize: Whether to canonicalize SMILES strings
        """
        self.tokenizer = AutoTokenizer.from_pretrained(chembert_model)
        self.max_length = max_length
        self.canonicalize = canonicalize
        logger.info(f"Initialized SMILESTokenizer with model {chembert_model}")
        
    def tokenize(
        self,
        smiles_list: List[str],
        return_tensors: str = "pt",
        padding: str = "max_length",
        truncation: bool = True
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Tokenize a list of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings to tokenize
            return_tensors: Format of the returned tensors ("pt" for PyTorch)
            padding: Padding strategy ("max_length" or "longest")
            truncation: Whether to truncate sequences longer than max_length
            
        Returns:
            Dictionary with tokenized outputs (input_ids, attention_mask)
        """
        # Canonicalize SMILES if specified
        if self.canonicalize:
            smiles_list = [canonicalize_smiles(s) for s in smiles_list]
        
        # Tokenize SMILES
        tokens = self.tokenizer(
            smiles_list,
            padding=padding,
            max_length=self.max_length,
            truncation=truncation,
            return_tensors=return_tensors
        )
        
        return tokens
    
    def batch_encode(
        self,
        smiles_list: List[str],
        batch_size: int = 32
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Encode a large list of SMILES strings in batches.
        
        Args:
            smiles_list: List of SMILES strings to encode
            batch_size: Batch size for encoding
            
        Returns:
            List of dictionaries with tokenized outputs
        """
        results = []
        
        # Process in batches
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i+batch_size]
            tokens = self.tokenize(batch)
            results.append(tokens)
            
        return results
    
    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode token IDs back to SMILES strings.
        
        Args:
            token_ids: Tensor of token IDs
            skip_special_tokens: Whether to skip special tokens in the decoded output
            
        Returns:
            List of decoded SMILES strings
        """
        return self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )
    
    def get_vocab_size(self) -> int:
        """
        Get the vocabulary size of the SMILES tokenizer.
        
        Returns:
            Vocabulary size
        """
        return len(self.tokenizer)
    
    def get_pad_token_id(self) -> int:
        """
        Get the pad token ID.
        
        Returns:
            Pad token ID
        """
        return self.tokenizer.pad_token_id
    
    def tokenize_for_training(
        self,
        smiles_list: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize SMILES strings for model training.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        # Special handling for training (e.g., data augmentation)
        # For now, just call the regular tokenize method
        return self.tokenize(smiles_list) 