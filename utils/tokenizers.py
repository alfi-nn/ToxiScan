"""
Tokenization utilities for the Bio-ChemTransformer model.
"""

import torch
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Union, Optional


class BioChemTokenizer:
    """
    Tokenizer for Bio-ChemTransformer that handles both biomedical text and SMILES.
    Uses Bio_ClinicalBERT tokenizer for text and ChemBERT tokenizer for SMILES.
    """
    
    def __init__(
        self,
        bio_clinical_bert_model: str = "emilyalsentzer/Bio_ClinicalBERT",
        chembert_model: str = "seyonec/ChemBERTa-zinc-base-v1",
        max_adr_length: int = 256,
        max_smiles_length: int = 256
    ):
        """
        Initialize the tokenizers for biomedical text and SMILES.
        
        Args:
            bio_clinical_bert_model: The Bio_ClinicalBERT model name or path
            chembert_model: The ChemBERT model name or path
            max_adr_length: Maximum length for ADR text tokens
            max_smiles_length: Maximum length for SMILES tokens
        """
        self.max_adr_length = max_adr_length
        self.max_smiles_length = max_smiles_length
        
        # Initialize the tokenizers
        self.text_tokenizer = AutoTokenizer.from_pretrained(bio_clinical_bert_model)
        self.smiles_tokenizer = AutoTokenizer.from_pretrained(chembert_model)
        
        # Special tokens
        self.text_tokenizer.pad_token = self.text_tokenizer.pad_token or "[PAD]"
        self.text_tokenizer.mask_token = self.text_tokenizer.mask_token or "[MASK]"
        
        # Ensure the SMILES tokenizer has the necessary special tokens
        if not hasattr(self.smiles_tokenizer, 'pad_token') or self.smiles_tokenizer.pad_token is None:
            self.smiles_tokenizer.pad_token = self.smiles_tokenizer.eos_token or "[PAD]"
        
        # Store vocabulary sizes for embedding layer initialization
        self.text_vocab_size = len(self.text_tokenizer)
        self.smiles_vocab_size = len(self.smiles_tokenizer)
    
    def tokenize_adr(
        self,
        text: str,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize ADR text using Bio_ClinicalBERT tokenizer.
        
        Args:
            text: The ADR text to tokenize
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            return_tensors: Return format ('pt' for PyTorch tensors)
            
        Returns:
            Dictionary of tokenized output
        """
        return self.text_tokenizer(
            text,
            padding=padding,
            truncation=truncation,
            max_length=self.max_adr_length,
            return_tensors=return_tensors
        )
    
    def tokenize_smiles(
        self,
        smiles: str,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize SMILES string using ChemBERT tokenizer.
        
        Args:
            smiles: The SMILES string to tokenize
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            return_tensors: Return format ('pt' for PyTorch tensors)
            
        Returns:
            Dictionary of tokenized output
        """
        return self.smiles_tokenizer(
            smiles,
            padding=padding,
            truncation=truncation,
            max_length=self.max_smiles_length,
            return_tensors=return_tensors
        )
    
    def tokenize_batch(
        self,
        adr_texts: List[str],
        smiles_strings: List[str],
        mask_probability: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of ADR texts and SMILES strings.
        
        Args:
            adr_texts: List of ADR text strings
            smiles_strings: List of SMILES strings
            mask_probability: Probability of masking tokens in ADR text (for training)
            
        Returns:
            Dictionary containing tokenized outputs for both ADR and SMILES
        """
        # Tokenize ADR texts
        adr_encodings = self.text_tokenizer(
            adr_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_adr_length,
            return_tensors="pt"
        )
        
        # Tokenize SMILES strings
        smiles_encodings = self.smiles_tokenizer(
            smiles_strings,
            padding="max_length",
            truncation=True,
            max_length=self.max_smiles_length,
            return_tensors="pt"
        )
        
        # Apply masking to ADR text if mask_probability > 0
        if mask_probability > 0:
            self._apply_masking(adr_encodings, mask_probability)
        
        return {
            "adr_input_ids": adr_encodings.input_ids,
            "adr_attention_mask": adr_encodings.attention_mask,
            "smiles_input_ids": smiles_encodings.input_ids,
            "smiles_attention_mask": smiles_encodings.attention_mask
        }
    
    def _apply_masking(self, encodings, mask_probability: float):
        """
        Apply random masking to tokens for masked language modeling.
        
        Args:
            encodings: The encodings to mask
            mask_probability: Probability of masking a token
        """
        input_ids = encodings.input_ids.clone()
        attention_mask = encodings.attention_mask
        
        # Create masking probability tensor
        mask_prob = torch.full(input_ids.shape, mask_probability)
        
        # Only mask tokens that are not special tokens and are not padding
        special_tokens_mask = self.text_tokenizer.get_special_tokens_mask(
            input_ids.tolist(), already_has_special_tokens=True
        )
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        
        # Also avoid masking padding tokens
        padding_mask = attention_mask.bool()
        
        # Combine masks to get tokens eligible for masking
        eligible_for_masking = (~special_tokens_mask) & padding_mask
        
        # Sample from mask_prob
        masked_indices = torch.bernoulli(mask_prob * eligible_for_masking).bool() & eligible_for_masking
        
        # Replace masked indices with mask token id
        input_ids[masked_indices] = self.text_tokenizer.mask_token_id
        
        # Update input_ids in the encodings
        encodings.input_ids = input_ids
    
    def decode_adr(self, token_ids: torch.Tensor) -> List[str]:
        """
        Decode ADR token IDs back to text.
        
        Args:
            token_ids: Tensor of token IDs
            
        Returns:
            List of decoded strings
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
            
        return [
            self.text_tokenizer.decode(ids, skip_special_tokens=True)
            for ids in token_ids
        ]
    
    def decode_smiles(self, token_ids: torch.Tensor) -> List[str]:
        """
        Decode SMILES token IDs back to SMILES strings.
        
        Args:
            token_ids: Tensor of token IDs
            
        Returns:
            List of decoded SMILES strings
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
            
        return [
            self.smiles_tokenizer.decode(ids, skip_special_tokens=True)
            for ids in token_ids
        ] 