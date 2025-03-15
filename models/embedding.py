"""
Embedding layer for the Bio-ChemTransformer.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Tuple, Optional


class BioChemEmbedding(nn.Module):
    """
    Embedding layer that combines Bio_ClinicalBERT and ChemBERT embeddings.
    """
    
    def __init__(
        self,
        bio_clinical_bert_model: str = "emilyalsentzer/Bio_ClinicalBERT",
        chembert_model: str = "seyonec/ChemBERTa-zinc-base-v1",
        bio_clinical_bert_dim: int = 768,
        chembert_dim: int = 768,
        projection_dim: int = 768,
        embedding_combination: str = "concat",
        max_adr_length: int = 256,
        max_smiles_length: int = 256
    ):
        """
        Initialize the embedding layer.
        
        Args:
            bio_clinical_bert_model: The Bio_ClinicalBERT model name or path
            chembert_model: The ChemBERT model name or path
            bio_clinical_bert_dim: Dimension of Bio_ClinicalBERT embeddings
            chembert_dim: Dimension of ChemBERT embeddings
            projection_dim: Dimension of the final embeddings after projection
            embedding_combination: Method to combine embeddings ("concat" or "sum")
            max_adr_length: Maximum length for ADR text tokens
            max_smiles_length: Maximum length for SMILES tokens
        """
        super().__init__()
        
        self.bio_clinical_bert_dim = bio_clinical_bert_dim
        self.chembert_dim = chembert_dim
        self.projection_dim = projection_dim
        self.embedding_combination = embedding_combination
        self.max_adr_length = max_adr_length
        self.max_smiles_length = max_smiles_length
        
        # Load pre-trained models
        self.bio_clinical_bert = AutoModel.from_pretrained(bio_clinical_bert_model)
        self.chembert = AutoModel.from_pretrained(chembert_model)
        
        # Load tokenizers
        self.bio_clinical_tokenizer = AutoTokenizer.from_pretrained(bio_clinical_bert_model)
        self.chembert_tokenizer = AutoTokenizer.from_pretrained(chembert_model)
        
        # Projection layers for dimensional alignment if needed
        self.bio_projection = None
        self.chem_projection = None
        
        if bio_clinical_bert_dim != projection_dim:
            self.bio_projection = nn.Linear(bio_clinical_bert_dim, projection_dim)
        
        if chembert_dim != projection_dim:
            self.chem_projection = nn.Linear(chembert_dim, projection_dim)
        
        # If using concatenation, we need a final projection to the desired dimension
        if embedding_combination == "concat":
            # The final projection should project from projection_dim to projection_dim
            # since we're projecting each embedding separately first
            self.final_projection = nn.Linear(projection_dim, projection_dim)
        else:
            self.final_projection = None
    
    def tokenize_adr_text(self, adr_text_list):
        """
        Tokenize a list of ADR text strings using Bio_ClinicalBERT tokenizer.
        
        Args:
            adr_text_list: List of ADR text strings
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        return self.bio_clinical_tokenizer(
            adr_text_list,
            padding='max_length',
            truncation=True,
            max_length=self.max_adr_length,
            return_tensors='pt'
        )
    
    def tokenize_smiles(self, smiles_list):
        """
        Tokenize a list of SMILES strings using ChemBERT tokenizer.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        return self.chembert_tokenizer(
            smiles_list,
            padding='max_length',
            truncation=True,
            max_length=self.max_smiles_length,
            return_tensors='pt'
        )
    
    def forward(
        self,
        adr_input_ids: torch.Tensor,
        adr_attention_mask: torch.Tensor,
        smiles_input_ids: torch.Tensor,
        smiles_attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute combined embeddings.
        
        Args:
            adr_input_ids: Input IDs for ADR text
            adr_attention_mask: Attention mask for ADR text
            smiles_input_ids: Input IDs for SMILES
            smiles_attention_mask: Attention mask for SMILES
            
        Returns:
            Tuple of (combined_embeddings, combined_attention_mask)
        """
        # Get Bio_ClinicalBERT embeddings for ADR text
        bio_outputs = self.bio_clinical_bert(
            input_ids=adr_input_ids,
            attention_mask=adr_attention_mask,
            return_dict=True
        )
        bio_embeddings = bio_outputs.last_hidden_state
        
        # Get ChemBERT embeddings for SMILES
        chem_outputs = self.chembert(
            input_ids=smiles_input_ids,
            attention_mask=smiles_attention_mask,
            return_dict=True
        )
        chem_embeddings = chem_outputs.last_hidden_state
        
        # Apply projections if needed
        if self.bio_projection is not None:
            bio_embeddings = self.bio_projection(bio_embeddings)
            
        if self.chem_projection is not None:
            chem_embeddings = self.chem_projection(chem_embeddings)
        
        # Combine embeddings
        if self.embedding_combination == "concat":
            # Concatenate along sequence length
            batch_size = bio_embeddings.shape[0]
            
            # Create a combined attention mask
            combined_attention_mask = torch.cat(
                [adr_attention_mask, smiles_attention_mask], dim=1
            )
            
            # Concatenate embeddings
            combined_embeddings = torch.cat(
                [bio_embeddings, chem_embeddings], dim=1
            )
            
            # Apply final projection
            if self.final_projection is not None:
                # Reshape the embeddings to (batch_size * seq_len, projection_dim)
                batch_size, seq_len, hidden_dim = combined_embeddings.shape
                combined_embeddings = combined_embeddings.reshape(-1, hidden_dim)
                combined_embeddings = self.final_projection(combined_embeddings)
                # Reshape back to (batch_size, seq_len, projection_dim)
                combined_embeddings = combined_embeddings.reshape(batch_size, seq_len, -1)
                
        elif self.embedding_combination == "sum":
            # Sum the embeddings (requires same sequence length)
            # Here we assume bio_embeddings and chem_embeddings have the same shape
            # If they don't, padding or truncation would be needed
            combined_embeddings = bio_embeddings + chem_embeddings
            combined_attention_mask = adr_attention_mask
            
        else:
            raise ValueError(f"Unsupported embedding combination: {self.embedding_combination}")
        
        return combined_embeddings, combined_attention_mask
    
    def get_output_dim(self) -> int:
        """
        Get the output dimension of the embedding layer.
        
        Returns:
            The dimension of the output embeddings
        """
        return self.projection_dim 