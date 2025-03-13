"""
Complete Bio-ChemTransformer model implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from .embedding import BioChemEmbedding
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder


class BioChemTransformer(nn.Module):
    """
    Bio-ChemTransformer model integrating Bio_ClinicalBERT and ChemBERT.
    """
    
    def __init__(
        self,
        bio_clinical_bert_model: str = "emilyalsentzer/Bio_ClinicalBERT",
        chembert_model: str = "seyonec/ChemBERTa-zinc-base-v1",
        bio_clinical_bert_dim: int = 768,
        chembert_dim: int = 768,
        projection_dim: int = 768,
        embedding_combination: str = "concat",
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        encoder_attention_heads: int = 12,
        decoder_attention_heads: int = 12,
        encoder_ffn_dim: int = 3072,
        decoder_ffn_dim: int = 3072,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_function: str = "gelu",
        layernorm_eps: float = 1e-12,
        encoder_layerdrop: float = 0.0,
        decoder_layerdrop: float = 0.0,
        max_position_embeddings: int = 512,
        vocab_size: int = None,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        use_dma: bool = True,
        dma_probability: float = 0.25
    ):
        """
        Initialize the Bio-ChemTransformer model.
        
        Args:
            bio_clinical_bert_model: The Bio_ClinicalBERT model name or path
            chembert_model: The ChemBERT model name or path
            bio_clinical_bert_dim: Dimension of Bio_ClinicalBERT embeddings
            chembert_dim: Dimension of ChemBERT embeddings
            projection_dim: Dimension of the final embeddings after projection
            embedding_combination: Method to combine embeddings ("concat" or "sum")
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            encoder_attention_heads: Number of attention heads in the encoder
            decoder_attention_heads: Number of attention heads in the decoder
            encoder_ffn_dim: Dimension of the encoder feed-forward network
            decoder_ffn_dim: Dimension of the decoder feed-forward network
            dropout: Dropout probability
            attention_dropout: Dropout probability for attention weights
            activation_dropout: Dropout probability for activation outputs
            activation_function: Activation function
            layernorm_eps: Epsilon for layer normalization
            encoder_layerdrop: Probability of dropping an encoder layer
            decoder_layerdrop: Probability of dropping a decoder layer
            max_position_embeddings: Maximum number of position embeddings
            vocab_size: Size of the output vocabulary (if None, uses Bio_ClinicalBERT's vocab)
            pad_token_id: ID of the padding token
            eos_token_id: ID of the end-of-sequence token
            use_dma: Whether to use Diagonal-Masked Attention
            dma_probability: Probability of masking in DMA
        """
        super().__init__()
        
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        
        # Combined embedding layer for Bio_ClinicalBERT and ChemBERT
        self.embedding_layer = BioChemEmbedding(
            bio_clinical_bert_model=bio_clinical_bert_model,
            chembert_model=chembert_model,
            bio_clinical_bert_dim=bio_clinical_bert_dim,
            chembert_dim=chembert_dim,
            projection_dim=projection_dim,
            embedding_combination=embedding_combination
        )
        
        # Encoder
        self.encoder = TransformerEncoder(
            embed_dim=projection_dim,
            num_layers=num_encoder_layers,
            num_heads=encoder_attention_heads,
            ffn_dim=encoder_ffn_dim,
            dropout=dropout,
            activation_fn=activation_function,
            layernorm_eps=layernorm_eps,
            layerdrop=encoder_layerdrop,
            use_dma=use_dma,
            dma_probability=dma_probability
        )
        
        # If vocab_size is not provided, use the vocab size from Bio_ClinicalBERT
        if vocab_size is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(bio_clinical_bert_model)
            vocab_size = len(tokenizer)
        
        # Decoder
        self.decoder = TransformerDecoder(
            embed_dim=projection_dim,
            num_layers=num_decoder_layers,
            num_heads=decoder_attention_heads,
            ffn_dim=decoder_ffn_dim,
            vocab_size=vocab_size,
            dropout=dropout,
            activation_fn=activation_function,
            layernorm_eps=layernorm_eps,
            layerdrop=decoder_layerdrop,
            max_position_embeddings=max_position_embeddings,
            pad_token_id=pad_token_id
        )
    
    def forward(
        self,
        adr_input_ids: torch.Tensor,
        adr_attention_mask: torch.Tensor,
        smiles_input_ids: torch.Tensor,
        smiles_attention_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the Bio-ChemTransformer.
        
        Args:
            adr_input_ids: Input IDs for ADR text
            adr_attention_mask: Attention mask for ADR text
            smiles_input_ids: Input IDs for SMILES
            smiles_attention_mask: Attention mask for SMILES
            decoder_input_ids: Input IDs for the decoder
            decoder_attention_mask: Attention mask for the decoder
            encoder_outputs: Pre-computed encoder outputs
            head_mask: Mask for encoder attention heads
            decoder_head_mask: Mask for decoder attention heads
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a dictionary or tuple
            
        Returns:
            Dictionary or tuple containing model outputs
        """
        # If decoder_input_ids not provided, create them by shifting the target by one position
        if decoder_input_ids is None:
            decoder_input_ids = torch.zeros(
                (adr_input_ids.shape[0], 1),
                dtype=torch.long,
                device=adr_input_ids.device
            ).fill_(self.pad_token_id)
        
        # Get combined embeddings from Bio_ClinicalBERT and ChemBERT
        if encoder_outputs is None:
            # Get combined embeddings from the embedding layer
            combined_embeddings, combined_attention_mask = self.embedding_layer(
                adr_input_ids=adr_input_ids,
                adr_attention_mask=adr_attention_mask,
                smiles_input_ids=smiles_input_ids,
                smiles_attention_mask=smiles_attention_mask
            )
            
            # Pass through the encoder
            encoder_outputs = self.encoder(
                hidden_states=combined_embeddings,
                attention_mask=combined_attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states
            )
        
        # Get encoder outputs
        encoder_hidden_states = encoder_outputs[0]
        
        # Pass through the decoder
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=combined_attention_mask if 'combined_attention_mask' in locals() else None,
            attention_mask=decoder_attention_mask,
            head_mask=decoder_head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        # Get decoder outputs
        logits = decoder_outputs[0]
        
        # Return outputs
        if return_dict:
            return {
                "logits": logits,
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_attentions": encoder_outputs[2] if output_attentions else None,
                "decoder_hidden_states": decoder_outputs[1] if output_hidden_states else None,
                "decoder_self_attentions": decoder_outputs[2] if output_attentions else None,
                "decoder_cross_attentions": decoder_outputs[3] if output_attentions else None
            }
        else:
            return (logits,) + encoder_outputs[1:] + decoder_outputs[1:]
    
    def generate(
        self,
        adr_input_ids: torch.Tensor,
        adr_attention_mask: torch.Tensor,
        smiles_input_ids: torch.Tensor,
        smiles_attention_mask: torch.Tensor,
        max_length: int = 128,
        min_length: int = 5,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1
    ) -> torch.Tensor:
        """
        Generate ADR predictions.
        
        Args:
            adr_input_ids: Input IDs for ADR text
            adr_attention_mask: Attention mask for ADR text
            smiles_input_ids: Input IDs for SMILES
            smiles_attention_mask: Attention mask for SMILES
            max_length: Maximum length of generated sequence
            min_length: Minimum length of generated sequence
            temperature: Temperature for sampling
            top_k: Number of highest probability tokens to keep for top-k sampling
            top_p: Cumulative probability for nucleus sampling
            repetition_penalty: Penalty for repeating tokens
            num_return_sequences: Number of sequences to return
            
        Returns:
            Generated token IDs
        """
        batch_size = adr_input_ids.shape[0]
        device = adr_input_ids.device
        
        # Get encoder outputs
        combined_embeddings, combined_attention_mask = self.embedding_layer(
            adr_input_ids=adr_input_ids,
            adr_attention_mask=adr_attention_mask,
            smiles_input_ids=smiles_input_ids,
            smiles_attention_mask=smiles_attention_mask
        )
        
        encoder_outputs = self.encoder(
            hidden_states=combined_embeddings,
            attention_mask=combined_attention_mask
        )
        
        encoder_hidden_states = encoder_outputs[0]
        
        # Start with batch_size * num_return_sequences
        expanded_batch_size = batch_size * num_return_sequences
        
        # Expand encoder outputs for beam search
        if num_return_sequences > 1:
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_return_sequences, dim=0)
            combined_attention_mask = combined_attention_mask.repeat_interleave(num_return_sequences, dim=0)
        
        # Start with a pad token
        input_ids = torch.full(
            (expanded_batch_size, 1),
            self.pad_token_id,
            dtype=torch.long,
            device=device
        )
        
        # Keeping track of which sequences are already finished
        unfinished_sequences = torch.ones(expanded_batch_size, dtype=torch.long, device=device)
        
        for step in range(max_length):
            # Prepare decoder inputs
            decoder_outputs = self.decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=combined_attention_mask
            )
            
            # Get next token logits
            next_token_logits = decoder_outputs[0][:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(expanded_batch_size):
                    for previous_token in input_ids[i]:
                        if previous_token in [self.pad_token_id, self.eos_token_id]:
                            continue
                        next_token_logits[i, previous_token] /= repetition_penalty
            
            # Mask tokens before min_length
            if step < min_length:
                next_token_logits[:, self.eos_token_id] = float('-inf')
            
            # Apply top-k sampling
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, 
                    index=sorted_indices, 
                    src=sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Convert logits to probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample next token
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            # Finished sequences should have their next token as the pad token
            next_tokens = next_tokens * unfinished_sequences.unsqueeze(1) + self.pad_token_id * (1 - unfinished_sequences.unsqueeze(1))
            
            # Add next token to sequence
            input_ids = torch.cat([input_ids, next_tokens], dim=1)
            
            # Update unfinished sequences
            unfinished_sequences = unfinished_sequences * (next_tokens.squeeze() != self.eos_token_id)
            
            # Check if all sequences are finished
            if unfinished_sequences.max() == 0:
                break
        
        # Reshape output if needed
        if num_return_sequences > 1:
            input_ids = input_ids.view(batch_size, num_return_sequences, -1)
        
        return input_ids 