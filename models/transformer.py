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
        dma_probability: float = 0.25,
        freeze_pretrained: bool = True
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
            freeze_pretrained: Whether to freeze the pretrained models
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
            embedding_combination=embedding_combination,
            freeze_pretrained=freeze_pretrained
        )
        
        # Apply layer normalization before encoder for stable training
        self.pre_encoder_norm = nn.LayerNorm(projection_dim, eps=layernorm_eps)
        
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
        
        # Initialize weights with smaller values to prevent exploding gradients
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize or reset parameters with careful initialization."""
        # Initialize decoder output projection with smaller values
        if hasattr(self.decoder, 'output_projection'):
            nn.init.normal_(self.decoder.output_projection.weight, mean=0.0, std=0.02)
            if hasattr(self.decoder.output_projection, 'bias') and self.decoder.output_projection.bias is not None:
                nn.init.zeros_(self.decoder.output_projection.bias)
    
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
        # Print input shapes for debugging
        print(f"Input shapes:")
        print(f"- ADR input_ids: {adr_input_ids.shape}")
        print(f"- ADR attention_mask: {adr_attention_mask.shape}")
        print(f"- SMILES input_ids: {smiles_input_ids.shape}")
        print(f"- SMILES attention_mask: {smiles_attention_mask.shape}")
        if decoder_input_ids is not None:
            print(f"- decoder_input_ids: {decoder_input_ids.shape}")
        if decoder_attention_mask is not None:
            print(f"- decoder_attention_mask: {decoder_attention_mask.shape}")
            
        # If decoder_input_ids not provided, create them
        if decoder_input_ids is None:
            # Create decoder input IDs with the same batch size as adr_input_ids
            decoder_input_ids = torch.zeros(
                (adr_input_ids.shape[0], adr_input_ids.shape[1]),
                dtype=torch.long,
                device=adr_input_ids.device
            ).fill_(self.pad_token_id)
            
            # Set the first token to the start token (using pad_token as start)
            decoder_input_ids[:, 0] = self.pad_token_id
            print(f"Created decoder_input_ids with shape: {decoder_input_ids.shape}")
        
        # Get combined embeddings from Bio_ClinicalBERT and ChemBERT
        combined_attention_mask = None
        if encoder_outputs is None:
            # Get combined embeddings from the embedding layer
            combined_embeddings, combined_attention_mask = self.embedding_layer(
                adr_input_ids=adr_input_ids,
                adr_attention_mask=adr_attention_mask,
                smiles_input_ids=smiles_input_ids,
                smiles_attention_mask=smiles_attention_mask
            )
            
            # Apply layer normalization for stability
            combined_embeddings = self.pre_encoder_norm(combined_embeddings)
            
            # Print shapes for debugging
            print(f"Combined embeddings shape: {combined_embeddings.shape}")
            print(f"Combined attention mask shape: {combined_attention_mask.shape}")
            
            # Ensure embeddings are contiguous before encoder
            combined_embeddings = combined_embeddings.contiguous()
            
            # Pass through the encoder
            encoder_outputs = self.encoder(
                hidden_states=combined_embeddings,
                attention_mask=combined_attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states
            )
        
        # Get encoder outputs and ensure they're contiguous
        encoder_hidden_states = encoder_outputs[0].contiguous()
        print(f"Encoder hidden states shape before processing: {encoder_hidden_states.shape}")
        
        # Ensure encoder_attention_mask is 2D for the decoder
        encoder_attention_mask = None
        if combined_attention_mask is not None:
            # Make sure it's 2D [batch_size, src_len]
            if len(combined_attention_mask.shape) == 2:
                encoder_attention_mask = combined_attention_mask
            elif len(combined_attention_mask.shape) == 3:
                # If it's 3D, squeeze out the middle dimension if possible
                if combined_attention_mask.shape[1] == 1:
                    encoder_attention_mask = combined_attention_mask.squeeze(1)
                else:
                    # If we can't squeeze, take the first slice
                    encoder_attention_mask = combined_attention_mask[:, 0, :]
            
            # Ensure mask is contiguous
            if encoder_attention_mask is not None:
                encoder_attention_mask = encoder_attention_mask.contiguous()
                print(f"Encoder attention mask shape: {encoder_attention_mask.shape}")
        
        # Crucial fix: Ensure encoder_hidden_states are in the correct format for the decoder
        # The PyTorch nn.MultiheadAttention used in the decoder expects [seq_len, batch_size, embed_dim] format
        batch_size = encoder_hidden_states.size(0)
        seq_len = encoder_hidden_states.size(1)
        embed_dim = encoder_hidden_states.size(2)
        
        print(f"Encoder hidden states dimensions: batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}")
        
        # The decoder expects encoder_hidden_states in [seq_len, batch_size, embed_dim] format
        # Transpose from [batch_size, seq_len, embed_dim] to [seq_len, batch_size, embed_dim]
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1).contiguous()
        print(f"Encoder hidden states shape after transpose: {encoder_hidden_states.shape}")
        
        # If decoder attention mask is not provided, create it from decoder input IDs
        if decoder_attention_mask is None and decoder_input_ids is not None:
            decoder_attention_mask = (decoder_input_ids != self.pad_token_id).to(dtype=torch.float)
            print(f"Created decoder attention mask with shape: {decoder_attention_mask.shape}")
        
        # Ensure decoder input tensors are contiguous
        decoder_input_ids = decoder_input_ids.contiguous()
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.contiguous()
        
        try:
            # Pass through the decoder
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                attention_mask=decoder_attention_mask,
                head_mask=decoder_head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states
            )
            
            print("Decoder forward pass completed successfully")
        except RuntimeError as e:
            print(f"Error in decoder forward pass: {e}")
            
            # Try with a simplified version if there's a shape error
            if "shape" in str(e) and "is invalid for input of size" in str(e):
                print("Attempting simplified forward pass with fixed tensor shapes")
                
                # Simplify by creating fresh tensors with expected shapes
                decoder_embed_dim = self.decoder.embed_dim
                src_len, batch_size, _ = encoder_hidden_states.shape
                
                # Create a simplified encoder_hidden_states with correct dimensions
                simplified_encoder_states = torch.zeros(
                    (src_len, batch_size, decoder_embed_dim),
                    dtype=encoder_hidden_states.dtype,
                    device=encoder_hidden_states.device
                )
                
                # Copy data if dimensions allow
                min_dim = min(encoder_hidden_states.shape[2], decoder_embed_dim)
                simplified_encoder_states[:, :, :min_dim] = encoder_hidden_states[:, :, :min_dim]
                
                print(f"Created simplified encoder states with shape: {simplified_encoder_states.shape}")
                
                # Try decoder with simplified inputs
                decoder_outputs = self.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=simplified_encoder_states,
                    encoder_attention_mask=encoder_attention_mask,
                    attention_mask=decoder_attention_mask,
                    head_mask=decoder_head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states
                )
                
                print("Simplified decoder forward pass completed")
            else:
                # Re-raise if not a shape error
                raise
        
        # Return outputs
        if not return_dict:
            # If not return_dict, return a tuple of tensors
            return (
                decoder_outputs["logits"],
                decoder_outputs.get("hidden_states", None),
                decoder_outputs.get("attentions", None),
                encoder_outputs,
            )
        
        # Combine decoder and encoder outputs into a dictionary
        outputs = {
            "logits": decoder_outputs["logits"],
            "encoder_hidden_states": encoder_outputs[0],
        }
        
        if "all_hidden_states" in decoder_outputs:
            outputs["decoder_hidden_states"] = decoder_outputs["all_hidden_states"]
            
        if "attentions" in decoder_outputs:
            outputs["decoder_attentions"] = decoder_outputs["attentions"]
            
        if len(encoder_outputs) > 1:
            if output_hidden_states:
                outputs["encoder_hidden_states"] = encoder_outputs[1]
            if output_attentions:
                outputs["encoder_attentions"] = encoder_outputs[2]
        
        return outputs
    
    def prepare_for_generation(self):
        """
        Prepare the model for generation by setting appropriate parameters and caching.
        """
        # Set model to evaluation mode
        self.eval()
        
        # Prepare decoder specific settings
        self.decoder.prepare_for_generation()
        
        return self
        
    def generate(
        self,
        adr_input_ids: Optional[torch.Tensor] = None,
        adr_attention_mask: Optional[torch.Tensor] = None,
        smiles_input_ids: torch.Tensor = None,
        smiles_attention_mask: torch.Tensor = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        max_length: int = 128,
        min_length: int = 5,
        num_beams: int = 5,
        temperature: float = 0.6,
        top_k: int = 50,
        top_p: float = 0.92,
        repetition_penalty: float = 1.2,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
        early_stopping: bool = True,
        num_return_sequences: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate ADR predictions.
        
        Args:
            adr_input_ids: Input IDs for ADR text (optional)
            adr_attention_mask: Attention mask for ADR text (optional)
            smiles_input_ids: Input IDs for SMILES
            smiles_attention_mask: Attention mask for SMILES
            decoder_input_ids: Initial input IDs for the decoder (optional)
            max_length: Maximum length of generated sequence
            min_length: Minimum length of generated sequence
            num_beams: Number of beams for beam search
            temperature: Temperature for sampling
            top_k: Number of highest probability tokens to keep for top-k sampling
            top_p: Cumulative probability for nucleus sampling
            repetition_penalty: Penalty for repeating tokens
            length_penalty: Penalty/reward for longer sequences
            no_repeat_ngram_size: Size of n-grams that shouldn't be repeated
            early_stopping: Whether to stop generation when all beams are finished
            num_return_sequences: Number of sequences to return
            pad_token_id: ID of the padding token (defaults to self.pad_token_id)
            eos_token_id: ID of the end-of-sequence token (defaults to self.eos_token_id)
            
        Returns:
            Generated token IDs
        """
        # Prepare model for generation
        self.prepare_for_generation()
        
        # Set default token IDs if not provided
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.eos_token_id
            
        # Make sure inputs are on the same device
        device = smiles_input_ids.device
        
        # Get batch size
        batch_size = smiles_input_ids.shape[0]
        
        # Get encoder outputs
        combined_embeddings, combined_attention_mask = self.embedding_layer(
            adr_input_ids=adr_input_ids,
            adr_attention_mask=adr_attention_mask,
            smiles_input_ids=smiles_input_ids,
            smiles_attention_mask=smiles_attention_mask
        )
        
        # Apply pre-encoder normalization for stability
        combined_embeddings = self.pre_encoder_norm(combined_embeddings)
        
        # Pass through encoder
        encoder_outputs = self.encoder(
            hidden_states=combined_embeddings,
            attention_mask=combined_attention_mask
        )
        
        # Get encoder hidden states
        encoder_hidden_states = encoder_outputs[0]
        
        # Ensure encoder hidden states are properly shaped for decoder
        # The decoder expects [seq_len, batch_size, embed_dim]
        if encoder_hidden_states.shape[0] == batch_size:
            # It's in [batch_size, seq_len, hidden_dim], need to transpose
            encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
            print(f"Transposed encoder hidden states to shape: {encoder_hidden_states.shape}")
        
        # Start with a pad token
        if decoder_input_ids is None:
            # Create initial decoder input
            decoder_input_ids = torch.full(
                (batch_size * num_beams, 1),
                pad_token_id,
                dtype=torch.long,
                device=device
            )
        
        # Expand encoder outputs for beam search
        if num_beams > 1:
            expanded_batch_size = batch_size * num_beams
            
            # Expand encoder hidden states
            seq_len, _, hidden_dim = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_beams, dim=1)
            
            # Expand attention mask
            combined_attention_mask = combined_attention_mask.repeat_interleave(num_beams, dim=0)
        else:
            expanded_batch_size = batch_size
        
        # Track finished sequences
        unfinished_sequences = torch.ones(expanded_batch_size, dtype=torch.long, device=device)
        
        # Track beam scores
        beam_scores = torch.zeros((expanded_batch_size,), dtype=torch.float, device=device)
        if num_beams > 1:
            # Reshape beam scores for easier handling
            beam_scores = beam_scores.view(batch_size, num_beams)
        
        # Generation loop
        for step in range(max_length):
            # Prepare decoder inputs
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=combined_attention_mask
            )
            
            # Get next token logits
            next_token_logits = decoder_outputs["logits"]
            
            # If logits are 2D [batch_size*seq_len, vocab_size], reshape to 3D
            if len(next_token_logits.shape) == 2:
                seq_len = decoder_input_ids.shape[1]
                next_token_logits = next_token_logits.view(expanded_batch_size, seq_len, -1)
            
            # We only need the last token's logits
            next_token_logits = next_token_logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(expanded_batch_size):
                    for previous_token in decoder_input_ids[i]:
                        if previous_token in [pad_token_id, eos_token_id]:
                            continue
                        next_token_logits[i, previous_token] /= repetition_penalty
            
            # Apply no repeat n-gram blocking
            if no_repeat_ngram_size > 0:
                # Implement n-gram blocking (simplified)
                for i in range(expanded_batch_size):
                    for ngram_size in range(2, min(no_repeat_ngram_size + 1, decoder_input_ids.shape[1])):
                        ngrams = decoder_input_ids[i, -(ngram_size-1):]
                        for j in range(len(decoder_input_ids[i]) - ngram_size + 1):
                            if torch.equal(decoder_input_ids[i, j:j+ngram_size-1], ngrams):
                                next_token = decoder_input_ids[i, j+ngram_size-1]
                                next_token_logits[i, next_token] = -float('inf')
            
            # Mask tokens below minimum length
            if step < min_length:
                next_token_logits[:, eos_token_id] = -float('inf')
            
            # Apply top-k sampling
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('inf')
            
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
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Convert logits to probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample next token
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            # Update beam scores
            if num_beams > 1:
                # Calculate new beam scores
                next_beam_scores = beam_scores.view(-1, 1) + torch.log(probs.gather(1, next_tokens))
                next_beam_scores = next_beam_scores.view(batch_size, -1)
                
                # Get top-k beam indices and token indices
                beam_indices = torch.topk(next_beam_scores, num_beams, dim=1)[1]
                next_tokens = torch.gather(next_tokens.view(batch_size, -1), 1, beam_indices)
                
                # Update beam scores
                beam_scores = torch.gather(next_beam_scores, 1, beam_indices)
                
                # Reshape for batch handling
                next_tokens = next_tokens.view(-1, 1)
            
            # Finished sequences should have their next token as the pad token
            next_tokens = next_tokens * unfinished_sequences.unsqueeze(1) + pad_token_id * (1 - unfinished_sequences.unsqueeze(1))
            
            # Add next token to sequence
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=1)
            
            # Update unfinished sequences
            unfinished_sequences = unfinished_sequences * (next_tokens.squeeze() != eos_token_id)
            
            # Early stopping
            if early_stopping and unfinished_sequences.max() == 0:
                break
        
        # Process beams to select the best ones
        if num_beams > 1 and num_return_sequences < num_beams:
            # Get top sequences based on beam scores
            _, indices = torch.topk(beam_scores.view(batch_size, num_beams), num_return_sequences, dim=1)
            indices = indices + torch.arange(batch_size, device=device).unsqueeze(1) * num_beams
            best_sequences = torch.index_select(decoder_input_ids, 0, indices.view(-1))
            decoder_input_ids = best_sequences.view(batch_size * num_return_sequences, -1)
        
        # Reshape output if needed
        if num_return_sequences > 1:
            decoder_input_ids = decoder_input_ids.view(batch_size, num_return_sequences, -1)
        
        return decoder_input_ids 