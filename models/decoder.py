"""
Decoder for the Bio-ChemTransformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class TransformerDecoderLayer(nn.Module):
    """
    Standard transformer decoder layer with self-attention and cross-attention.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation_fn: str = "gelu",
        layernorm_eps: float = 1e-12
    ):
        """
        Initialize the decoder layer.
        
        Args:
            embed_dim: Dimension of the embeddings
            num_heads: Number of attention heads
            ffn_dim: Dimension of the feed-forward network
            dropout: Dropout probability
            activation_fn: Activation function
            layernorm_eps: Epsilon for layer normalization
        """
        super().__init__()
        
        # Store dimensions
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False  # [seq_len, batch, embed_dim]
        )
        
        # Cross-attention layer
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False  # [seq_len, batch, embed_dim]
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=layernorm_eps)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=layernorm_eps)
        self.layer_norm3 = nn.LayerNorm(embed_dim, eps=layernorm_eps)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # Feed-forward network
        self.activation_fn = getattr(F, activation_fn)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        self_attention_mask: Optional[torch.Tensor] = None,
        self_head_mask: Optional[torch.Tensor] = None,
        cross_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass of the decoder layer.
        
        Args:
            hidden_states: Input tensor [seq_len, batch_size, embed_dim]
            encoder_hidden_states: Hidden states from encoder [src_len, batch_size, embed_dim]
            encoder_attention_mask: Attention mask for encoder states [batch_size, src_len]
            self_attention_mask: Self-attention mask [tgt_len, tgt_len]
            self_head_mask: Mask for self-attention heads
            cross_head_mask: Mask for cross-attention heads
            output_attentions: Whether to return attention weights
        """
        # Self-attention block
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        
        # Ensure hidden states are in [seq_len, batch_size, embed_dim]
        if hidden_states.size(1) == self.embed_dim:
            hidden_states = hidden_states.transpose(0, 1)
            print(f"Transposed hidden states to: {hidden_states.shape}")
        
        # Self-attention
        self_attn_output, self_attn_weights = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            attn_mask=self_attention_mask,
            key_padding_mask=None,
            need_weights=output_attentions
        )
        
        # Apply dropout and residual connection
        hidden_states = residual + self.dropout1(self_attn_output)
        
        # Cross-attention block
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        
        # Ensure encoder hidden states are in [src_len, batch_size, embed_dim]
        if encoder_hidden_states.size(1) == self.embed_dim:
            encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
            print(f"Transposed encoder hidden states to: {encoder_hidden_states.shape}")
        
        # Debug print shapes
        print(f"Cross attention shapes:")
        print(f"Query shape: {hidden_states.shape}")
        print(f"Key/Value shape: {encoder_hidden_states.shape}")
        
        # Handle encoder attention mask
        if encoder_attention_mask is not None:
            # Ensure mask is float
            if encoder_attention_mask.dtype != torch.float32:
                encoder_attention_mask = encoder_attention_mask.float()
        
        # Cross-attention
        cross_attn_output, cross_attn_weights = self.cross_attn(
            query=hidden_states,
            key=encoder_hidden_states,
            value=encoder_hidden_states,
            attn_mask=None,
            key_padding_mask=encoder_attention_mask,
            need_weights=output_attentions
        )
        
        # Apply dropout and residual connection
        hidden_states = residual + self.dropout2(cross_attn_output)
        
        # Feed-forward block
        residual = hidden_states
        hidden_states = self.layer_norm3(hidden_states)
        
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        
        hidden_states = residual + self.dropout3(hidden_states)
        
        if output_attentions:
            return hidden_states, self_attn_weights, cross_attn_weights
        return hidden_states, None, None


class TransformerDecoder(nn.Module):
    """
    Full transformer decoder with multiple decoder layers.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        vocab_size: int,
        dropout: float = 0.1,
        activation_fn: str = "gelu",
        layernorm_eps: float = 1e-12,
        layerdrop: float = 0.0,
        max_position_embeddings: int = 512,
        pad_token_id: int = 0
    ):
        """
        Initialize the decoder.
        
        Args:
            embed_dim: Dimension of the embeddings
            num_layers: Number of decoder layers
            num_heads: Number of attention heads
            ffn_dim: Dimension of the feed-forward network
            vocab_size: Size of the output vocabulary
            dropout: Dropout probability
            activation_fn: Activation function
            layernorm_eps: Epsilon for layer normalization
            layerdrop: Probability of dropping a layer during training
            max_position_embeddings: Maximum number of position embeddings
            pad_token_id: ID of the padding token
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.layerdrop = layerdrop
        self.embed_dim = embed_dim
        self.pad_token_id = pad_token_id
        
        # Embeddings
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.embed_positions = nn.Embedding(max_position_embeddings, embed_dim)
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                activation_fn=activation_fn,
                layernorm_eps=layernorm_eps
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim, eps=layernorm_eps)
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # For positional embeddings
        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1))
        )
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize the model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def get_input_embeddings(self) -> nn.Embedding:
        """Get the input embeddings."""
        return self.embed_tokens
    
    def set_input_embeddings(self, embeddings: nn.Embedding):
        """Set the input embeddings."""
        self.embed_tokens = embeddings
    
    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the decoder.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            encoder_hidden_states: Hidden states from encoder [batch_size, src_len, embed_dim]
            encoder_attention_mask: Attention mask for encoder states [batch_size, src_len]
            attention_mask: Attention mask for decoder [batch_size, seq_len]
            head_mask: Mask for attention heads
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            
        Returns:
            Dictionary containing:
                - logits: Output logits [batch_size * seq_len, vocab_size]
                - hidden_states: All hidden states (optional)
                - attentions: All attention weights (optional)
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attns = () if output_attentions else None
        
        # Get input embeddings
        input_shape = input_ids.shape
        batch_size, seq_length = input_shape
        
        print(f"Input shape: {input_shape}")
        
        # Create position IDs
        positions = self.position_ids[:, :seq_length]
        
        # Create token embeddings
        inputs_embeds = self.embed_tokens(input_ids)  # [batch_size, seq_len, embed_dim]
        
        # Create position embeddings
        position_embeds = self.embed_positions(positions)  # [1, seq_len, embed_dim]
        
        # Combine token and position embeddings
        hidden_states = inputs_embeds + position_embeds  # [batch_size, seq_len, embed_dim]
        
        # Ensure hidden states are in [seq_len, batch_size, embed_dim]
        hidden_states = hidden_states.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        print(f"Hidden states after transpose: {hidden_states.shape}")
        
        # Ensure encoder hidden states are in [src_len, batch_size, embed_dim]
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)  # [src_len, batch_size, embed_dim]
        print(f"Encoder hidden states after transpose: {encoder_hidden_states.shape}")
        
        # Handle attention masks
        # Create causal mask for autoregressive decoding
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length),
                dtype=torch.bool,
                device=input_ids.device
            )
        
        # Create causal mask for self-attention
        causal_mask = torch.triu(
            torch.ones(
                (seq_length, seq_length),
                dtype=torch.bool,
                device=input_ids.device
            ),
            diagonal=1
        )
        
        print(f"Created causal mask shape: {causal_mask.shape}")
        
        # Handle encoder attention mask
        if encoder_attention_mask is not None:
            # Convert to float mask and flip the values since PyTorch attention masks are inverted
            encoder_attention_mask = (1.0 - encoder_attention_mask.float()) * torch.finfo(torch.float).min
        
        # Process through each decoder layer
        for i, layer in enumerate(self.layers):
            # Apply layerdrop during training
            if self.training and torch.rand(1).item() < self.layerdrop:
                continue
            
            # Save hidden states if needed
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            # Get layer-specific head mask
            layer_head_mask = head_mask[i] if head_mask is not None else None
            
            # Process through the layer
            layer_outputs = layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                self_attention_mask=causal_mask,
                self_head_mask=layer_head_mask,
                cross_head_mask=layer_head_mask,
                output_attentions=output_attentions
            )
            
            hidden_states = layer_outputs[0]
            
            # Save attention weights if needed
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                all_cross_attns += (layer_outputs[2],)
        
        # Apply final layer normalization
        hidden_states = self.layer_norm(hidden_states)  # [seq_len, batch_size, embed_dim]
        
        # Save final hidden states if needed
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # Project to vocabulary space
        logits = self.output_projection(hidden_states)  # [seq_len, batch_size, vocab_size]
        
        # Transpose back to [batch_size, seq_len, vocab_size]
        logits = logits.transpose(0, 1)
        
        # Debug print shapes before reshaping
        print(f"Logits shape before reshape: {logits.shape}")
        
        # Reshape to [batch_size * seq_len, vocab_size] for loss calculation
        logits = logits.reshape(-1, logits.size(-1))
        
        # Debug print final shapes
        print(f"Final logits shape: {logits.shape}")
        
        # Create output dictionary
        outputs = {
            "logits": logits,
        }
        
        if output_hidden_states:
            outputs["hidden_states"] = all_hidden_states
            
        if output_attentions:
            outputs["attentions"] = {
                "self_attentions": all_self_attns,
                "cross_attentions": all_cross_attns
            }
        
        return outputs 