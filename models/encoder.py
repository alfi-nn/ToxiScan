"""
Encoder for the Bio-ChemTransformer with Diagonal-Masked Attention (DMA).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class DiagonalMaskedAttention(nn.Module):
    """
    Implementation of Diagonal-Masked Attention (DMA) to prevent information leakage.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        dma_probability: float = 0.25
    ):
        """
        Initialize the DMA layer.
        
        Args:
            embed_dim: Dimension of the embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
            dma_probability: Probability of masking in DMA
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.dma_probability = dma_probability
        
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.scaling = self.head_dim ** -0.5
        
        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def _apply_diagonal_mask(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Apply diagonal masking to the attention weights.
        
        Args:
            attn_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
            
        Returns:
            Masked attention weights
        """
        batch_size, num_heads, seq_len, _ = attn_weights.shape
        
        # Create a mask for the diagonal and below
        # This implements the "information leakage prevention" from the paper
        mask = torch.tril(torch.ones(seq_len, seq_len, device=attn_weights.device), diagonal=0)
        
        # Randomly mask some elements in the diagonal with probability dma_probability
        # This introduces a form of regularization
        if self.training and self.dma_probability > 0:
            diagonal_mask = torch.rand(seq_len, device=attn_weights.device) > self.dma_probability
            diagonal_mask = torch.diag(diagonal_mask.float())
            mask = mask * (1.0 - diagonal_mask) + diagonal_mask
        
        # Apply the mask
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        return attn_weights
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with diagonal-masked attention.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, embed_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            layer_head_mask: Mask for attention heads
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project for queries, keys, values
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention weights
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        
        # Apply diagonal mask
        attn_weights = self._apply_diagonal_mask(attn_weights)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert from [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply dropout
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply head mask if provided
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        # Compute context vector
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Project output
        output = self.out_proj(context)
        
        return output, attn_weights if output_attentions else None


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with Diagonal-Masked Attention.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation_fn: str = "gelu",
        layernorm_eps: float = 1e-12,
        dma_probability: float = 0.25
    ):
        """
        Initialize the encoder layer.
        
        Args:
            embed_dim: Dimension of the embeddings
            num_heads: Number of attention heads
            ffn_dim: Dimension of the feed-forward network
            dropout: Dropout probability
            activation_fn: Activation function
            layernorm_eps: Epsilon for layer normalization
            dma_probability: Probability of masking in DMA
        """
        super().__init__()
        
        # Attention layer with Diagonal-Masked Attention
        self.self_attn = DiagonalMaskedAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            dma_probability=dma_probability
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=layernorm_eps)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=layernorm_eps)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed-forward network
        self.activation_fn = getattr(F, activation_fn)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the encoder layer.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, embed_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            layer_head_mask: Mask for attention heads
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Self-attention block
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        
        attn_output, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions
        )
        
        hidden_states = residual + self.dropout1(attn_output)
        
        # Feed-forward block
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        
        hidden_states = residual + self.dropout2(hidden_states)
        
        return hidden_states, attn_weights


class TransformerEncoder(nn.Module):
    """
    Full transformer encoder with multiple encoder layers.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation_fn: str = "gelu",
        layernorm_eps: float = 1e-12,
        layerdrop: float = 0.0,
        use_dma: bool = True,
        dma_probability: float = 0.25
    ):
        """
        Initialize the encoder.
        
        Args:
            embed_dim: Dimension of the embeddings
            num_layers: Number of encoder layers
            num_heads: Number of attention heads
            ffn_dim: Dimension of the feed-forward network
            dropout: Dropout probability
            activation_fn: Activation function
            layernorm_eps: Epsilon for layer normalization
            layerdrop: Probability of dropping a layer during training
            use_dma: Whether to use Diagonal-Masked Attention
            dma_probability: Probability of masking in DMA
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.layerdrop = layerdrop
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                activation_fn=activation_fn,
                layernorm_eps=layernorm_eps,
                dma_probability=dma_probability if use_dma else 0.0
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim, eps=layernorm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass of the encoder.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, embed_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            head_mask: Mask for attention heads
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            
        Returns:
            Tuple of (output, all_hidden_states, all_attentions)
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # Process through each encoder layer
        for i, layer in enumerate(self.layers):
            # Apply layerdrop during training
            if self.training and torch.rand(1).item() < self.layerdrop:
                continue
                
            # Get layer-specific head mask
            layer_head_mask = head_mask[i] if head_mask is not None else None
            
            # Save hidden states if needed
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            # Process through the layer
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions
            )
            
            hidden_states = layer_outputs[0]
            
            # Save attention weights if needed
            if output_attentions:
                all_attentions += (layer_outputs[1],)
        
        # Apply final layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        # Save final hidden states if needed
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        return (hidden_states, all_hidden_states, all_attentions) 