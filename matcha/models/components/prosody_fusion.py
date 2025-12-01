"""
Prosody Fusion & Conditioning Module
Combines prosody features from LLM with text encoder outputs
"""

import torch
import torch.nn as nn
from typing import Optional


class ProsodyFusion(nn.Module):
    """
    Fuses prosody features with text encoder outputs.
    Implements attention-based fusion and feature conditioning.
    """
    
    def __init__(
        self,
        text_channels: int = 512,
        prosody_channels: int = 256,
        fusion_channels: int = 512,
        use_attention: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args:
            text_channels: Number of channels in text encoder output
            prosody_channels: Number of channels in prosody features
            fusion_channels: Number of channels after fusion
            use_attention: Whether to use attention for fusion
            dropout: Dropout rate
        """
        super().__init__()
        
        self.text_channels = text_channels
        self.prosody_channels = prosody_channels
        self.fusion_channels = fusion_channels
        self.use_attention = use_attention
        
        # Project prosody features to match text dimension
        self.prosody_proj = nn.Conv1d(prosody_channels, text_channels, 1)
        
        if use_attention:
            # Cross-attention: text attends to prosody
            self.text_query = nn.Conv1d(text_channels, text_channels, 1)
            self.prosody_key = nn.Conv1d(text_channels, text_channels, 1)
            self.prosody_value = nn.Conv1d(text_channels, text_channels, 1)
            self.attn_dropout = nn.Dropout(dropout)
        
        # Fusion layers
        self.fusion_net = nn.Sequential(
            nn.Conv1d(text_channels * 2, fusion_channels, kernel_size=1),
            nn.LayerNorm(fusion_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(fusion_channels, fusion_channels, kernel_size=1),
        )
        
        # Gating mechanism to control prosody influence
        self.gate = nn.Sequential(
            nn.Conv1d(text_channels * 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        text_features: torch.Tensor,
        prosody_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            text_features: Text encoder output [batch, text_channels, seq_len]
            prosody_features: Prosody features [batch, prosody_channels, seq_len]
            mask: Mask for attention [batch, 1, seq_len]
        
        Returns:
            fused_features: Fused features [batch, fusion_channels, seq_len]
        """
        # Project prosody to text dimension
        prosody_proj = self.prosody_proj(prosody_features)
        
        if self.use_attention:
            # Cross-attention
            query = self.text_query(text_features)  # [batch, text_channels, seq_len]
            key = self.prosody_key(prosody_proj)
            value = self.prosody_value(prosody_proj)
            
            # Compute attention scores
            scores = torch.bmm(
                query.transpose(1, 2),  # [batch, seq_len, text_channels]
                key  # [batch, text_channels, seq_len]
            )  # [batch, seq_len, seq_len]
            
            scores = scores / (self.text_channels ** 0.5)
            
            if mask is not None:
                mask_expanded = mask.squeeze(1).unsqueeze(1)  # [batch, 1, seq_len]
                scores = scores.masked_fill(mask_expanded == 0, -1e9)
            
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            
            # Apply attention to values
            prosody_attended = torch.bmm(
                attn_weights,  # [batch, seq_len, seq_len]
                value.transpose(1, 2)  # [batch, seq_len, text_channels]
            ).transpose(1, 2)  # [batch, text_channels, seq_len]
        else:
            # Simple addition
            prosody_attended = prosody_proj
        
        # Concatenate text and attended prosody
        combined = torch.cat([text_features, prosody_attended], dim=1)
        
        # Compute gating values
        gate_values = self.gate(combined)
        
        # Apply gating to prosody
        prosody_gated = prosody_attended * gate_values
        
        # Fuse features
        combined_gated = torch.cat([text_features, prosody_gated], dim=1)
        fused = self.fusion_net(combined_gated)
        
        if mask is not None:
            fused = fused * mask
        
        return fused


class ProsodyConditioner(nn.Module):
    """
    Conditions the decoder on prosody features.
    Used in the CFM decoder to incorporate prosody information.
    """
    
    def __init__(
        self,
        prosody_dim: int = 256,
        decoder_channels: int = 512,
        condition_method: str = "add",  # "add", "concat", "film"
    ):
        """
        Args:
            prosody_dim: Dimension of prosody features
            decoder_channels: Number of channels in decoder
            condition_method: Method to condition ("add", "concat", "film")
        """
        super().__init__()
        
        self.prosody_dim = prosody_dim
        self.decoder_channels = decoder_channels
        self.condition_method = condition_method
        
        if condition_method == "add":
            # Project prosody to decoder dimension
            self.prosody_proj = nn.Conv1d(prosody_dim, decoder_channels, 1)
        elif condition_method == "concat":
            # Concatenation requires no projection, handled externally
            pass
        elif condition_method == "film":
            # FiLM: Feature-wise Linear Modulation
            self.film_scale = nn.Conv1d(prosody_dim, decoder_channels, 1)
            self.film_shift = nn.Conv1d(prosody_dim, decoder_channels, 1)
        else:
            raise ValueError(f"Unknown condition method: {condition_method}")
    
    def forward(
        self,
        decoder_input: torch.Tensor,
        prosody_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            decoder_input: Decoder input [batch, decoder_channels, seq_len]
            prosody_features: Prosody features [batch, prosody_dim, seq_len]
        
        Returns:
            conditioned_input: Conditioned decoder input
        """
        if self.condition_method == "add":
            prosody_proj = self.prosody_proj(prosody_features)
            return decoder_input + prosody_proj
        
        elif self.condition_method == "concat":
            return torch.cat([decoder_input, prosody_features], dim=1)
        
        elif self.condition_method == "film":
            scale = self.film_scale(prosody_features)
            shift = self.film_shift(prosody_features)
            return decoder_input * (1 + scale) + shift
        
        return decoder_input
