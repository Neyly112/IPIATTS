"""
LLM-based Prosody Analysis Module
Analyzes input text to extract prosody features (pitch, energy, duration patterns)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import AutoModel, AutoTokenizer


class LLMProsodyAnalyzer(nn.Module):
    """
    Uses a pre-trained language model (PhoBERT/XLM-R) to analyze prosody from text.
    Extracts features related to:
    - Pitch patterns (intonation)
    - Energy/stress patterns
    - Duration/rhythm patterns
    """
    
    def __init__(
        self,
        llm_model_name: str = "vinai/phobert-base",
        prosody_dim: int = 256,
        n_prosody_features: int = 3,  # pitch, energy, duration
        freeze_llm: bool = True,
        use_adapter: bool = True,
        adapter_dim: int = 64,
    ):
        """
        Args:
            llm_model_name: Name of the pre-trained LLM (default: PhoBERT)
            prosody_dim: Dimension of prosody features
            n_prosody_features: Number of prosody feature types
            freeze_llm: Whether to freeze the LLM weights
            use_adapter: Whether to use adapter layers for efficiency
            adapter_dim: Dimension of adapter layers
        """
        super().__init__()
        
        self.prosody_dim = prosody_dim
        self.n_prosody_features = n_prosody_features
        self.use_adapter = use_adapter
        
        # Load pre-trained LLM
        try:
            self.llm = AutoModel.from_pretrained(llm_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            llm_hidden_size = self.llm.config.hidden_size
        except Exception as e:
            print(f"[!] Could not load {llm_model_name}, using dummy LLM. Error: {e}")
            # Fallback to a simple embedding layer
            llm_hidden_size = 768
            self.llm = None
            self.tokenizer = None
        
        # Freeze LLM if specified
        if freeze_llm and self.llm is not None:
            for param in self.llm.parameters():
                param.requires_grad = False
        
        # Adapter layers (lightweight fine-tuning)
        if use_adapter and self.llm is not None:
            self.adapter_down = nn.Linear(llm_hidden_size, adapter_dim)
            self.adapter_up = nn.Linear(adapter_dim, llm_hidden_size)
            self.adapter_activation = nn.ReLU()
            self.adapter_ln = nn.LayerNorm(llm_hidden_size)
        
        # Prosody prediction heads
        self.pitch_predictor = nn.Sequential(
            nn.Linear(llm_hidden_size, prosody_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(prosody_dim, prosody_dim),
        )
        
        self.energy_predictor = nn.Sequential(
            nn.Linear(llm_hidden_size, prosody_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(prosody_dim, prosody_dim),
        )
        
        self.duration_predictor = nn.Sequential(
            nn.Linear(llm_hidden_size, prosody_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(prosody_dim, prosody_dim),
        )
        
        # Fusion layer to combine all prosody features
        self.prosody_fusion = nn.Sequential(
            nn.Linear(prosody_dim * n_prosody_features, prosody_dim),
            nn.LayerNorm(prosody_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
    def forward(
        self,
        text_input: torch.Tensor,
        text_lengths: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            text_input: Phoneme sequence [batch, seq_len]
            text_lengths: Lengths of sequences [batch]
            attention_mask: Attention mask [batch, seq_len]
        
        Returns:
            prosody_features: Combined prosody features [batch, prosody_dim, seq_len]
            prosody_dict: Dictionary containing individual prosody predictions
        """
        batch_size, seq_len = text_input.shape
        
        if self.llm is not None:
            # Get LLM embeddings
            with torch.no_grad() if not self.training else torch.enable_grad():
                llm_outputs = self.llm(
                    input_ids=text_input,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                llm_hidden = llm_outputs.last_hidden_state  # [batch, seq_len, hidden_size]
            
            # Apply adapter if using
            if self.use_adapter:
                adapted = self.adapter_down(llm_hidden)
                adapted = self.adapter_activation(adapted)
                adapted = self.adapter_up(adapted)
                llm_hidden = self.adapter_ln(llm_hidden + adapted)
        else:
            # Dummy embeddings if LLM not available
            llm_hidden = torch.randn(
                batch_size, seq_len, 768,
                device=text_input.device, dtype=torch.float32
            )
        
        # Predict prosody features
        pitch_features = self.pitch_predictor(llm_hidden)  # [batch, seq_len, prosody_dim]
        energy_features = self.energy_predictor(llm_hidden)
        duration_features = self.duration_predictor(llm_hidden)
        
        # Concatenate all prosody features
        all_prosody = torch.cat([pitch_features, energy_features, duration_features], dim=-1)
        
        # Fuse prosody features
        prosody_fused = self.prosody_fusion(all_prosody)  # [batch, seq_len, prosody_dim]
        
        # Transpose to [batch, prosody_dim, seq_len] for compatibility with Conv1d
        prosody_fused = prosody_fused.transpose(1, 2)
        
        prosody_dict = {
            "pitch": pitch_features.transpose(1, 2),
            "energy": energy_features.transpose(1, 2),
            "duration": duration_features.transpose(1, 2),
        }
        
        return prosody_fused, prosody_dict


class SimpleProsodyAnalyzer(nn.Module):
    """
    Lightweight prosody analyzer without LLM (for faster inference)
    Uses only the text embeddings to predict prosody
    """
    
    def __init__(
        self,
        n_vocab: int = 256,
        embedding_dim: int = 512,
        prosody_dim: int = 256,
        n_prosody_features: int = 3,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(n_vocab, embedding_dim)
        self.prosody_dim = prosody_dim
        
        # Prosody prediction network
        # Use BatchNorm1d instead of LayerNorm for Conv1d layers
        self.prosody_net = nn.Sequential(
            nn.Conv1d(embedding_dim, prosody_dim * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(prosody_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(prosody_dim * 2, prosody_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(prosody_dim),
            nn.ReLU(),
        )
        
    def forward(
        self,
        text_input: torch.Tensor,
        text_lengths: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            text_input: Phoneme sequence [batch, seq_len]
            text_lengths: Lengths of sequences [batch]
        
        Returns:
            prosody_features: Combined prosody features [batch, prosody_dim, seq_len]
            prosody_dict: Empty dict for compatibility
        """
        # Embed text
        x = self.embedding(text_input)  # [batch, seq_len, embedding_dim]
        x = x.transpose(1, 2)  # [batch, embedding_dim, seq_len]
        
        # Predict prosody
        prosody_features = self.prosody_net(x)  # [batch, prosody_dim, seq_len]
        
        return prosody_features, {}
