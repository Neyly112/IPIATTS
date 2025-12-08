"""
LLM-based Prosody Analysis Module
Analyzes input text to extract prosody features (pitch, energy, duration patterns)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

import matcha.utils as utils

log = utils.get_pylogger(__name__)


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
        n_prosody_features: int = 1,
        freeze_llm: bool = True,
        use_adapter: bool = True,
        adapter_dim: int = 64,
    ):
        """
        Args:
            llm_model_name: Name of the pre-trained LLM (default: PhoBERT)
            prosody_dim: Dimension of prosody features
            n_prosody_features: Number of prosody feature streams (default 1 for global conditioning)
            freeze_llm: Whether to freeze the LLM weights
            use_adapter: Reserved for future adapter support
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

            model_positions = getattr(
                self.llm.config, "max_position_embeddings", 512)

            emb_layer = getattr(self.llm, "embeddings", None)
            if emb_layer is not None and hasattr(emb_layer, "position_embeddings"):
                actual_positions = emb_layer.position_embeddings.num_embeddings
            else:
                actual_positions = model_positions

            # Giới hạn an toàn cho position (trừ bớt 1 để tránh off-by-one)
            self.llm_max_position_embeddings = max(
                1, min(model_positions, actual_positions) - 1
            )

        except Exception as e:
            print(
                f"[!] Could not load {llm_model_name}, using dummy LLM. Error: {e}")
            # Fallback to a simple embedding layer
            llm_hidden_size = 768
            self.llm = None
            self.tokenizer = None
            self.llm_max_position_embeddings = 1024
            self.llm_pad_token_id = 0

        # Freeze LLM if specified
        if freeze_llm and self.llm is not None:
            for param in self.llm.parameters():
                param.requires_grad = False

        self.prosody_projection = nn.Linear(llm_hidden_size, prosody_dim)
        self.prosody_fusion = nn.Sequential(
            nn.Linear(prosody_dim * n_prosody_features, prosody_dim),
            nn.LayerNorm(prosody_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self._llm_truncation_warning_logged = False

    def forward(
        self,
        text_input: torch.Tensor,
        text_lengths: torch.Tensor,
        raw_texts: Optional[list[str]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            text_input: Phoneme sequence [batch, seq_len]
            text_lengths: Lengths of sequences [batch]
            raw_texts: Optional list of original Vietnamese sentences for each batch entry

        Returns:
            prosody_features: Combined prosody features [batch, prosody_dim, seq_len]
            prosody_dict: Dictionary containing the global prosody vector
        """
        batch_size, seq_len = text_input.shape

        prosody_dict = {}
        if self.llm is not None and raw_texts:
            prosody_dict = {}

        if self.llm is not None and raw_texts:
            if len(raw_texts) != batch_size:
                raise ValueError("raw_texts length must match batch size")

            # Số special tokens PhoBERT tự thêm (CLS, SEP, …)
            special_tokens = self.tokenizer.num_special_tokens_to_add(
                pair=False)

            # Số token text tối đa mình cho phép (không tính special)
            max_text_tokens = max(
                1,
                self.llm_max_position_embeddings - special_tokens,
            )

            # 1) Tokenize raw_texts với truncation an toàn
            encoding = self.tokenizer(
                raw_texts,
                truncation=True,
                max_length=max_text_tokens,  # chỉ tính phần text
                padding=True,
                return_tensors="pt",
            )
            encoding = {k: v.to(text_input.device)
                        for k, v in encoding.items()}
            input_ids = encoding["input_ids"].long()
            attn_mask = encoding["attention_mask"].long()

            # 2) Nếu vì lý do nào đó chiều dài vẫn vượt limit → cắt thêm
            b, t = input_ids.shape
            if t > self.llm_max_position_embeddings:
                if not self._llm_truncation_warning_logged:
                    log.warning(
                        "LLMProsodyAnalyzer: seq_len=%d > llm_max_position_embeddings=%d, "
                        "clamping to safe range.",
                        t,
                        self.llm_max_position_embeddings,
                    )
                    self._llm_truncation_warning_logged = True

                input_ids = input_ids[:,
                                      : self.llm_max_position_embeddings].contiguous()
                attn_mask = attn_mask[:,
                                      : self.llm_max_position_embeddings].contiguous()
                t = self.llm_max_position_embeddings

            # 3) Tự tạo position_ids và clamp cho chắc
            pos_ids = torch.arange(t, device=text_input.device)
            pos_ids = pos_ids.clamp(max=self.llm_max_position_embeddings - 1)
            pos_ids = pos_ids.unsqueeze(0).expand(b, -1).contiguous()

            # 4) Forward PhoBERT
            grad_ctx = torch.enable_grad() if self.training else torch.no_grad()
            with grad_ctx:
                outputs = self.llm(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    position_ids=pos_ids,
                    return_dict=True,
                )

            # 5) Dùng CLS hidden làm global prosody
            cls_hidden = outputs.last_hidden_state[:, 0, :]  # [B, hidden]
            global_prosody = self.prosody_projection(
                cls_hidden)  # [B, prosody_dim]

            # 6) Broadcast global prosody theo chiều phoneme seq_len
            prosody_repeated = (
                global_prosody.unsqueeze(1)
                .expand(-1, seq_len, -1)
                .contiguous()
            )
            prosody_dict["global"] = global_prosody
        else:
            prosody_repeated = torch.zeros(
                batch_size,
                seq_len,
                self.prosody_dim,
                device=text_input.device,
            )
            prosody_dict["global"] = torch.zeros(
                batch_size,
                self.prosody_dim,
                device=text_input.device,
            )

        prosody_fused = self.prosody_fusion(prosody_repeated)
        prosody_fused = prosody_fused.transpose(1, 2)

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
            nn.Conv1d(embedding_dim, prosody_dim *
                      2, kernel_size=5, padding=2),
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
