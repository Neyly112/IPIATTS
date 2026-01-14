import datetime as dt
import math
import random

import torch
import torch.nn.functional as F

import matcha.utils.monotonic_align as monotonic_align  # pylint: disable=consider-using-from-import
from matcha import utils
from matcha.models.baselightningmodule import BaseLightningClass
from matcha.models.components.flow_matching import CFM
from matcha.models.components.text_encoder import TextEncoder
from matcha.models.components.prosody_analyzer import LLMProsodyAnalyzer, SimpleProsodyAnalyzer
from matcha.models.components.prosody_fusion import ProsodyFusion
from matcha.utils.model import (
    denormalize,
    duration_loss,
    fix_len_compatibility,
    generate_path,
    sequence_mask,
)

log = utils.get_pylogger(__name__)


class MatchaTTS(BaseLightningClass):  # 🍵
    def __init__(
        self,
        n_vocab,
        n_spks,
        spk_emb_dim,
        n_feats,
        encoder,
        decoder,
        cfm,
        data_statistics,
        out_size,
        optimizer=None,
        optimizer_kwargs=None,
        scheduler=None,
        prior_loss=True,
        use_precomputed_durations=False,
        llm_model_name="vinai/phobert-base",
        prosody_dim=256,
        use_token_level_prosody=True,
        finetune_llm=False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_feats = n_feats
        self.out_size = out_size
        self.prior_loss = prior_loss
        self.use_precomputed_durations = use_precomputed_durations
        self.prosody_dim = prosody_dim
        self.acoustic_loss_weight = 0.25

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)

        # Prosody analyzer với PhoBERT (luôn bật)
        self.prosody_analyzer = LLMProsodyAnalyzer(
            llm_model_name=llm_model_name,
            prosody_dim=prosody_dim,
            n_prosody_features=1,
            freeze_llm=not finetune_llm,  # Freeze if NOT finetuning
            use_adapter=True,
            use_token_level_prosody=use_token_level_prosody,
            finetune_llm=finetune_llm,
        )

        # Prosody fusion module
        self.prosody_fusion = ProsodyFusion(
            text_channels=encoder.encoder_params.n_feats,
            prosody_channels=prosody_dim,
            fusion_channels=encoder.encoder_params.n_feats,
            use_attention=True,
        )

        self.encoder = TextEncoder(
            encoder.encoder_type,
            encoder.encoder_params,
            encoder.duration_predictor_params,
            n_vocab,
            n_spks,
            spk_emb_dim,
        )

        # Acoustic prosody predictors (token-level pitch/energy) and conditioning
        feat_ch = encoder.encoder_params.n_feats
        self.pitch_predictor = torch.nn.Sequential(
            torch.nn.Conv1d(feat_ch, feat_ch, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(feat_ch, 1, kernel_size=1),
        )
        self.energy_predictor = torch.nn.Sequential(
            torch.nn.Conv1d(feat_ch, feat_ch, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(feat_ch, 1, kernel_size=1),
        )

        # NEW: Pause predictor (predicts pause duration between words)
        self.pause_predictor = torch.nn.Sequential(
            torch.nn.Conv1d(feat_ch, feat_ch, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(feat_ch, 1, kernel_size=1),
            torch.nn.Softplus(),  # Ensure non-negative pause duration
        )

        # NEW: Boundary detector (detects phrase/word boundaries)
        self.boundary_detector = torch.nn.Sequential(
            torch.nn.Conv1d(feat_ch, feat_ch, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(feat_ch, 1, kernel_size=1),
            torch.nn.Sigmoid(),  # Binary classification
        )

        self.pitch_cond = torch.nn.Conv1d(1, feat_ch, kernel_size=1)
        self.energy_cond = torch.nn.Conv1d(1, feat_ch, kernel_size=1)
        self.pause_cond = torch.nn.Conv1d(1, feat_ch, kernel_size=1)
        self.boundary_cond = torch.nn.Conv1d(1, feat_ch, kernel_size=1)

        self.decoder = CFM(
            in_channels=2 * encoder.encoder_params.n_feats,
            out_channel=encoder.encoder_params.n_feats,
            cfm_params=cfm,
            decoder_params=decoder,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        self.update_data_statistics(data_statistics)

    @torch.inference_mode()
    def synthesise(
        self,
        x,
        x_lengths,
        n_timesteps,
        temperature=1.0,
        spks=None,
        length_scale=1.0,
        raw_texts=None,
    ):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            spks (bool, optional): speaker ids.
                shape: (batch_size,)
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.

        Returns:
            dict: {
                "encoder_outputs": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Average mel spectrogram generated by the encoder
                "decoder_outputs": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Refined mel spectrogram improved by the CFM
                "attn": torch.Tensor, shape: (batch_size, max_text_length, max_mel_length),
                # Alignment map between text and mel spectrogram
                "mel": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Denormalized mel spectrogram
                "mel_lengths": torch.Tensor, shape: (batch_size,),
                # Lengths of mel spectrograms
                "rtf": float,
                # Real-time factor
            }
        """
        # For RTF computation
        t = dt.datetime.now()

        if self.n_spks > 1:
            # Get speaker embedding
            spks = self.spk_emb(spks.long())

        # LLM Prosody Analysis với PhoBERT
        prosody_features, prosody_dict = self.prosody_analyzer(
            text_input=x,
            text_lengths=x_lengths,
            raw_texts=raw_texts,
        )

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks)

        # Prosody Fusion & Conditioning
        mu_x = self.prosody_fusion(mu_x, prosody_features, x_mask)

        # Predict token-level prosody features
        pitch_pred_tokens = self.pitch_predictor(mu_x) * x_mask
        energy_pred_tokens = self.energy_predictor(mu_x) * x_mask
        pause_pred_tokens = self.pause_predictor(mu_x) * x_mask
        boundary_pred_tokens = self.boundary_detector(mu_x) * x_mask

        # Apply prosody conditioning
        mu_x = mu_x + self.pitch_cond(pitch_pred_tokens) + \
            self.energy_cond(energy_pred_tokens) + \
            self.pause_cond(pause_pred_tokens) + \
            self.boundary_cond(boundary_pred_tokens)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = y_lengths.max()
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(
            y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(
            1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(
            1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Generate sample tracing the probability flow
        decoder_outputs = self.decoder(
            mu_y, y_mask, n_timesteps, temperature, spks)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        t = (dt.datetime.now() - t).total_seconds()
        rtf = t * 22050 / (decoder_outputs.shape[-1] * 256)

        return {
            "encoder_outputs": encoder_outputs,
            "decoder_outputs": decoder_outputs,
            "attn": attn[:, :, :y_max_length],
            "mel": denormalize(decoder_outputs, self.mel_mean, self.mel_std),
            "mel_lengths": y_lengths,
            "rtf": rtf,
        }

    def forward(
        self,
        x,
        x_lengths,
        y,
        y_lengths,
        spks=None,
        out_size=None,
        cond=None,
        durations=None,
        raw_texts=None,
        pitch=None,
        energy=None,
    ):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotonic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. flow matching loss: loss between mel-spectrogram and decoder outputs.

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            y (torch.Tensor): batch of corresponding mel-spectrograms.
                shape: (batch_size, n_feats, max_mel_length)
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
                shape: (batch_size,)
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
            spks (torch.Tensor, optional): speaker ids.
                shape: (batch_size,)
        """
        if self.n_spks > 1:
            # Get speaker embedding
            spks = self.spk_emb(spks)

        # LLM Prosody Analysis với PhoBERT
        prosody_features, prosody_dict = self.prosody_analyzer(
            text_input=x,
            text_lengths=x_lengths,
            raw_texts=raw_texts,
        )

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks)

        # Prosody Fusion & Conditioning (text-level)
        mu_x = self.prosody_fusion(mu_x, prosody_features, x_mask)

        # Predict token-level prosody features
        # Predictors output shape: [B, n_spks, seq_len] -> squeeze to [B, seq_len]
        pitch_pred_tokens = self.pitch_predictor(
            mu_x).squeeze(1) * x_mask.squeeze(1)
        energy_pred_tokens = self.energy_predictor(
            mu_x).squeeze(1) * x_mask.squeeze(1)
        pause_pred_tokens = self.pause_predictor(
            mu_x).squeeze(1) * x_mask.squeeze(1)
        boundary_pred_tokens = self.boundary_detector(
            mu_x).squeeze(1) * x_mask.squeeze(1)

        # Apply prosody conditioning
        mu_x = mu_x + self.pitch_cond(pitch_pred_tokens) + \
            self.energy_cond(energy_pred_tokens) + \
            self.pause_cond(pause_pred_tokens) + \
            self.boundary_cond(boundary_pred_tokens)

        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        if self.use_precomputed_durations:
            attn = generate_path(durations.squeeze(1), attn_mask.squeeze(1))
        else:
            # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
            with torch.no_grad():
                const = -0.5 * math.log(2 * math.pi) * self.n_feats
                factor = -0.5 * \
                    torch.ones(mu_x.shape, dtype=mu_x.dtype,
                               device=mu_x.device)
                y_square = torch.matmul(factor.transpose(1, 2), y**2)
                y_mu_double = torch.matmul(
                    2.0 * (factor * mu_x).transpose(1, 2), y)
                mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)
                log_prior = y_square - y_mu_double + mu_square + const

                attn = monotonic_align.maximum_path(
                    log_prior, attn_mask.squeeze(1))
                attn = attn.detach()  # b, t_text, T_mel

        # Token-level acoustic targets from frame-level pitch/energy via alignment
        acoustic_losses = {
            "pitch_loss": torch.tensor(0.0, device=y.device),
            "energy_loss": torch.tensor(0.0, device=y.device),
            "pause_loss": torch.tensor(0.0, device=y.device),
            "boundary_loss": torch.tensor(0.0, device=y.device),
        }
        if pitch is not None and energy is not None:
            attn_token = attn  # (batch, x_len, y_len)
            denom = torch.sum(attn_token, dim=-1, keepdim=True) + 1e-5
            # pitch, energy shape: (batch, y_len) -> unsqueeze(1) -> (batch, 1, y_len)
            pitch_token_gt = torch.sum(
                # (batch, x_len, 1)
                attn_token * pitch.unsqueeze(1), dim=-1, keepdim=True) / denom
            energy_token_gt = torch.sum(
                # (batch, x_len, 1)
                attn_token * energy.unsqueeze(1), dim=-1, keepdim=True) / denom

            # Remove last dimension and squeeze for loss computation
            pitch_token_gt = pitch_token_gt.squeeze(-1)  # (batch, x_len)
            energy_token_gt = energy_token_gt.squeeze(-1)  # (batch, x_len)

            pitch_mask = x_mask  # (batch, 1, x_len)
            energy_mask = x_mask
            acoustic_losses["pitch_loss"] = F.mse_loss(
                pitch_pred_tokens * pitch_mask.squeeze(1),
                pitch_token_gt * pitch_mask.squeeze(1),
                reduction="sum",
            ) / torch.sum(pitch_mask)
            acoustic_losses["energy_loss"] = F.mse_loss(
                energy_pred_tokens * energy_mask.squeeze(1),
                energy_token_gt * energy_mask.squeeze(1),
                reduction="sum",
            ) / torch.sum(energy_mask)

            # Pause prediction loss (simple heuristic: detect blank tokens)
            # Blank tokens (ID=0) should have higher pause
            blank_mask = (x == 0).float().unsqueeze(1)  # [B, 1, seq_len]
            pause_target = blank_mask.squeeze(
                1) * x_mask.squeeze(1)  # [B, seq_len]
            acoustic_losses["pause_loss"] = F.mse_loss(
                pause_pred_tokens * x_mask.squeeze(1),
                pause_target,
                reduction="sum",
            ) / torch.sum(x_mask)

            # Boundary detection loss (heuristic: boundaries at blank tokens)
            boundary_target = blank_mask.squeeze(
                1) * x_mask.squeeze(1)  # [B, seq_len]
            acoustic_losses["boundary_loss"] = F.binary_cross_entropy(
                boundary_pred_tokens * x_mask.squeeze(1),
                boundary_target,
                reduction="sum",
            ) / torch.sum(x_mask)

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        # refered to as prior loss in the paper
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        #   - "Hack" taken from Grad-TTS, in case of Grad-TTS, we cannot train batch size 32 on a 24GB GPU without it
        #   - Do not need this hack for Matcha-TTS, but it works with it as well
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(
                zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset = torch.LongTensor(
                [torch.tensor(random.choice(range(start, end)) if end >
                              start else 0) for start, end in offset_ranges]
            ).to(y_lengths)
            attn_cut = torch.zeros(
                attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
            y_cut = torch.zeros(
                y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)

            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + \
                    (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]

            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)

            attn = attn_cut
            y = y_cut
            y_mask = y_cut_mask

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(
            1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        # Compute loss of the decoder
        diff_loss, _ = self.decoder.compute_loss(
            x1=y, mask=y_mask, mu=mu_y, spks=spks, cond=cond)

        if self.prior_loss:
            prior_loss = torch.sum(
                0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
            prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        else:
            prior_loss = 0

        return dur_loss, prior_loss, diff_loss, attn, acoustic_losses

    def get_losses(self, batch):
        x, x_lengths = batch["x"], batch["x_lengths"]
        y, y_lengths = batch["y"], batch["y_lengths"]
        spks = batch["spks"]
        raw_texts = batch.get("raw_texts")
        pitch = batch.get("pitch")
        energy = batch.get("energy")

        dur_loss, prior_loss, diff_loss, attn, acoustic_losses = self(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            spks=spks,
            out_size=self.out_size,
            durations=batch["durations"],
            raw_texts=raw_texts,
            pitch=pitch,
            energy=energy,
        )

        acoustic_loss = self.acoustic_loss_weight * (
            acoustic_losses["pitch_loss"] +
            acoustic_losses["energy_loss"] +
            0.5 * acoustic_losses["pause_loss"] +  # Lower weight for pause
            0.3 * acoustic_losses["boundary_loss"]  # Lower weight for boundary
        )

        return {
            "dur_loss": dur_loss,
            "prior_loss": prior_loss,
            "diff_loss": diff_loss,
            "acoustic_loss": acoustic_loss,
            "pitch_loss": acoustic_losses["pitch_loss"],
            "energy_loss": acoustic_losses["energy_loss"],
            "pause_loss": acoustic_losses["pause_loss"],
            "boundary_loss": acoustic_losses["boundary_loss"],
        }
