"""SAM2-UNet CD-FSS (aggressive) blueprint implementation.

This module is meant to live *inside* the SAM2-UNet repo.
It reuses:
  - SAM2 encoder construction + Adapter idea from SAM2UNet.py (WZH0120/SAM2-UNet)
  - PATNet's hypercorrelation + HPNLearner (slei109/PATNet)

Expected episode batch format is compatible with PATNet dataloaders:
  batch['query_img']:   (B, 3, H, W)
  batch['support_imgs']:(B, K, 3, H, W) or (B, 3, H, W)
  batch['support_masks']:(B, K, H, W) or (B, H, W)  (binary 0/1)

Output logits are 2-channel (bg, fg).

NOTE: This is an *architecture file*; training loop / loss is separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Reuse from SAM2-UNet ---
from sam2.build_sam import build_sam2

# We reuse Up + DoubleConv + Adapter from the original SAM2UNet implementation.
# If you prefer no cross-file import, you can copy these small classes from SAM2UNet.py.
from SAM2UNet import Adapter, Up

# --- Reuse from PATNet (vendor these files into your repo as described in the instructions) ---
# e.g. place them at: cdfss/patnet/base/{conv4d.py,correlation.py} and cdfss/patnet/learner.py
from cdfss.patnet.base.correlation import Correlation
from cdfss.patnet.learner import HPNLearner


# -------------------------
# Utility ops (mask/proto)
# -------------------------

def _resize_mask(mask: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    """Resize a binary mask to match a feature map spatial size.

    Args:
        mask: (B, H, W) or (B, 1, H, W) tensor (0/1 or 0/255 ok).
        size_hw: (H_feat, W_feat)

    Returns:
        mask_rs: (B, 1, H_feat, W_feat) float tensor in [0, 1].
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()
    # If mask is 0/255 style, squash to 0/1
    if mask.max() > 1.5:
        mask = (mask > 127).float()
    mask_rs = F.interpolate(mask, size=size_hw, mode="nearest")
    return mask_rs


def masked_avg_pool(feat: torch.Tensor, mask_rs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Masked average pooling to create a prototype.

    Args:
        feat: (B, C, H, W)
        mask_rs: (B, 1, H, W) in {0,1}

    Returns:
        proto: (B, C)
    """
    masked = feat * mask_rs
    denom = mask_rs.sum(dim=(2, 3)) + eps  # (B,1)
    proto = masked.sum(dim=(2, 3)) / denom  # (B,C)
    return proto


def masked_topk_tokens(
    feat: torch.Tensor,
    mask_rs: torch.Tensor,
    k: int,
    fallback_to_full: bool = True,
) -> torch.Tensor:
    """Select top-k tokens (by feature L2 norm) inside a mask.

    Deterministic (no random sampling) so 5-shot eval is stable.

    Args:
        feat: (B, C, H, W)
        mask_rs: (B, 1, H, W) in {0,1}
        k: number of tokens to return

    Returns:
        tokens: (B, k, C)
    """
    b, c, h, w = feat.shape
    scores = feat.pow(2).sum(dim=1).sqrt()  # (B,H,W)
    mask_flat = mask_rs.view(b, -1)  # (B,H*W)

    if fallback_to_full:
        # If a support mask is empty after resize, fall back to full image.
        valid = (mask_flat.sum(dim=1) > 0)
        if not torch.all(valid):
            # set empty masks to ones
            mask_rs = mask_rs.clone()
            for i in range(b):
                if not valid[i]:
                    mask_rs[i] = 1.0
            mask_flat = mask_rs.view(b, -1)

    scores = scores.view(b, -1)
    scores = scores.masked_fill(mask_flat < 0.5, float("-inf"))

    # If k > H*W, clamp
    k_eff = min(k, scores.size(1))
    idx = torch.topk(scores, k_eff, dim=1).indices  # (B,k)

    feat_flat = feat.view(b, c, -1)  # (B,C,H*W)
    idx_exp = idx.unsqueeze(1).expand(-1, c, -1)  # (B,C,k)
    tok = feat_flat.gather(dim=2, index=idx_exp)  # (B,C,k)
    tok = tok.transpose(1, 2).contiguous()  # (B,k,C)

    # If we clamped k, pad by repeating last token
    if k_eff < k:
        pad = tok[:, -1:, :].expand(-1, k - k_eff, -1)
        tok = torch.cat([tok, pad], dim=1)

    return tok


# -------------------------
# PAT-style anchor transform
# -------------------------


class PATAnchorTransform(nn.Module):
    """PATNet-style feature transformation via prototypes + learnable reference anchors.

    We implement the same closed-form transform used in PATNet:
        P = pinv(C) @ R
    where C = [p_bg, p_fg] (normalized), R are learnable reference anchors (normalized).

    IMPORTANT: To keep it computationally feasible, apply this in a reduced channel space
    (e.g., 256) via 1x1 projections.
    """

    def __init__(self, dim: int, num_levels: int):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels

        # One reference layer per level; weight has shape (2, dim)
        self.reference_layers = nn.ModuleList()
        for _ in range(num_levels):
            layer = nn.Linear(dim, 2, bias=True)
            nn.init.kaiming_normal_(layer.weight, a=0, mode="fan_in", nonlinearity="linear")
            nn.init.constant_(layer.bias, 0)
            self.reference_layers.append(layer)

    def _compute_P(
        self,
        proto_fg: torch.Tensor,
        proto_bg: torch.Tensor,
        level_idx: int,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Compute transform matrix P for a given level.

        Args:
            proto_fg/proto_bg: (B, dim)

        Returns:
            P: (B, dim, dim)
        """
        b, d = proto_fg.shape
        C = torch.stack([proto_bg, proto_fg], dim=1)  # (B,2,dim)

        # R from reference anchors (2,dim) -> (B,2,dim)
        R = self.reference_layers[level_idx].weight.unsqueeze(0).expand(b, -1, -1)

        # Normalize C and R across channel dim
        C = C / (C.pow(2).sum(dim=2, keepdim=True).sqrt() + eps)
        R = R / (R.pow(2).sum(dim=2, keepdim=True).sqrt() + eps)

        # P in PATNet: P = pinv(C) @ R, then transpose
        P = torch.matmul(torch.pinverse(C), R)  # (B,dim,dim)
        P = P.transpose(1, 2).contiguous()      # (B,dim,dim)
        return P

    @staticmethod
    def apply_P_to_feat(P: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """Apply P to a feature map.

        Args:
            P: (B, dim, dim)
            feat: (B, dim, H, W)

        Returns:
            feat_t: (B, dim, H, W)
        """
        b, d, h, w = feat.shape
        feat_flat = feat.view(b, d, -1)
        out = torch.matmul(P, feat_flat).view(b, d, h, w)
        return out

    @staticmethod
    def apply_P_to_vec(P: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Apply P to a prototype/token vector.

        Args:
            P: (B, dim, dim)
            vec: (B, dim)

        Returns:
            vec_t: (B, dim)
        """
        b, d = vec.shape
        v = vec.view(b, d, 1)
        out = torch.matmul(P, v).view(b, d)
        return out

    def forward(
        self,
        query_feats: List[torch.Tensor],
        support_feats: List[torch.Tensor],
        support_mask: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Transform query/support features for each level.

        Args:
            query_feats: list of length L, each (B, dim, H_l, W_l)
            support_feats: list of length L, each (B, dim, H_l, W_l)
            support_mask: (B, H, W) or (B,1,H,W)

        Returns:
            query_feats_t: transformed query feats
            support_feats_t: transformed support feats
            proto_fg_t: list of transformed fg prototypes (B,dim)
            proto_bg_t: list of transformed bg prototypes (B,dim)
            P_list: list of P matrices (B,dim,dim)
        """
        assert len(query_feats) == len(support_feats) == self.num_levels

        query_feats_t: List[torch.Tensor] = []
        support_feats_t: List[torch.Tensor] = []
        proto_fg_t: List[torch.Tensor] = []
        proto_bg_t: List[torch.Tensor] = []
        P_list: List[torch.Tensor] = []

        for li, (q, s) in enumerate(zip(query_feats, support_feats)):
            mask_rs = _resize_mask(support_mask, s.shape[-2:])
            proto_fg = masked_avg_pool(s, mask_rs)
            proto_bg = masked_avg_pool(s, 1.0 - mask_rs)

            P = self._compute_P(proto_fg, proto_bg, li)
            P_list.append(P)

            q_t = self.apply_P_to_feat(P, q)
            s_t = self.apply_P_to_feat(P, s)

            query_feats_t.append(q_t)
            support_feats_t.append(s_t)

            # recompute prototypes in transformed space (more consistent for attention)
            proto_fg_t.append(masked_avg_pool(s_t, mask_rs))
            proto_bg_t.append(masked_avg_pool(s_t, 1.0 - mask_rs))

        return query_feats_t, support_feats_t, proto_fg_t, proto_bg_t, P_list


# -------------------------
# Cross-attention fusion
# -------------------------


class SupportQueryCrossAttention(nn.Module):
    """Support->Query cross-attention on flattened spatial tokens."""

    def __init__(self, dim: int, num_heads: int, ff_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "embed dim must be divisible by num_heads"

        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        # Post-attn fusion: concat(original, attn_out) -> 1x1 conv
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        # Light FFN on feature map (optional)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * ff_mult, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim * ff_mult),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * ff_mult, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.ffn_act = nn.ReLU(inplace=True)

    def forward(self, query_feat: torch.Tensor, support_tokens: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            query_feat: (B, dim, H, W)
            support_tokens: (B, S, dim)

        Returns:
            out: (B, dim, H, W)
        """
        b, d, h, w = query_feat.shape
        q = query_feat.flatten(2).transpose(1, 2).contiguous()  # (B, N, dim)

        # Cross-attn: Q=query tokens, K/V=support tokens
        attn_out, _ = self.mha(q, support_tokens, support_tokens)  # (B, N, dim)
        attn_map = attn_out.transpose(1, 2).contiguous().view(b, d, h, w)

        x = self.fuse(torch.cat([query_feat, attn_map], dim=1))

        # FFN residual
        x2 = self.ffn(x)
        x = self.ffn_act(x + x2)
        return x


# -------------------------
# The main SAM2-UNet CD-FSS model (aggressive)
# -------------------------


@dataclass
class SAM2CDFSSConfig:
    sam2_model_cfg: str = "sam2_hiera_l.yaml"
    sam2_checkpoint: Optional[str] = None

    # Encoder channel dims for SAM2-Hiera-L (as used in SAM2UNet.py)
    encoder_channels: Tuple[int, int, int, int] = (144, 288, 576, 1152)

    # Reduced feature dim for few-shot heads (keeps PAT transform feasible)
    embed_dim: int = 256

    # Cross-attn
    attn_heads: int = 8
    num_fg_tokens: int = 32  # per level

    # Levels used (index into x1..x4): we use x2,x3,x4 for PAT+attn+hypercorr
    # and use x1 as finest skip in decoder A
    use_levels: Tuple[int, int, int] = (1, 2, 3)  # corresponds to [x2,x3,x4]


class SAM2UNetCDFSSAggressive(nn.Module):
    """Aggressive CD-FSS model:

    - SAM2-Hiera encoder + Adapters (SAM2-UNet style)
    - PAT anchor transform (domain alignment)
    - Support->Query cross-attention (feature adaptation)
    - Hypercorrelation pyramid + HPN learner (correspondence)
    - Gated fusion of logits

    Output: 2-channel logits (bg, fg) at input resolution.
    """

    def __init__(self, cfg: SAM2CDFSSConfig):
        super().__init__()
        self.cfg = cfg

        # 1) Build SAM2 and keep only the encoder trunk (same as SAM2UNet.py)
        if cfg.sam2_checkpoint:
            model = build_sam2(cfg.sam2_model_cfg, cfg.sam2_checkpoint)
        else:
            model = build_sam2(cfg.sam2_model_cfg)

        # Strip unused SAM2 modules (same list as SAM2UNet.py)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck

        self.encoder = model.image_encoder.trunk

        # 2) Freeze encoder trunk weights, insert Adapters (SAM2-UNet style)
        for p in self.encoder.parameters():
            p.requires_grad = False
        blocks = []
        for blk in self.encoder.blocks:
            blocks.append(Adapter(blk))
        self.encoder.blocks = nn.Sequential(*blocks)

        # 3) 1x1 projections to a manageable embed dim
        c1, c2, c3, c4 = cfg.encoder_channels
        d = cfg.embed_dim

        # Projections for query/support features
        self.proj1 = nn.Conv2d(c1, d, kernel_size=1, bias=False)
        self.proj2 = nn.Conv2d(c2, d, kernel_size=1, bias=False)
        self.proj3 = nn.Conv2d(c3, d, kernel_size=1, bias=False)
        self.proj4 = nn.Conv2d(c4, d, kernel_size=1, bias=False)

        # Optional norm after projection
        self.proj_norm1 = nn.BatchNorm2d(d)
        self.proj_norm2 = nn.BatchNorm2d(d)
        self.proj_norm3 = nn.BatchNorm2d(d)
        self.proj_norm4 = nn.BatchNorm2d(d)

        # 4) PAT anchor transform for levels x2,x3,x4
        self.pat = PATAnchorTransform(dim=d, num_levels=3)

        # 5) Cross-attn blocks per level (x2,x3,x4)
        self.ca2 = SupportQueryCrossAttention(dim=d, num_heads=cfg.attn_heads)
        self.ca3 = SupportQueryCrossAttention(dim=d, num_heads=cfg.attn_heads)
        self.ca4 = SupportQueryCrossAttention(dim=d, num_heads=cfg.attn_heads)

        # 6) Decoder A (U-Net style) using existing Up blocks
        self.up43 = Up(d * 2, d)
        self.up32 = Up(d * 2, d)
        self.up21 = Up(d * 2, d)
        self.head_a = nn.Conv2d(d, 2, kernel_size=1)

        # 7) Branch B: Hypercorrelation + HPN learner (PATNet)
        self.stack_ids = [1, 2, 3]  # pick [x4,x3,x2] from [x2,x3,x4]
        self.hpn = HPNLearner([1, 1, 1])

        # 8) Fusion gate on logits
        self.fuse_gate = nn.Conv2d(4, 1, kernel_size=1)

    # ---------- feature extraction helpers ----------

    def _encode(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run SAM2-Hiera trunk and return 4 feature maps."""
        x1, x2, x3, x4 = self.encoder(img)
        return x1, x2, x3, x4

    def _project(self, feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Project encoder feats to embed_dim."""
        x1, x2, x3, x4 = feats
        x1 = F.relu(self.proj_norm1(self.proj1(x1)), inplace=True)
        x2 = F.relu(self.proj_norm2(self.proj2(x2)), inplace=True)
        x3 = F.relu(self.proj_norm3(self.proj3(x3)), inplace=True)
        x4 = F.relu(self.proj_norm4(self.proj4(x4)), inplace=True)
        return x1, x2, x3, x4

    def _build_support_tokens(self, support_feat_t: torch.Tensor, support_mask: torch.Tensor, k_fg: int) -> torch.Tensor:
        """Build a token set for cross-attention.

        Token set = [bg_proto, fg_proto] + topK foreground patch tokens.
        """
        mask_rs = _resize_mask(support_mask, support_feat_t.shape[-2:])
        fg_proto = masked_avg_pool(support_feat_t, mask_rs)
        bg_proto = masked_avg_pool(support_feat_t, 1.0 - mask_rs)

        fg_tokens = masked_topk_tokens(support_feat_t, mask_rs, k=k_fg)

        proto_tokens = torch.stack([bg_proto, fg_proto], dim=1)  # (B,2,dim)
        tokens = torch.cat([proto_tokens, fg_tokens], dim=1)     # (B,2+k,dim)
        return tokens

    # ---------- single-shot forward ----------

    def forward_oneshot(
        self,
        query_img: torch.Tensor,
        support_img: torch.Tensor,
        support_mask: torch.Tensor,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward for a single support image/mask."""

        # Encode + project
        q_feats = self._project(self._encode(query_img))   # (q1,q2,q3,q4)
        s_feats = self._project(self._encode(support_img)) # (s1,s2,s3,s4)

        q1, q2, q3, q4 = q_feats
        s1, s2, s3, s4 = s_feats

        # PAT transform on levels x2,x3,x4 (lists length 3)
        q_list = [q2, q3, q4]
        s_list = [s2, s3, s4]
        q_t_list, s_t_list, _, _, _ = self.pat(q_list, s_list, support_mask)
        q2_t, q3_t, q4_t = q_t_list
        s2_t, s3_t, s4_t = s_t_list

        # Branch A: cross-attn per level (x2,x3,x4)
        tok2 = self._build_support_tokens(s2_t, support_mask, self.cfg.num_fg_tokens)
        tok3 = self._build_support_tokens(s3_t, support_mask, self.cfg.num_fg_tokens)
        tok4 = self._build_support_tokens(s4_t, support_mask, self.cfg.num_fg_tokens)

        q2_a = self.ca2(q2_t, tok2)
        q3_a = self.ca3(q3_t, tok3)
        q4_a = self.ca4(q4_t, tok4)

        # Decode branch A (U-Net style)
        x = self.up43(q4_a, q3_a)
        x = self.up32(x, q2_a)
        x = self.up21(x, q1)
        logit_a = self.head_a(x)
        logit_a = F.interpolate(logit_a, size=query_img.shape[-2:], mode="bilinear", align_corners=False)

        # Branch B: hypercorrelation pyramid + HPN
        mask2 = _resize_mask(support_mask, s2_t.shape[-2:])
        mask3 = _resize_mask(support_mask, s3_t.shape[-2:])
        mask4 = _resize_mask(support_mask, s4_t.shape[-2:])
        s2_m = s2_t * mask2
        s3_m = s3_t * mask3
        s4_m = s4_t * mask4

        corr_pyr = Correlation.multilayer_correlation([q2_t, q3_t, q4_t], [s2_m, s3_m, s4_m], self.stack_ids)
        logit_b = self.hpn(corr_pyr)
        logit_b = F.interpolate(logit_b, size=query_img.shape[-2:], mode="bilinear", align_corners=False)

        # Fusion (gated)
        gate = torch.sigmoid(self.fuse_gate(torch.cat([logit_a, logit_b], dim=1)))  # (B,1,H,W)
        logit = gate * logit_a + (1.0 - gate) * logit_b

        if not return_aux:
            return logit

        aux = {
            "logit_a": logit_a,
            "logit_b": logit_b,
            "gate": gate,
        }
        return logit, aux

    # ---------- K-shot forward ----------

    def forward(
        self,
        query_img: torch.Tensor,
        support_imgs: torch.Tensor,
        support_masks: torch.Tensor,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward for K-shot support.

        We aggregate by averaging logits across shots (differentiable).
        """

        if support_imgs.dim() == 4:
            support_imgs = support_imgs.unsqueeze(1)
        if support_masks.dim() == 3:
            support_masks = support_masks.unsqueeze(1)

        b, k, _, h, w = support_imgs.shape

        logits: List[torch.Tensor] = []
        logits_a: List[torch.Tensor] = []
        logits_b: List[torch.Tensor] = []
        gates: List[torch.Tensor] = []

        for si in range(k):
            out = self.forward_oneshot(query_img, support_imgs[:, si], support_masks[:, si], return_aux=True)
            logit, aux = out
            logits.append(logit)
            logits_a.append(aux["logit_a"])
            logits_b.append(aux["logit_b"])
            gates.append(aux["gate"])

        logit = torch.stack(logits, dim=0).mean(dim=0)

        if not return_aux:
            return logit

        aux = {
            "logit_a": torch.stack(logits_a, dim=0).mean(dim=0),
            "logit_b": torch.stack(logits_b, dim=0).mean(dim=0),
            "gate": torch.stack(gates, dim=0).mean(dim=0),
        }
        return logit, aux

    # ---------- convenience for PATNet-style batches ----------

    def forward_batch(self, batch: Dict[str, torch.Tensor], return_aux: bool = False):
        return self(batch["query_img"], batch["support_imgs"], batch["support_masks"], return_aux=return_aux)


