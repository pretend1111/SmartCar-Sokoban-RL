"""SAGE-PR 神经评分器 (PyTorch 实现).

参考 docs/FINAL_ARCH_DESIGN.md §4.

架构总览:
    Inputs:
        X_grid:   [B, 30, 10, 14]  (空间张量, 已裁外圈墙)
        X_cand:   [B, 64, 128]     (候选集合)
        u_global: [B, 16]           (全局标量)
        mask:     [B, 64]           (合法性 mask)

    Grid Encoder:
        Conv 3x3 30->32 + ReLU
        FixupResBlock(DSConv 32->32) x2
        FixupTransition(DSConv 32->48, dilation=2)
        FixupResBlock(DSConv 48->48) x2
        GAP -> [B, 48]
        FC 48->96, ReLU -> z_grid [B, 96]

    Candidate Encoder:
        Linear 128->96, ReLU
        Linear 96->96, ReLU
        e_i: [B, 64, 96]
        z_set = mean over 64 -> [B, 96]

    Context Fusion:
        concat(z_grid, z_set, u_global) -> [B, 208]
        FC 208->128, ReLU
        FC 128->96, ReLU
        c: [B, 96]

    Score Head (per candidate):
        e_tilde_i = e_i + c (broadcast)
        Linear 96->96, ReLU
        Linear 96->1 -> score [B, 64]
        mask + softmax -> π [B, 64]

    Aux heads (per candidate, share e_tilde):
        deadlock_head:    Linear 96->1, sigmoid
        progress_head:    Linear 96->1
        info_gain_head:   Linear 96->1

    Value head (global):
        FC 96 (c) -> 32 -> 1

约束: 仅使用 RFC §4.4 允许的 op (Conv2D / DSConv / Linear / ReLU / Sigmoid /
       AvgPool2D / Softmax / Concat / Add / Reshape).

参数量目标 ~98K (P2.3 sanity check).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 基础块 ────────────────────────────────────────────────

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积 = depthwise 3x3 + pointwise 1x1.

    (TFLite Micro: Conv2D + DepthwiseConv2D 都允许.)
    """
    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 3, dilation: int = 1):
        super().__init__()
        pad = (kernel // 2) * dilation
        self.dw = nn.Conv2d(in_ch, in_ch, kernel, padding=pad,
                            dilation=dilation, groups=in_ch, bias=True)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=True)

    def forward(self, x):
        x = F.relu(self.dw(x), inplace=True)
        x = self.pw(x)
        return x


class FixupResBlock(nn.Module):
    """Fixup-style 残差块 (channel 不变).

    残差路径上加可学 scalar α (init 0). 训练后可 fold 进 conv weights.

    block:
        y = DSConv(x + bias_a)
        y = ReLU(y + bias_b)
        y = DSConv2(y) * alpha
        return ReLU(x + y)

    Fixup 论文 (Zhang et al. 2019) 简化版.
    """
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.dw1 = nn.Conv2d(channels, channels, 3, padding=dilation,
                             dilation=dilation, groups=channels, bias=False)
        self.pw1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.dw2 = nn.Conv2d(channels, channels, 3, padding=dilation,
                             dilation=dilation, groups=channels, bias=False)
        self.pw2 = nn.Conv2d(channels, channels, 1, bias=False)

        # Fixup scalars
        self.bias_a = nn.Parameter(torch.zeros(1))
        self.bias_b = nn.Parameter(torch.zeros(1))
        self.bias_c = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        identity = x
        y = self.dw1(x + self.bias_a)
        y = self.pw1(y)
        y = F.relu(y + self.bias_b, inplace=True)
        y = self.dw2(y)
        y = self.pw2(y) * self.alpha
        return F.relu(identity + y + self.bias_c, inplace=True)


class FixupTransition(nn.Module):
    """Channel 变化的过渡块 (无残差, 直接 DSConv)."""
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        self.ds = DepthwiseSeparableConv(in_ch, out_ch, dilation=dilation)

    def forward(self, x):
        return F.relu(self.ds(x), inplace=True)


# ── Grid Encoder ──────────────────────────────────────────

class GridEncoder(nn.Module):
    def __init__(self, in_ch: int = 30, mid: int = 32, out: int = 48,
                 z_grid: int = 96):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, mid, 3, padding=1, bias=True)
        self.block1 = FixupResBlock(mid)
        self.block2 = FixupResBlock(mid)
        self.transition = FixupTransition(mid, out, dilation=2)
        self.block3 = FixupResBlock(out)
        self.block4 = FixupResBlock(out)
        self.fc = nn.Linear(out, z_grid)

    def forward(self, x_grid: torch.Tensor) -> torch.Tensor:
        # x_grid: [B, in_ch, 10, 14]
        x = F.relu(self.stem(x_grid), inplace=True)
        x = self.block1(x)
        x = self.block2(x)
        x = self.transition(x)
        x = self.block3(x)
        x = self.block4(x)
        # Global average pool
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # [B, out]
        z = F.relu(self.fc(x), inplace=True)
        return z


# ── Candidate Encoder ────────────────────────────────────

class CandidateEncoder(nn.Module):
    """共享 MLP per candidate (Deep Sets-style)."""
    def __init__(self, in_dim: int = 128, hidden: int = 96, out: int = 96):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out)

    def forward(self, x_cand: torch.Tensor) -> torch.Tensor:
        # x_cand: [B, N, in_dim]
        h = F.relu(self.fc1(x_cand), inplace=True)
        h = F.relu(self.fc2(h), inplace=True)
        return h  # [B, N, out]


# ── Context Fusion ────────────────────────────────────────

class ContextFusion(nn.Module):
    def __init__(self, z_grid: int = 96, z_set: int = 96, u_dim: int = 16,
                 hidden: int = 128, out: int = 96):
        super().__init__()
        in_dim = z_grid + z_set + u_dim
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out)

    def forward(self, z_grid, z_set, u_global):
        h = torch.cat([z_grid, z_set, u_global], dim=-1)
        h = F.relu(self.fc1(h), inplace=True)
        h = F.relu(self.fc2(h), inplace=True)
        return h  # [B, out]


# ── Score & Aux Heads ────────────────────────────────────

class ScoreAndAuxHeads(nn.Module):
    def __init__(self, e_dim: int = 96, hidden: int = 96, value_hidden: int = 32):
        super().__init__()
        # Score head
        self.score_fc1 = nn.Linear(e_dim, hidden)
        self.score_fc2 = nn.Linear(hidden, 1)

        # Aux heads (输入 = e_tilde, [B, N, e_dim])
        self.deadlock_fc = nn.Linear(e_dim, 1)
        self.progress_fc = nn.Linear(e_dim, 1)
        self.info_gain_fc = nn.Linear(e_dim, 1)

        # Value head: 输入 = c [B, e_dim]
        self.value_fc1 = nn.Linear(e_dim, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

    def forward(self, e_i: torch.Tensor, c: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, ...]:
        # e_i: [B, N, e_dim]; c: [B, e_dim]
        e_tilde = e_i + c.unsqueeze(1)  # broadcast → [B, N, e_dim]
        e_tilde = F.relu(e_tilde, inplace=False)  # 不能 inplace (e_i 上游用)

        # Score
        s = F.relu(self.score_fc1(e_tilde), inplace=True)
        score_logits = self.score_fc2(s).squeeze(-1)  # [B, N]
        if mask is not None:
            # mask=1 合法; mask=0 非法 → -inf
            score_logits = score_logits.masked_fill(mask < 0.5, -1e9)

        # Aux per-candidate
        deadlock = torch.sigmoid(self.deadlock_fc(e_tilde).squeeze(-1))   # [B, N]
        progress = self.progress_fc(e_tilde).squeeze(-1)                  # [B, N]
        info_gain = self.info_gain_fc(e_tilde).squeeze(-1)                # [B, N]

        # Value (global)
        v = F.relu(self.value_fc1(c), inplace=True)
        value = self.value_fc2(v).squeeze(-1)                             # [B]

        return score_logits, value, deadlock, progress, info_gain


# ── 整网 ──────────────────────────────────────────────────

class SAGEPolicyRanker(nn.Module):
    """SAGE-PR 候选评分器主体."""

    def __init__(self,
                 grid_in_ch: int = 30,
                 grid_mid: int = 32,
                 grid_out: int = 48,
                 z_grid: int = 96,
                 cand_in_dim: int = 128,
                 cand_hidden: int = 96,
                 e_dim: int = 96,
                 u_dim: int = 16,
                 fusion_hidden: int = 128,
                 c_dim: int = 96,
                 value_hidden: int = 32):
        super().__init__()
        self.grid_encoder = GridEncoder(grid_in_ch, grid_mid, grid_out, z_grid)
        self.cand_encoder = CandidateEncoder(cand_in_dim, cand_hidden, e_dim)
        self.context = ContextFusion(z_grid, e_dim, u_dim, fusion_hidden, c_dim)
        self.heads = ScoreAndAuxHeads(e_dim, e_dim, value_hidden)

        self._init_fixup()

    def _init_fixup(self):
        """Fixup-style init: stem & 普通 conv He init; ResBlock 第二 conv 置 0."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Fixup: 残差路径第二 pw conv 权重置 0 + alpha=0 (默认就是 0)
        for m in self.modules():
            if isinstance(m, FixupResBlock):
                nn.init.zeros_(m.pw2.weight)

    def forward(self,
                x_grid: torch.Tensor,
                x_cand: torch.Tensor,
                u_global: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_grid: [B, 30, 10, 14]
            x_cand: [B, 64, 128]
            u_global: [B, 16]
            mask: [B, 64] (1=legal, 0=illegal)

        Returns:
            score_logits: [B, 64]
            value:        [B]
            deadlock:     [B, 64] in [0,1]
            progress:     [B, 64]
            info_gain:    [B, 64]
        """
        z_grid = self.grid_encoder(x_grid)              # [B, z_grid]
        e_i = self.cand_encoder(x_cand)                  # [B, N, e_dim]

        if mask is not None:
            # z_set: 只对合法候选求平均 (避免 pad 把均值拉低)
            mask_f = mask.unsqueeze(-1)                  # [B, N, 1]
            z_set = (e_i * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        else:
            z_set = e_i.mean(dim=1)                       # [B, e_dim]

        c = self.context(z_grid, z_set, u_global)        # [B, c_dim]

        score, value, dl, pg, ig = self.heads(e_i, c, mask)
        return score, value, dl, pg, ig

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── 默认实例化辅助 ────────────────────────────────────────

def build_default_model() -> SAGEPolicyRanker:
    """按 FINAL_ARCH_DESIGN §4 默认配置."""
    return SAGEPolicyRanker(
        grid_in_ch=30,
        grid_mid=32,
        grid_out=48,
        z_grid=96,
        cand_in_dim=128,
        cand_hidden=96,
        e_dim=96,
        u_dim=16,
        fusion_hidden=128,
        c_dim=96,
        value_hidden=32,
    )


def build_large_model() -> SAGEPolicyRanker:
    """更大版本 ~250K params, 用于 phase 4/5/6 hard cases."""
    return SAGEPolicyRanker(
        grid_in_ch=30,
        grid_mid=48,
        grid_out=72,
        z_grid=128,
        cand_in_dim=128,
        cand_hidden=128,
        e_dim=128,
        u_dim=16,
        fusion_hidden=192,
        c_dim=128,
        value_hidden=48,
    )


def detect_model_size(state_dict: dict) -> str:
    """根据 state_dict 推断是 default 还是 large 模型."""
    if "grid_encoder.stem.weight" in state_dict:
        # default: shape[0]=32, large: shape[0]=48
        return "large" if state_dict["grid_encoder.stem.weight"].shape[0] >= 48 else "default"
    return "default"


def build_model_from_ckpt(ckpt_path: str, device=None) -> SAGEPolicyRanker:
    """根据 ckpt 自动构建 default/large 并加载权重."""
    import torch
    ck = torch.load(ckpt_path, map_location=device)
    sd = ck["model_state_dict"]
    size = ck.get("model_size", detect_model_size(sd))
    builder = build_large_model if size == "large" else build_default_model
    model = builder()
    if device is not None:
        model = model.to(device)
    model.load_state_dict(sd)
    return model
