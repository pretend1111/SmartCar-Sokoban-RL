"""CNN-over-walls 策略 — TFLite Micro 友好.

输入约定: 观测 obs = concat([entities (62 维), walls (12×16 = 192 维)]).
walls 是 row-major (row 0..11, col 0..15), 与 oracle_features.encode_wall_layout
保持一致.

只用 TFLite Micro 核心 op: Conv2D, FullyConnected, ReLU, Reshape, Concat.
不用 BatchNorm / Dropout / LayerNorm.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from smartcar_sokoban.rl.high_level_env import (
    MAP_COLS, MAP_ROWS, MAP_LAYOUT_DIM, STATE_DIM, STATE_DIM_WITH_MAP,
)


class MaskedConvBCPolicy(nn.Module):
    """墙体走 Conv2D 提空间嵌入, 实体走 FC, concat 后接 trunk.

    默认尺寸 (16x12 网格):
      Conv1: 1 → 16  (3x3, pad=1, stride=1)   → (16, 12, 16)   ~160 params
      Conv2: 16 → 32 (3x3, pad=1, stride=2)   → (32, 6, 8)     ~4.6K params
      flatten: 1536
      wall_head: 1536 → 64                                       ~98K params
      trunk: 126 → hidden → hidden → n_actions                    根据 hidden
      hidden=256 时合计 ~215K params.
    """

    def __init__(self,
                 entity_dim: int = STATE_DIM,
                 wall_h: int = MAP_ROWS,
                 wall_w: int = MAP_COLS,
                 n_actions: int = 54,
                 hidden_dim: int = 256,
                 wall_emb_dim: int = 64,
                 conv1_ch: int = 16,
                 conv2_ch: int = 32):
        super().__init__()
        self.entity_dim = entity_dim
        self.wall_h = wall_h
        self.wall_w = wall_w
        self.wall_dim = wall_h * wall_w
        if self.wall_dim != MAP_LAYOUT_DIM:
            raise ValueError(
                f"wall_h*wall_w={self.wall_dim} != MAP_LAYOUT_DIM={MAP_LAYOUT_DIM}"
            )

        self.conv1 = nn.Conv2d(1, conv1_ch, kernel_size=3, padding=1)
        # stride=2 下采样, (12,16) → (6,8)
        self.conv2 = nn.Conv2d(conv1_ch, conv2_ch, kernel_size=3,
                               padding=1, stride=2)

        out_h = wall_h // 2
        out_w = wall_w // 2
        self.conv_out_dim = conv2_ch * out_h * out_w

        self.wall_head = nn.Linear(self.conv_out_dim, wall_emb_dim)

        self.fc1 = nn.Linear(entity_dim + wall_emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, entity_dim + wall_dim) — 与 oracle_features.build_oracle_obs
        # 在 include_map_layout=True 下的输出格式一致
        ent = obs[:, :self.entity_dim]
        walls_flat = obs[:, self.entity_dim:self.entity_dim + self.wall_dim]
        walls = walls_flat.view(-1, 1, self.wall_h, self.wall_w)

        x = F.relu(self.conv1(walls))
        x = F.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        wall_emb = F.relu(self.wall_head(x))

        h = torch.cat([ent, wall_emb], dim=1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        logits = self.fc_out(h)
        return logits

    def expected_obs_dim(self) -> int:
        return self.entity_dim + self.wall_dim


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def summarize_model(model: MaskedConvBCPolicy) -> dict:
    p_conv1 = sum(p.numel() for p in model.conv1.parameters())
    p_conv2 = sum(p.numel() for p in model.conv2.parameters())
    p_wall_head = sum(p.numel() for p in model.wall_head.parameters())
    p_fc1 = sum(p.numel() for p in model.fc1.parameters())
    p_fc2 = sum(p.numel() for p in model.fc2.parameters())
    p_fc_out = sum(p.numel() for p in model.fc_out.parameters())
    total = count_params(model)
    return {
        "conv1": p_conv1,
        "conv2": p_conv2,
        "wall_head": p_wall_head,
        "fc1": p_fc1,
        "fc2": p_fc2,
        "fc_out": p_fc_out,
        "total": total,
        "expected_obs_dim": model.expected_obs_dim(),
        "n_actions": model.fc_out.out_features,
        "hidden_dim": model.fc1.out_features,
    }


if __name__ == "__main__":
    # sanity check
    m = MaskedConvBCPolicy()
    print(summarize_model(m))
    obs = torch.randn(512, m.expected_obs_dim())
    out = m(obs)
    print("forward shape:", out.shape)
