"""P2 bench: forward batch=512 在 cuda 上 < 5ms."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from experiments.solver_bc.policy_conv import MaskedConvBCPolicy, summarize_model


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = MaskedConvBCPolicy().to(dev).eval()
    print("model:", summarize_model(m))
    obs = torch.randn(512, m.expected_obs_dim(), device=dev)

    # warmup
    for _ in range(20):
        with torch.no_grad():
            _ = m(obs)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        with torch.no_grad():
            _ = m(obs)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n * 1000
    print(f"forward bs=512 on {dev}: {dt:.2f} ms")

    # 单样本推理 (模拟 OpenART)
    obs1 = torch.randn(1, m.expected_obs_dim())
    m_cpu = MaskedConvBCPolicy().eval()
    for _ in range(5):
        with torch.no_grad():
            _ = m_cpu(obs1)
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        with torch.no_grad():
            _ = m_cpu(obs1)
    dt = (time.perf_counter() - t0) / n * 1000
    print(f"forward bs=1 on cpu (fp32): {dt:.3f} ms (单样本, 无量化)")


if __name__ == "__main__":
    main()
