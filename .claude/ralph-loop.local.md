---
active: true
iteration: 1
session_id: 
max_iterations: 60
completion_promise: "DONE"
started_at: "2026-05-10T15:51:14Z"
---

按 TODO.md 推进 SAGE-PR 训练。每次循环先读 TODO.md 顶部『下一步指针』和 §9 不做的事,然后干当前任务(P4 用 V2 数据训 + 补 L_info/L_ranking 损失 → 全 phase 评估)。所有 python 命令用 D:/anaconda3/Scripts/conda.exe run -n rl python ...。长跑前用 scripts/monitor_resources.py --tag <task> 起监控。每完成一个 ☐ 改 ☑、更新顶部指针和最新评估数字、git commit 一次。任一 phase < 95% 不停,走 §7 故障排查表挑下一招;同一招连续 2 次无效切下一招。完成条件:phase 1-6 全部 deterministic 通关率 ≥ 95%(纯神经网络,不挂 solver 和 plan_exploration),输出 <promise>DONE</promise>。
