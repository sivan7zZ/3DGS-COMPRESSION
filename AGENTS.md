# 本仓库约定

- 开始修改前阅读 WORKFLOW.md；以该文档和用户最新要求为准。
- 仓库位置固定，不迁移。所有实验结果放在本仓库 output/ 内。
- 当前流程是预训练3DGS → 边界加权PUP六维Fisher评分 + 原PWSR RIR/DPRA + 实例/拓扑保护、合并和预算回填 → 标准SH3 checkpoint/PLY；不运行蒸馏。当前边界评分为PUP_logdet+log(1+5B)，B=0.4SAM+0.4depth+0.2RGB，三通道按训练视角alpha*T投影汇总为连续值；入口 run_continuous_boundary_experiments.py，范围与续跑见 CONTINUOUS_BOUNDARY_COMPRESSION.md；历史v3.2.0、SPS和蒸馏入口保留供复现。
- 不重新引入 3D-DoG、负响应字段或对应渲染器；输出标准 SH PLY。
- SH 教师和学生均从 checkpoint 初始化：教师用原始未剪枝 3DGS，学生用 PWSR 剪枝结果；不得静默复用同一 checkpoint。
- SH3→SH2 初始化创建新优化器；SH2 续训才恢复优化器和配套 sidecar。
- 新默认蒸馏 5000 次（虚拟 35000→40000）。旧实验 30000→40000 是10000次，不混淆。
- output/city2_stage3、output/city2_stage3_sh2、历史报告和 integration_backups 不覆盖。
- 不因整理代码自动重跑全量训练。静态 / 单元测试：bash scripts/run.sh -m unittest discover -s tests -v。
- LightGaussian 蒸馏依赖必须在本仓库内；不得依赖外部 LightGaussian checkout 或旧结果目录中的动态库。
- 新流程的边界信息仅用于加权评分、原拓扑构建和恢复损失，不作为独立剪枝/合并硬约束；实例与拓扑保护不变。人工City语义标签仅用于下游评测，10帧分独立测试2帧和训练视角辅助8帧，辅助评估器需分折排除当前评测帧。
