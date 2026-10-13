# 本仓库约定

- 开始修改前阅读 WORKFLOW.md；以该文档和用户最新要求为准。
- 仓库位置固定，不迁移。所有实验结果放在本仓库 output/ 内。
- 当前流程是预训练 3DGS → PWSR RIR/DPRA/SPS → checkpoint → LightGaussian SH 蒸馏。
- 不重新引入 3D-DoG、负响应字段或对应渲染器；输出标准 SH PLY。
- SH 教师和学生均从 checkpoint 初始化：教师用原始未剪枝 3DGS，学生用 PWSR 剪枝结果；不得静默复用同一 checkpoint。
- SH3→SH2 初始化创建新优化器；SH2 续训才恢复优化器和配套 sidecar。
- 新默认蒸馏 5000 次（虚拟 35000→40000）。旧实验 30000→40000 是10000次，不混淆。
- output/city2_stage3、output/city2_stage3_sh2、历史报告和 integration_backups 不覆盖。
- 不因整理代码自动重跑全量训练。静态 / 单元测试：bash scripts/run.sh -m unittest discover -s tests -v。
- LightGaussian 蒸馏依赖必须在本仓库内；不得依赖外部 LightGaussian checkout 或旧结果目录中的动态库。
