# 无语义标签的小实例保护

在原有 RIR / DPRA / SPS → 原始 SH3 教师 checkpoint → 5000 步 SH2 蒸馏流程上增加可选实例保护。不引入 3D-DoG，也不读取人工语义标签。

## 输入及划分

原始 City RGB/COLMAP 和 iteration_30000 PLY 的路径保持 `configs/city.json` 中定义。此前全 70 视角生成的实例模型位于 `/home/zhangys/Documents/Codex/2026-10-23/gt-semantic-3dgs-ply-ply-ply/outputs/city_label_free/`。正式压缩实验不直接使用这套含测试视角的标签，而是仅用 LLFF hold=8 划分出的 61 个训练视角重新关联，得到 480 个类别无关自动区域。9 个测试视角的 RGB 与 SAM 仅用于最终评估。

新的数据与全部实验输出位于 `output/city2_instance_guard_20261026/`，不覆盖旧结果。`inputs/train_labels/` 保存逐点实例 NPZ、区域目录和二维来源；`inputs/profiles/` 保存训练视角的投影面积统计、区域 / 边界损失权重。

## 剪枝约束

小实例的操作性定义为训练视角中位投影面积不超过影像的 0.5%；这不等同于已确认的车辆、树木等类别或物理实例。

每次仍由 SPS 计算全局空间和频率贡献，由 DPRA 决定本轮全局数量 K，但选择集合满足：

```text
保留总数严格等于 K
每个受保护实例至少保留其原始配额 floor_i
```

配额从原模型固定计算，不随轮次乘法递减：`min(N_i, 256, max(32, ceil(N_i*(.02+.38*smallness_i))))`。可信度门槛见 `configs/city_instance_policy.json` 及 helper 默认项；unknown、缺少多视角直接证据的区域不产生伪配额。所有配额若超过最终预算即显式报错，不隐式放松或超预算。

配额内优先覆盖原实例的空间网格，再结合实例内 SPS 排名与观测置信度选择点；其余预算继续按综合 SPS 竞争。每轮输出 `stage3/instance_audits/iteration_*.json`。其中 counterfactual 是当前模型、当前分数上的全局选择，仅为诊断，不冒充独立训练的基线。

恢复训练、SPS 评分及 RIR 的恢复检查加入训练小区域权重，使少量像素的误差不被整图面积稀释。蒸馏对原始训练视角增加区域 L1 和边界梯度损失；扰动视角没有对应的二维标签，跳过这些额外区域项。原始未剪枝教师始终冻结，学生 SH3→SH2 创建新 AdamW。

## 逐点身份与输出

标准 PLY 保持原生 SH 属性，不把大特征写入模型。每个 checkpoint 配套 `.instances.npz`，各阶段 PLY 目录及根目录配套 `instance_priors.npz`。`original_ids` 跟随剪枝索引严格同步，支持精确追踪；源 XYZ 哈希与训练后当前 XYZ 哈希分开保存。输入未知编号 -1 在这些训练旁车中规范化为 0。

SH2 蒸馏不再增删高斯，所有实例配额应持续满足。`evaluate_instance_retention.py` 精确检查实例消失与配额违规；`evaluate_small_roi.py` 独立比较测试视角的小区域渲染误差。点数保留并不能单独证明视觉目标仍清晰，不能把伪标签覆盖率或非零点数当成检测召回率。

`export_instance_ply.py` 可把当前 PLY 与同目录的实例旁车合为带实例属性的 PLY。它核验当前 XYZ 哈希、逐行长度及原始索引，完整保留原有渲染属性，追加 `instance_id`、`instance_confidence`、`instance_views`、`instance_source` 和 `original_id`。0 是未知编号；正数只表示自动区域，不表示语义类别。输出采用原子写入且拒绝覆盖已有文件。标准 SH2 PLY 与带实例字段 PLY 都可保留，前者体积更小。

## 运行

```bash
# 新方法：目标178809个高斯，PWSR到25199，SH2蒸馏35000→40000
bash scripts/run.sh run_pipeline.py --config configs/city_instance.json

# 同预算、同原始教师、同蒸馏步数的全局SPS对照；保留identity旁车，但不启用保护/ROI
bash scripts/run.sh run_pipeline.py --config configs/city_instance_control.json

# 流程完成后导出带实例字段的 SH2 PLY（输出文件必须尚不存在）
bash scripts/run.sh export_instance_ply.py \
  --ply output/city2_instance_guard_20261026/protected/sh2/point_cloud/iteration_40000/point_cloud.ply \
  --metadata output/city2_instance_guard_20261026/protected/sh2/point_cloud/iteration_40000/instance_priors.npz \
  --output output/city2_instance_guard_20261026/protected/sh2/point_cloud/iteration_40000/point_cloud_instances.ply
```

数据准备使用 `prepare_instance_protection.py`，严格拒绝来源中出现非训练相机。单元测试遵循仓库既有命令。最终对外结论以实际训练、独立测试 ROI 和指标报告为准，不承诺自动 SAM 未发现的真实小目标也一定获得保护。

`audit_instance_visibility.py` 分别统计 sigmoid(opacity) 超过 0.01 / 0.05 的点数；这是不透明度弱代理，不代替逐视角的实际贡献。`make_small_roi_comparison.py` 同时选择改善、接近中位数和退化案例，保留原生像素裁剪。实际项目的 65 项单元检查与短程 GPU 剪枝、蒸馏、实例导出已通过；完整实验指标保存在独立运行目录，不使用短程联调的质量数字。

## v2.0.0 已完成实验

保护版与同预算对照的完整 City 实验均已完成，最终各 178809 个 SH2 高斯。9 个测试视角的 813 个小区域平均 PSNR 从 38.1593 提高到 38.3351 dB，66.54% 区域改善；整图 PSNR 从 34.8479 降至 34.4456 dB，存在质量代价。两版均未出现自动区域整体归零，不能据此宣称所有真实小目标消失问题已解决。完整进度、模型位置和结果摘要见 `PROGRESS.md`。
