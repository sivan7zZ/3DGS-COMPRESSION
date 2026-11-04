## v3.1.0 版本快照（2026-11-04）

- 本版本保存v3.0.0之后的自适应可行性约束压缩、City 10×/20×实验配置、统一基线比较、语义及地物保护评测源代码、测试和进度。
- 最新分析以地物保护指标为准；按用户要求不再继续开展消融。既有实验文件和蒸馏前后模型完整保留在本机output中。
- 可随Git查看的轻量结果与核验摘要位于 `reports/v3.1.0/`，版本说明见 `RELEASE_NOTES_v3.1.0.md`。本次使用新标签v3.1.0，历史v3.0.0/v2.0.0标签保留。

## 2026-11-04 地物保护指标评测（完成）

- 按最新要求仅计算现有模型的保护指标，不启动训练或消融；旧实验文件保留。新评测目录为 `output/city2_comparison_20261104/preservation_metrics/`。
- 覆盖LightGaussian、PUP-3DGS、Speedy-Splat、PWSR、本次方法的10×/20×，以及原始3DGS、GT RGB；既有v3另存CSV。全部为蒸馏前SH3。
- 现有City标注没有独立电线、井盖实例或细线；按用户确认使用现有六类标签。仅2张有标注的测试图：frame_000/frame_056。Small-object Recall明确为语义连通小区域代理，clDice明确为语义骨架指标，不当作物理电线/井盖测量。
- 固定248个GT小区域（面积16像素至整图1%，IoU≥0.5一对一匹配），其中Vehicle 59个。Boundary F1为2像素欧氏容差、六类宏平均；clDice使用硬Zhang骨架，两帧先汇总计数再宏平均。分割器沿用已冻结权重，不重训或调参。
- ROI统一使用GT区域包围框加8像素上下文、原生边长至少32、不缩放；逐ROI计算后等权平均PSNR/LPIPS。LPIPS用VGG0.1、标准[-1,1]输入；另列只在目标像素上的PSNR，以及此前固定6处地物ROI的局部质量。

| 本次方法 | Small Recall | Boundary F1 | 语义clDice | ROI-PSNR | ROI-LPIPS |
| --- | --- | --- | --- | --- | --- |
| 10× | 60.48% (150/248) | 39.22% | 77.85% | 37.1349 | 0.06230 |
| 20× | 59.68% (148/248) | 37.54% | 76.90% | 34.8280 | 0.10628 |

- PWSR20对应58.87% (146/248)、38.06%、77.50%、35.1289、0.09966。本次方法小区域召回增加2个，但边界、语义骨架和248区域平均视觉质量未同步改善；不能写成全面解决地物消失或细结构断裂。
- 评测入口 `evaluate_preservation.py --phase prepare/lpips/report`；CPU骨架阶段用已有VGGT环境，LPIPS用项目环境。仓库测试共176项：175通过、1项因项目环境无skimage跳过；6项新指标测试在VGGT环境全部通过。原始标签自比较三项指标为1，GT RGB所有ROI自比较LPIPS为0。
- `validation.json`独立核验通过：用欧氏距离变换重算所有边界匹配，重算硬骨架clDice及全部3556项ROI像素误差，核对LPIPS来源/聚合、模型权重及248/59分母。新评测脚本、依赖源文件和结果SHA另存 `reproduction_source/manifest.json`。
- 详细协议、结果、逐ROI/逐类CSV、输入SHA与区域概览见 `preservation_metrics/README_地物保护指标.md`。独立核验入口 `validate_preservation.py`。

## 2026-11-04 统一基线比较、City语义评测与逐模块消融（历史结果保留）

- 找到并核验LightGaussian、PUP-3DGS、Speedy-Splat、PWSR的10×/20×，加本方法、原始3DGS、v3共13个SH3模型。全部用本仓库原生渲染器重渲染9个固定测试视图；相机和GT像素一致。Speedy训练初始化不同，LG10多1点，主报告单列说明。
- City人工标签仅10视图，其中测试集仅frame_000、frame_056有标签；按实际COLMAP相机将原畸变标注配准到渲染坐标。语义主指标仅2帧，视觉指标9帧。
- 固定ResNet34 U-Net：6帧训练、2帧内层验证1500步，选择1500步后同ImageNet初始化对8个非测试标注帧重训，冻结SHA35a50751d2e145f489b4f17016fe7397272f72e373e40913b73fb0ff88643519。压缩训练不使用City语义真值。
- 完成全部方法的PSNR/SSIM/LPIPS、语义区域Recall@0.5、B-mIoU、mIoU/mAcc。主指标与定义见 `COMPARISON_PROTOCOL.md`、输出根目录 `README_统一对比与消融.md`。当前20×区域Recall为61.84%，mIoU71.04%，B-mIoU60.07%；PWSR分别60.78%、71.68%、60.74%，不能宣称全面超过基线。
- 原串行实验队列已经结束，旧消融文件保留。按最新要求，本轮地物保护指标不读取消融模型，也不继续启动消融实验。
- 全部新数据在 `output/city2_comparison_20261104/`，队列入口 `run_city_benchmark_queue.py --resume`，状态 `benchmark_queue.json` 与 `ablations20/ablation_manifest.json`。

## 2026-11-03 文档方法实现与 10× / 20×实验（完成）

- 按 `3DGS_methods_for_Codex.md` 完成原始全局 SPS 可行性约束、实例存活/边界配额、稀疏PCA图与割点保护、动态恢复轮次、矩匹配合并、硬区域预算回填。原v3入口保留；没有逐实例排序或修改基础评分。
- 61训练视图构建1021个无类别伪实例，407414个未知高斯。9测试视图未参与关联、约束、恢复触发或回填。
- 10×实际12轮、4019次恢复更新，保留178809个高斯；20×实际13轮、4224次更新，保留89405个高斯。两组各完成原始未剪枝教师监督的5000次SH2蒸馏。
- 10×累计合并14318次、回填75590次；20×累计合并14211次、回填73070次。这些是跨轮累计事件数，不是最终独立高斯数。
- 132项测试通过；最终四个PLY与对应checkpoint全部原生参数逐值一致；各轮实例/边界配额零违反，SH2行标签完全一致，教师未变。诊断在 `final_validation.json`。

| 倍率 | 阶段 | PNG PSNR | SSIM | LPIPS |
| --- | --- | --- | --- | --- |
| 10× | SH3 | 34.6614 | 0.97312 | 0.05148 |
| 10× | SH2 | 34.7466 | 0.97386 | 0.05040 |
| 20× | SH3 | 33.2462 | 0.95387 | 0.09377 |
| 20× | SH2 | 33.4255 | 0.95596 | 0.08968 |

- 所有结果在 `output/city2_adaptive_constraints_20261102/`，子目录 `full_10x` 和 `full_20x` 均保留 `stage3`/`sh2`，含PLY、checkpoint、辅助状态、原生测试渲染及指标。主报告 `README_自适应压缩结果.md`。
- SH3对比先前v3同预算PSNR分别提高约0.094/0.084dB；固定6处裁剪对LG20优势明显，但v3/PWSR同样保留地物，未确认新增模块独有的地物挽救。不同方法恢复步数不同，不应将当前比较视为等计算量消融。
- 标准PLY无额外特征或解码器；实例/结构信息保存为独立元数据。源支撑16px格回填、伪实例误差及候选图连通性的适用限制见 `ADAPTIVE_COMPRESSION.md` 和审计回执。
- 本段记录2026-11-03实验；其代码与运行进度随v3.1.0版本保存，历史v3.0.0标签保留。

## v3.0.0 继续工作入口

在仓库根目录查看计划，或用新的输出子目录重跑，避免覆盖已完成的 SH3 / SH2 结果：

```bash
bash scripts/run.sh run_pipeline.py --config configs/city_structure20_adaptive.json --dry_run
bash scripts/run.sh run_pipeline.py --config configs/city_structure20_adaptive.json \
  --output_root output/city2_structure_v3_reproduce
```

97 项测试与两阶段 checkpoint 恢复验证在本轮实验收尾时已通过；本次版本保存不重新启动训练。后续改进应以同预算原 PWSR20 为对照，检查实际渲染缺口和结构连续性，不能仅以源支撑覆盖或区域点数宣称物体被救回。

以下是已提交并推送的 v2.0.0 历史快照；本轮未改写该标签。

# v2.0.0 进度快照（2026-10-27）

当前已完成 City 无语义标签实例保护压缩的实现、全量训练、同预算对照和独立测试评估。本版本保留 PWSR 的 RIR / DPRA / SPS，不含 3D-DoG；随后使用原始未剪枝 SH3 教师进行 LightGaussian SH3→SH2 蒸馏。

## 已完成

- 自动实例标签：只用 RGB、COLMAP、原始 3DGS 与 SAM ViT-H；61 个训练视角关联得到 480 个类别无关区域，其中 427 个小投影区域、182 个可靠 ROI；9 个测试视角不参与标签关联或训练 ROI。
- 实例保护：不可逐轮缩水的最低点数配额、空间覆盖、ROI/边界恢复与蒸馏损失、SPS 区域项；同一个索引掩膜同步剪枝所有高斯属性和 original_id，checkpoint 旁车保存实例和恢复状态。
- 完整实验：保护版和关闭保护的全局 SPS 对照均运行 PWSR 10199 次更新、SH2 蒸馏 5000 次更新；教师相同且冻结，最终各 178809 个高斯。
- 校验：65 项单元测试通过；短程 GPU 剪枝/蒸馏/实例导出联调通过；最终实例 PLY 验证标准渲染属性保持不变、身份与行数一致。

## 实测结果

| SH2 指标 | 全局 SPS 对照 | 实例保护 |
|---|---:|---:|
| 标准 PLY 大小 / MiB | 27.97 | 27.97 |
| PNG 全图 PSNR ↑ | 34.8479 | 34.4456 |
| PNG SSIM ↑ | 0.974562 | 0.969090 |
| PNG LPIPS ↓ | 0.048472 | 0.061130 |
| 813 个测试小区域的平均 PSNR ↑ | 38.1593 | 38.3351 |
| 427 个受保护小区域的最低配额违规数 | 333 | 0 |
| 480 个自动区域完全失去高斯数 | 0 | 0 |

66.54% 的测试小区域 PSNR 提高，平均提高 0.1758 dB，但全图 PSNR 下降 0.4024 dB，LPIPS 也退化。保护版标准 SH2 PLY 相对原始 SH3 模型压缩 15.12 倍；增加实例字段后为 30.53 MiB、13.85 倍压缩。

SAM 自动区域不是人工核验的完整物体，配额保证不等于每视角视觉可见保证。当前数据中对照也没有整个自动区域点数归零，不能把这些结果写成已经解决所有真实小目标消失问题。点数、不透明度弱代理和独立测试 ROI 指标应分开解释。

## 本地数据与模型

- 原项目：`/home/zhangys/Prune-Wisely-Reconstruct-Sharply`。
- 原始 RGB / COLMAP：`/home/zhangys/gaussian-splatting（原生）/data/city`。
- 原始 PLY：`/home/zhangys/gaussian-splatting（原生）/output/city2/point_cloud/iteration_30000/point_cloud.ply`。
- 冻结教师：`/home/zhangys/gaussian-splatting（原生）/output/city2/chkpnt30000.pth`。
- 实验及复现输入：`output/city2_instance_guard_20261026/`。其中 `inputs/train_labels/` 为训练专用标签，`inputs/profiles/` 为 ROI 与面积统计；不可直接用旧的全 70 视角标签替换。
- 最终标准模型：`output/city2_instance_guard_20261026/protected/sh2/point_cloud/iteration_40000/point_cloud.ply`。
- 最终实例模型：同目录 `point_cloud_instances.ply`。
- 对外交付及完整评估：`/home/zhangys/Documents/Codex/2026-10-23/gt-semantic-3dgs-ply-ply-ply/outputs/city_instance_compression/`，含中文 README、summary.json、artifacts.json、日志、完整报告、对比图、模型及源文件修改包。最终实例 PLY 的 SHA-256 为 `f814c6cf8fba0f7038cf602afcce80e96456b51f753ac3f4c578e6920b8b81ea`。

## 复现入口

先准备上述本地训练标签及 ROI 依赖，复制配置并把 `output_root` 改到新目录后再重跑，避免覆盖本次实验。配置中的绝对数据路径需在其他机器上调整。

```bash
bash scripts/run.sh -m unittest discover -s tests -v
bash scripts/run.sh run_pipeline.py --config configs/city_instance.json
bash scripts/run.sh run_pipeline.py --config configs/city_instance_control.json
```

`INSTANCE_PROTECTION.md` 记录算法、配额和实例 PLY 导出命令；`WORKFLOW.md` 记录 checkpoint 交接、SH2 新优化器与恢复方式。训练 checkpoint 必须保留配套 `.state.pth` 与 `.instances.npz`。大型数据、模型、checkpoint 与运行缓存保留本地，Git 仅保存源代码、配置、测试和进度/结果摘要。
