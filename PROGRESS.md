# v3.0.0 进度快照（2026-10-29 保存）

本版本保存 2026-10-27 完成的三类结构保护代码、实验结论及复现入口。大型模型与运行数据仍保留本机项目 output/，Git 保存源码、配置、测试及进度摘要。

已按最新要求完成小目标、边界、细长结构三类源可见支撑保护及 City 全量 20×实验。主入口 `configs/city_structure20_adaptive.json`；无人工2D语义标签、不使用gt_semantic模型、不含3D-DoG。所有新模型、checkpoint、SAM标签、原生渲染、证据与报告在本仓库 `output/city2_structure_guard_20261027/`，保留SH3与SH2。

- 主模型 `protected20_adaptive/`：源1,788,093→89,405点，SH3标准PLY约21.15MiB（约20×），SH2约13.98MiB（约30.24×），实例字段额外占空间。PWSR10199次更新，原始SH3教师蒸馏5000次更新。
- 61训练视图构建296,910个源可见候选单元。最终剪枝从实例SPS出发，8,677对高斯交换补齐23,549个初始无支撑单元，无回退，配额0违规。固定证书仅在中间阶段保证可行性；最终允许旧锚点替换且高斯属性自由优化。
- 对照保留用户给出的原生LG10，并新增真实LG20与原PWSR20，两种20×对照同为89,405点；原PWSR10、v2实例10×及固定锚点20×消融也保留。外部LG恢复调度不同，不冒充完全相同训练设置。
- 主要结论来自SH3同相机1592×894图。固定六例中，新版相比LG20保留黑车可辨性、白色庭院小物、屋面轮廓与细窗框；相比原PWSR20整体相当，没有确认新增模块独有的目标挽救。全量自动候选边界召回略升，小区域与细线略降，不能宣称三类问题已被全面解决。测试视图用于开发诊断，本轮不是新的盲测基准。
- 固定锚点初版使恢复受限、整体结构退化，保存为 `protected20/` 消融，不作为最终主模型。source覆盖/配额保证不等于渲染可见性或真实物体召回率。
- 详细主报告：`output/city2_structure_guard_20261027/protected20_adaptive/README_结构保护结果.md`；五列图：`reports/primary_adaptive/compact_visual_review/reviewed_native_comparison.png`；机制说明：`STRUCTURE_PROTECTION.md`。

训练源码快照保存在主模型 `source_snapshot/`。训练结束后仅补充早期预算一致性和保护模式续训的配额校验，不改变本次选择器或优化计算；全局SPS对照仍可加载低于保护配额的旁车。

验证完成：97项单元测试全通过，SH3/SH2已完成checkpoint均能加载完整续训状态并零新增更新退出；两个实例PLY导出核验通过。共同原生PNG评估中，原PWSR20 PSNR33.6782/SSIM0.9581/LPIPS0.0856，adaptive SH3为33.1624/0.9520/0.0958，存在整图质量代价；不把局部结构证据当作全面质量提升。

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
