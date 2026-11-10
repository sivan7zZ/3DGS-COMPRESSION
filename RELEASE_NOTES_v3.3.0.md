# v3.3.0 — SAM/深度/RGB连续边界PUP压缩

保存日期：2026-11-10（Asia/Shanghai）。本版保存v3.2.0之后的无蒸馏流程、二值边界参考及最终连续边界版本，包含已完成的City 10×/20×实验和统一评测。

## 当前方法

- 标准SH3压缩，保留PUP六维xyz/log-scale Fisher logdet与原PWSR RIR/DPRA；实例/拓扑保护、恢复损失继续保留。
- 三通道为SAM轮廓距离强度、渲染深度边缘和RGB灰度边缘。SAM使用sigma=2像素的距离高斯；深度/RGB用q95归一化Sobel。仅从61训练视角按原生各向异性alpha×T贡献汇总，不使用人工语义真值。
- `B=0.4*SAM+0.4*depth+0.2*RGB`，`score=PUP_logdet+log(1+5B)`；分量为[0,1]连续值，奇异Fisher的负无穷保持。权重在运行前固定，未用评测标签调参。
- 边界不设独立剪枝配额或合并否决；原边界掩膜仍供拓扑构造和兼容恢复监督。每轮合并上限1%、回填比例2%，最终预算精确。新流程不运行SH蒸馏。
- 三连续字段随剪除、合并、池保存、回填及续训同步；合并按原质量权重计算归一化凸平均，lineage保存权重用于独立复算。
- 保留二值参考及历史v3.2.0入口供复现，增加SH3串行运行、连续先验、独立验证与冻结五折语义评测入口，并隔离跨环境动态库。

方法与运行：[CONTINUOUS_BOUNDARY_COMPRESSION.md](CONTINUOUS_BOUNDARY_COMPRESSION.md)。

## 完成的实验

两组各10199次训练/恢复更新、5次剪枝，最终分别178809和89405个高斯。视觉指标为9个固定独立测试视角的原生PNG；LPIPS为标准VGG0.1、输入[-1,1]。下表语义为全部10张City标签的混合汇总，其中2张独立测试、8张为分折留出的3DGS训练视角辅助，不能称为10张独立测试。

| 倍率 | PSNR↑ | SSIM↑ | LPIPS↓ | 区域Recall↑ | B-mIoU↑ | mIoU↑ | mAcc↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10× | 34.92340 | 0.97676 | 0.06313 | 64.822% | 68.437% | 75.948% | 86.776% |
| 20× | 33.49404 | 0.96338 | 0.10300 | 64.229% | 67.237% | 74.876% | 85.973% |

相对二值边界版，PSNR提高，但两倍率独立主测试和10帧汇总的B-mIoU、mIoU、mAcc均下降，20×LPIPS略有变差。独立主测试Recall为182/283、174/283，分别比二值版多1个和持平。当前连续参数未显示更好的边界保护，不作为“全面保留小目标/修复边界和细线”的结论。

固定六处原生裁剪中，连续版、二值版和PWSR均保留主要目标，尚未确认新增模块独有的整物体挽救。City标签没有电线、井盖独立实例真值；Recall是同类8连通语义区域IoU≥0.5的一对一召回，最小16像素，不是物理实例召回。B-mIoU沿用2%对角线内边界带协议，0类忽略。

## 验证与保存范围

- 保存前重新运行261项测试：260通过、1项因缺scikit-image跳过；本次保存没有重新训练模型。
- 两倍率的PLY/checkpoint标准62字段、精确预算、原RIR/DPRA重放、实例配额和连续元数据行对齐核验通过。315个实验产物SHA在发布前逐一复核。
- 共19模型+GT、3范围60组统计；同一组5个冻结语义评估器，180张预测按checkpoint+RGB SHA复用，新增20张。独立CPU核验200预测、60组confusion/mIoU/mAcc，误差均为0。Recall/B-mIoU使用已测试统一实现，未声称另行独立重算。
- [完整指标CSV](reports/v3.3.0/comparison_all_metrics.csv)、[核验摘要](reports/v3.3.0/verification_summary.json)、[源/结果指纹](reports/v3.3.0/artifact_manifest.json)、[轻量记录索引](reports/v3.3.0/README.md)。
- [10×原生裁剪](reports/v3.3.0/figures/native_comparison_10x.png)、[20×原生裁剪](reports/v3.3.0/figures/native_comparison_20x.png)、[连续强度分布](reports/v3.3.0/figures/continuous_strength_histograms.png)。

完整模型、checkpoint、连续先验、渲染、逐轮评分和恢复状态保留本机`/home/zhangys/Prune-Wisely-Reconstruct-Sharply/output/city2_continuous_boundary_20261109/`；二值参考保留`output/city2_boundary_weighted_20261106/`。Git保存代码、配置、测试、方法说明和轻量结果；其他机器复现仍需准备配置中的数据、先验、权重及运行环境。

实验结束时的`source_changes_committed:false`等历史状态保持原样；本次版本由v3.3.0 Git标签和本目录发布清单识别，既有实验指纹与历史标签不改写。
