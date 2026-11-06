# v3.2.0 — PUP Fisher评分与RIR/DPRA保护压缩

保存日期：2026-11-06（Asia/Shanghai）。基于v3.1.0，保留实例存活、边界配额、拓扑保护、高斯矩匹配合并、预算回填及恢复损失，替换评分与剪枝调度。不含3D-DoG。

## 方法与代码

- 评分改为PUP原始位置xyz与log-scale六维Fisher块的SVD logdet，使用全部61个训练相机和原生像素。不加阻尼、频域权重或ROI分数；奇异矩阵产生的负无穷保留，稳定按当前行序处理并列。
- 恢复原PWSR RIR/DPRA，虚拟15001–25199、10199次更新；每500次检查，最长等待1500次，beta=0.99。预算按1/2、1/4、1/8、1/16和剩余量分5次执行，保留可行性保护事务。
- 新训练、串行实验、续跑、独立核验和对比入口；10×/20×配置及评分/调度测试。方法和运行说明见 [PUP_RIR_DPRA.md](PUP_RIR_DPRA.md)。
- 本仓库编译的Fisher CUDA源码来自PUP后端固定提交 `eaef2faba21f076867c1db84c3f998e5f673762f`。保留版权、许可证与文件来源SHA；见 [第三方说明](THIRD_PARTY_NOTICES.md) 和 [来源记录](submodules/pup-fisher/PORT_PROVENANCE.json)。平台编译产物保留本地。

## 完成的实验

10×保留178809个高斯，20×保留89405个高斯。两组各10199次训练/恢复更新；使用原始未剪枝教师分别完成5000次SH3→SH2蒸馏。两阶段均单独保存。

| 倍率 | 阶段 | PSNR↑ | SSIM↑ | 标准LPIPS↓ |
|---|---|---:|---:|---:|
| 10× | SH3 | 34.7279 | 0.97632 | 0.06455 |
| 10× | SH2 | 34.8489 | 0.97579 | 0.06624 |
| 20× | SH3 | 33.4243 | 0.96082 | 0.10968 |
| 20× | SH2 | 33.4729 | 0.96131 | 0.10800 |

视觉指标使用9个固定原生测试相机、标准VGG0.1 LPIPS的[-1,1]输入；历史[0,1]结果不混用。语义及保护评测使用同一冻结City分割器、2张有标注测试图与248个小语义区域，协议沿用 [COMPARISON_PROTOCOL.md](COMPARISON_PROTOCOL.md) 及评测源代码。压缩训练不使用City语义真值。

| SH3 | 小区域Recall↑ | Boundary F1↑ | 语义clDice↑ | ROI-PSNR↑ | ROI-LPIPS↓ |
|---|---:|---:|---:|---:|---:|
| 10× | 60.48% (150/248) | 39.58% | 77.91% | 37.6756 | 0.05279 |
| 20× | 60.08% (149/248) | 39.10% | 78.21% | 35.2468 | 0.09266 |

与v3.1.0相比，10×小区域Recall持平，20×多召回1个小区域，248区域平均ROI质量改善；20×固定6处ROI-LPIPS由0.11476变差为0.12995。固定裁剪中主要目标在旧方法中也保留，尚未确认新版独有的整物体挽救。评分、调度及总恢复步数同时改变，不能把改善单独归因于PUP。City没有电线、井盖独立标注，语义区域Recall和语义骨架不能直接表示其物理实例召回或连通性。

## 验证与保存范围

- 项目测试182项：181通过、1项因项目环境缺scikit-image跳过。Fisher CUDA与逐像素Jacobian核对相对误差约7.38e-8，模型梯度缓冲未改变。
- 两组PLY/checkpoint原生参数一致、预算精确、实例/边界配额零违规、SH2元数据行序一致、教师未变。五次非零剪枝发生于15700、17200、18700、20200、21700；调度与原RIR/DPRA重放一致。
- 保护评测独立核验14组、3556项ROI，重算边界匹配、硬骨架及像素误差，并核对LPIPS聚合和输入SHA。冻结评估器未重训，只推理4张新图，24张不变GT/基线预测经SHA核对后复用。

[完整比较CSV](reports/v3.2.0/comparison_all_metrics.csv) 包含LightGaussian、PUP-3DGS、Speedy-Splat、PWSR、本版本、v3参考、原始3DGS及GT；模型比较使用SH蒸馏前结果。另保存 [蒸馏前后摘要](reports/v3.2.0/compression_summary.json)、[相对v3.1.0保护指标](reports/v3.2.0/preservation_comparison.json)、[10×核验](reports/v3.2.0/validation_10x.json)、[20×核验](reports/v3.2.0/validation_20x.json)、[评测核验](reports/v3.2.0/preservation_validation.json) 和 [源文件/结果清单](reports/v3.2.0/artifact_manifest.json)。

完整输出位于本机 `/home/zhangys/Prune-Wisely-Reconstruct-Sharply/output/city2_pup_rir_dpra_20261105/`，其中 `full_{10x,20x}/{stage3,sh2}` 保存标准PLY、checkpoint和元数据；总览为 `README_实验总览.md`。Git保存代码、配置、许可、测试及轻量摘要；大型模型、数据、渲染、逐ROI文件、编译产物和完整日志留在本机。其他机器需准备配置中的输入数据、训练专用元数据及运行环境。

原实验清单 `final_artifact_manifest.json` 按实验结束时状态原样留存，其中 `source_changes_committed: false` 是当时记录；本次版本归档由v3.2.0标签及本目录的发布清单标识，不改写旧证据。历史标签和输出保留，本次保存不重跑训练。
