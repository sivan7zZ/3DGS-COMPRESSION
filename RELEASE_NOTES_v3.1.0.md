# v3.1.0 — 自适应约束压缩与地物保护评测

保存日期：2026-11-04（Asia/Shanghai）。本版本基于v3.0.0，保留PWSR RIR/DPRA/SPS与LightGaussian SH蒸馏，不含3D-DoG。

## 本次保存

- 自适应压缩入口：原始全局SPS评分下的实例存活配额、边界配额、稀疏PCA图/割点保护、自适应相对剪枝与恢复、高斯矩匹配合并、硬区域预算回填。标准PLY保持原生渲染字段，实例和结构元数据另存。
- 10×/20×完整实验：分别保留178809/89405个高斯；SH3分别4019/4224次恢复更新，SH2均使用原始未压缩教师蒸馏5000次，蒸馏前后结果分别保存。
- 四种既有基线LightGaussian、PUP-3DGS、Speedy-Splat、PWSR，原始3DGS、GT及v3参考；统一9个原生测试相机、SH3渲染。Speedy初始化不同、LG10多1点的比较限制已记录。
- 冻结City下游分割器及配准标签；仅8张非测试标注图参与拟合。2张带标签测试图用于语义及地物保护指标，压缩训练不使用这些语义真值。
- Small-object Recall、Boundary F1、clDice、ROI-PSNR/ROI-LPIPS、逐类与逐区域数据及独立核验。LPIPS统一新评测采用VGG0.1标准[-1,1]输入，历史[0,1]结果留存，不跨协议混用。
- 保存已有消融配置和工具；按最新要求，当前地物保护评测不启动或读取消融实验。

## 最新结果

Small-object Recall在此明确为同类别8连通小语义区域代理，面积16像素至整图1%、IoU≥0.5一对一匹配。Boundary F1为2像素容差、逐类汇总后宏平均；clDice为硬Zhang语义骨架指标。全部248个小区域，Vehicle其中59个。ROI为固定GT包围框加8像素上下文、边长至少32像素、原生不缩放，逐ROI等权平均。

| 本次方法 | Small Recall↑ (%) | Boundary F1↑ (%) | clDice↑ (%) | ROI-PSNR↑ | ROI-LPIPS↓ |
|---|---:|---:|---:|---:|---:|
| 10× | 60.48 | 39.22 | 77.85 | 37.1349 | 0.06230 |
| 20× | 59.68 | 37.54 | 76.90 | 34.8280 | 0.10628 |

PWSR20对应58.87%、38.06%、77.50%、35.1289、0.09966。本方法召回多2个小语义区域，边界、语义骨架和248区域平均视觉质量未同步改善，不能宣称全面优于基线。只有两张标注测试图，且没有电线/井盖独立标注；这些代理指标不能直接证明物理电线或井盖是否丢失。

[各方法完整CSV](reports/v3.1.0/preservation_metrics.csv)、[机器可读摘要](reports/v3.1.0/preservation_summary.json)、[独立核验摘要](reports/v3.1.0/preservation_validation.json)、[原始结果文件SHA](reports/v3.1.0/artifact_manifest.json)。GT RGB的语义值经过同一分割器，不设为100%；视觉自比较在表中留空。

## 验证

- 项目测试176项：175通过，1项因项目环境无scikit-image跳过；新指标6项测试在已有VGGT环境全部通过。
- 独立核验14组、3556项ROI：欧氏距离变换核对边界匹配，重算硬骨架clDice与ROI像素误差，核对ROI-LPIPS来源和聚合、模型权重及区域分母。GT标签自比较三项为1，GT RGB所有ROI自比较LPIPS为0。
- 标准PLY与checkpoint的原生参数一致性、预算与配额检查保留在各实验输出核验文件中。

## 本地继续工作

仓库：`/home/zhangys/Prune-Wisely-Reconstruct-Sharply`。

- 新方法模型：`output/city2_adaptive_constraints_20261102/full_10x/{stage3,sh2}`、`full_20x/{stage3,sh2}`。
- 统一比较：`output/city2_comparison_20261104/`。
- 当前重点报告：`output/city2_comparison_20261104/preservation_metrics/README_地物保护指标.md`。
- 完整标签、模型、权重、渲染、逐ROI文件与运行日志保留本机，未纳入Git；`artifact_manifest.json`记录原始位置与SHA。在其他机器运行需准备这些依赖并调整配置绝对路径。

```bash
bash scripts/run.sh -m unittest discover -s tests -v
/home/zhangys/anaconda3/envs/VGGT/bin/python -m unittest discover -s tests -p test_preservation_metrics.py -v
```

评测可按以下阶段复现；先保存旧结果，或指定新输出目录，避免覆盖当前证据：

```bash
/home/zhangys/anaconda3/envs/VGGT/bin/python evaluate_preservation.py --phase prepare --output output/city_preservation_reproduce
bash scripts/run.sh evaluate_preservation.py --phase lpips --output output/city_preservation_reproduce
bash scripts/run.sh evaluate_preservation.py --phase report --output output/city_preservation_reproduce
/home/zhangys/anaconda3/envs/VGGT/bin/python validate_preservation.py --output output/city_preservation_reproduce
```

压缩复现与方法范围见 [ADAPTIVE_COMPRESSION.md](ADAPTIVE_COMPRESSION.md)、[WORKFLOW.md](WORKFLOW.md)；全部历史进度见 [PROGRESS.md](PROGRESS.md)。
