# 3DGS COMPRESSION

当前流程使用边界加权PUP Fisher评分、PWSR RIR/DPRA调度，以及实例/拓扑保护、合并和预算回填，输出标准SH3模型，不运行SH蒸馏。
**不包含 3D-DoG，也不是上游论文完整四阶段的复现版本。**
历史v3.2.0及更早的SPS、边界硬配额和SH蒸馏入口保留供复现，无需调用外部LightGaussian仓库。

详细输入输出、迭代编号、checkpoint 语义和历史结果说明见 [WORKFLOW.md](WORKFLOW.md)。

## v3.3.0：SAM/深度/RGB连续边界PUP，无蒸馏

2026-11-09按用户要求将边界从二值掩膜改为三通道连续先验：`B=0.4SAM+0.4depth+0.2RGB`，`score=PUP_logdet+log(1+5B)`。三信号由61个训练视角按高斯实际alpha*T可见贡献汇总；权重预先固定，未用人工语义标签调参。实例、拓扑、RIR/DPRA、恢复损失、合并1%/回填2%不变。方法与运行见[CONTINUOUS_BOUNDARY_COMPRESSION.md](CONTINUOUS_BOUNDARY_COMPRESSION.md)。新输出为`output/city2_continuous_boundary_20261109/`，10×/20×及统一评测、独立核验均已完成，结果不与历史目录混用。PSNR为34.9234/33.4940，标准LPIPS为0.06313/0.10300。独立主测试Recall为182/283和174/283；相对二值版，两倍率B-mIoU、mIoU、mAcc下降，未确认连续版独有的整物体挽救。这组参数不能称为更好的边界保护。完整结果、全部方法比较与模型链接见该输出目录的`README_实验总览.md`和`comparison_all_metrics.csv`。

[v3.3.0版本说明](RELEASE_NOTES_v3.3.0.md) · [完整比较CSV](reports/v3.3.0/comparison_all_metrics.csv) · [验证摘要](reports/v3.3.0/verification_summary.json) · [保存清单](reports/v3.3.0/artifact_manifest.json)。

## 20261106：历史二值边界加权PUP，无蒸馏

按2026-11-06要求，新流程仅生成SH3压缩结果：原始PUP logdet加 `2 × boundary_mask`，边界不再设独立剪枝配额或合并禁令；实例/拓扑保护与恢复监督保持。合并上限从0.5%提高到1%，回填比例从1%提高到2%。方法、定义和新运行入口见 [BOUNDARY_WEIGHTED_COMPRESSION.md](BOUNDARY_WEIGHTED_COMPRESSION.md)。

City匹配标签共10张，2张独立测试视角和8张训练视角辅助结果分开报告，辅助语义评估器采用分折拟合。10×/20×及全部标签评测已完成，9视角PSNR分别为34.8140/33.4494，标准LPIPS为0.06314/0.10282。10张标签汇总指标均较v3.2.0小幅改善，但20×独立测试Recall下降1.41个百分点，未确认相对v3.2/PWSR独有的整物体挽救。

新输出为 `output/city2_boundary_weighted_20261106/`，本机报告 `README_实验总览.md`、`README_完整结果对比.md` 和完整 `comparison_all_metrics.csv` 均位于该目录。入口 `bash scripts/run.sh run_boundary_weighted_experiments.py --resume`，评测入口见方法说明；历史结果保留。

## v3.2.0：PUP评分 + RIR/DPRA

保留v3.1.0全部保护、合并、预算回填和恢复损失，把评分改为PUP六维Fisher的SVD logdet，并恢复原RIR/DPRA调度。方法、配置、运行及续跑方式见 [PUP_RIR_DPRA.md](PUP_RIR_DPRA.md)。入口为 `bash scripts/run.sh run_pup_rir_experiments.py`，10×/20×均已完成，SH3与SH2结果分别保存在 `output/city2_pup_rir_dpra_20261105/`；同一冻结分割器和固定ROI复评及独立核验已完成。

SH3的10×/20× PSNR分别为34.7279/33.4243，标准LPIPS为0.06455/0.10968。相比v3.1.0，20×多召回1个小语义区域，248区域平均ROI质量提高；固定6处ROI的20× LPIPS变差，尚未确认新版独有的整物体挽救。评分、调度和恢复步数同时改变，不能把差异单独归因于PUP。

[v3.2.0版本说明](RELEASE_NOTES_v3.2.0.md) · [完整比较CSV](reports/v3.2.0/comparison_all_metrics.csv) · [蒸馏前后摘要](reports/v3.2.0/compression_summary.json) · [实验文件SHA](reports/v3.2.0/artifact_manifest.json)。大型模型和渲染保留本机，Git保存源代码及轻量实验记录。

## v3.1.0：自适应约束压缩与地物保护评测

在v3基础上新增原始全局SPS排序下的实例/边界配额、拓扑保护、自适应恢复、高斯矩匹配合并与预算回填。City的10×/20×实验已完成，SH3剪枝结果与SH2蒸馏结果分别保留。方法与配置见 [ADAPTIVE_COMPRESSION.md](ADAPTIVE_COMPRESSION.md)。

现有LightGaussian、PUP-3DGS、Speedy-Splat、PWSR、原始3DGS及GT已纳入统一比较，新增小区域Recall、Boundary F1、语义骨架clDice和ROI-PSNR/LPIPS。当前方法20×小区域Recall较PWSR提高0.81个百分点，边界、语义骨架与平均ROI质量未同步改善。现有City标签不含独立电线、井盖标注，不能据此宣称这类目标已经完整保留。

[v3.1.0版本说明](RELEASE_NOTES_v3.1.0.md) · [结果CSV](reports/v3.1.0/preservation_metrics.csv) · [比较协议](COMPARISON_PROTOCOL.md) · [完整进度](PROGRESS.md)。模型、渲染和完整逐ROI结果留在本机 `output/`；Git保存代码、配置、测试及轻量结果摘要。

## v3.0.0：无语义标签的实例与结构保护

新增训练视角可见支撑约束、小区域配额、边界对比监督及细线连续性监督；主配置为 `configs/city_structure20_adaptive.json`。City 20×实验保留 89,405 个高斯，SH3 与 SH2 分别保存，97 项测试通过。

主分析使用 SH 蒸馏前 SH3。新版在固定例子中保留了 LightGaussian20×下退化的外观和细节，但相对原 PWSR20×尚未确认新增模块独有的目标挽救，且存在整图质量代价。当前不能宣称三类问题已全面解决。

方法与复现见 [STRUCTURE_PROTECTION.md](STRUCTURE_PROTECTION.md)，版本进度、结果与本地模型位置见 [PROGRESS.md](PROGRESS.md)。大型模型、数据和渲染保留本机 `output/`，不随源码上传。

## 当前机器运行

```bash
cd /home/zhangys/Prune-Wisely-Reconstruct-Sharply
# 仅查看新SH3流程，不训练、不创建输出
bash scripts/run.sh run_boundary_weighted_pipeline.py --config configs/city_continuous_boundary_10x.json --dry_run
# 中断续跑本轮10×/20×；完成的步骤核验后跳过
bash scripts/run.sh run_continuous_boundary_experiments.py --resume
```

从头复现时复制配置并改为新的output_root，避免覆盖当前模型。历史蒸馏入口如下，仅用于复现旧实验：

```bash
bash scripts/run.sh run_pipeline.py --config configs/city.json --stage distill \
  --stage3_checkpoint output/city2_stage3/chkpnt25199.pth \
  --output_root output/city2_checkpoint_sh2_5k
```

默认 SH3 → SH2，实际更新 5,000 次，编号 35001–40000；高斯数量不变。
这是本整合流程的显式设置，不代表上游脚本始终只运行 5,000 次。
启动脚本默认使用本机已有 `prune_wisely` 环境，可通过 `PWSR_PYTHON` 指定其他 Python。

## 环境与标准渲染器

需要 NVIDIA GPU、兼容的 CUDA toolkit 和环境中的 PyTorch。
`environment.yml` 提供依赖清单。已有环境无需重新创建。
渲染器从本仓库加载；修改 CUDA 源码后，在同一环境重新编译：

```bash
conda activate prune_wisely
cd submodules/gaussian-rasterization
python setup.py build_ext --inplace
```

本机已使用 CUDA 11.8、GCC 11 和 `TORCH_CUDA_ARCH_LIST=7.5` 编译通过。
其他显卡应选择对应架构，不要直接照搬 7.5。

## 入口脚本

| 文件 | 用途 |
| --- | --- |
| `run_pipeline.py` | 统一调度、日志、配置和输出冲突检查 |
| `train_prune.py` | 预训练 3DGS → RIR / DPRA / SPS → 标准 SH3 checkpoint / PLY |
| `distill_train.py` | checkpoint 教师 / 学生 → SH2 蒸馏 |
| `render.py`、`metrics.py` | 标准 PLY 渲染、PSNR / SSIM / LPIPS |
| `full_eval.py` | 统一调度入口的兼容别名，参数同 `run_pipeline.py` |

```bash
bash scripts/run.sh -m unittest discover -s tests -v
```

## 上游来源

基于 PWSR 上游代码修改，并适配 LightGaussian 的蒸馏逻辑；保留原有版权和许可。
LightGaussian 来源和本地适配范围见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

**CVPR 2026**

[[Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Prune_Wisely_Reconstruct_Sharply_Compact_3D_Gaussian_Splatting_via_Adaptive_CVPR_2026_paper.html)]
[[arXiv](https://arxiv.org/abs/2602.24136)]

## License

This project is derived from the official 3D Gaussian Splatting implementation
and retains its non-commercial research and evaluation license. See
`LICENSE.md`.
