# 3DGS COMPRESSION

本分支将 PWSR 的 RIR / DPRA / SPS 剪枝与 LightGaussian 的 SH 蒸馏整合在同一仓库。
**不包含 3D-DoG，也不是上游论文完整四阶段的复现版本。**
蒸馏的教师、学生均从标准 checkpoint 初始化，PLY 仅作为导出结果。
教师使用未剪枝的原始 3DGS checkpoint，学生使用 PWSR 剪枝后的 checkpoint。
无需调用外部 LightGaussian 仓库，不包含 LightGaussian 的重复剪枝或 VecTree 量化。

详细输入输出、迭代编号、checkpoint 语义和历史结果说明见 [WORKFLOW.md](WORKFLOW.md)。

## v3.0.0：无语义标签的实例与结构保护

新增训练视角可见支撑约束、小区域配额、边界对比监督及细线连续性监督；主配置为 `configs/city_structure20_adaptive.json`。City 20×实验保留 89,405 个高斯，SH3 与 SH2 分别保存，97 项测试通过。

主分析使用 SH 蒸馏前 SH3。新版在固定例子中保留了 LightGaussian20×下退化的外观和细节，但相对原 PWSR20×尚未确认新增模块独有的目标挽救，且存在整图质量代价。当前不能宣称三类问题已全面解决。

方法与复现见 [STRUCTURE_PROTECTION.md](STRUCTURE_PROTECTION.md)，版本进度、结果与本地模型位置见 [PROGRESS.md](PROGRESS.md)。大型模型、数据和渲染保留本机 `output/`，不随源码上传。

## 当前机器运行

```bash
cd /home/zhangys/Prune-Wisely-Reconstruct-Sharply
# 仅查看命令，不训练、不创建输出
bash scripts/run.sh run_pipeline.py --config configs/city.json --dry_run
# 完整流程；默认输出到 output/city2_unified，不覆盖旧结果
bash scripts/run.sh run_pipeline.py --config configs/city.json
```

只使用已有 PWSR checkpoint 做蒸馏：

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
