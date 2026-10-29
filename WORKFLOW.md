# 固定流程：PWSR 前三阶段 → checkpoint → LightGaussian SH 蒸馏

新增的无语义实例保护分支见 `INSTANCE_PROTECTION.md`。使用 `configs/city_instance.json` 时，在原流程中加入逐实例最低配额、空间覆盖及训练视角小区域恢复 / 蒸馏损失；全局目标数量不变。`configs/city_instance_control.json` 是同预算、同教师、同蒸馏步数的全局 SPS 对照。默认 `configs/city.json` 保持原流程。

2026-10-27 的三类结构保护扩展见 `STRUCTURE_PROTECTION.md`。主入口为 `configs/city_structure20_adaptive.json`，精确保留 89,405 个高斯；训练视角可见支撑约束覆盖小区域、边界段及细线段，恢复阶段允许高斯自由优化。主对比先使用 SH3，SH2 结果另存。固定锚点初版保留为消融，不能与自适应版混称。实际渲染证据、局限及输出见 `output/city2_structure_guard_20261027/README_结构保护结果.md`。

## 范围

仓库固定为 `/home/zhangys/Prune-Wisely-Reconstruct-Sharply`。
这是本地整合版：保留 RIR / DPRA / SPS，移除 3D-DoG 的训练、字段和 CUDA 分支。
LightGaussian 仅整合 SH 蒸馏；不重复运行其剪枝、恢复和 VecTree 量化。
教师是原始未剪枝 3DGS 的 SH3 模型；学生是 PWSR 剪枝后的 SH3 模型。

## 输入

- 影像与 COLMAP：`/home/zhangys/gaussian-splatting（原生）/data/city`。
- 默认预训练输入：`/home/zhangys/gaussian-splatting（原生）/output/city2/point_cloud/iteration_30000/point_cloud.ply`。
- 可替代输入：`/home/zhangys/gaussian-splatting（原生）/output/city2/chkpnt30000.pth`。
- 蒸馏教师：`/home/zhangys/gaussian-splatting（原生）/output/city2/chkpnt30000.pth`。

在 `configs/city.json` 中，`input_ply` 与 `input_checkpoint` 必须且只能保留一个。
默认沿用旧实验的 PLY 剪枝初始化；改用 checkpoint 时通过 `--init_checkpoint` 读取模型，
仍重新建立剪枝优化器并重编号，不会把来源中的 30000 当成新的剪枝起点。
输入必须来自可信来源，PyTorch checkpoint 反序列化不是安全的未知文件读取方式。

## 默认调用链与输出

`run_pipeline.py` 使用当前 Python 和本仓库脚本，不调用外部 LightGaussian 文件夹。
以下路径相对于仓库；完整路径前缀均为 `/home/zhangys/Prune-Wisely-Reconstruct-Sharply/`。

| 脚本 | 输入 | 输出 |
| --- | --- | --- |
| `train_prune.py` | 影像 / COLMAP + 预训练模型 | `output/city2_unified/stage3/chkpnt25199.pth`、`stage3/point_cloud/iteration_25199/point_cloud.ply` |
| `distill_train.py` | 影像 / 相机 + 原始 3DGS 教师 checkpoint + 上一步学生 checkpoint | `output/city2_unified/sh2/chkpnt40000.pth`、`sh2/point_cloud/iteration_40000/point_cloud.ply` |
| `render.py`（两次） | 两阶段各自的标准 PLY | 各模型目录 `test/ours_<iteration>/renders/` 和 `gt/` |
| `metrics.py` | 渲染图与真实图 | 各模型目录 `results.json`、`per_view.json` |

两个表中缩写的 PLY 路径均位于同一个 `output/city2_unified/` 下。
每个训练 checkpoint 同时生成 `<checkpoint>.state.pth`，用于保存调度器、随机状态等续训信息。
蒸馏默认每 1000 次更新保存一次，先保存 checkpoint，再导出 PLY 和执行评估。
`run_args.json` 记录来源与配置；统一入口在输出根目录记录 `pipeline_<stage>.json` 和每阶段日志。
`metrics_float.json` 是训练内浮点渲染评估，不应和 PNG 离线评估混用。

## 迭代含义

PWSR 默认把已训练模型放在虚拟迭代 15000，运行 15001–25199，共 10199 次更新；
没有重新从稀疏点云训练 3DGS。RIR 默认每 500 次检查恢复情况，最大等待 1500 次，
实现中存在最大等待后的强制剪枝，因此不是无条件“只有恢复才允许剪”。
DPRA 控制剪枝率，SPS 保留空间和频域评分；不会运行 25200 起的 DoG。
具体剪枝事件由训练状态决定，查看 `run_args.json`，不保证每次与旧实验迭代完全一致。

SH 蒸馏分别读取原始 3DGS checkpoint 和剪枝 checkpoint：教师保持原始 SH3 并冻结；
学生从剪枝后的 SH3 保留低阶系数形成 SH2。教师和学生的高斯数量可以不同，监督发生在渲染图像上。
默认重新编号为 35000，实际更新 35001–40000，**共 5000 次**。
35000 只是蒸馏阶段编号，不代表中间又执行了 9801 次训练。
默认使用视点扰动，以教师渲染图计算 L1 + DSSIM 损失，更新学生参数；不改变高斯数量。
不运行重要性量化，因此本流程不生成 `imp_score.npz`。

## 为什么使用 checkpoint

用户提出使用 checkpoint 是合理的：原 LightGaussian 蒸馏支持教师 / 学生 checkpoint 输入。
标准 PLY 可以恢复渲染参数，但不保存优化器、迭代和随机状态，不能代替完整续训状态。
本版 `--start_checkpoint` 初始化剪枝后的学生，`--teacher_model` 初始化原始未剪枝教师；
新任务必须显式提供两者，不会把学生 checkpoint 静默复用为教师。两者张量独立，不共享可训练存储。

注意：**跨 SH3 → SH2 的阶段初始化不等于恢复原优化器。**
因为系数形状改变且这是新的蒸馏阶段，学生建立新的 AdamW 优化器，教师不使用优化器。
真正继续已经开始的 SH2 蒸馏，使用 `--resume_checkpoint`，恢复该阶段优化器、学习率调度、
随机状态和未用视点队列，且必须保留配套 `.state.pth` 及原教师文件。

```bash
bash scripts/run.sh distill_train.py \
  --resume_checkpoint output/city2_unified/sh2/chkpnt36000.pth
```

PWSR 自身续训使用 `train_prune.py --start_checkpoint ...`，也必须有本版配套 sidecar，
并传入与原剪枝任务一致的配置。旧标准 checkpoint 可用于阶段初始化，不冒充完整续训。

## 旧结果保留

- `output/city2_stage3/`：旧 PWSR 25199 结果，178809 个高斯。
- `output/city2_stage3_sh2/`：旧 SH2 40000 结果；历史实际运行 10000 次（30000 → 40000），不要误标为新默认 5000 次。
- 旧蒸馏 checkpoint 曾从 PLY 重建，不能据此声称完整保留了当时优化器状态。
- `output/city_pipeline_report.md`：旧实验报告，不作为新配置训练结果。

旧目录未覆盖。新代码的迭代保存顺序、checkpoint 交接和默认蒸馏步数已明确化，
不能保证重新训练与旧数值完全一致。历史整合阶段完成了短程 GPU 链路验证和 13 项单元测试，
当时没有重跑全量实验；这段记录不代表当前版本进度。

2026-10-27 保存的 v2.0.0 已完成实例保护版与同预算全局 SPS 对照的完整 City 实验，
两版均为 PWSR 10199 次更新和原始教师 SH2 蒸馏 5000 次更新，65 项单元测试通过。
实际结果、质量代价、本地模型位置和复现入口见 `PROGRESS.md`。

删除的 DoG 代码 / 修改前 Python 源文件保存在
`output/integration_backups/2026-09-18/pwsr_python_before_merge.tar.gz`；
旧编译产物在 `output/integration_backups/rasterizer_build_EjcboL/`。
上游已跟踪的 DoG CUDA 源码还可以通过 Git 历史找回。许可证与标准 SPS 后端保留。
