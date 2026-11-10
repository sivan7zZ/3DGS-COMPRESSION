# 边界加权PUP压缩（无SH蒸馏）

本轮以v3.2.0为基础，只生成SH3压缩模型。实例最低配额、拓扑/细结构安全检查、RIR/DPRA及恢复损失沿用；边界由硬生存配额改成评分权重，同时移除独立的边界合并禁令。历史版本及输出保留。

## 评分和保护

对当前模型重新计算原始PUP六维Fisher评分，然后使用：

```text
score_i = sum(log(svdvals(F_i))) + 2 * boundary_mask_i
```

剪枝仍按分数从低到高、全局稳定排序。对有限边界分数加2，等价于行列式重要性乘exp(2)，避免把负logdet直接乘大权重后反而更易被删。奇异Fisher的负无穷保持，不能凭边界标记创造原本为零的Fisher证据。边界来自既有训练视角SAM/深度/RGB先验，不使用人工语义标签。

新模式不设边界最低数量，也不在合并时单独否决边界点。原实例配额与拓扑/细结构候选保护不变；这些保护若覆盖某个边界点，仍可能阻止其删除或合并。边界恢复监督保持，合并后的边界元数据用布尔OR传播，后续按合并后的当前模型重新计算Fisher再加权。

每次评分保存raw_pup_scores、scores、boundary_mask、row_ids；合并后补剪枝也单独保存评分。验证器逐值重算权重，检查奇异值约定、行顺序、空边界配额、原调度、实例配额与精确预算。

## 合并与回填

- 合并上限从每轮进入高斯数的0.5%提高到1%；不放宽几何距离、尺度、SH相似性、可靠实例及拓扑条件。上限不等于实际合并量。
- 回填预算从每轮目标数的1%提高到2%。先为这些槽位额外剪枝，再按训练视图困难区域恢复分数Q从删除池回填；合并生成的新高斯不能作为原始删除点回填。
- 每轮最终数量仍精确等于DPRA目标，10×为178809、20×为89405。两组各10199次更新，不运行SH蒸馏。

## 运行

```bash
bash scripts/run.sh run_boundary_weighted_pipeline.py --config configs/city_boundary_weighted_10x.json --dry_run
bash scripts/run.sh run_boundary_weighted_experiments.py
# 同一冻结源码与配置中断续跑：
bash scripts/run.sh run_boundary_weighted_experiments.py --resume
```

输出根为 `output/city2_boundary_weighted_20261106/`。两个倍率串行运行，`full_{10x,20x}/stage3/` 保存标准PLY、checkpoint、元数据和原生渲染。新入口不含教师checkpoint、SH2阶段或蒸馏参数，历史蒸馏文件仅为旧版本复现保留。

## 标签和比较

按更正后的路径 `/home/zhangys/semantic_data/3D-AS_dataset/City/scene0/segmentation`，共有10张匹配当前模型的人工标签。标签须使用模型自身data/city中的COLMAP相机注册到1592×894，不能用另一套重建替代。

PSNR/SSIM/标准VGG0.1 LPIPS使用固定9个独立测试视角。Recall为同类别语义连通区域IoU≥0.5一对一匹配召回；B-mIoU、mIoU、mAcc沿用统一语义统计定义。City类别0忽略、1建筑、2树、3道路、4车辆、5草地、6地面，不把区域召回当成物理实例召回。

全部10张标签分三组报告：2张3DGS独立测试视角为主表，8张3DGS训练视角为辅助表，另列10张汇总。主表用历史冻结评估器；其余8张分4折，每折仅用其他6张非测试原RGB标签拟合评估器，固定1500步，不用被评测帧选步。两张独立测试标签始终不参与评估器训练；压缩训练也不读取这些人工标签。

对比新10×/20×、v3.2.0、v3.1.0、既有LightGaussian/PUP-3DGS/Speedy-Splat/PWSR、v3参考、原始3DGS及GT RGB，全部使用SH蒸馏前模型及同一相机。LightGaussian10×多1个高斯，Speedy初始化不同。新实验同时改变边界处理和合并/回填比例，结果不能用于单独归因某一模块。

准备与评测入口如下；完整比较会等待压缩队列释放GPU，然后串行执行辅助渲染、固定步数分折拟合、推理与统计：

```bash
/home/zhangys/anaconda3/envs/VGGT/bin/python evaluate_boundary_weighted_semantics.py --phase prepare --protocol city_oof
bash scripts/run.sh run_boundary_weighted_evaluation.py --wait --resume
```

评测启动器对VGGT工作进程清除继承的动态库注入，防止两套Torch/cuDNN混用；原生渲染工作进程保留自身环境。失败后仅从未完成的评测步骤继续，压缩和已完成渲染不重跑。标签和配置不变时不要重复准备阶段，以保留既有协议文件。
