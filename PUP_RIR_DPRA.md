# PUP评分 + RIR/DPRA（保留v3.1保护模块）

本版本按2026-11-05要求，只替换评分和剪枝调度。实例存活、边界配额、拓扑保护、矩匹配合并、预算回填、原恢复损失和训练专用元数据均沿用v3.1.0；历史代码、标签及结果保留。

## 评分

PUP作者原始位置xyz与log-scale六维Fisher块，累积61个训练相机、原生1592×894像素，评分为 `sum(log(svdvals(F)))`，从低到高删除；不加入SPS频域权重、ROI/结构分数，不用trace/梯度平方标量替代，不添加对角阻尼。奇异矩阵原样产生−inf，按当前行序稳定打破并列；可行性约束仅跳过当前不可删除点，不改变评分。

官方Fisher CUDA源码固定为 `eaef2faba21f076867c1db84c3f998e5f673762f`，放在本仓库 `submodules/pup-fisher/`，由本仓库自行编译，不依赖外部checkout动态库。来源：[PUP](https://github.com/j-alex-hanson/gaussian-splatting-pup)、[Fisher后端](https://github.com/j-alex-hanson/rasterization_and_pup_fisher/tree/eaef2faba21f076867c1db84c3f998e5f673762f)。端口来源和文件SHA在 `PORT_PROVENANCE.json`。

## 调度

沿用 `utils/schedule_utils.py` 的原DPRA，虚拟15001–25199，共10199次更新。15200初始化预算，之后每500次以全部训练图的L1+共同ROI项检查恢复；`current_loss < 0.99 * pre_prune_loss_stamp`时剪枝，最长等待1500次强制触发。全局预算按原来的1/2、1/4、1/8、1/16及最终剩余量分5次执行；到达预算后继续恢复至25199，不再进行零预算的模型变更。

关闭相对自适应轮次，不使用其恢复率、滑动窗口、动态比例或相对恢复步数。每次PUP排序之后执行同一套保护/合并/回填事务，精确达到该次DPRA目标数。合并后需要补剪时重新计算PUP Fisher，不能给新高斯臆造评分。

## 运行与输出

```bash
bash scripts/run.sh run_pup_rir_experiments.py
# 中断后续跑同一配置，保留已完成步骤：
bash scripts/run.sh run_pup_rir_experiments.py --resume
```

配置为 `configs/city_pup_rir_dpra_10x.json`、`configs/city_pup_rir_dpra_20x.json`，目标分别178809/89405点；输入仍为同一个1,788,093点原始city2 PLY。输出在 `output/city2_pup_rir_dpra_20261105/full_{10x,20x}/{stage3,sh2}`，两个倍率串行运行。SH3为主分析；原始未剪枝教师监督的SH2蒸馏5000次，另存结果。

每次原始PUP评分及行ID、保护配额、RIR判断、合并/回填事件、checkpoint、标准PLY、元数据、续训状态与原生测试图都会保存。最终回执核对RIR/DPRA与原逻辑、预算、原生字段和教师完整性；LPIPS标准[-1,1]值另存，不覆盖旧[0,1]评测。

此次评分与调度同时改变，恢复更新数也不同于v3.1.0，因此结果差异不能单独归因于PUP。保护配额和候选图约束不等于已证明实际地物被挽救。
