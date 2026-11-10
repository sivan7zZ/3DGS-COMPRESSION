# SAM、深度和RGB连续边界评分

2026-11-09：按用户要求替换20261106版本的`2*boundary_mask`。本版仍只输出标准SH3，不运行SH蒸馏或3D-DoG。历史二值边界实验与v3.2.0均保留。

## 定义

仅使用61个3DGS训练视角，逐视角构造三张原生分辨率连续图：

- SAM：相邻SAM区域的轮廓到像素的欧氏距离为d，强度为`exp(-d²/(2*2²))`，没有轮廓时全0；不先膨胀、不二值阈值化。
- 深度：原始模型渲染深度的Sobel梯度幅值；仅使用有限深度且alpha>0.5的有效支撑，腐蚀有效区排除无效值边缘，以有效梯度95%分位归一化并截断到[0,1]。
- RGB：灰度Sobel梯度幅值，以95%分位归一化并截断到[0,1]。

按原始高斯的各向异性投影足迹、前向透射率T与不透明度alpha计算每个高斯的可见贡献，三通道分别使用`sum(alpha*T*edge)/sum(alpha*T)`，跨所有训练视角汇总。既不使用高斯中心单点采样，也不只取最前方高斯。不可见高斯三项为0。

固定`B=0.4*SAM+0.4*depth+0.2*RGB`，`W=1+5*B`，最终评分为：

`score = PUP_logdet + log(W)`

三项与B均为[0,1]连续浮点数，det重要性乘数W位于[1,6]，奇异Fisher的负无穷保持。SAM和深度共同强调区域轮廓及几何不连续，较低RGB权重减少纯纹理边缘影响。这是实验前固定的启发式权重，未使用人工语义标签调参，也不宣称最优。

旧`boundary_mask`只为原拓扑构造和兼容恢复监督保留，不用于本版评分，也没有独立边界剪枝配额或边界合并否决。实例配额、拓扑保护、RIR/DPRA、恢复损失保持20261106设置；每轮合并上限为输入高斯1%，回填为目标2%，最终数量精确固定。

## 行对齐与审计

三个连续字段`sam_boundary_strength`、`depth_boundary_strength`、`rgb_boundary_strength`随高斯剪除、池保存、回填、续训同步移动。合并使用与原矩匹配相同的可见质量权重，对三字段计算归一化凸平均，保证仍在[0,1]；原高斯参数和其他元数据算法不变。lineage记录合并权重，独立核验器逐行重算。

预处理输出记录源PLY、坐标、共享元数据、SAM、深度、RGB、相机、训练/测试划分、实现与运行时SHA。每帧均与原可见贡献核验；小图另有独立逐像素参考。两倍率冻结同一份连续先验及清单。每轮原PUP、三通道、融合B、乘数、最终评分和row_id均保存；合并后重新计算PUP时也使用相同连续权重。

## 运行

在仓库根目录先准备连续先验（使用历史预处理的NumPy运行时以保持投影数值一致，不需要GPU）：

```bash
/home/zhangys/anaconda3/envs/prune_wisely/bin/python prepare_continuous_boundary.py
bash scripts/run.sh run_continuous_boundary_experiments.py
bash scripts/run.sh run_continuous_boundary_evaluation.py --wait
```

中断后两个运行器均可加`--resume`。已有但内容或配置不同的实验不会被覆盖。首次联调使用`configs/city_continuous_boundary_smoke.json`与独立`smoke/`目录，不替代全量结果。

全部输出为`output/city2_continuous_boundary_20261109/`，两倍率模型分别位于`full_{10x,20x}/stage3/`。最终PLY是标准SH3；连续信息另存sidecar，不给标准渲染器增加字段依赖。

## 公平比较

PSNR、SSIM和标准VGG0.1 LPIPS[-1,1]来自9张固定独立测试视角。使用City/scene0/segmentation的全部10张匹配标签，分2张独立测试主结果、8张训练视角辅助、10张混合汇总报告；Recall为同类别8连通语义区域IoU≥0.5的一对一召回，最小16像素，不能代称人工物理实例召回。B-mIoU、mIoU、mAcc沿用既有定义，0类忽略。

本轮原封沿用20261106的五个冻结语义评估器，不重新拟合或按新结果选择权重；每个辅助折排除其2张预测帧，并排除两张独立测试帧。只有checkpoint与RGB内容SHA均相同才复用旧预测，旧预测清单另外由上轮已完成指标中的SHA锚定。新模型与全部旧方法、原始3DGS、GT共享对应折的权重。GT RGB也经过同一分割器，不将其指标设为100%。

固定六处原生裁剪沿用上轮坐标，保留二值边界版、连续边界版、PWSR、LightGaussian对照。不能仅凭整图或语义汇总指标宣称电线、井盖或整物体已经被挽救；现有标签没有这两类的独立标注。
