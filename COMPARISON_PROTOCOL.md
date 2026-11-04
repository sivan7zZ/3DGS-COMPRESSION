# City 统一对比与模块消融协议

本实验按用户要求，把已存在的 LightGaussian、PUP-3DGS、Speedy-Splat、PWSR 的10×/20×结果与本次方法、原始3DGS、真实RGB及人工语义标签进行比较。原实验不覆盖。所有新输出在 `output/city2_comparison_20261104/`。

## 视图、模型与来源

视觉测试为 LLFF hold=8 的9视图：000、008、016、024、032、040、048、056、064；分辨率1592×894。所有模型统一使用本仓库标准3DGS渲染器，关闭抗锯齿，黑背景、SH3。原始3DGS为1788093高斯。主结果采用SH蒸馏前模型；既有SH2保留，不混入本次主表。所有GT RGB经过逐像素哈希和相机内外参核验。

已找到全部基线，具体PLY/checkpoint/来源SHA和历史渲染目录见 `inventory/baseline_inventory.json`。10×目标178809点，LightGaussian阈值剪枝实际178810点；20×均89405点。实际点数与倍率单独列出。

PWSR一栏指本项目不含3D-DoG的RIR/DPRA/SPS全局剪枝恢复对照。

Speedy-Splat使用其已有COLMAP稀疏点训练模型，在默认iter20000的292912点模型处续训到目标预算；其余主要基线以原始city2模型压缩。统一渲染解决评价渲染器差异，但不会消除训练初始化与优化步数差异。本表是这些现存结果的比较，不能解释为全部方法同初始化、同计算量的严格消融。

## City 标签与冻结分割器

City仅有10张人工标注：000、007、014、021、028、035、042、049、056、063。与3DGS测试集重合的只有000、056。视觉指标用9帧；语义、边界、区域召回主指标只用这2帧，逐类与逐帧数据同时输出。不能把未标注帧伪装成GT，也不能把训练视图指标当作测试性能。

类别0 unannotated忽略；1 Building，2 Tree，3 Road，4 Vehicle，5 Grass，6 Ground。原标签在1593×896畸变影像上，不能直接resize。按实际city COLMAP的OPENCV→PINHOLE相机映射，用最近邻把六类binary mask配准至1592×894；不确定或越界区域为0。原mask无类别重叠；类别union与标注mask精确一致。配准参数、原文件与输出SHA见 `semantic/dataset_manifest.json`。

仅训练一个场景专用的ResNet34 U-Net下游评估器，骨干初始化为本机ImageNet权重。内层6帧原RGB训练(007/014/028/035/049/063)，2帧验证(021/042)，最多1500步，以验证mIoU选步数。再从同一ImageNet初始化与随机种子，在全部8张非测试标注帧上训练所选步数，冻结最终模型。两个测试标签、所有压缩图像均不参与训练或选择权重。模型使用原生像素crop与增强；推理512方块、步长384、重叠softmax平均，整图不缩放。

所有原始RGB、原始3DGS和压缩结果均使用同一个权重SHA和同一推理设置。GT行的语义值是“分割器对真实RGB的预测”与人工标签比较，并非强制100%，也不是理论上界。此实验衡量压缩对下游2D语义任务的影响，不给原生PLY加入语义字段，也不宣称完成了逐高斯3D语义分割。评估器自身误差、仅两张测试标注的样本量需与结果一起解释。

## 指标定义

- PSNR↑、SSIM↑、LPIPS↓：9张保存的原生PNG，与对应真实RGB比较，按视图平均。PSNR/SSIM沿用本仓库 `metrics.py`。主表LPIPS为VGG v0.1，使用 `evaluate_comparison_lpips.py` 将uint8 RGB先除255再转换到[-1,1]，原生整图推理、无缩放；这与[LPIPS作者实现](https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py)的输入范围一致。本仓库旧脚本直接输入[0,1]，所有旧results.json/per_view.json保留原样，另在报告附表与 `reports/lpips_protocol_comparison.csv` 追溯。所有方法及消融统一重算，不能跨协议混用；不改变模型、PSNR、SSIM、分割权重或语义指标。每模型 `comparison_lpips.json` 记录各帧值、PNG/代码/权重SHA及真实GT自比较为0的控制检查，根 `lpips_standard_manifest.json` 汇总。GT RGB视觉自比较在表中以“—”表示，不用无穷大参加排名。
- mIoU↑：累计两帧混淆矩阵后，六个非void类别对正union类别求均值。GT不存在但有预测的类仍因假阳性参与IoU。预测0在有效GT上仍计为漏检。
- mAcc↑：累计混淆矩阵，按具有GT支持的类别平均 `TP/(TP+FN)`。它已经是类别平均像素Recall。
- B-mIoU↑：按类别使用内边界 `mask - erode(mask,d)`，`d=max(1,round(0.02*图像对角线))`，原生尺寸为37像素。3×3结构元重复侵蚀，图像外填0；边界交并限制在远离GT void至少d像素的有效域，避免虚假的未标注区边界。有效域侵蚀时图像外视为有效，保留正常图像边缘。累计类别边界交/并后平均正union类别。方法概念参考[Boundary IoU作者仓库](https://github.com/bowenc0221/boundary-iou-api)，此处明确列出语义类别及void适配。
- Recall↑：主列为 **语义区域Recall@IoU≥0.5**。把GT同类别8连通区域作为代理目标，忽略面积<16像素的小碎片，与预测同类别连通区域做最大基数的一对一匹配。`匹配GT区域数/有效GT区域总数`。另列面积不超过全图1%的small-region Recall与Vehicle区域Recall，并保留TP/GT分母。它不是人工实例检测Recall，也不等于三维物体召回率；同类接触物体可能合并成一个GT区域。所有阈值在查看测试预测前固定。

## 20×消融

复用完整方法20× SH3，并执行7个leave-one-module-out：no_instance_survival、no_boundary_quota、no_topology_guard、no_adaptive_schedule、no_merging、no_budget_reallocation、no_structure_loss。每项最终89405点、相同9视觉测试视图、相同2语义测试视图、同冻结分割器，禁用SH蒸馏。只改对应开关与输出目录；原v3恢复损失共用。

no_boundary仅关闭边界配额，合并的边界限制、风险图与恢复损失仍共用；no_topology还取消合并时的PCA细结构排除，按实际实现的联合模块解释；no_structure_loss仅关闭新增风险加权L1，不关闭共同的v3 ROI/边界/细线损失。no_adaptive改为固定20%相对剪枝与非末轮200步恢复，末轮1500步；各项的实际轮次与更新数另列，因此不是等计算量归因，也不把单次运行差值当作统计显著性。

复现入口：`run_comparison_renders.py`、`city_semantic.py`、`run_comparison_ablations.py`，总队列 `run_city_benchmark_queue.py --resume`。模型和标签源有SHA保护；训练/渲染GPU任务全部串行。评估表、逐类指标、固定裁剪、语义图与消融delta由 `finalize_city_benchmark.py` 生成。
