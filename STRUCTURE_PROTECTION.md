# 无语义标注的实例与可见结构保护压缩

当前主配置为 `configs/city_structure20_adaptive.json`。它在原有 **PWSR RIR / DPRA / SPS → LightGaussian SH 蒸馏** 流程中加入小区域、边界和细长结构保护，不含 3D-DoG。

本轮先用 **SH 蒸馏前的标准 SH3 PLY** 判断结构是否保留。原始 City 模型有 1,788,093 个高斯；20 倍数量压缩预算为 **89,405 个高斯**，即 `round(1788093 / 20)`。所有同预算 20 倍分支均使用这一数量。标准 SH3 每点字段相同，因此其 PLY 文件大小也约缩小 20 倍；带实例附加字段的 PLY 和 checkpoint 大小应单独报告。

adaptive 主实验、两阶段导出及续训状态核验已完成。本文说明实现、输入约束、复现方式和评估协议；最终质量数值与目视结论以本次实验的最终报告为准，不能把可行性检查或源模型覆盖率写成最终效果。

## 要解决的三个问题

| 问题 | 单纯逐点全局排序的不足 | 本次增加的约束与恢复监督 |
|---|---|---|
| 小目标变得不可辨认或消失 | 屏幕占比小的区域对全图总分贡献低，重要支撑可能集中被剪掉 | 训练视角 SAM 跨视图伪实例、少量实例配额、小区域可见单元覆盖、区域恢复损失 |
| 边界变弱、断裂或相邻面混合 | 保留区域内的高斯数量不代表保留轮廓所需的高斯 | 照片与原始教师共同支持的边界单元、边界方向与相对对比度监督 |
| 细长结构缺失 | 很少的支撑点就可能决定一段窗框、围栏或细枝是否存在 | SAM 窄分支与 RGB 细线响应、较细屏幕网格、细线对比和局部弱连接损失 |

这些机制保护自动识别出的图像证据。SAM 区域可能是物体的一部分或背景，结构响应也可能包括纹理、阴影和屋顶棱线；它们不等于人工真实物体标签或完整物理拓扑。

## 输入与训练、测试隔离

实际项目位置固定为 `/home/zhangys/Prune-Wisely-Reconstruct-Sharply`。主分支使用以下输入：

- 影像、COLMAP：`/home/zhangys/gaussian-splatting（原生）/data/city`。
- 原始未剪枝 SH3 PLY：`/home/zhangys/gaussian-splatting（原生）/output/city2/point_cloud/iteration_30000/point_cloud.ply`。
- 原始 SH3 教师 checkpoint：`/home/zhangys/gaussian-splatting（原生）/output/city2/chkpnt30000.pth`。
- 自动伪实例：`output/city2_instance_guard_20261026/inputs/train_labels/instance_priors.npz`，由训练视角 RGB、SAM 掩膜、COLMAP 和原始高斯模型关联得到。
- 训练结构先验及 ROI：`output/city2_structure_guard_20261027/inputs/structure/`。

按排序后的帧名每 8 帧留出 1 帧，得到 **61 个训练视角、9 个测试视角**。伪实例关联、实例面积统计、可见贡献、结构网格和训练 ROI 都只使用 61 个训练视角。代码检查实例图和训练帧列表一致，拒绝混入测试视角。

**不读取人工 2D 语义标签，不读取 `gt_semantic_3dgs.ply`，也不使用 Building / Tree / Vehicle 等语义类别。** SAM 只提供自动区域划分。测试视角 SAM 和固定人工检查框仅供最终评估，不能反过来选择训练锚点、配额或损失位置。

## 从可见图像证据建立结构单元

`prepare_structure_protection.py` 首先在原始模型上计算原生分辨率可见性。它使用高斯的各向异性投影、深度顺序和 alpha 合成，记录每个像素的主导源高斯及其 `alpha × T` 贡献。这里的 `T` 是前方高斯透射率。仅把点中心投到某个掩膜里不足以证明可见，因此不以“投影中心落入区域”作为覆盖证据。

当前声明的像素有效条件为：主导贡献至少 `0.035`，合成 alpha 至少 `0.5`。小区域还要求主导高斯的伪实例 ID 与该像素对应区域一致。每个关键单元保存最多 8 个源高斯候选及其主导可见贡献；这是原始模型的候选支持记录，不是所有高斯贡献的完整枚举。

三类证据分别构建：

1. **小区域**：在训练视角投影较小、且满足多视图直接关联和置信度条件的伪实例。本组共有 427 个候选小区域。
2. **边界**：SAM 区域交界与 RGB 梯度相结合，并要求原始教师在邻域中提供相同方向的图像支持。这样不会把每条 SAM 人工分割缝都强制当成真实边缘。
3. **细长结构**：SAM 区域中经形态学 opening 去除宽面后的窄分支，经过细化和长度筛选；再补充 RGB Hessian 多尺度细线响应，以覆盖 SAM 可能漏掉的窗框、线状构件等。照片与教师共同支持的邻域约束用于抑制不可靠响应。

小区域与边界采用 24 像素屏幕单元，细线采用 16 像素单元，每单元至少包含 2 个有效关键像素。共享屏幕位置的不同小区域仍保留各自单元。当前训练先验的实际计数为：

| 类型 | 单元数量 |
|---|---:|
| 小区域单元 | 48,900 |
| 边界段单元 | 162,639 |
| 细线段单元 | 85,371 |
| 合计 | **296,910** |

计数来自 `inputs/structure/structure_manifest.json`。它们是训练图像上的可见支持单元，不能表述为 296,910 个地物或真实物体。

## 预算内的可行覆盖证书

`utils/structure_pruning.py` 用确定性的贪心集合覆盖，在这些候选中找出一组能覆盖全部声明单元的源高斯。本组初始证书含 **27,488 个高斯**，未超过显式的证书预算 31,291（最终预算的 35%）。

实例配额与结构证书先做并集成本检查。当前可行性记录为：

| 项目 | 数量 |
|---|---:|
| 逐实例最低配额合计 | 19,170 |
| 初始结构证书高斯 | 27,488 |
| 证书与配额并集的最低成本 | **33,959** |
| 最终预算 | **89,405** |
| 该可行证书方案留下的 SPS 余量 | 55,446 |

19,170 与 27,488 有重叠，不能直接相加。33,959 是一个已验证可行方案的并集成本；它不是 adaptive 最终必须冻结的点数。若约束超过预算，代码在昂贵训练开始前报错，不会悄悄扩大最终模型或删除约束单元。

“每个源可见单元至少保留一个候选”也不等于保留了全部源贡献质量。一个高斯可覆盖很多单元，一个单元也可能需要多个高斯共同重建。源主导贡献质量的保留比例在覆盖审计中另行记录，不能用单元覆盖 100% 代替质量保留 100%。

## 当前 adaptive 选择规则

初始固定证书是中间轮次的可行性后备。adaptive 最终选择由 `StructurePruningState.select_keep_adaptive_final()` 实现：

1. 中间剪枝轮次仍保留完整证书，保证后续存在已知可行的覆盖选择。
2. 达到声明的最终预算 `K=89,405` 时，先执行普通的实例配额约束 SPS，得到恰好 K 个点的候选集合；此时不强制保留全部 27,488 个旧证书点。
3. 检查这个集合还没有覆盖哪些**源可见单元**。只为未覆盖单元补入目前仍可用的源候选，优先补能覆盖更多缺口、且 SPS 分数较好的候选。
4. 补点之后，按 SPS 从低到高移除冗余点。不能移除某单元最后一个源可见候选，也不能破坏实例最低配额。
5. 最终同时检查：**精确 K、全部声明单元有候选、实例配额不违反、源点 lineage 对齐**。若局部贪心修复不能回到精确 K，使用已验证的固定证书选择作为后备，并在审计中记录原因；不默默放宽约束。

这是确定性的贪心修复，不是全局最优组合求解器。最终保留了多少原证书点、补入与移除多少点、是否调用后备，均写入剪枝事件 `instance_audits/iteration_*.json` 的 `adaptive_final` 字段。

实例配额配置为 `configs/city_structure_policy.json`：基础比例 `0.005`、小区域额外比例系数 `0.06`、最低 8、最高配额 48。实例内部排名附加权重、小区域全局分数加成均为 0，空间分格为 1；尽量让全局 SPS 决定正常区域，仅给可能灭失的区域留下少量支撑。

## 恢复阶段允许自由优化

adaptive 配置明确使用：

```json
{
  "structure_selection_mode": "adaptive",
  "structure_freeze_anchors": false,
  "structure_anchor_weight": 0
}
```

因此它不冻结证书点的坐标、缩放、旋转、opacity 或 SH 系数，也不施加固定源锚点的 trust-region 损失。选中的高斯继续随恢复训练优化。CLI 拒绝 adaptive 与非零锚点约束或冻结开关混用，以免配置名称与实际行为不一致。

除原有 L1 + DSSIM 外，当前配置加入适度的恢复监督：

| 配置项 | 当前权重 | 作用 |
|---|---:|---|
| `instance_roi_weight` | 0.02 | 小区域 RGB 重建 |
| `instance_edge_weight` | 0.01 | 区域边界梯度误差 |
| `instance_score_roi_weight` | 0.05 | 剪枝评分中的区域贡献 |
| `structure_boundary_weight` | 0.005 | 边界方向和相对对比度存活 |
| `structure_thin_weight` | 0.005 | 细线局部对比与弱连接 |
| `structure_score_weight` | 0.1 | SPS 评分中的结构权重 |

`boundary_survival_loss()` 比较当前与参考图像的有符号 RGB 差分，惩罚削弱、缺失或反向的边界，而不是仅鼓励任意高频锐化。`thin_survival_loss()` 比较灰度高通响应，在 16 像素局部单元内用 soft minimum 增强弱段的训练信号，并附带区域重建项。

这些都是图像恢复代理损失。局部 soft minimum 不等同于物理图结构连通性证明，二维 skeleton 也不能保证任意新视角的三维拓扑。若优化后高斯漂移、变透明、被其他点遮挡或外观改变，即使源单元覆盖审计通过，最终结构仍可能变弱。因此必须独立检查最终渲染。

## 固定证书初版作为消融保留

`configs/city_structure20.json` 对应 `output/city2_structure_guard_20261027/protected20/`。它强制保留初始证书，在 SH3 恢复中冻结这些源高斯，并使用更强的结构损失和锚点约束。这个初版在已完成实验中出现明显整体质量退化，保留用于消融和查因，不作为当前 adaptive 主结果。

**“27,488 个锚点的 SH3 全部 62 个标准 PLY 字段与原始源点逐位一致”只适用于 fixed 初版的属性审计。** adaptive 允许这些点重新优化，最终也可以换成其他可覆盖同一单元的源候选，因此不具有也不承诺该逐位一致性。即使 fixed 的源点属性不变，其他高斯的遮挡和合成顺序变化仍可改变它的最终可见贡献。

不得把 fixed 的属性一致性审计、源覆盖证书或其 SH2 结果移植成 adaptive 的质量结论。

## 复现与输出

在项目根目录运行，先检查实际计划：

```bash
cd /home/zhangys/Prune-Wisely-Reconstruct-Sharply
bash scripts/run.sh run_pipeline.py \
  --config configs/city_structure20_adaptive.json --dry_run
```

复现新实验时使用新的 output 子目录，保留本轮及历史结果：

```bash
bash scripts/run.sh run_pipeline.py \
  --config configs/city_structure20_adaptive.json \
  --output_root output/city2_structure_guard_reproduce_adaptive
```

默认本轮根目录为 `output/city2_structure_guard_20261027/protected20_adaptive/`。入口拒绝覆盖非空训练输出，已有实验续训应使用对应脚本的 checkpoint 与配套状态，而非重新覆盖目录。配置引用的是已生成的训练先验；重新生成先验时也必须保留相同训练 / 测试划分，并核对原始 PLY 的 SHA 和 XYZ lineage。

| 阶段 | 输出相对于该实验根目录的位置 |
|---|---|
| SH3 剪枝与恢复 | `stage3/point_cloud/iteration_25199/point_cloud.ply` |
| SH3 checkpoint | `stage3/chkpnt25199.pth` 及 `.instances.npz`、`.state.pth` |
| SH3 原生测试图 | `stage3/test/ours_25199/renders/` |
| SH2 蒸馏后模型 | `sh2/point_cloud/iteration_40000/point_cloud.ply` |
| SH2 checkpoint | `sh2/chkpnt40000.pth` 及配套 sidecar |
| SH2 原生测试图 | `sh2/test/ours_40000/renders/` |

`pre_distill_analysis=true` 使统一入口先完成 SH3 渲染，再进行 SH2 蒸馏。SH3 与 SH2 的 PLY、checkpoint 和渲染图分别保存，不互相覆盖。标准 PLY 用于统一渲染；`export_instance_ply.py` 可根据同目录对齐的实例 sidecar 另存带实例字段的 PLY，保留标准模型。

SH3 阶段沿用 PWSR 的虚拟 15000→25199，共 10,199 次实际更新；RIR 控制恢复检查，DPRA 控制剪枝比例，SPS 计算评分。阶段初始化不重新从 COLMAP 稀疏点训练 3DGS，也不运行 3D-DoG。

SH 蒸馏的教师始终是原始未剪枝 SH3 checkpoint；学生从本次 SH3 剪枝 checkpoint 初始化并截断到 SH2，创建新的 AdamW。虚拟 35000→40000 是 **5,000 次实际更新**。SH2 恢复已经开始的蒸馏时才读取其优化器及状态 sidecar，不能把 SH3 优化器当作 SH2 续训状态。蒸馏不改变高斯数量；训练 2D ROI 只用于原训练相机，扰动视角跳过不再对齐的 2D ROI 项。

## 对照与主分析协议

主分析必须使用 SH 蒸馏前 SH3 的相同测试相机与渲染器，保留以下分支：

| 分支 | 数量 / 倍率 | 用途 |
|---|---:|---|
| 原始模型 | 1,788,093 / 1× | 确认待评估结构本就在 3DGS 中可见 |
| 原生 LightGaussian 10× | 178,810 / 约 10× | 用户指定外部剪枝基线 |
| 原生 LightGaussian 20× | 89,405 / 约 20× | 更强压缩下的外部方法 |
| 原 PWSR 10× | 178,809 / 约 10× | 原流程的较宽松预算参考 |
| v2 实例配额版 SH3 | 178,809 / 约 10× | 上一版本参考，不是 20×同预算消融 |
| 原 PWSR 20× | 89,405 / 约 20× | 新模块必须面对的同预算对照 |
| fixed 结构保护 20× | 89,405 / 约 20× | 冻结证书与较强损失的初版消融 |
| adaptive 结构保护 20× | 89,405 / 约 20× | 当前主实现，完整实验与渲染核验已完成 |

用户给定的 LG10 位于 `/home/zhangys/LightGaussian/output/city2_native_ckpt_prune10x/01_pruned`。本轮保留它的来源，使用当前项目渲染器重新渲染到 `output/city2_structure_guard_20261027/reference/lightgaussian10/`。原生 LG 的贡献计数与体积重要性排序不冒充 PWSR SPS；其约 5,000 轮恢复与 PWSR 10,199 次更新也不是相同训练调度。外部方法比较与 PWSR 内部同预算消融应分别解释。

评估不以局部 PSNR 排名作为“目标获救”的证据：

- `evaluate_visible_structure.py` 检查源 RGB 与索引 GT 的逐像素映射、原生分辨率、图像及掩膜 SHA。只有照片与原始教师共同支持的结构进入参考分母。
- 边界存活以有方向的相对梯度衡量，模型不经过绝对 Canny 迟滞阈值，以免把轻微变弱但仍可见的边缘误报为整段消失。它还报告连续边缘幅度、缺失片段和额外伪边缘。
- 自动 SAM 细区域、无 SAM 线段及屏幕网格提供候选；图像边界距离与内部区域统计单独记录。自动数值不等于真实物体召回率。
- `make_reviewed_structure_evidence.py` 导出事先固定框的 GT、教师和所有方法 native 裁剪。保存的 crop 不缩放、不锐化、不填补；只有总览用 nearest-neighbor 放大。同时保存 source SHA 与坐标，并保留退化和负对照。
- 整图 PSNR / SSIM / LPIPS 仍报告为质量代价，不把整图质量损失藏在局部改善中。训练内 float 指标与 PNG 离线指标分开记录。

## 已审查的六个固定例子及结论边界

固定框文件为 `inputs/reviewed_examples_20x.json`，证据图在同目录 `reviewed_six_cases/`。以下六框均未被图像边界截断，且在当前新模型检查前已固定：

| 帧与原生 bbox `[x0,y0,x1,y1]` | 检查内容 | LG20 已观察到的现象 |
|---|---|---|
| `frame_024` `[1321,30,1380,69]` | 黑色停泊车辆 | 车顶、车窗和车身变成不可辨认的深色块，仍有弥散残留 |
| `frame_064` `[1350,821,1373,844]` | 小型白色庭院物体，精确语义类别不确定 | 15×15 像素的明亮顶面和轮廓近乎消失成灰色残影 |
| `frame_056` `[1352,70,1382,101]` | 屋顶小单元的 L 形黑色轮廓 | 右侧轮廓段缺失或显著变弱，浅色主体仍在 |
| `frame_008` `[1183,77,1232,116]` | 白色附属温室的细窗框 | 多条独立窄杆合并成光滑白面，建筑本身仍在 |
| `frame_000` `[157,31,247,70]` | 浅色庭院围栏分隔 | 面板边界与邻近颜色混合，不能声称整条围栏消失 |
| `frame_024` `[83,192,111,225]` | 红色停泊车辆负对照 | 各方法仍可见，不能列为消失物体 |

**原 PWSR20 已保留这些 LG20 退化案例。** 它们证明外部压缩方法可破坏可辨目标外观、边界和细结构，但尚不能证明新增 adaptive 模块独有地挽救了目标。新模块的实际增益必须在相同固定框及全量评估中相对 PWSR20 判断；无改善或出现退化时应如实记录。

目前没有人工完整实例真值，不能据此报告真实物体召回率，也不能把弥散残影、几何混合或纹理变弱统一写成“物体的全部高斯都被删除”。最终可写出的结论应对应已核实的可见现象，并附 SH3 native 证据。

## 相关实现与审计文件

- `prepare_structure_protection.py`：训练结构 ROI、可见候选单元及初始覆盖证书。
- `utils/structure_visibility.py`：各向异性高斯的原生 CPU 可见贡献。
- `utils/structure_pruning.py`：固定证书、adaptive 最终修复和精确预算审计。
- `utils/structure_losses.py`：边界、细线及 fixed 消融使用的锚点恢复约束。
- `train_prune.py` / `distill_train.py` / `run_pipeline.py`：剪枝、恢复与 checkpoint 蒸馏交接。
- `audit_structure_source_coverage.py`：源模型主导贡献与 lineage 支持审计；输出不是最终可见性真值。
- `evaluate_visible_structure.py` / `make_reviewed_structure_evidence.py`：独立的原生 SH3 图像评估与固定框证据。
- `inputs/structure/{structure_manifest.json,feasibility.json}`、各轮 `instance_audits/` 与各模型目录 `structure_coverage.json`：具体来源和不变量记录。

静态与单元检查入口：

```bash
bash scripts/run.sh -m unittest discover -s tests -v
```

最终报告已记录 SH3/SH2 文件、覆盖审计和逐例目视结论。97 项测试通过，两阶段已完成 checkpoint 的零更新续训校验通过；整图质量代价另见本轮主索引。

## 本轮完成结果

完整 adaptive 实验已完成：SH3 10,199 次更新，SH2 原始教师蒸馏 5,000 次更新；两阶段各 89,405 个高斯。主分析始终使用 SH3。详细结果位于 `output/city2_structure_guard_20261027/protected20_adaptive/README_结构保护结果.md`，五列原生图位于同根 `reports/primary_adaptive/compact_visual_review/reviewed_native_comparison.png`。最终一轮相对实例SPS交换 8,677 对高斯，补齐 23,549 个初始未覆盖单元，无固定证书回退；296,910 个源候选单元全部有代表，实例配额无违规。这是源可见支撑保证，不是最终物体召回率或拓扑保证。

六例目视复核表明：相对LG20保留小物体可辨形状、屋顶轮廓与细窗框；相对原PWSR20整体相当，未确认新增模块独有的目标挽救。全量候选中边界召回略升，小区域对比与细区域中心线略降。原PWSR20同样没有自动区域点数归零；不能宣称三类问题已被全面解决。所有70视图SAM缓存与预览保存在本轮 `inputs/sam_masks/`，其 `split_and_origin.json` 记录61训练/9测试用途。
