# 来源与适配范围

- 基础仓库：WangHaoran16/Prune-Wisely-Reconstruct-Sharply。
- SH 蒸馏参考：VITA-Group/LightGaussian，本地来源 revision `6676b983e77baadd909effc56a6aaadafa964dcc`。
  适配其 `distill_train.py`、`utils/pose_utils.py` 中的教师监督、SH 降阶和视点扰动逻辑。
- 本地入口 `distill_train.py` 和 `utils/distill_utils.py` 使用 PWSR 的标准 GaussianModel 与 SPS 渲染后端，
  新增 checkpoint 校验、原子保存、阶段编号、状态续训和来源记录；不是上游 LightGaussian 的逐字复制。
- 不包括 LightGaussian 的额外剪枝、VecTree / 码本量化流程，不包含 PWSR 的 3D-DoG。
- 根目录 LICENSE.md、源码中的 Inria / GRAPHDECO 版权声明及各 submodule 的许可证原样保留；
  本整合没有授予额外使用许可。
