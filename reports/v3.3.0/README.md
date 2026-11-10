# v3.3.0 轻量实验记录

保存日期：2026-11-10。实验日期为2026-11-09，保存不重跑训练。

[版本说明](../../RELEASE_NOTES_v3.3.0.md) · [完整七指标CSV](comparison_all_metrics.csv) · [验证摘要](verification_summary.json) · [源/快照指纹](artifact_manifest.json)。

本目录保存连续版10×/20×的指标、完整核验回执、五折评测协议、先验统计、源码清单与固定裁剪；binary_boundary_reference/保存二值版轻量参考。JSON中的绝对路径用于定位本机原始产物，不表示大型模型随Git上传。

主语义比较2张独立测试与辅助8张分开，CSV另含全部10张混合汇总。比较含LightGaussian、PUP-3DGS、Speedy-Splat、PWSR、v3/v3.1/v3.2、二值边界版、当前连续版、原始3DGS及GT。

本机完整输出：`/home/zhangys/Prune-Wisely-Reconstruct-Sharply/output/city2_continuous_boundary_20261109/`。模型路径与SHA见verification_summary.json的model_paths；标准PLY和checkpoint位于full_{10x,20x}/stage3。

实验原始清单按产生时状态精确复制。保存版本后的Git状态由v3.3.0标签确定，不把历史source_changes_committed标记改成当前状态。
