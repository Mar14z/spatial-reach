# 论文库：按推进阶段索引

> 组织原则：**按「工程推进到哪一步」索引**，不按年份、不按影响力。每一步先读能直接改代码的那篇。
> 最后更新：2026-10-25。新增条目请追加到对应阶段，并在文末「待核验」记录疑点。
>
> 【核实深度】标记：
> `深读` = 读过摘要 + 正文片段；`摘要` = 只读了摘要；`元数据` = 只核实标题/年份/venue/链接存在。
> 标 `子代理` 的条目由并行检索代理产出，条目元数据已核对，但未逐篇精读全文。

---

## 0. 先看这张索引表

| 阶段 | 要解决的病 | 首选论文 | 落到哪个文件 |
|---|---|---|---|
| **S0 评测与不确定性** | 不知道「什么时候敢信这个位姿」 | SLUE、Geodesic-CP、Keypoint Self-Consistency | `scripts/ablation_report.py`、`touch/perception/pose.py` |
| **S1 几何确定性** | 平面歧义、外点、样本太少 | IPPE、TEASER++ | `touch/perception/pose.py` |
| **S2 学习范式** | 真机没标注、共面退化 | CtRNet、DRHeC | `scripts/train_keypoints.py`、新增可微渲染模块 |
| **S3 免人工标定** | 每次开机都要人摆位 | ARC-Calib、Hydra、Calib3R | 新增标定会话脚本 |
| **S4 多视角与 3D 基础模型** | 单帧不可观 | VGGT、DUSt3R | 离线标定器（PC 端） |
| **S5 物体侧** | 射线∩桌面这个弱假设 | FoundationPose | M2 空间认知 |
| **背景** | 为了能说清「为什么不做」 | NARF/GS-HEC/主动标定 | 只写进 limitations |

---

## 1. S0 · 评测与不确定性（决定「敢不敢信」）

| 文献 | 年 / venue | 链接 | 优点（为什么读） | 风险 / 局限 | 核实深度 |
|---|---|---|---|---|---|
| SLUE: Uncertainty Quantification for Visual Object Pose Estimation | 2025 / T-RO 2026 | https://arxiv.org/abs/2511.21666 | 只要 2D 关键点的**高概率像素噪声界**，不假设分布；用 S-lemma 求最小体积椭球，可拆成独立的平移界与旋转界 | 需要先给出像素噪声界；凸松弛不是精确解 | 摘要（子代理） |
| Deterministic Object Pose Confidence Region Estimation | 2025 / ICCV | https://arxiv.org/abs/2506.22720 | conformal 标定 + **隐函数定理**直接把关键点置信域传播成 6D 位姿置信域，免采样免 ensemble；报告旋转体积降 99.9%、平移降 99.8% | conformal 只给边际覆盖 | 摘要（子代理） |
| Adaptive Geodesic Conformal Prediction for Egocentric Camera Pose | 2026 / arXiv | https://arxiv.org/abs/2605.00233 | **点破了标准 conformal 的致命伤**：边际 90% 覆盖，最难 25% 帧只有 ~60%（约 30 个点差距）。改用 SE(3) 测地线非一致性分数 + 难度自适应，把最难帧从 ~0.75 提到 ~0.93 | 需要难度估计器；跨场景迁移未验证 | 摘要（子代理） |
| Detecting Pose Estimation Failures via Keypoint Self-Consistency | 2026 / arXiv | https://arxiv.org/abs/2608.03516 | 最便宜的一招：只用**关键点两两距离、重投影一致性、mask/render 一致性**做特征 + 逻辑回归，就打败了纯 conformal 不确定性 | 只在关键点+PnP 范式下验证 | 摘要（子代理） |
| VeriCalib: Verifiable Target-Based LiDAR-Camera Calibration | 2026 / arXiv | https://arxiv.org/abs/2610.28791 | 把「RMSE 好就收工」换成**结构化协方差 + 逐方向精度 + 最小可检测偏差**，输出 accept / re-extract / recollect / add observation 四态 | 面向靶标式标定，需重写为你的「无靶」场景 | 摘要（子代理） |
| CLOSURE: Fast Quantification of Pose Uncertainty Sets | 2024 / RSS | https://www.roboticsproceedings.org/rss20/p072.pdf | 从**学习得到的关键点**直接算位姿不确定集，不采样 | 依赖关键点噪声模型 | 元数据 |
| Consensus-Driven Uncertainty for Robotic Grasping (RGB) | 2025 / IROS | https://arxiv.org/abs/2506.20045 | 直接回答「这次位姿会不会导致抓取失败」，让代理**选择不动作** | 与夹爪/物体强绑定，换场景要重训 | 摘要（子代理） |

> **这一阶段的产出应该是**：每个位姿附一个椭球 + 一个二值「可执行/拒答」。指标从 `median/p95 误差` 换成 `覆盖率@容差` 与 `可用帧率`。

---

## 2. S1 · 几何确定性（PnP / 鲁棒 / 歧义）

| 文献 | 年 / venue | 链接 | 优点 | 风险 / 局限 | 核实深度 |
|---|---|---|---|---|---|
| IPPE: Infinitesimal Plane-based Pose Estimation | 2014 / IJCV | https://github.com/tobycollins/IPPE | **显式返回两个解**，是处理平面双解的正统工具；速度极快 | 只对共面点成立（你的 9 点近共面，可用但要注明假设） | 元数据 |
| TEASER / TEASER++ | 2020 / CVPR | https://arxiv.org/abs/2001.07715 | 可证明鲁棒，抗 **99% 外点**；C++/Python/MATLAB 绑定 | 需要点对距离不变性 → 用于「3D-3D」最自然，3D-2D 要改造 | 元数据 |
| Optimal Uncertainty-Aware Calibration for the AX=YB Problem | 2026 / arXiv（IJRR 在审） | https://arxiv.org/abs/2605.04809 | Lie 代数上迭代、保结构、近似全局最优；高不确定条件精度提升 ≥67% | 未开源（待查） | 摘要（子代理） |
| Simultaneous Hand-Eye and Robot-World Calibration (AX=YB) | 2016 / — | https://rpk.lcsr.jhu.edu/wp-content/uploads/2016/08/Li16_Simultaneous-Hand-Eye-and-Robot-World-Calibration.pdf | 同时解 X、Y，避免两步法误差累积；给出可解性条件 | 需多姿态 | 元数据 |
| An Extended Robot–World/Hand–Eye Calibration Method | 2018 / Sensors 18(11):3949 | https://www.mdpi.com/1424-8220/18/11/3949 | 带实验对比的基线，做对照时引用 | 同上 | 元数据 |
| A PnP Algorithm for Two-Dimensional Pose Estimation | 2024 / arXiv | https://arxiv.org/abs/2312.08488 | 相机被约束在二维运动时降自由度，极小样本解 | 你的相机是 3D 任意放置，只在你做「桌面平放」变体时用 | 元数据 |

---

## 3. S2 · 学习范式（keypoint → 自监督 → 可微渲染）

| 文献 | 年 / venue | 链接 | 优点 | 风险 / 局限 | 核实深度 |
|---|---|---|---|---|---|
| DREAM: Camera-to-Robot Pose Estimation from a Single Image | 2020 / ICRA | https://arxiv.org/abs/1911.09231 | 你的 baseline 正统出处：用运动学渲染关键点图 + 关键点网络 + PnP | 依赖仿真标注；单帧在平面臂上退化 | 元数据 + 引用片段 |
| CtRNet: Markerless Camera-to-Robot Pose Estimation via Self-supervised Sim-to-Real | 2023 / CVPR | https://arxiv.org/abs/2302.14332 | **本项目最重要的范式升级**：前景分割 + 可微渲染做图像级自监督，**不需要真机 3D 标注**；整条 pipeline 可微，已集成视觉伺服 | 成败取决于前景分割质量；金属细长臂分割是硬骨头 | 深读 |
| DRHeC: Differentiable Rendering for Hand-Eye Calibration with RGB-Based Gradients | 2026 / T-IM | https://arxiv.org/abs/2609.36779 | 用 **RGB 梯度**替代二值掩码梯度；真机抓取成功率 88.9%，比 EasyHeC 高 46.3 个点 | 手眼（eye-in-hand）设定，需改造成 eye-to-hand | 摘要（子代理） |
| MonoSE(3)-Diffusion | 2025 / arXiv | https://arxiv.org/abs/2510.10434 | 生成式建模位姿，报告 AUC 66.75（较 SOTA +32.3%） | 推理成本高；端侧不可行 | 摘要（子代理） |
| RoPE-Flow: 2D Graph-Conditioned 3D Flow Matching | 2026 / arXiv | https://arxiv.org/abs/2610.21086 | 流匹配 + 图条件，跨场景零样本基准 | 太新，未复现 | 摘要（子代理） |
| RoboTAG: End-to-end Robot Configuration Estimation via Topological Alignment Graph | 2025 / arXiv | https://arxiv.org/abs/2511.07717 | 2D/3D 双分支图对齐，**弱标签**训练 | 未给精度数字 | 摘要（子代理） |
| Learning Markerless Robot-Depth Camera Calibration and EE Pose | 2023 / PMLR v205 | https://proceedings.mlr.press/v205/sefercik23a/sefercik23a.pdf | 两阶段框架（单帧末端 6D → 多帧外参），且**不依赖仿真数据** | 需要深度相机 | 摘要 |
| SAM 2: Segment Anything in Images and Videos | 2024 / arXiv | https://arxiv.org/abs/2408.00714 | CtRNet 式自监督的前置分割件；视频记忆库利于连续帧 | 端侧难；细长目标易漏 | 元数据 |

---

## 4. S3 · 免人工标定的标定流程

| 文献 | 年 / venue | 链接 | 优点 | 风险 / 局限 | 核实深度 |
|---|---|---|---|---|---|
| ARC-Calib: Autonomous Markerless Camera-to-Robot Calibration | 2025 / arXiv | https://arxiv.org/abs/2503.14701 | 让机械臂**自己动出可跟踪的轨迹图案**当靶子，用共面/共线约束精化；不依赖预训练跟踪网络 | 需机器人可自主运动；摘要无数字 | 摘要（子代理） |
| Hydra: Marker-Free RGB-D Hand-Eye Calibration | 2025 / arXiv | https://arxiv.org/abs/2504.20584 | 点云 + Lie 代数 ICP；任务空间 **5 mm**（经典法 7 mm），3 个随机构型约 90% 收敛 | 需深度相机 | 摘要（子代理） |
| Calib3R: Hand-Eye Calibration and 3D Metric-Scaled Reconstruction with 3D Foundation Models | 2025 / arXiv | https://arxiv.org/abs/2509.08813 | 用 3D 基础模型的点图，**不到 10 张图**完成免靶标定 + 米制重建 | 依赖 3D 基础模型（PC 端） | 摘要（子代理） |
| Automatic Robot Hand-Eye Calibration Enabled by Learning-Based 3D Vision | 2023 / arXiv | https://arxiv.org/abs/2311.01335 | 无标记、无人工，声称 **1 秒**完成 | 需 3D 视觉，非纯 RGB | 元数据 |
| LRBO2: Improved 3D Vision Based Hand-Eye Calibration | 2025 / arXiv | https://arxiv.org/abs/2504.21619 | 近期的工程化对比基线 | 同上 | 元数据 |
| Continual Hand-Eye Calibration for Open-world Manipulation | 2026 / arXiv | https://arxiv.org/abs/2604.15814 | 空间回放 + 结构保持蒸馏，缓解「换场景就忘」 | 未给数字 | 摘要（子代理） |
| Continual learning approaches to hand–eye calibration | 2024 / MVA | https://dl.acm.org/doi/abs/10.1007/s00138-024-01572-w | 持续学习视角的系统研究，适合做 related work | 非主流设定 | 元数据 |

---

## 5. S4 · 多视角与 3D 基础模型（手持场景的免费三维）

| 文献 | 年 / venue | 链接 | 优点 | 风险 / 局限 | 核实深度 |
|---|---|---|---|---|---|
| VGGT: Visual Geometry Grounded Transformer | 2025 / CVPR **Best Paper** | https://arxiv.org/abs/2503.11651 | 一次前向直出**相机参数、点图、深度、点轨迹**；可当离线真值生成器 | 参数量大，端侧不可行；弱纹理金属未验证 | 元数据 |
| DUSt3R: Geometric 3D Vision Made Easy | 2024 / CVPR | https://cvpr.thecvf.com/virtual/2024/poster/29338 | 两张**未标定**图像直接出稠密点图，不需要位姿先验 | 两视图为主；尺度仍需外部锚 | 元数据 |
| MUSt3R: Multi-view Network for Stereo 3D Reconstruction | 2025 / arXiv | https://arxiv.org/abs/2503.01661 | 把 DUSt3R 扩到多视图，正好对应「手持拍一圈」 | 同上 | 元数据 |
| MASt3R | 2024 | 见 DUSt3R 项目页 | 在 DUSt3R 上加精确匹配，更利于与 FK 点做对应 | 同上 | 元数据 |

> **用法**：多视角点云 ↔ FK 的 3D 关键点做**带尺度 Umeyama 对齐** → 一次得到「尺度 + 各位姿」。尺度偏离 1 就是 URDF/关键点定义有系统误差的证据。

---

## 6. S5 · 物体侧

| 文献 | 年 / venue | 链接 | 优点 | 风险 / 局限 | 核实深度 |
|---|---|---|---|---|---|
| FoundationPose | 2024 / CVPR Highlight | https://arxiv.org/abs/2312.08344 | 只要网格/CAD 就能做 6D 位姿 + 跟踪，**替代「射线∩桌面」的弱假设** | 需模型；纯 RGB 下精度低于 RGB-D | 摘要 |
| MarUco: Markerless 6D Pose for Continuum Manipulators | 2026 / arXiv | https://arxiv.org/abs/2602.16365 | 0.78 mm / 3.07°，52.4 ms/双目对；给了「运动学无关的基座外参」思路 | 手术连续体臂，构型不同 | 摘要（子代理） |

---

## 7. 背景资料：读它是为了能说清「为什么不做」

| 文献 | 年 / venue | 链接 | 一句话 |
|---|---|---|---|
| NARF22: Neural Articulated Radiance Fields | 2022 / arXiv | https://arxiv.org/abs/2210.01166 | 关节物体神经场，联合估关节与 6D 位姿 |
| NARF24: Estimating Articulated Object Structure for Implicit Rendering | 2024 / arXiv | https://arxiv.org/abs/2409.09829 | 部件级 NeRF 估移动/转动关节 |
| A-NeRF | 2021 / NeurIPS | https://openreview.net/forum?id=lwwEh0OM61b | 无真值联合优化关节姿态与体密度 |
| CLA-NeRF | 2022 / ICRA | https://weichengtseng.github.io/project_website/icra22/index.html | 类别级关节 NeRF：视图合成 + 部件分割 + 位姿 |
| Gaussian Splatting-Based Hand-Eye Calibration | 2026 / arXiv | https://arxiv.org/abs/2603.08983 | 3DGS 做手眼标定，精度高但重（只做离线） |
| Observability-Aware Active Calibration of Multi-Sensor Systems | 2025 / arXiv | https://arxiv.org/abs/2506.13420 | 可观测性感知的主动标定，你的 U5 方法论模板 |
| GenNBV: Generalizable Next-Best-View Policy | 2024 / CVPR | https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_GenNBV_Generalizable_Next-Best-View_Policy_for_Active_3D_Reconstruction_CVPR_2024_paper.pdf | 可泛化的下一最佳视角策略 |
| Configuration-Dependent Robot Kinematics Model and Calibration | 2025 / arXiv | https://arxiv.org/abs/2510.19962 | 全工作空间的构型相关运动学标定 |
| Multi-Agent Pose Uncertainty: A Differentiable Rendering Cramér-Rao Bound | 2025 / arXiv | https://arxiv.org/abs/2510.21785 | 用可微渲染推 Cramér-Rao 界，判断位姿可观测性上限 |

---

## 8. 阅读顺序（4 周，每周 2 篇 + 一张卡）

| 周 | 读什么 | 读完要能回答 |
|---|---|---|
| W1 | DREAM → CtRNet | 我的 baseline 到底弱在哪一段？自监督信号从哪来？ |
| W2 | IPPE → TEASER++ | 平面双解怎么显式拿到？RANSAC 在小样本下为什么失效？ |
| W3 | SLUE → Geodesic-CP → Keypoint Self-Consistency | 我怎么证明「这个位姿可信」？ |
| W4 | VGGT/DUSt3R → Calib3R → ARC-Calib | 手持多视角能不能替代单帧？ |

**每篇产出统一格式**：`WHY / HOW / WHAT / 能不能插进我的哪一环 / 卡在哪里`。

---

## 9. 待核验（不要当成结论引用）

- `RoPE-Flow`(2610.21086)、`MonoSE(3)-Diffusion`(2510.10434) 的会议录用状态未确认，仅 arXiv 预印本。
- `DRHeC` 的 eye-in-hand 设定能否直接迁移到 eye-to-hand，摘要未说明，需读正文。
- `Optimal Uncertainty-Aware Calibration for AX=YB` 未确认是否开源实现。
- 2025–2026 年**没有**检索到「可观测性判据驱动的主动选视角来标定相机—机械臂外参」的完整工作；这条线目前需要 VeriCalib（补数据方向）+ ARC-Calib（自主运动）两篇拼起来用。**「检索范围内未发现」不等于「不存在」。**
- 上表所有「声称精度」均为论文自述，未经本项目复现。
