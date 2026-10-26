# SpatialReach — 单目空间认知机械臂

> 项目代号 **Touch**。一台 PhantomX Pincher 机械臂 + 一个单目摄像头。
> 目标不是「看见一个像素框」，而是让开发板在空间里知道：**我在哪、臂在哪、物体在哪、它们是什么关系**。

当前阶段：**仿真优先**。先把「单目 → 空间概念 → 抓取」的链路在仿真里完整跑通，再用同一套接口切真机。

> 论文库与技术路线：[docs/papers.md](docs/papers.md)（按推进阶段索引，含链接、优点与局限）。

---

## 1. 核心想法：单目摄像头如何建立空间概念

单张图像没有深度，也没有尺度。我们不靠深度网络去「猜」，而是**把机械臂本身当成一把已知的尺子**：

```
关节角（编码器/真值） + URDF
        ↓ FK
关节在基座系下的 3D 坐标  ←── 已知尺度
        ↓ 与图像中的 2D keypoints 对应
        PnP
        ↓
T_base←camera（相机相对基座的 6DoF 位姿）
        ↓
图像上任意像素 → 基座系下的一条射线
        ↓ 与桌面平面求交
物体在基座系下的 3D 位置
```

这条链成立后，单目图像第一次有了**度量**，并且这个坐标系锚定在机械臂上。相机随手放，只要画面里能看到臂，开发板就能重新把自己「放进」空间里——这就是「免现场标定」的来源。

### 空间概念的可验证定义

不做口号，只回答五个可量化的问题：

| # | 问题 | 输出 | 依赖 |
|---|---|---|---|
| 1 | 相机在哪 | `T_base←camera`（6DoF） | keypoints + PnP |
| 2 | 臂在哪 | 关节角 / 末端位姿 | keypoints + URDF |
| 3 | 物体在哪 | 基座系下的 3D 位置 | 检测 + 相机位姿 + 桌面平面 |
| 4 | 什么关系 | 左/右、前/后、远/近、可抓/不可抓 | 基座系坐标 |
| 5 | 动过之后 | 重锚定后的同一套表示 | 空间记忆与跟踪 |

### 为什么不是「跑个深度估计」

- 单目深度模型的绝对尺度不稳定，误差随场景漂移；
- 机械臂几何是**已知且精确**的，用它定尺度比网络可靠；
- 桌面平面是天然约束：物体本来就在桌上，射线求交即可得到 3D 位置；
- 深度模型可以作为可选先验（估计物体高度），但不作为主链路的尺度来源。

## 2. 为什么把重心放在仿真

1. **真值齐全**：相机位姿、关节角、物体 6D 位姿、深度、分割掩码，真机很难同时拿到。
2. **误差可分解**：keypoint 噪声 / 相机位姿误差 / 物体定位误差可以逐项注入，而不是只看到一个成功率。
3. **验证核心卖点**：相机可以任意摆、批量随机，「换个位置也能用」这件事只有仿真能大规模验证。
4. **不被硬件阻塞**：臂和板子未完全就位时，算法、数据、评测管线可以先行。
5. **同一接口切真机**：`ArmBackend` / `CameraSource` 两个协议把仿真与真机隔开，上层代码不感知后端。

## 3. 系统架构

```
┌──────────────────────────────────────────────────────────┐
│ 【已实现】仿真后端  PyBullet + TinyRenderer               │
│   sim/camera · sim/renderer · data/dataset                │
└───────────────────────────┬──────────────────────────────┘
                            │ ArmBackend / CameraSource
                            ▼
┌──────────────────────────────────────────────────────────┐
│ 【已实现】Learning / Perception                            │
│   learning/keypoint_model · perception/geometry · pose    │
│   图像 → heatmap → 2D 关键点 → PnP → 相机相对基座位姿     │
└───────────────────────────┬──────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────┐
│ 【未实现】Spatial（核心）  frames · scene_graph · relations │
│           基座系下的物体、关系、记住并更新                 │
└───────────────────────────┬──────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────┐
│ 【部分实现】Grasp  kinematics · planner · executor          │
│   MuJoCo 任务层：IK → 笛卡尔下探 → 合爪 → 抬升（开环）      │
└───────────────────────────┬──────────────────────────────┘
                            ▼
                    actions → ArmBackend
```

## 4. 目录结构

```
touch/
├── README.md
├── .gitignore
├── requirements.txt
├── configs/
│   ├── camera_acepro.yaml    # Ace Pro UVC 实测参数
│   └── camera_intrinsics.yaml# 1280×720 标定结果（K / dist，被 touch/camera.py 读取）
├── assets/
│   ├── robots/phantomx_pincher_cad/  # CAD URDF + 零件 STL（来源/许可见目录内 README）
│   ├── objects/              # 目标物体模型
│   ├── textures/             # 域随机化纹理
│   └── backgrounds/          # 域随机化背景
├── touch/                    # 库：仿真与真机共用，不放一次性逻辑
│   ├── __init__.py
│   ├── camera.py             # 真实相机：内参加载、去畸变、16:9→4:3 裁剪
│   ├── interfaces.py         # ArmBackend / CameraSource：仿真与真机的接缝
│   ├── data/
│   │   └── dataset.py        # M0Dataset + heatmap 标签
│   ├── learning/
│   │   ├── evaluate.py       # 统一评测口径：2D / 位姿 / 桌面 / 可靠性
│   │   └── keypoint_model.py # Tiny / ResNet-18 / MobileNetV3 三种 heatmap 网络
│   ├── perception/
│   │   ├── geometry.py       # 投影、反投影、射线-平面求交、位姿误差
│   │   └── pose.py           # PnP：单帧 / 多候选 / 多帧联合
│   └── sim/
│       ├── camera.py         # 虚拟相机：FOV→K、投影、投影矩阵
│       └── renderer.py       # PyBullet TinyRenderer 离屏渲染
├── scripts/                  # 19 个可执行入口
│   ├── build_cad_urdf.py         # 建模：STEP/GLB → CAD 运动学 URDF
│   ├── build_keypoint_urdf.py    # 建模：追加 9 个 keypoint
│   ├── calibrate_camera.py       # 建模：棋盘格内参标定
│   ├── generate_frames.py        # 数据：生成合成帧（RGB/深度/分割/2D 真值）
│   ├── check_dataset_coverage.py # 数据：数据集覆盖面体检（分布 / 边距 / 差异）
│   ├── merge_parts.py            # 数据：并行分片合并为 train/val
│   ├── train_keypoints.py        # 训练：--model tiny|resnet18|mobilenet_v3
│   ├── predict_keypoints.py      # 推理：模型 → 预测 JSON
│   ├── eval_report.py            # 评测：唯一入口，含按帧 bootstrap CI
│   ├── multiframe_select.py      # 实验：单帧 vs 跨帧仲裁 vs 多帧联合（合成第 2 帧）
│   ├── multiframe_pairs.py       # 实验：真实帧的帧数 × 转角扫描（几帧、动多少度）
│   ├── render_urdf.py            # 渲染：URDF 离屏出图
│   ├── mujoco_scene.py           # 任务层：MuJoCo 场景（URDF + 桌面 + 方块 + 位置执行器）
│   ├── mujoco_fk_check.py        # 任务层：PyBullet ↔ MuJoCo FK 对拍
│   ├── mujoco_pick.py            # 任务层：IK + 笛卡尔下探 + 合爪 + 抬升（开环抓取）
│   ├── mujoco_pick_sweep.py      # 任务层：抓取成功率扫描（尺寸 × 半径 × 方位角）
│   ├── tb_progress.py            # 工具：TensorBoard 曲线转文本
│   ├── demo_board_pose.py        # 真机：棋盘格位姿
│   └── demo_color_pose.py        # 真机：颜色块 3D 定位
├── tests/                    # 8 个测试：相机 / 数据 / PnP / 投影 / CAD
├── data/                     # 数据（不入库）
│   ├── synthetic/
│   └── real/
└── outputs/                  # 渲染、权重、日志（不入库）
    ├── checkpoints/  eval/  renders/  runs/  tb/  calibration/
    └── _archive/             # 一次性分析脚本归档（96 个，只查不跑）
```

## 5. 里程碑（仿真优先）

| 阶段 | 内容 | 交付物 | 退出条件（初始目标，M0 后校准） |
|---|---|---|---|
| **M0 场景与渲染** | PyBullet 场景、5 轴臂模型、可任意摆放的虚拟相机、TinyRenderer 离屏渲染 | 一条命令渲染出 RGB + 深度 + 分割 + 2D keypoints + 相机/关节真值 | 连续 100 帧无泄漏；标注叠加图与渲染图对齐 |
| **M1 单目自标定** | 关节角 + URDF → 3D keypoints；图像 → 2D keypoints；PnP 求 `T_base←camera` | 自标定模块 + 误差评测 | 仿真中平移误差 ≤1cm、旋转 ≤2° `待实测` |
| **M2 空间认知** | scene graph、物体 3D 定位（射线-平面求交）、关系判定、空间记忆与重锚定 | 空间查询 demo：`红色方块在机械臂左边多少厘米` | 物体 3D 误差 ≤1.5cm；关系查询准确率 ≥95%；挪相机后自动恢复 |
| **M3 闭环抓取** | 检测 → 空间定位 → IK → 抓取 → 放置 | 仿真闭环抓取 demo | 简单方块抓取成功率 ≥90%（20 次统计） |
| **M4 随机化与 sim2real** | 材质/光照/背景/相机随机化；实拍小样本评测；X5 导出与量化 | 量化报告 + 域随机化数据管线 | 实拍 PCK 达标（阈值待 M1/M2 实测后定） |

M0–M3 全程 PC，零真机依赖；M4 才引入实拍与开发板。

## 6. 技术栈

- **仿真**：PyBullet（物理 + TinyRenderer 离屏渲染）、trimesh（网格）
- **视觉**：PyTorch（keypoint heatmap / 检测）、OpenCV（PnP、标定、几何）
- **数值**：NumPy / SciPy
- **部署（后续）**：RDK X5 + TROS + Horizon `hb_mapper`（INT8/BPU）
- **Python**：3.11–3.13（若 pybullet 轮子不兼容，退 3.11）

## 7. 开发环境

PyBullet 在 Windows 上没有 pip 轮子，建议用 conda（Python 3.11）：

```bash
conda create -n arm_sim python=3.11
conda activate arm_sim
conda install -c conda-forge pybullet
pip install -r requirements.txt

# CUDA 12.8 PyTorch（RTX 5070 / sm_120）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Linux/WSL 下 PyBullet 有 manylinux 轮子，可直接 `pip install -r requirements.txt`。

> 请先 `conda activate arm_sim` 再运行脚本。直接调用 `...\arm_sim\python.exe` 可能因 DLL 搜索路径冲突导致 numpy 崩溃。

渲染 URDF：

```bash
python scripts/render_urdf.py
# 输出到 outputs/renders/
```

> 渲染与臂控留在 Windows；训练可先在 Windows 原生；X5 量化工具链需要 Linux/WSL2。

### 机械臂模型

`assets/robots/phantomx_pincher_cad/` 是当前默认模型：由 PhantomX Pincher Mark II STEP 装配生成的 CAD URDF，包含底座 + 全臂网格，7 个关节，运动学来自 CAD 装配。

尺寸（2026-10-20 确认以 CAD 为准）：

- 基座 → 肩 pan：**9.44cm**
- 肩 lift → 肘：**10.70cm**
- 肘 → 腕：**10.70cm**

![CAD model](assets/robots/phantomx_pincher_cad/pincher_cad_upright_a.png)

当前配置：CAD URDF 由 `--urdf` 显式传入，指向 [phantomx_pincher_cad_keypoints_v1.urdf](file:///D:/documents/Code/touch/assets/robots/phantomx_pincher_cad/phantomx_pincher_cad_keypoints_v1.urdf)（各数据集的配方与 keypoint 顺序见 [data/manifest.json](file:///D:/documents/Code/touch/data/manifest.json)）；基础模型由 [build_cad_urdf.py](file:///D:/documents/Code/touch/scripts/build_cad_urdf.py) 生成、关键点标记由 [build_keypoint_urdf.py](file:///D:/documents/Code/touch/scripts/build_keypoint_urdf.py) 注入，渲染见 [pincher_cad_views.png](file:///D:/documents/Code/touch/outputs/renders/pincher_cad_views.png)。

旧的手工 URDF 仅本地保留作对照；它的 5.4cm 基座高度是另一个参考点，不再作为默认。

### 真实相机（Ace Pro）

- `configs/camera_acepro.yaml`：UVC `1280×720@30` MJPG，`undistort: true`
- `configs/camera_intrinsics.yaml`：1280×720 标定结果，RMS `0.249px`
- `touch/camera.py`：`load_intrinsics` / `AceProCamera` / `prepare_model_input`
- 真实 16:9 画面先中心裁成 4:3，再送 640×480 keypoint 模型；K 会同步修正
  （`fx=419.645, fy=415.061, cx=321.297, cy=240.738`）
- 裁切后垂直 FOV 约 60°，与仿真虚拟相机一致
- 标定/真实相机脚本需要 OpenCV 环境：本机用 `conda activate cv_env`
- Demo：`scripts/demo_board_pose.py`（棋盘格位姿）/ `scripts/demo_color_pose.py`（颜色块 3D）

### 复现

```bash
# 1) 生成 9 点合成数据
#    --urdf / --kp-names 都必须显式给：URDF 里有不止一种 keypoint 布局，
#    拿错不会报错但和 checkpoint 对不上。本仓库主力数据用的就是下面这套。
URDF=assets/robots/phantomx_pincher_cad/phantomx_pincher_cad_keypoints_v1.urdf
KP=kp_base,kp_shoulder_pan,kp_shoulder_lift,kp_upper_mid,kp_elbow,kp_forearm_mid,kp_wrist,kp_wrist_mid,kp_gripper_base
python scripts/generate_frames.py --urdf $URDF --kp-names $KP --frames 1000 --seed 500 \
  --out data/synthetic/m0_kp9b/train --yaw-min 0 --yaw-max 360 \
  --dist-min 0.35 --dist-max 0.60 --pitch-min -25 --pitch-max -5 --no-depth
python scripts/generate_frames.py --urdf $URDF --kp-names $KP --frames 100 --seed 501 \
  --out data/synthetic/m0_kp9b/val --yaw-min 0 --yaw-max 360 \
  --dist-min 0.35 --dist-max 0.60 --pitch-min -25 --pitch-max -5 --no-depth

# 1b) 序列训练集：同序列相机/桌面/光照固定，只转 pan（800 序列 x 4 帧）
#     同时扩了分布：dist 0.35-0.80、pitch -45~-5、阴影/噪声/JPEG/逆光随机
python scripts/generate_frames.py --urdf $URDF --kp-names $KP \
  --frames 3200 --seed 6000 --out data/synthetic/m0_seq/train --no-depth \
  --seq-len 4 --pan-span 30 \
  --dist-min 0.35 --dist-max 0.80 --pitch-min -45 --pitch-max -5 \
  --shadow-p 0.5 --light-y-min -1.0 --light-y-max 0.6 \
  --noise-max 3 --jpeg-min 60 --jpeg-max 95 --blur-p 0.3
python scripts/generate_frames.py --urdf $URDF --kp-names $KP \
  --frames 320 --seed 6001 --out data/synthetic/m0_seq/val --no-depth \
  --seq-len 4 --pan-span 30 \
  --dist-min 0.35 --dist-max 0.80 --pitch-min -45 --pitch-max -5 \
  --shadow-p 0.5 --light-y-min -1.0 --light-y-max 0.6 \
  --noise-max 3 --jpeg-min 60 --jpeg-max 95 --blur-p 0.3

# 1c) 分布外测试集（配方见 data/manifest.json 的 m0_ood）
python scripts/check_dataset_coverage.py data/synthetic/m0_ood/mild --ref data/synthetic/m0_kp9b/train

# 2) 训练（GPU，cache + AMP）
python scripts/train_keypoints.py --model resnet18 \
  --train-root data/synthetic/m0_kp9b/train --val-root data/synthetic/m0_kp9b/val \
  --steps 1000 --batch-size 64 --workers 0 --cache --lr 3e-3 \
  --stride 4 --sigma 3.0 --eval-every 500 \
  --ckpt outputs/checkpoints/m0_kp9_resnet18.pth

# 2b) 序列集训练（宽分布 + 退化；lr 必须衰减，否则末期在两个解之间来回跳）
# --cache 实测无收益：1.0 step/s vs 无 cache 1.1 step/s，却多占 4.7GB
python scripts/train_keypoints.py --model resnet18 \
  --train-root data/synthetic/m0_seq/train --val-root data/synthetic/m0_seq/val \
  --steps 3200 --batch-size 64 --workers 2 --lr 3e-3 --lr-schedule cosine \
  --stride 4 --sigma 3.0 --eval-every 200 --preprocess gamma \
  --ckpt outputs/checkpoints/m0_seq_cos.pth

# 3) TensorBoard
tensorboard --logdir outputs/tb

# 4) 测试
pytest -q
```

## 8. 当前状态

- [x] 目录骨架与接口契约
- [x] PhantomX Pincher URDF（PyBullet 加载 + FK 验证通过）
- [x] 导师模式规则、skill 与学习文档
- [x] M0 相机内参与投影链路（5 tests passed，已接入渲染标注）
- [x] Ace Pro 内参标定 + 真实相机源（1280×720，RMS 0.249px；`touch/camera.py`，4 tests passed）
- [x] 真实 CAD 装配转换 + CAD 模型驱动 URDF（底座 + 全臂；零位与 CAD 误差 0；7 joints）
- [x] M0 合成数据：10 帧 RGB / depth / seg / 2D keypoints（`data/synthetic/m0/`）
- [x] M0 数据加载与 heatmap 标签（7 tests passed）
- [x] M0 最小过拟合训练（10 帧，mean 2.02px）
- [x] M0 数据扩增 + 逐帧随机化（600 train / 100 val）
- [x] M0 ResNet-18 基线（验证集平均像素误差 3.91px，软最大值解码 3.71px）
- [x] M0 热图步长 2 ResNet-18（验证集软最大值解码 4.22px，PCK@3 57.5%）
- [x] CAD 模型重训（验证集软最大值解码 5.30px，PCK@3 55.2%）
- [x] M1 PnP 对照：真值 6 点通过；预测 6 点 13.9cm / 16.1°
- [x] 模型对比：Tiny U-Net / ResNet-18 / MobileNetV3；ResNet-18 最好
- [x] 夹爪 keypoint 对照：去掉后 3.97px，但 5 点 PnP 仍不稳
- [x] 新 keypoint 彩色标记（K0–K5）：val 3.56px/60.2%；预测 PnP 7.14cm/10.23°
- [x] 9 点外抱环标记（D 组）+ gamma 预处理：test 500 帧 2D **关键点级** median 1.79px / PCK@3 76.1%（帧级 2.03px / 81.4%，两级不可混用）/ 平移 median 2.00cm / p95 7.58cm / 翻转 14 帧（2.8%）
- [x] PnP 误差分解（6 点 100 帧）：误差是**双峰**——`rot<5°` 的 46/100 帧平移 p50 仅 1.96cm；翻转与 `|pan|` 无关（corr 0.18）；`gt_5`（完美 2D 只少 1 点）仍翻转 1/100
- [x] 工作区减法：移除 11 个 1 行占位模块与 2 个未被引用的 config；一次性脚本/日志归档至 `outputs/_archive/`
- [x] 代码结构整理：评测入口统一为 `scripts/eval_report.py`（合并 `ablation_report.py`，含 CI 与 markdown 输出）；删除 `eval_pnp.py`（已被 `touch/learning/evaluate.py` 取代）、`render_cad_urdf.py`（被 `render_urdf.py` 取代）、`train_overfit.py`（被 `--model tiny` 取代）、`phantomx_pincher_cad_keypoints_v1.urdf`（旧 9 点链内版，无人引用）与 4 个空壳目录
- [x] **P0 求解器**：`solve_pnp` 默认改 SQPNP；新增 `solve_pnp_candidates`（IPPE+SQPNP 多候选）与 `select_candidate`（物理先验）。翻转 14→9/500，但 14 个翻转帧里只有 6 帧的正确解在候选集中 → 选解不是瓶颈，几何才是
- [x] **P0 指标**：一级指标改为 `flip_rate` + `usable@容差`，`scripts/eval_report.py` 已输出
- [x] **P0 误差预算**：主副本在 Obsidian `单目空间认知/工程文档/误差预算与PnP方案.md`。关键结论：**旋转误差才是瓶颈**——0.4m 处 5° 会放大成 3.5cm
- [x] ~~P0 凸耳设计~~ **已放弃**：δ 扇扫曾定出 ≥3cm、推荐 5cm（可用率 71.6%→80.0%，δ≤1cm 反而更差），但改硬件要重做 URDF + 重生成 + 重训，收益不如多帧。脚本归档在 `outputs/_archive/flip_vs_lever.py`，`build_keypoint_urdf.py --lug-cm` 保留仅用于复现该结论
- [x] **P0 多帧联合（主路线）**：相机不动、机械臂转 pan，各帧 FK 3D 点拼起来重解**同一个** `T_base←camera`。本质上不是逐帧投票，而是**一次外参自标定**：标好之后新帧直接用 T，不再逐帧解 PnP，翻转路径从根上消失。在 `m0_seq/val`（80 组 × 4 帧，真实渲染 + 真实预测，pan 每帧 +10°）上扫帧数与转角：

  | 口径 | 联解可用@3cm·5° | 翻转 |
  |---|---:|---:|
  | 单帧 | 74.4% | 3.1% |
  | 双帧 10° | 76.7% | 1.7% |
  | 双帧 20° | 81.9% | 1.9% |
  | 双帧 30° | **85.0%** | **1.2%** |
  | 四帧 30° | 81.2% | 1.2% |

  ① 收益来自**转角**不来自帧数：10° 只 +2%，20° +7.5%，30° +10.6%；四帧没有比双帧 30° 更好，多拍无益。② 运行时触发判据没有捷径——候选分歧 >5° 会 100% 触发，2D 近共线在 50% 触发率下只召回 8/10 翻转，所以**固定用相隔 20–30° 的两帧**即可。③ 联解对各帧残差做离群剔除，翻转率的尾差来自 1 组四帧全翻的序列
- [x] **多帧兜底（合成第 2 帧）**：`solve_pnp_multiframe` 在 m0_kp9b_test 上把可用率 98.2%→99.0–99.8%、两帧全合成 100%；两个反直觉结论：① 跨帧仲裁（`select_candidate_multiframe`）基本无效——9 个错帧里 8 帧的候选集根本没有正确答案；② 必须多初值启动，单初值陷局部极小（112°）。**该数字的第 2 帧是合成噪声，仅供参考，以真实帧扫描为准**
- [x] **宽分布重训 + lr 余弦衰减**（`m0_seq_cos`）：同分布 seq_val 上 2D 关键点级 median **1.27px** / PCK@3 **88.0%** / 平移 median 1.92cm / 翻转 3.1% / usable@3cm·5° **74.4%**；lr 不衰减时同一配方只有 2.97px / 50.7% / 3.52cm / 7.8% / 39.4%。桌面反投影 median **0.87cm**，首次进 1cm
- [x] **GT 关键点 PnP 上限**：GT 2D 直接喂 PnP，seq_val 与 OOD-hard 均 **100% 可用、0 翻转**，含 26% 近共线档也是 100% —— 共面镜像歧义在无噪声时不发作，与完美的差距**全部**来自 2D 误差
- [x] **瓶颈换位**：按距离分箱，kp median 全距离段 1.12–1.20px 持平，但 usable 从 90.0%（0.45m）掉到 46.9%（0.75m）——瓶颈是**尺度**不是精度；翻转率与 `bias_v_max` 同向（0.39px→3.5%，1.68px→22.5%），是**偏差**不是噪声
- [ ] M1：真机多帧自标定验证——固定相机，pan 两次相隔 20–30° 各拍一帧，联解外参；当前 85.0% 可用 / 1.2% 翻转是仿真数字，真机待实测
- [ ] M1：把「标一次 T、之后不再逐帧解 PnP」接进推理路径（现在是逐帧求解 + 序列覆写）
- [ ] M1：真机 10–20 帧 + ArUco 真值最小验证
- [x] **MuJoCo 任务层（2026-10-26）**：感知链路不动，执行层迁 MuJoCo 3.17。URDF 导入后手补位置执行器（`armature=0.02` / `damping=0.2` / `kv=2√(kp·armature)`）、给第二指补反向滑动关节 + `mjEQ_JOINT` 等式 `q2=q1` 还原齿条联动、`forcerange=±8N`。FK 与 PyBullet 对拍最大差 **7.9e-9 m**。
- [x] **夹爪几何纠正**：之前「不对称夹爪、抓取点=固定指面+半块」是拿滑块伸出的姿态当零位量错了。真实是**齿条联动对称双指**，指腹 ±14.8mm，**开 61.7mm / 合到底 ~2mm**，抓取点 = 指腹中点（腕部系 y=0）；URDF 下限 `0.002 → -0.0276`。按错误结论放方块，合爪时会被推 31mm 再挤出。
- [x] **开环抓取 + 成功率扫描**：`mujoco_pick.py`（多起点 IK → 笛卡尔分步下探 → 合爪 → 笛卡尔抬升）。判据对齐 <10mm、夹持内偏移 <20mm、抬升 >5cm。3/4cm × 半径 0.09–0.15 × 方位角 −0.4/0/+0.4 = **24/24**（抬升 68–74mm）。注入 0–15mm 指令点偏移仍 **20/20**，20mm 起开始失败——**夹爪横向容差 > 反投影中位误差 0.72cm，抓取成败的瓶颈不是定位精度**。
  两个坑值得记：① 抬升必须走笛卡尔分步，关节空间插值会让手腕中途转 30°+、方块 0.15s 内滑脱 15.3mm；② 方位角要**相对臂正前方**（`arm_base_joint` 绕 z 转了 −90°，臂朝世界 −y，`pan ≈ az_world + π/2`），按世界 +x 扫会在 +0.2 rad 撞 pan 限位，看着像"抓不到"。
- [ ] M2：空间认知层
- [ ] M3：闭环抓取（仿真开环已通，见上；闭环视觉伺服未做）
- [ ] M4：域随机化 / sim2real / X5

> 文档（论文库、误差预算、合成数据说明、学习笔记）主副本在 Obsidian `D:\documents\note\02 - Notes\02 - 项目\机械臂视觉抓取\单目空间认知\`；本仓库只保留这一份 README。

## 9. 边界说明

- **「免标定」免的是外参/现场标定**，相机内参仍需一次性标定（定焦镜头标一次长期有效）。
- Pincher 本体是 4 个旋转关节 + 1 个夹爪自由度，末端姿态能力有限，不做任意姿态的精细操作；聚焦桌面俯视抓取与分拣。
- 单目尺度来自**机械臂已知几何 + 桌面平面**，不依赖单目深度网络的绝对尺度。
- 仿真结论不等于真机结论：臂本体精度、sim2real gap、BPU 实测必须在真机上复核。
