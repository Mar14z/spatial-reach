# Touch — 单目空间认知机械臂

> 一台 PhantomX Pincher 机械臂 + 一个单目摄像头。
> 目标不是「看见一个像素框」，而是让开发板在空间里知道：**我在哪、臂在哪、物体在哪、它们是什么关系**。

当前阶段：**仿真优先**。先把「单目 → 空间概念 → 抓取」的链路在仿真里完整跑通，再用同一套接口切真机。

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
│                    仿真后端 (PyBullet + pyrender)          │
│   world / arm / camera / renderer / randomization         │
└───────────────────────────┬──────────────────────────────┘
                            │ ArmBackend / CameraSource
                            ▼
┌──────────────────────────────────────────────────────────┐
│ Perception        keypoints · detector · geometry · pose  │
│                   图像 → 2D 结构 → PnP → 相机位姿           │
└───────────────────────────┬──────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────┐
│ Spatial（核心）    frames · scene_graph · relations · memory│
│                   基座系下的物体、关系、记住并更新           │
└───────────────────────────┬──────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────┐
│ Grasp             kinematics · planner · executor         │
│                   IK → 接近 → 抓取 → 放置                  │
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
│   ├── sim.yaml              # 仿真后端、物理步长、随机种子
│   ├── camera.yaml           # 虚拟相机分辨率、视场角、裁剪面
│   ├── camera_acepro.yaml    # Ace Pro UVC 实测 + 标定引用
│   └── camera_intrinsics.yaml# 1280×720 标定结果（K / dist）
├── assets/
│   ├── robots/phantomx_pincher_cad/  # CAD URDF + 零件 STL（来源/许可见目录内 README）
│   ├── objects/              # 目标物体模型
│   ├── textures/             # 域随机化纹理
│   └── backgrounds/          # 域随机化背景
├── touch/
│   ├── interfaces.py         # ArmBackend / CameraSource：仿真与真机的接缝
│   ├── sim/
│   │   ├── world.py          # PyBullet 世界：桌面、物体、物理
│   │   ├── arm.py            # ArmBackend 的仿真实现
│   │   ├── camera.py         # 可任意摆放的虚拟相机
│   │   ├── renderer.py       # pyrender 离屏渲染：RGB / 深度 / 分割 / keypoints
│   │   └── randomization.py  # 材质、光照、背景、相机位姿随机化
│   ├── perception/
│   │   ├── keypoints.py      # 臂关节 keypoint 检测（heatmap）
│   │   ├── detector.py       # 目标检测
│   │   ├── geometry.py       # 投影、反投影、射线-平面求交
│   │   └── pose.py           # PnP：相机相对基座位姿
│   ├── spatial/              # 核心：空间概念
│   │   ├── frames.py         # world / base / camera / image 坐标帧与变换
│   │   ├── scene_graph.py    # 物体、位置、属性
│   │   ├── relations.py      # 左右 / 前后 / 远近等关系判定
│   │   └── memory.py         # 空间记忆：相机移动后的重锚定
│   ├── grasp/
│   │   ├── kinematics.py     # 5 轴 FK / IK
│   │   ├── planner.py        # 抓取点与接近轨迹
│   │   └── executor.py       # 动作执行（经 ArmBackend）
│   ├── data/
│   │   ├── generate.py       # 合成数据工厂入口
│   │   └── dataset.py        # 数据集读取与增强
│   ├── learning/
│   │   ├── train_keypoints.py
│   │   └── evaluate.py       # PCK / 位姿误差 / 成功率
│   ├── deploy/
│   │   ├── export.py         # ONNX 导出
│   │   └── quantize.py       # X5 hb_mapper 量化脚本
│   └── pipeline/
│       └── loop.py           # 感知 → 空间 → 抓取主循环
├── scripts/                  # 可执行入口（M0 起添加）
├── tests/
├── docs/
├── data/                     # 数据（不入库）
│   ├── synthetic/
│   └── real/
└── outputs/                  # 渲染、日志、权重（不入库）
    ├── renders/
    ├── checkpoints/
    └── runs/
```

## 5. 里程碑（仿真优先）

| 阶段 | 内容 | 交付物 | 退出条件（初始目标，M0 后校准） |
|---|---|---|---|
| **M0 场景与渲染** | PyBullet 场景、5 轴臂模型、可任意摆放的虚拟相机、pyrender 离屏渲染 | 一条命令渲染出 RGB + 深度 + 分割 + 2D keypoints + 相机/关节真值 | 连续 100 帧无泄漏；标注叠加图与渲染图对齐 |
| **M1 单目自标定** | 关节角 + URDF → 3D keypoints；图像 → 2D keypoints；PnP 求 `T_base←camera` | 自标定模块 + 误差评测 | 仿真中平移误差 ≤1cm、旋转 ≤2° `待实测` |
| **M2 空间认知** | scene graph、物体 3D 定位（射线-平面求交）、关系判定、空间记忆与重锚定 | 空间查询 demo：`红色方块在机械臂左边多少厘米` | 物体 3D 误差 ≤1.5cm；关系查询准确率 ≥95%；挪相机后自动恢复 |
| **M3 闭环抓取** | 检测 → 空间定位 → IK → 抓取 → 放置 | 仿真闭环抓取 demo | 简单方块抓取成功率 ≥90%（20 次统计） |
| **M4 随机化与 sim2real** | 材质/光照/背景/相机随机化；实拍小样本评测；X5 导出与量化 | 量化报告 + 域随机化数据管线 | 实拍 PCK 达标（阈值待 M1/M2 实测后定） |

M0–M3 全程 PC，零真机依赖；M4 才引入实拍与开发板。

## 6. 技术栈

- **仿真**：PyBullet（物理）、pyrender（离屏渲染）、trimesh（网格）
- **视觉**：PyTorch（keypoint heatmap / 检测）、OpenCV（PnP、标定、几何）
- **数值**：NumPy / SciPy
- **部署（后续）**：RDK X5 + TROS + Horizon `hb_mapper`（INT8/BPU）
- **Python**：3.11–3.13（若 pybullet/pyrender 轮子不兼容，退 3.11）

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

![CAD model](docs/assets/pincher_cad_upright_a.png)

当前配置：`configs/sim.yaml` 的 `robot_urdf` 指向 [phantomx_pincher_cad.urdf](file:///D:/documents/Code/touch/assets/robots/phantomx_pincher_cad/phantomx_pincher_cad.urdf)。生成脚本 [build_cad_urdf.py](file:///D:/documents/Code/touch/scripts/build_cad_urdf.py)，渲染见 [pincher_cad_views.png](file:///D:/documents/Code/touch/outputs/renders/pincher_cad_views.png)。

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
# 1) 生成 CAD 合成数据
python scripts/generate_frames.py --frames 500 --seed 200   --out data/synthetic/m0_cad/train --yaw-min 0 --yaw-max 360   --dist-min 0.35 --dist-max 0.60 --pitch-min -25 --pitch-max -5 --no-depth

# 2) 训练 keypoint 模型
python scripts/train_keypoints.py --model resnet18   --train-root data/synthetic/m0_cad/train --val-root data/synthetic/m0_cad/val   --steps 3000 --batch-size 8 --lr 1e-3 --stride 4 --sigma 3.0   --ckpt outputs/checkpoints/m0_cad_keypoints.pth

# 3) 测试
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
- [ ] 确认基座参考 / 优化腕、夹爪 keypoint
- [ ] M1：用预测 keypoints 跑 PnP
- [ ] M1：keypoint + PnP 自标定
- [ ] M2：空间认知层
- [ ] M3：闭环抓取
- [ ] M4：域随机化 / sim2real / X5

## 9. 边界说明

- **「免标定」免的是外参/现场标定**，相机内参仍需一次性标定（定焦镜头标一次长期有效）。
- Pincher 本体是 4 个旋转关节 + 1 个夹爪自由度，末端姿态能力有限，不做任意姿态的精细操作；聚焦桌面俯视抓取与分拣。
- 单目尺度来自**机械臂已知几何 + 桌面平面**，不依赖单目深度网络的绝对尺度。
- 仿真结论不等于真机结论：臂本体精度、sim2real gap、BPU 实测必须在真机上复核。
