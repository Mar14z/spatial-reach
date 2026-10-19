# M0 合成数据格式

输出目录：`data/synthetic/m0/<frame_id>/`（`data/` 不入库）

| 文件 | 内容 |
|---|---|
| `rgb.png` | 640×480 RGB |
| `depth.npy` | float32 `(480, 640)`，单位米 |
| `depth.png` | 深度归一化可视化 |
| `seg.npy` | int32 `(480, 640)`，PyBullet 分割缓冲 |
| `seg.png` | 分割彩色可视化 |
| `overlay.png` | 2D keypoints 叠加 |
| `meta.json` | 相机参数、关节角、3D/2D keypoints |

## meta.json

| 字段 | 说明 |
|---|---|
| `K` | 3×3 内参 |
| `R_cb`, `t_cb` | base→camera 外参 |
| `keypoints_3d_base` | `(6, 3)` 基座系 3D 点 |
| `keypoints_2d` | `(6, 2)` 像素坐标 |
| `link_names` | 6 个 link 名称 |
| `joints` | 关节名 → 角度 |
| `valid` | 每个 keypoint 是否在相机前方 |
| `camera` | yaw / pitch / distance / target |
| `near`, `far`, `fov_deg`, `width`, `height` | 相机参数 |

## 生成

```bash
conda activate arm_sim

# 小样本（带 depth/seg）
python scripts/generate_frames.py --frames 10 --seed 0

# 训练集：600 帧，360° yaw，dist 0.45–0.95，逐帧随机桌面颜色/光照
python scripts/generate_frames.py --frames 600 --seed 50 \
  --out data/synthetic/m0/train_dr \
  --yaw-min 0 --yaw-max 360 --dist-min 0.45 --dist-max 0.95 \
  --pitch-min -25 --pitch-max -5 --no-depth

# 验证集：100 帧，同分布，不同 seed
python scripts/generate_frames.py --frames 100 --seed 60 \
  --out data/synthetic/m0/val_dr \
  --yaw-min 0 --yaw-max 360 --dist-min 0.45 --dist-max 0.95 \
  --pitch-min -25 --pitch-max -5 --no-depth
```

随机姿态范围见 `scripts/generate_frames.py` 的 `JOINT_RANGES`。  
`--no-depth` 用于大数据集，只保存 RGB / overlay / meta，节省磁盘。

切换机械臂模型：

```bash
python scripts/generate_frames.py \
  --urdf assets/robots/phantomx_pincher_cad/phantomx_pincher_cad.urdf \
  --frames 500 --seed 200 --out data/synthetic/m0_cad/train \
  --yaw-min 0 --yaw-max 360 --dist-min 0.35 --dist-max 0.60 \
  --pitch-min -25 --pitch-max -5 --no-depth
```

> 默认 URDF 已切到 CAD 模型 `assets/robots/phantomx_pincher_cad/phantomx_pincher_cad.urdf`。CAD 模型的基座 link 名是 `base_link`；旧模型是 `arm_base_link`。脚本会自动识别。

## 当前限制

- `seg.npy` 是 PyBullet TinyRenderer 的粗分割（body 级），不是逐 link 掩码。
- 深度是米制，背景值为 `far`。
- 已随机：关节姿态、相机位姿、桌面颜色、光照方向/颜色/强度。
- 尚未随机：背景纹理、机械臂材质/颜色、运动模糊、相机噪声。
