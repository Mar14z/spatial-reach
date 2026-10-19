# PhantomX Pincher Mark II — CAD 模型

## 来源

- 原始文件：PhantomX Pincher Robot Arm Kit Mark II STEP（2018-01-04，约 9.1 MB，本地保留）
- 装配：12 个产品、47 个实例
- 转换：`cascadio` + `trimesh`

## 仓库内资产

| 文件 | 说明 |
|---|---|
| `phantomx_pincher_cad.urdf` | 默认 URDF；7 个关节；运动学来自 CAD 装配 |
| `parts/*.stl` | 运行时依赖的主要结构件网格 |
| `material.mtl` | 材质定义 |
| `docs/assets/pincher_cad_assembly.png` | 装配渲染图 |
| `docs/assets/pincher_cad_urdf_views.png` | URDF 多视角渲染 |

> 大体积源文件 `pincher.step.glb`（约 5.2 MB）和 `pincher_assembly.obj`（约 10.4 MB）**不入库**，保留本地。`scripts/build_cad_urdf.py` 需要本地 GLB 才能重新生成 URDF。

## CAD URDF

- 2026-10-20 起，`configs/sim.yaml` 默认指向本模型。
- `base_link` 包含 `PincherBase` + `PincherBase_1`；`base_footprint` 把 CAD 的 Y-up 转成 Z-up。
- link frame = 零件局部坐标系；joint origin = 父→子相对变换。
- 关节轴：pan = Z，lift/elbow/wrist = Y，夹爪 = -Y（在各自 link frame 下）。
- 验证：零位时 7 个 link 的位置/朝向与 CAD 装配误差 0；PyBullet 7 joints；`tests/test_cad_urdf.py` 通过。

## 尺寸

| 段 | CAD 值 | 说明 |
|---|---|---|
| 基座 → 肩 pan | **9.44 cm** | 2026-10-20 用户确认以此为准 |
| 肩 lift → 肘 | **10.70 cm** | — |
| 肘 → 腕 | **10.70 cm** | — |

旧的手工 URDF 中基座→肩 pan 为 5.4 cm，是不同参考点；旧模型仅本地保留作对照，不入库。

## 渲染姿态

STEP 默认装配是折叠位；渲染脚本使用限位内的直立 home 位：

```text
pan=0.002, bicep=0.077, forearm=-1.366, wrist=0.0, gripper=0.0
```

## 零件映射

| CAD 零件 | 对应 link | 说明 |
|---|---|---|
| `PincherBase` | `base_link` | 底座/安装板 |
| `PincherBase_1` | `base_link` | 底座小件 |
| `Component1` | `arm_shoulder_pan_link` | 肩部组件 |
| `LowerArm v2` | `arm_bicep_link` | 下臂 |
| `UpperArm v2` | `arm_forearm_link` | 上臂 |
| `GripperBody v4` | `arm_wrist_flex_link` | 腕/夹爪本体 |
| `LeftFinger v2` ×2 | `gripper_active_link` / `gripper_active2_link` | 两指 |

## 说明

- STEP 是静态 CAD 装配，不包含 joint、axis、limit，不能直接当 URDF。
- 当前默认模型由 CAD 装配变换生成；旧手工 URDF 只作为本地对照。
- 关节正方向、零位偏置仍建议和实机做一次对照。
