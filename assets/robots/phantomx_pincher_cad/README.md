# PhantomX Pincher Mark II — 真实 CAD 装配

## 来源

- 原始文件：`C:\Users\1\Downloads\JCI54550240030\PhantomX Pincher Robot Arm Kit Mark II.step`
- 文件日期：2018-01-04
- 大小：9.1 MB
- 装配：12 个产品、47 个实例

## 已转换

| 文件 | 说明 |
|---|---|
| `pincher.step.glb` | 带层级、零件名和装配变换的 GLB（约 5.2 MB） |
| `pincher_assembly.obj` | 合并后的装配网格，用于快速渲染（约 10.4 MB） |
| `parts/*.stl` | 主要结构件：底座、肩部组件、下臂、上臂、腕/夹爪本体、手指 |
| `D:\documents\Code\touch\docs\assets\pincher_cad_assembly.png` | PyBullet 渲染图 |

转换工具：`cascadio` + `trimesh`（已装入 `arm_sim`）。

## CAD URDF（模型驱动版）

- 文件：`phantomx_pincher_cad.urdf`
- 生成脚本：`scripts/build_cad_urdf.py`（从 GLB 的零件装配变换直接生成）
- 运动学：直接来自 CAD；link frame = 零件局部坐标系，joint origin = 父→子相对变换
- 底座：`base_link` 包含 `PincherBase` + `PincherBase_1`；`base_footprint` 把 CAD 的 Y-up 转成 Z-up
- 关节轴：pan = Z，lift/elbow/wrist = Y，夹爪 = -Y（在各自 link frame 下）
- 验证：零位时 7 个 link 的位置/朝向与 CAD 装配误差 **0**；PyBullet 7 joints；`tests/test_cad_urdf.py` 通过
- 渲染：`outputs/renders/pincher_cad_views.png`
- 渲染姿态：STEP 的默认装配是**折叠位**，不是装错；渲染脚本使用限位内的**直立 home 位**：
  `pan=0.002, bicep=0.077, forearm=-1.366, wrist=0.0, gripper=0.0`
- 注意：CAD 的 base→shoulder 约 **9.44cm**，与实测 5.4cm 不同；本版按模型数据生成，旧 URDF 仍是实测基线。关节正方向、零位偏置需要和实机对照。

## 零件映射（待最终确认）

| CAD 零件 | 可能对应 | 说明 |
|---|---|---|
| `PincherBase` | `base_link` | 底座/安装板 |
| `PincherBase_1` | 底座小件 | 约 4.25cm 的圆形件 |
| `Component1` | 肩部组件 | 在 `ShoulderServo v5` 下 |
| `LowerArm v2` | 下臂 | 外形约 0.1295m，含关节外伸 |
| `UpperArm v2` | 上臂 | 与下臂同尺寸 |
| `GripperBody v4` | 腕/夹爪本体 | 约 5.9 × 7.2 × 8.3cm |
| `LeftFinger v2` ×2 | 两指 | 约 4.3 × 2.1 × 4.1cm |
| `pins v2` / `10mm standoff v1` / `ArbotiX-M v4` | — | 螺钉/铜柱/控制板，仿真可省略或静态挂载 |

## 重要说明

- STEP 是**静态 CAD 装配**，不包含 joint、axis、limit，不能直接当 URDF。
- 当前 `assets/robots/phantomx_pincher_arm/urdf/phantomx_pincher_arm.urdf` 仍是**运动学基线**（实测 5.4 / 10.8 / 10.8 cm）。
- 下一步建议：用 CAD 网格 + 现有 joint 定义生成 `phantomx_pincher_cad.urdf`，保留旧 URDF 做对照，并在 PyBullet 里验证 FK 和关节运动。
