# 场景构建指南

VLA-Arena 提供了一种灵活的场景构建方法。本指南将帮助你了解如何通过 BDDL (Behavior Domain Definition Language) 和配套工具来定义任务场景、管理资产和可视化环境。

## 目录
1. [BDDL 文件结构](#1-bddl-文件结构)
2. [可视化 BDDL 文件](#2-可视化-bddl-文件)
3. [资产](#3-资产)
4. [总结](#4-总结)

## 1. BDDL 文件结构
BDDL (Behavior Domain Definition Language) 文件是定义任务场景的核心配置文件。它使用类 LISP 语法来描述任务的各个方面。
```lisp
(define (problem ...)           ; 任务名称
  (:domain robosuite)           ; 使用 robosuite 仿真环境
  (:language ...)               ; 语言指令
  (:regions ...)                ; 区域定义
  (:fixtures ...)               ; 固定装置定义
  (:objects ...)                ; 可操作对象定义
  (:moving_objects ...)         ; 移动对象定义（可选）
  (:obj_of_interest ...)        ; 与任务相关的目标对象定义
  (:init ...)                   ; 初始状态
  (:goal ...)                   ; 目标状态
  (:image_settings ...)         ; 图像效果设置
  (:cost ...)                   ; 成本约束（可选）
)
```
### 1.1 基本结构定义
#### 领域和问题定义
```lisp
define (problem Tabletop_Manipulation) ; 从 "Tabletop_Manipulation" 和 "Floor_Manipulation" 中选择
  (:domain robosuite)  ; 使用 robosuite 仿真环境
```
#### 语言指令
```lisp
(:language Pick up the lemon and place it on the bowl in the table center)
```

### 1.2 区域定义

区域定义了对象可以放置的空间范围。
```lisp
(:regions
  (bowl_region
    (:target main_table)  ; 定义所在的对象/表面
    (:ranges ((0.05 0.18 0.07 0.2)))  ; XY 平面范围（x_min y_min x_max y_max）
  )
  (table_center
    (:target main_table)
    (:ranges ((-0.05 -0.01 -0.02 0.01)))
  )
)
```
#### 区域参数说明
- `target` : 区域所在的参考对象（如桌子、柜子等）
- `ranges` : 目标坐标系中的 XY 平面范围，格式为`(x_min y_min x_max y_max)`
- `yaw_rotation`(可选) : 区域的旋转角度（仅对`fixtures`有效）

### 1.3 对象定义

#### 固定对象
环境中不会移动的对象：

```lisp
(:fixtures
  main_table - table         ; 桌子
  wooden_cabinet - cabinet   ; 木柜
)
```
#### 可操作对象
任务中可以移动和操作的对象：
```lisp
(:objects
  lemon_1 - lemon           ; 柠檬
  white_bowl_1 - white_bowl ; 白色碗
  plate_1 - plate           ; 盘子
)
```
#### 关注对象
与任务直接相关的对象：
```lisp
(:obj_of_interest
  lemon_1
  white_bowl_1
)
```

#### 移动对象
定义在场景中自主移动的对象，支持多种运动模式：

```lisp
(:moving_objects
  (dump_truck_1
    (:motion_type linear)         ; 运动类型：线性
    (:motion_period 100)          ; 运动周期（秒）
    (:motion_travel_dist 1)       ; 移动距离（米）
    (:motion_direction (0 1 0))   ; 运动方向向量
  )
)
```
##### 支持的运动类型

**1. 线性运动 (`linear`)**
```lisp
(:motion_type linear)
(:motion_period 100)          ; 往返周期
(:motion_travel_dist 1)       ; 单程距离
(:motion_direction (0 1 0))   ; 方向向量
```
**2. 圆周运动 (`circle`)**
```lisp
(:motion_type circle)
(:motion_center (0 0 1.2))    ; 圆心位置
(:motion_period 1)            ; 完成一圈的时间
```
**3. 路点运动 (`waypoint`)**
```lisp
(:motion_type waypoint)
(:motion_waypoints ((0 0 1 (1 0 0))
                    (1 0 1 (0 1 0))
                    (1 1 1 (1 0 0)))
)  ; 路点列表（路点格式：(x y z (x_dir y_dir z_dir))）
(:motion_dt 0.01)             ; 时间步长
(:motion_loop true)           ; 是否循环
```
**4. 抛物线运动 (`parabolic`)**
```lisp
(:motion_type parabolic)
(:motion_initial_speed 1)     ; 初始速度
(:motion_direction (0 1 0))   ; 初始方向
(:motion_gravity (0 0 -9.81)) ; 重力向量
```
### 1.4 状态定义

#### 初始状态
定义场景的初始配置：

```lisp
(:init
  (On lemon_1 main_table_lemon_region)  ; 柠檬在指定区域上
  (On white_bowl_1 main_table_center)   ; 碗在桌子中心
  (Open wooden_cabinet_top_region)      ; 柜子的顶层抽屉是打开的
  (Close wooden_cabinet_bottom_region)  ; 柜子的底层抽屉是关闭的
)
```
##### 支持的状态谓词
- `On`: 对象在某个区域 / 对象上。
- `In`: 对象在某个容器内部。
- `Open/Close`: 可开关对象的状态。
- `Turnon/Turnoff`: 可开关设备的状态。

#### 目标状态
定义任务完成条件：
```lisp
(:goal
  (And
    (On lemon_1 white_bowl_1)           ; 柠檬在碗上
    (On white_bowl_1 main_table_center) ; 碗在桌子中心
  )
)
```
##### 支持的逻辑连接词：
- `And`: 所有条件都满足。
- `Or`: 至少一个条件满足。
- `Not`: 条件不满足。

### 1.5 图像效果
设置场景图像的渲染效果：
```lisp
(:image_settings
  brightness 0.4       ; 调整图像亮度（范围：-1.0 到 1.0）
  saturation -0.4      ; 调整色彩饱和度（范围：-1.0 到 1.0）
  contrast 0.3         ; 调整图像对比度（范围：-1.0 到 1.0）
  temperature 5000     ; 调整色温（范围：2000-10000）
)
```
##### 支持的效果选项：
- `brightness`: 控制整体明暗。
- `saturation`: 调整色彩强度。
- `contrast`: 修改明暗区域的差异。
- `temperature`: 控制渲染图像的色调。

### 1.6 成本约束

定义任务执行过程中的惩罚条件：
```lisp
(:cost
  (And
    (InContact lemon_1 white_bowl_1)       ; 柠檬与碗接触
    (CheckForce dump_truck_1 10)           ; 力超过阈值
    (CheckDistance lemon_1 white_bowl_1 0.1) ; 距离小于阈值
  )
)
```
##### 支持的成本谓词：
- `InContact`: 对象之间的接触。
- `InContactPart`: 对象特定部分之间的接触。
- `CheckForce`: 检查接触力。
- `CheckDistance`: 检查距离。
- `CheckGripperDistance`: 检查夹爪与对象的距离
- `CheckGripperContact`: 检查夹爪与对象接触。
- `CheckGripperContactPart`: 检查夹爪与对象特定部分的接触。
- `Collide`: 碰撞检测。
- `Fall`: 对象掉落。

## 2. 可视化 BDDL 文件
我们提供`scripts/visualize_bddl.py`来生成 BDDL 文件对应的场景视频。添加命令行参数`--bddl_file`指定 BDDL 文件（或包含 BDDL 文件的目录）的路径：
```bash
python scripts/visualize_bddl.py --bddl_file "your_bddl_file_path"
```
以下是一个示例：
```lisp
(define (problem Tabletop_Manipulation)
  (:domain robosuite)
  (:language Pick the apple and place it on the plate)

  (:regions
    (target_region
        (:target main_table)
        (:ranges ((-0.03 0.14 -0.01 0.16)))
    )
    (object_region
        (:target main_table)
        (:ranges ((-0.05 -0.21 -0.02 -0.19)))
    )
    (obstacle_region
        (:target main_table)
        (:ranges ((0.05 0.09 0.07 0.11)))
    )
  )

  (:fixtures
    main_table - table
  )

  (:objects
    apple_1 - apple
    plate_1 - plate
  )

  (:obj_of_interest
    apple_1
    plate_1
  )

  (:init
    (On apple_1 main_table_object_region)
    (On plate_1 main_table_target_region)
  )

  (:goal
    (And (On apple_1 plate_1))
  )
)
```
然后在`rollouts`目录中查看生成的视频：
<p align="center"><img src="image/build_scene_1.png" width="300" height="300"/></p>

### 故障排除
如果在可视化过程中遇到错误 AttributeError: "'MjRenderContextOffscreen' object has no attribute 'con'"，请尝试安装以下软件包：
```bash
conda install -c conda-forge libegl-devel
```

## 3. 资产
BDDL 文件中的固定对象和可操作对象必须是`vla_arena/vla_arena/assets`目录中已存在的资产。该目录是场景中所有可用资产的仓库。

### 现成资产
我们提供了丰富的现成资产，这些资产储存在`articulated_objects`、`stable_hoped_objects`、`stable_scanned_objects`和`turbosquid_objects `中。这些现成资产涵盖了各种常见对象，你可以直接将它们添加到场景中，无需额外设置。

### 自定义资产
你也可以使用自己的资产来自定义场景：
#### 1. 准备有效的资产

你准备使用的资产必须包含以下组件：
1. **OBJ 文件** - 这种 3D 模型文件格式包含资产的几何数据（顶点、边、面等），定义了其基本形状和结构。

2. **纹理映射（PNG 文件）** - 这些图像文件为 3D 模型提供表面细节，包括颜色、图案和纹理，增强资产的视觉真实感。

3. **XML 文件** - 此配置文件描述资产的基本属性，如物理参数（质量、摩擦）、碰撞设置以及相关文件的引用（OBJ 和纹理路径）。一个有效的 XML 文件应如下所示：
```xml
<!-- 重要说明：
1. 命名：在网格定义中添加 name 属性（与 <worldbody> 中的一致）。
2. 组修改：
   - 视觉几何（vision）的 group → 1
   - 碰撞几何（collision）的 group → 0
3. 物理属性：定义 <mesh> 时添加 'scale'，定义 <geom> 时添加 'density' 和 'mass'（不能为 0）。
4. 将 body 名称设置为 "object"，并将其放在 worldbody 内的另一个 body 下。
5. 站点定义：添加三个必要的站点，它们应放在对象 body 外部和 worldbody 内的 body 内部。
6. 参数：参数的具体值需要根据资产的实际大小进行修改。
7. 文件：确保所有这些文件在 XML 配置中正确链接，并放置在正确的目录结构中，以保证资产正常运行。
-->
<mujoco model="apple_41">
  <asset>
    <!-- visual mesh -->
    <mesh file="visual/model_normalized_0.obj" name="model_normalized_0_vis" scale="0.06750 0.06750 0.06750" refquat="1.0 0.0 0.0 0.0" />
    ...
    <!-- collision mesh -->
    <mesh file="collision/model_normalized_collision_22.obj" name="model_normalized_collision_22._coll" scale="0.06750 0.06750 0.06750" refquat="1.0 0.0 0.0 0.0" />
    ...
    <!-- texture map -->
    <texture type="2d" name="image0" file="visual/image0.png" />
    ...
    <!-- texture material -->
    <material name="defaultMat.011" texture="image0" specular="0.5" shininess="0.25" />
    ...
  </asset>
  <worldbody>
    <body>
      <!-- body -->
      <body name="object">
        <!-- visual geom -->
        <geom solimp="0.998 0.998 0.001" solref="0.001 1" density="100" friction="0.95 0.3 0.1" type="mesh" mesh="model_normalized_0_vis" conaffinity="0" contype="0" group="1" material="defaultMat.011" />
        ...
        <!-- collision geom -->
        <geom solimp="0.998 0.998 0.001" solref="0.001 1" density="100" friction="0.95 0.3 0.1" type="mesh" mesh="model_normalized_collision_22._coll" group="0" rgba="0.8 0.8 0.8 0.0" />
        ...
      </body>
        <!-- necessary sites -->
        <site rgba="0 0 0 0" size="0.005" pos="-0.00026 -0.00018 -0.03315" name="bottom_site" />
        <site rgba="0 0 0 0" size="0.005" pos="-0.00026 -0.00018 0.03324" name="top_site" />
        <site rgba="0 0 0 0" size="0.005" pos="0.03313 0.03298 0.00004" name="horizontal_radius_site" />
    </body>
  </worldbody>
</mujoco>
```

#### 2. 注册资产
你应该在`vla_arena/vla_arena/envs/objects/`下的相应文件中注册新资产，以便 BDDL 文件解析器能够正确识别相应的对象：
```python
@register_object
class Apple(GoogleScannedObject):
    def __init__(self, name="apple", obj_name="apple"):
        super().__init__(name, obj_name, duplicate_collision_geoms=True)
        self.rotation = (np.pi/2, np.pi/2)
        self.rotation_axis = "z"
```
