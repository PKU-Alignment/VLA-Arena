# Scene Construction Guide

VLA-Arena provides a flexible method for scene construction. This guide will help you understand how to define task scenarios, manage assets, and visualize environments through the BDDL (Behavior Domain Definition Language) and supporting tools.

## Table of Contents
1. [BDDL File Structure](#1-bddl-file-structure)
2. [Visualize BDDL File](#2-visualize-bddl-file)
3. [Assets](#3-assets)
4. [Summary](#4-summary)

## 1. BDDL File Structure
The BDDL (Behavior Domain Definition Language) file is the core configuration file for defining task scenarios. It uses LISP-like syntax to describe various aspects of a task.
```lisp
(define (problem ...)           ; Task name
  (:domain robosuite)           ; Use the robosuite simulation environment
  (:language ...)               ; Language instruction
  (:regions ...)                ; Definition of regions
  (:fixtures ...)               ; Definition of fixtures
  (:objects ...)                ; Definition of manipulable objects
  (:moving_objects ...)         ; Definition of moving objects (Optional)
  (:obj_of_interest ...)        ; Definition of task-related target objects
  (:init ...)                   ; Initial states
  (:goal ...)                   ; Target states
  (:image_settings ...)         ; Image effect settings
  (:cost ...)                   ; Cost constraints (Optional)
)
```
### 1.1 Basic Structure Definition
#### Domain and Problem Definition
```lisp
define (problem Tabletop_Manipulation) ; Chosen from "Tabletop_Manipulation" and "Floor_Manipulation"
  (:domain robosuite)  ; Use the robosuite simulation environment
```
#### Language Instruction
```lisp
(:language Pick up the lemon and place it on the bowl in the table center)
```

### 1.2 Region Definition

Regions define the spatial scope where objects can be placed.
```lisp
(:regions
  (bowl_region
    (:target main_table)  ; Define which object/surface it is on
    (:ranges ((0.05 0.18 0.07 0.2)))  ; XY-plane range (x_min y_min x_max y_max)
  )
  (table_center
    (:target main_table)
    (:ranges ((-0.05 -0.01 -0.02 0.01)))
  )
)
```
#### Region Parameter Description
- `target` : The reference object where the region is located (such as a table, cabinet, etc.)
- `ranges` : The XY-plane range in the target coordinate system, formatted as `(x_min y_min x_max y_max)`
- `yaw_rotation`(Optional) : Rotation angle of the region (only valid for `fixtures`)

### 1.3 Object Definition

#### Fixtures
Objects that do not move in the environment:

```lisp
(:fixtures
  main_table - table         ; Table
  wooden_cabinet - cabinet   ; Cabinet
)
```
#### Objects
Objects that can be moved and manipulated in the task:
```lisp
(:objects
  lemon_1 - lemon           ; Lemon
  white_bowl_1 - white_bowl ; White bowl
  plate_1 - plate           ; Plate
)
```
#### Object of Interest
Objects directly related to the task:
```lisp
(:obj_of_interest
  lemon_1
  white_bowl_1
)
```

#### Moving Objects
Define objects that move autonomously in the scene, supporting multiple motion modes:

```lisp
(:moving_objects
  (dump_truck_1
    (:motion_type linear)         ; Motion type: linear
    (:motion_period 100)          ; Motion period (seconds)
    (:motion_travel_dist 1)       ; Travel distance (meters)
    (:motion_direction (0 1 0))   ; Motion direction vector
  )
)
```
##### Supported Motion Types

**1. Linear Motion (`linear`)**
```lisp
(:motion_type linear)
(:motion_period 100)          ; Round-trip period
(:motion_travel_dist 1)       ; One-way distance
(:motion_direction (0 1 0))   ; Direction vector
```
**2. Circular Motion (`circle`)**
```lisp
(:motion_type circle)
(:motion_center (0 0 1.2))    ; Center position
(:motion_period 1)            ; Time to complete one circle
```
**3. Waypoint Motion (`waypoint`)**
```lisp
(:motion_type waypoint)
(:motion_waypoints ((0 0 1 (1 0 0))
                    (1 0 1 (0 1 0))
                    (1 1 1 (1 0 0)))
)  ; Waypoint list (waypoint: (x y z (x_dir y_dir z_dir)))
(:motion_dt 0.01)             ; Time step
(:motion_loop true)           ; Whether to loop
```
**4. Parabolic Motion (`parabolic`)**
```lisp
(:motion_type parabolic)
(:motion_initial_speed 1)     ; Initial speed
(:motion_direction (0 1 0))   ; Initial direction
(:motion_gravity (0 0 -9.81)) ; Gravity vector
```
### 1.4 State Definition

#### Initial State
Defines the initial configuration of the scene:

```lisp
(:init
  (On lemon_1 main_table_lemon_region)  ; The lemon is in the specified region
  (On white_bowl_1 main_table_center)   ; The bowl is at the center of the table
  (Open wooden_cabinet_top_region)      ; The top drawer of the cabinet is open
  (Close wooden_cabinet_bottom_region)  ; The bottom drawer of the cabinet is closed
)
```
##### Supported State Predicates
- `On`: An object is on a certain region/object.
- `In`: An object is inside a certain container.
- `Open/Close`: The state of objects that can be opened or closed.
- `Turnon/Turnoff`: The state of devices that can be turned on or off.

#### Goal State
Defines the task completion conditions:
```lisp
(:goal
  (And
    (On lemon_1 white_bowl_1)           ; The lemon is on the bowl
    (On white_bowl_1 main_table_center) ; The bowl is at the center of the table
  )
)
```
##### Supported Logical Connectives:
- `And`: All conditions are satisfied.
- `Or`: At least one condition is satisfied.
- `Not`: The condition is not satisfied.

### 1.5 Image Effect
Set the rendering effect of the scene image:
```lisp
(:image_settings
  brightness 0.4       ; Adjust image brightness (range: -1.0 to 1.0)
  saturation -0.4      ; Adjust color saturation (range: -1.0 to 1.0)
  contrast 0.3         ; Adjust image contrast (range: -1.0 to 1.0)
  temperature 5000     ; Adjust color temperature (range: 2000-10000)
)
```
##### Supported Effect Options:
- `brightness`: Controls overall lightness/darkness.
- `saturation`: Adjusts color intensity.
- `contrast`: Modifies difference between light and dark areas.
- `temperature`: Controls the color tone of the rendered image.

### 1.6 Cost Constraints

Define penalty conditions during task execution:
```lisp
(:cost
  (And
    (InContact lemon_1 white_bowl_1)       ; The lemon touches the bowl
    (CheckForce dump_truck_1 10)           ; Force exceeds the threshold
    (CheckDistance lemon_1 white_bowl_1 0.1) ; Distance is less than the threshold
  )
)
```
##### Supported cost predicates:
- `InContact`: Contact between objects.
- `InContactPart`: Contact between specific parts of objects.
- `CheckForce`: Check contact force.
- `CheckDistance`: Check distance.
- `CheckGripperDistance`: Check distance between the gripper and the object.
- `CheckGripperContact`: Check gripper contact.
- `CheckGripperContactPart`: Check contact between the gripper and specific parts of an object.
- `Collide`: Collision detection.
- `Fall`: Object falls.

## 2. Visualize BDDL File
We provide `scripts/visualize_bddl.py` to generate a video of the scene corresponding to the BDDL file. Add the command-line argument `--bddl_file` to specify the path of the BDDL file (or a directory containing BDDL files):
```bash
python scripts/visualize_bddl.py --bddl_file your_bddl_file_path
```
Here is an example:
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
Then view the generated video in the `rollouts` directory:
<p align="center"><img src="image/build_scene_1.png" width="300" height="300"/></p>

### Trouble Shooting
If you encounter error AttributeError: "'MjRenderContextOffscreen' object has no attribute 'con'" during visualization, please try installing the following package:
```bash
conda install -c conda-forge libegl-devel
```

## 3. Assets
Both fixtures and objects in the BDDL file must be existing assets in the `vla_arena/vla_arena/assets` directory. This directory serves as the central repository for all usable assets within the scene.

### Ready-made Assets
We have provided a rich collection of pre-made assets organized in `articulated_objects`, `stable_hoped_objects`, `stable_scanned_objects` and `turbosquid_objects`. These ready-to-use assets cover a wide range of common objects and fixtures, allowing you to directly integrate them into your scenes without additional setup.

### Your Own Assets
You can also use your own assets to customize your scenes:
#### 1. Prepare Valid Assets

The assets you intend to use must include the following components:
1. **OBJ file** - This 3D model file format contains the geometric data (vertices, edges, faces, etc.) of the asset, defining its basic shape and structure.

2. **Texture maps (PNG files)** - These image files provide surface details for the 3D model, including colors, patterns, and textures that enhance the visual realism of the asset.

3. **XML file** - This configuration file describes essential properties of the asset such as physical parameters (mass, friction), collision settings, and references to associated files (OBJ and texture paths). A valid XML file should be like:
```xml
<!-- Important Notes:
1. Naming: Add a name attribute to the mesh definition (consistent with that in <worldbody>).
2. Group modification:
   - Group of visual geometry (vision) → 1
   - Group of collision geometry (collision) → 0
3. Physical properties: Add 'scale' when defining <mesh>, add 'density' and 'mass'(cannot be 0) when defining <geom>.
4. Set the body name to "object" and place it under another body within worldbody.
5. Site definition: Add three necessary sites, which should be placed outside the object body and inside the body under worldbody.
6. Parameters: The specific values of parameters need to be modified according to the actual size of the asset.
7. Files: Ensure all these files are properly linked within the XML configuration and placed in the correct directory structure to guarantee the asset functions as expected.
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

#### 2. Register Assets
You should register the new asset in the corresponding file under `vla_arena/vla_arena/envs/objects/` so that the BDDL file parser can correctly recognize the corresponding object:
```python
@register_object
class Apple(GoogleScannedObject):
    def __init__(self, name="apple", obj_name="apple"):
        super().__init__(name, obj_name, duplicate_collision_geoms=True)
        self.rotation = (np.pi/2, np.pi/2)
        self.rotation_axis = "z"
```
