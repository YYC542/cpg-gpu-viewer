# Author:Ach1n
# Data:2026/3/28
# Flag:I earned it.
# ============================================================================
# 文件名: Norne_viewer_VR_V10.py
# 作者: Ach1n (Yang Yuchen)
# 日期: 2026/3/28
# 版本: v10 - VR沉浸式版本
# ============================================================================
#
# ▌总体说明:
# ▌本程序是一个基于 Modern OpenGL 3.3 Core Profile + 几何着色器 (Geometry Shader)
# ▌的 石油油藏三维可视化工具，在V9基础上新增SteamVR/OpenVR头显支持。
# ▌它读取 Norne 油田的 GRDECL 格式角点网格数据，
# ▌将约 11.3 万个活跃网格单元实时渲染为三维体积模型，
# ▌并支持在VR头显（如Lenovo Explorer）中进行沉浸式油藏探索。
# ▌
# ▌V10 版本的新增功能（相比 V9）：
# ▌  1) SteamVR/OpenVR 头显支持 (通过openvr库)
# ▌  2) 双眼立体渲染 (左右眼各一套MVP矩阵，提供真实3D视差)
# ▌  3) 头部姿态追踪 (头显方向控制相机视角)
# ▌  4) 自动降级模式 (头显不可用时自动回退到普通窗口模式)
# ▌  5) VR/普通模式实时切换 (V键切换)
# ▌
# ▌V11 版本的新增功能（相比 V10）：
# ▌  1) 近端内表面封闭渲染 (解决进入油藏后看到空洞的问题)
# ▌     — 几何着色器新增 emitQuadInterior()，翻转绕序发射内表面
# ▌     — N/B 键调节封闭半径，半径内的单元自动切换为内表面渲染
# ▌     — 无需修改 OpenGL 状态机，零额外 draw call，性能无损
# ▌  2) VR 近裁剪面修正 (near plane: sz*0.001 → sz*0.00005)
# ▌     — 消除 VR 近距离穿模和视觉畸变问题
# ▌     — OpenVR getProjectionMatrix() 使用修正后的 near/far 值
# ▌
# ▌VR渲染架构:
# ▌  头显每帧提供左右眼投影矩阵 + 头部姿态矩阵
# ▌  → 合成左右眼MVP → 分别渲染到左右眼FBO
# ▌  → 提交给SteamVR compositor显示到头显
# ▌  → 同时在桌面窗口显示左眼画面（镜像模式）
# ▌
# ▌V9 保留功能：
# ▌  1) 断层检测与高亮显示 (F 键切换，白色 = 断层区域)
# ▌  2) 井轨迹可视化 (W 键切换，显示 Norne 油田公开井位)
# ▌  3) 屏幕上的颜色图例条 (Colorbar legend)
# ▌  4) 三点光照模型
# ▌  5) 双面法线处理
# ▌  6) 多边形偏移减少 z-fighting
# ▌  7) 背面剔除提升渲染效率
# ▌
# ▌核心架构思想——"一点膨胀"(Point-to-Hexahedron Expansion):
# ▌  CPU 端为每个网格单元只提交一个 GL_POINTS 点，
# ▌  几何着色器在 GPU 端将该点"膨胀"为一个完整的六面体（6个面×4个顶点 = 24顶点），
# ▌  所有切片裁剪、垂直放大、法线计算都在 GPU 端完成，
# ▌  从而避免了 CPU 端生成海量三角形的性能瓶颈。
#
# ============================================================================
# ▌功能特性总列表:
# ▌  - 全 3D 体积渲染（所有活跃网格单元，非仅表面）
# ▌  - 彩虹深度色图 (蓝=浅层, 红=深层)
# ▌  - 双轴切片: X 轴 (Q/A 键) 和 Z 轴深度 (E/D 键)
# ▌  - 垂直夸大 (Z/X 键调节)
# ▌  - 断层检测 & 高亮 (F 键切换)
# ▌  - 井轨迹显示 (W 键切换)
# ▌  - 屏幕颜色图例条
# ▌  - 三点光照 + 双面法线
# ▌  - 多边形偏移减少 z-fighting
# ============================================================================

# ============================================================================
# 第一部分: 导入依赖库
# ============================================================================

import glfw
# ▌glfw (Graphics Library Framework):
# ▌  一个跨平台的窗口管理库，负责:
# ▌  1) 创建 OpenGL 上下文窗口
# ▌  2) 处理键盘/鼠标输入事件
# ▌  3) 管理渲染循环的帧刷新 (swap buffers / poll events)
# ▌  对比: 类似 GLUT/SDL 的功能，但更现代化，是当前 OpenGL 开发的主流选择。

import ctypes
# ▌ctypes: Python 标准库，用于与 C 语言数据类型交互。
# ▌  在本程序中主要用途:
# ▌  ctypes.c_void_p(offset) —— 为 glVertexAttribPointer 提供字节偏移量指针。
# ▌  OpenGL 的很多函数参数需要 C 风格的指针类型，ctypes 在这里起桥梁作用。

from OpenGL.GL import *
# ▌PyOpenGL: Python 对 OpenGL API 的封装。
# ▌  "from OpenGL.GL import *" 导入了所有 OpenGL 核心函数和常量，例如:
# ▌  - glCreateShader, glCompileShader, glLinkProgram 等着色器操作
# ▌  - glGenBuffers, glBindBuffer, glBufferData 等缓冲区操作
# ▌  - glDrawArrays, glClear 等绘制命令
# ▌  - GL_VERTEX_SHADER, GL_FLOAT, GL_TRIANGLES 等常量枚举
# ▌  注意: 这里导入的是 OpenGL 3.3 Core Profile 的子集。

import numpy as np
# ▌NumPy: Python 科学计算核心库。
# ▌  在本程序中的用途:
# ▌  1) 矩阵运算: 构建 MVP(Model-View-Projection) 矩阵，用 4x4 的 np.array 表示
# ▌  2) 数据存储: 将网格单元的角点坐标、深度值等存入 np.float32 数组
# ▌  3) 向量运算: 叉积、归一化、点积等 (np.cross, np.linalg.norm, np.dot)
# ▌  4) 三角函数: np.sin, np.cos, np.radians 用于摄像机轨道计算
# ▌  5) 数据转换: 将 Python list 高效转换为 GPU 可接受的连续内存数组

import os
# ▌os: Python 标准库，文件系统操作。
# ▌  本程序仅用 os.path.exists() 检查 GRDECL 文件是否存在。

# -------- OpenVR / SteamVR (可选，头显不可用时自动降级) --------
# ════════════════════════════════════════════════════════════════
# V17 (本版本) 相对 V16 的关键修复:
#   ★ 修复"每次运行模型位置都不一样、模型出现在头顶"的 bug ★
#
#   根因: V16 直接把 SteamVR 追踪空间的绝对坐标 (含头部高度Y≈1.6米)
#         乘以 VR_WORLD_SCALE=20000，导致用户被传送到模型上空 32 公里。
#         而且每次启动追踪空间的绝对原点都有漂移，所以每次结果不同。
#
#   修复: 首帧记录头显的初始 pose (hmd_init_inv)，
#         之后每帧用 T_relative = hmd_init_inv @ hmd_current 算出
#         "你相对启动那一刻移动了多少"。这样:
#         - 无论你戴头显时站房间哪里，模型一定在你正前方 8000 米处
#         - 无论你启动时朝向哪里，模型一定在你视线正中央
#         - 每次运行的初始状态完全一致
#
#   附加: 重新开启调试输出 (每 60 帧打印一次位置)，方便排查
#         按 R 键可以随时重新锚定初始位置（站歪了重置用）
# ════════════════════════════════════════════════════════════════

try:
    import openvr
    OPENVR_AVAILABLE = True
    # ▌openvr: Python绑定的OpenVR API，用于与SteamVR通信。
    # ▌  openvr.init() 初始化VR系统
    # ▌  openvr.IVRSystem 提供头部姿态追踪
    # ▌  openvr.IVRCompositor 负责将渲染结果提交到头显
except ImportError:
    OPENVR_AVAILABLE = False
    print("[VR] openvr库未安装，将以普通窗口模式运行")


# ============================================================================
# 第二部分: 全局配置参数
# ============================================================================

# ------- 文件与窗口 -------
FILENAME = "NORNE_ATW2013.GRDECL"
# ▌要加载的 GRDECL 格式油藏数据文件名。
# ▌GRDECL 是 Schlumberger ECLIPSE 油藏模拟器定义的文本格式，
# ▌包含角点网格 (Corner-Point Grid) 的几何描述信息。
# ▌Norne ATW2013 是 Statoil（现 Equinor）公开发布的基准数据集，
# ▌网格维度为 46×112×22 (i×j×k)，约 11.3 万个活跃单元。

WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 720
# ▌渲染窗口的像素分辨率: 1280×720 (720p HD)。
# ▌宽高比 = 1280/720 ≈ 1.778，这个值会传入透视投影矩阵。

# ------- 摄像机参数 -------
cam_dist  = 20000.0
# ▌摄像机到场景原点的距离（初始值，后面会被 sz*2.0 覆盖）。
# ▌单位与数据坐标一致（米）。这个值越大，看到的模型越小/越远。

cam_yaw   = 30.0
# ▌偏航角 (Yaw): 摄像机绕 Y 轴的水平旋转角度，单位: 度。
# ▌0° = 正对 Z 轴正方向，30° = 略偏向右侧。
# ▌设为 30° 是为了启动时就能看到截面切口的内部结构，而不是正面平视。

cam_pitch = 30.0
# ▌俯仰角 (Pitch): 摄像机的上下倾斜角度，单位: 度。
# ▌0° = 水平平视，正值 = 从上方俯视。
# ▌30° 让你同时能看到模型的顶面和侧面，产生良好的 3D 透视感。
# ▌在渲染循环中会被限制在 [-89°, +89°] 范围内，防止万向锁 (gimbal lock)。

last_x = last_y = 0
# ▌记录鼠标上一帧的屏幕坐标 (像素)。
# ▌用于计算鼠标拖拽的位移量 delta_x = mx - last_x, delta_y = my - last_y，
# ▌从而转换为摄像机旋转角度的增量。

mouse_pressed = False
# ▌鼠标左键是否处于按下状态的标志。
# ▌True = 用户正在拖拽旋转摄像机；
# ▌False = 鼠标释放，不进行旋转。
# ▌用于在首次按下时记录起始位置 (last_x, last_y)，避免跳动。

# ------- 切片与显示控制 -------
slice_x     = 1.0
# ▌X 轴切面位置，范围 [0.0, 1.0]。
# ▌1.0 = 不切割（显示全部），0.5 = 只显示 X 坐标前 50% 的网格单元。
# ▌V7 默认是 0.6（启动即显示内部），V8 改为 1.0（启动时显示完整模型）。
# ▌用 Q 键减小（切掉更多）、A 键增大（恢复显示）。

slice_z     = 1.0
# ▌Z 轴（深度方向）切面位置，范围 [0.0, 1.0]。
# ▌1.0 = 不切割，0.5 = 只显示 Z 坐标前 50%（即较浅的层）。
# ▌用 E 键减小、D 键增大。

vexag       = 4.0
# ▌垂直夸大系数 (Vertical Exaggeration)。
# ▌油藏在水平方向可能跨越数公里，但垂直厚度只有几十到一百多米，
# ▌如果按真实比例渲染，模型看起来会像一张极薄的纸片。
# ▌vexag = 4.0 表示 Z 轴坐标被放大 4 倍，使地层结构清晰可见。
# ▌V7 默认 10x，V8 改为 4x（更接近真实比例，配合背面剔除效果更好）。
# ▌用 Z 键减小 (最小 1.0 = 真实比例)，X 键增大 (最大 50.0)。

show_faults = True
# ▌是否显示断层高亮，True = 开启（断层区域叠加白色），False = 关闭。
# ▌V8 新增功能。按 F 键切换。

show_wells  = True
# ▌是否显示井轨迹，True = 显示，False = 隐藏。
# ▌V8 新增功能。按 W 键切换。

# ------- VR 相关参数 (V10 新增) -------
vr_mode     = True  # V16: 默认尝试进入VR模式，启动时自动检测头显
# ▌是否启用VR模式，True = 尝试连接头显，False = 普通窗口模式。
# ▌V16 改动：启动时自动检测 OpenVR + 头显是否可用：
# ▌  - 如果 SteamVR 已启动且头显已连接 → 直接进入 VR 模式
# ▌  - 如果未检测到头显 → 自动降级为桌面模式
# ▌运行时按 V 键可手动在两种模式之间切换。

VR_EYE_SEP  = 0.063
# ▌两眼间距 (Inter-Pupillary Distance, IPD)，单位: 米。
# ▌标准值约63mm。实际值由openvr从头显获取，此为备用默认值。

vr_system   = None
# ▌openvr VR系统对象，初始化后赋值。

vr_compositor = None
# ▌openvr compositor对象，用于提交渲染帧给头显。

# ------- Proximity Culling 参数 (V10 新增) -------
proximity_radius = 0
shell_radius = 1500.0  # 球壳变形半径（米）- 自动激活，无需按键

# ════════════════════════════════════════════════════════════════════
# V14 新增: VR 6DoF 配置常量
# ════════════════════════════════════════════════════════════════════
# 模型固定在世界原点 (0,0,0)，用户初始位置在模型前方 VR_USER_OFFSET 米处。
# VR_WORLD_SCALE 控制头显的物理位移如何映射到虚拟世界:
#   用户在房间走 1 米 → 虚拟世界中走 VR_WORLD_SCALE 米。
# 因为油藏模型尺度约 9000 米，需要较大缩放让用户能用脚步穿越模型。
VR_USER_OFFSET = 8000.0   # 用户初始距模型中心的距离（米），模型半径≈7642 → 起点在模型边缘外
VR_WORLD_SCALE = 3000.0   # 修改: 20000→3000。配合颈部补偿，走0.5-1米进入模型，转头时模型基本不动
                          # 走进距离 = (8000-7642)/3000 ≈ 0.12米起就开始进入，走~0.5米深入内部
# ▌近距离裁剪半径，世界坐标单位。
# ▌0.0 = 关闭（不裁剪任何单元）。
# ▌按 N 键增大（裁剪更大范围），B 键减小。
# ▌在VR模式下当你"走进"模型时，增大此值可看到内部结构。


# ============================================================================
# 第三部分: Norne 油田井位坐标数据
# ============================================================================

NORNE_WELLS = {
    "C-1H":  [(456478, 7321504, 2578), (456502, 7321598, 2700)],
    "C-2H":  [(456823, 7321210, 2580), (456901, 7321350, 2710)],
    "C-3H":  [(456234, 7321890, 2575), (456180, 7321760, 2690)],
    "C-4AH": [(456650, 7321650, 2582), (456720, 7321800, 2720)],
    "D-1H":  [(457100, 7322100, 2590), (457200, 7322250, 2730)],
    "D-2H":  [(457300, 7321950, 2588), (457380, 7322080, 2715)],
    "E-1H":  [(457650, 7322400, 2595), (457700, 7322550, 2740)],
    "E-2H":  [(457450, 7322600, 2592), (457520, 7322750, 2725)],
    "E-3H":  [(457850, 7322200, 2598), (457920, 7322350, 2745)],
    "B-2H":  [(456100, 7320800, 2570), (456050, 7320650, 2680)],
    "B-4H":  [(455900, 7321100, 2568), (455850, 7320950, 2675)],
}
# ▌Norne 油田的公开井位坐标字典。
# ▌数据来源: Norne 基准数据集 (SPE/Statoil public release)。
# ▌坐标系: UTM Zone 32N (北欧地区通用的投影坐标系)。
# ▌
# ▌字典结构:
# ▌  键 (key) = 井名 (string)，例如 "C-1H", "D-2H" 等。
# ▌    命名规则: 字母表示井组 (C, D, E, B)，数字 + H 表示水平井编号。
# ▌  值 (value) = 包含若干三维坐标点的列表 [(x, y, z), (x, y, z), ...]。
# ▌    每个元组 = 井轨迹上的一个控制点:
# ▌    x ≈ 455000~458000 —— 东向坐标 (Easting, 单位: 米)
# ▌    y ≈ 7320000~7323000 —— 北向坐标 (Northing, 单位: 米)
# ▌    z ≈ 2568~2745 —— 垂直深度 (True Vertical Depth, 单位: 米)
# ▌
# ▌每口井只有 2 个控制点 (起点和终点)，
# ▌渲染时用 GL_LINE_STRIP 连成直线段表示简化的井轨迹。
# ▌实际井轨迹可能是弯曲的 (水平井)，但这里是简化表示。
# ▌
# ▌注意: 这些坐标是原始 UTM 坐标，在渲染前会减去 origin 做归一化，
# ▌与油藏网格使用相同的坐标变换。


# ============================================================================
# 第四部分: GLSL 着色器源代码
# ============================================================================

# ============================================================================
# 4.1 油藏网格顶点着色器 (Reservoir Vertex Shader)
# ============================================================================

VERTEX_SHADER_SRC = """
#version 330 core
// ▌声明使用 GLSL 3.30 版本，对应 OpenGL 3.3 Core Profile。
// ▌"core" 表示不允许使用已弃用的固定管线功能。

layout(location = 0) in float aDepth;
// ▌顶点属性输入 (Vertex Attribute Input):
// ▌  layout(location = 0) —— 绑定到属性位置 0 (对应 glVertexAttribPointer 的第一个参数)
// ▌  in float aDepth —— 每个顶点接收一个浮点数，表示该网格单元的平均 Z 深度值。
// ▌
// ▌重要: 在本程序的架构中，每个"顶点"实际代表一个完整的网格单元 (cell)，
// ▌而不是传统意义上的三角形顶点。真正的几何体在几何着色器中生成。

out float vDepth;
// ▌顶点着色器的输出变量，传递给下一阶段 (几何着色器)。
// ▌"out" 表示这个变量会被管线中的下一个着色器读取。

void main() {
    vDepth = aDepth;
    // ▌原样传递深度值给几何着色器，不做任何变换。

    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
    // ▌将顶点位置设为原点 (0,0,0)。
    // ▌这看似无意义，但关键在于: 我们根本不需要顶点着色器计算位置，
    // ▌因为真正的 8 个角点坐标存在纹理缓冲 (TBO) 里，
    // ▌将由几何着色器通过 texelFetch() 读取并处理。
    // ▌顶点着色器在这里只是充当一个"传话人"和"触发器"。
    // ▌vec4(x, y, z, w) 中 w=1.0 是齐次坐标的标准写法。
}
"""

# ============================================================================
# 4.2 油藏网格几何着色器 (Reservoir Geometry Shader) —— 核心渲染引擎
# ============================================================================

GEOMETRY_SHADER_SRC = """
#version 330 core
// ▌GLSL 3.30, Core Profile

layout(points) in;
// ▌输入图元类型: 点 (GL_POINTS)。
// ▌每次执行 main()，几何着色器会收到来自顶点着色器的 1 个点。
// ▌这个"点"在逻辑上代表 1 个网格单元。

layout(triangle_strip, max_vertices = 48) out;
// ▌V11 最终版: 同时发射外表面+内表面，确保模型从任何视角看都是封闭实体
// ▌6面 × 4顶点 × 2(内+外) = 48 顶点
// ▌输出图元类型: 三角带 (triangle_strip)。
// ▌max_vertices = 24: 最多输出 24 个顶点。
// ▌为什么是 24? 一个六面体有 6 个面，每个面用一个 4 顶点的 triangle_strip 绘制:
// ▌  三角带 (A,B,C,D) 会自动生成 2 个三角形: ABC 和 BCD。
// ▌  6 面 × 4 顶点 = 24 顶点。
// ▌每次 EndPrimitive() 结束一个三角带，所以实际输出 6 个独立的四边形面。

in  float vDepth[];
// ▌从顶点着色器接收的深度值数组。
// ▌因为输入是 "points" (1 个顶点)，所以 vDepth[0] 就是这个网格单元的深度。
// ▌在几何着色器中，所有 "in" 变量都是数组形式 (即使只有 1 个元素)。

out float fDepth;
// ▌输出给片段着色器的深度值，用于颜色映射。
// ▌"f" 前缀表示 "fragment"。

out vec3  fNormal;
// ▌输出给片段着色器的法线向量，用于光照计算。

out float fFault;
// ▌输出给片段着色器的断层标志 (0.0 或 1.0)。
// ▌V8 新增: 如果该网格单元位于断层区域，fFault = 1.0，否则 = 0.0。

uniform mat4  uMVP;
// ▌Model-View-Projection 矩阵 (4×4)。
// ▌将三维世界坐标变换到裁剪空间 (clip space) 坐标。
// ▌"uniform" = 在一次 draw call 中对所有顶点保持不变的值，由 CPU 每帧传入。
// ▌MVP = Projection × View × Model (本程序 Model = 单位矩阵，所以 MVP = P × V)。

uniform float uSliceX;
// ▌X 轴切片阈值 (世界坐标，不是归一化值)。
// ▌如果网格单元的中心 X 坐标超过此值，该单元不会被渲染 (被"切掉")。

uniform float uSliceZ;
// ▌Z 轴切片阈值 (世界坐标)。
// ▌同理: 中心 Z 坐标超过此值的单元被切掉。

uniform float uVExag;
// ▌垂直夸大系数。Z 坐标会乘以此值来放大垂直方向的尺度。
// ▌在 vex() 函数中使用。

uniform samplerBuffer uCorners;
// ▌纹理缓冲对象 (Texture Buffer Object, TBO) 采样器。
// ▌存储所有网格单元的角点坐标数据:
// ▌  每个单元 8 个角点 × 3 个坐标 (x,y,z) = 24 个 float32。
// ▌  总大小 = 网格单元数 × 24 × 4 字节。
// ▌
// ▌为什么用 TBO 而不是普通 Vertex Buffer?
// ▌  普通 VBO 的 attribute 只能按顶点线性传递数据，
// ▌  但这里每个"顶点"(=网格单元) 需要访问 24 个值（8个角点×3坐标），
// ▌  而 vertex attribute 每个顶点最多传 4 个 float (vec4)。
// ▌  TBO 允许在着色器中通过 texelFetch(sampler, index) 随机访问任意位置的数据，
// ▌  非常适合这种"每个单元需要大量额外数据"的场景。
// ▌
// ▌数据布局 (Memory Layout):
// ▌  [cell_0_corner_0_x, cell_0_corner_0_y, cell_0_corner_0_z,
// ▌   cell_0_corner_1_x, cell_0_corner_1_y, cell_0_corner_1_z,
// ▌   ... (共8个角点)
// ▌   cell_1_corner_0_x, cell_1_corner_0_y, cell_1_corner_0_z,
// ▌   ...]

uniform samplerBuffer uFaultFlag;
// ▌断层标志的纹理缓冲采样器。V8 新增。
// ▌每个网格单元 1 个 float 值: 1.0 = 断层, 0.0 = 正常。
// ▌通过 texelFetch(uFaultFlag, cell) 在几何着色器中读取。

uniform vec3  uCamPos;
// ▌摄像机在世界坐标系中的位置（VR模式下为头显实际位置）。
// ▌用于 proximity culling: 距离摄像机太近的单元自动隐藏，
// ▌产生"进入油藏内部"时周围地层消失的沉浸式效果。

uniform float uProximityRadius;
uniform float uShellRadius;  // V11最终版: 球壳半径，用于观察者周围封闭壳生成
// ▌proximity culling 的半径阈值（世界坐标单位）。
// ▌距摄像机小于此距离的网格单元不渲染。按 B/N 键实时调节。

vec3 getCorner(int cell, int ci) {
// ▌从 TBO 中读取指定网格单元 (cell) 的第 ci 个角点 (0~7) 的三维坐标。
// ▌参数:
// ▌  cell: 网格单元的全局索引 (0, 1, 2, ..., ncells-1)
// ▌  ci:   角点编号 (0~7), 对应六面体的 8 个顶点
// ▌返回: vec3(x, y, z) 世界坐标 (已经过原点归一化)

    int base = cell * 24 + ci * 3;
    // ▌计算该角点在 TBO 中的起始索引:
    // ▌  cell * 24: 跳过前面所有单元 (每个单元 8 角点 × 3 坐标 = 24 个 float)
    // ▌  ci * 3:    在当前单元内跳到第 ci 个角点 (每个角点 3 个坐标)
    // ▌  例如: cell=5, ci=2 → base = 5*24 + 2*3 = 126

    return vec3(
        texelFetch(uCorners, base    ).r,
        texelFetch(uCorners, base + 1).r,
        texelFetch(uCorners, base + 2).r
    );
    // ▌texelFetch(sampler, index): 从纹理缓冲中读取第 index 个纹素 (texel)。
    // ▌因为 TBO 格式是 GL_R32F (每个纹素 = 1 个 float32, 存在红色通道 .r 中)，
    // ▌所以连续读取 3 个纹素就得到一个角点的 (x, y, z)。
    // ▌
    // ▌texelFetch 是 GLSL 内置函数，不涉及纹理过滤或 mipmap，直接按整数索引精确读取。
}

vec3 vex(vec3 p) {
// ▌垂直夸大函数 (Vertical Exaggeration)。
// ▌只对 Z 轴坐标乘以夸大系数，X 和 Y 保持不变。
// ▌参数: p = 原始世界坐标
// ▌返回: Z 轴被放大后的坐标

    return vec3(p.x, p.y, p.z * uVExag);
    // ▌p.z * uVExag: 垂直放大。
    // ▌例如 uVExag = 4.0 时，Z=100 的点会变成 Z=400，
    // ▌使原本很薄的地层在视觉上变得更厚更清晰。
    // ▌
    // ▌为什么在 GPU 端做而不是 CPU 端?
    // ▌  1) CPU 端修改数据需要重新上传 TBO，非常慢
    // ▌  2) GPU 端只需改一个 uniform 值，下一帧立即生效
    // ▌  3) 用户按 Z/X 键实时调节时，效果是即时流畅的
}

// GRDECL corner layout (ECLIPSE 角点编号约定):
// ▌角点编号与网格单元 (i, j, k) 坐标的对应关系:
// ▌
// ▌  c0 = (i,   j,   k_top)    c1 = (i+1, j,   k_top)
// ▌  c2 = (i,   j+1, k_top)    c3 = (i+1, j+1, k_top)
// ▌  c4 = (i,   j,   k_bot)    c5 = (i+1, j,   k_bot)
// ▌  c6 = (i,   j+1, k_bot)    c7 = (i+1, j+1, k_bot)
// ▌
// ▌其中 _top = 单元的顶面 (k层的上边界), _bot = 底面 (k层的下边界)。
// ▌在油藏坐标系中，k 增大通常表示更深 (Z值更大)。
// ▌
// ▌可视化 (俯视图, 看顶面):
// ▌       c2 -------- c3
// ▌      / |         / |
// ▌    c0 -------- c1  |     ← 顶面
// ▌     |  c6 -----|-- c7
// ▌     | /        | /
// ▌    c4 -------- c5        ← 底面

void emitQuad(vec3 a, vec3 b, vec3 c, vec3 d, float depth, float fault) {
// ▌发射一个四边形面 (Quad) 到管线中。
// ▌参数:
// ▌  a, b, c, d: 四边形的 4 个顶点 (世界坐标，未经垂直夸大)
// ▌  depth: 该网格单元的深度值 (用于片段着色器中的颜色映射)
// ▌  fault: 断层标志 (0.0 或 1.0)
// ▌
// ▌输出: 一个 triangle_strip 图元 (4个顶点 → 2个三角形 → 1个四边形面)

    vec3 ae=vex(a), be=vex(b), ce=vex(c), de=vex(d);
    // ▌对四个顶点应用垂直夸大。
    // ▌重要: 在计算法线之前做夸大，这样法线是在"夸大后的空间"中计算的，
    // ▌光照效果才能正确反映放大后的几何形状。

    vec3 n = normalize(cross(be - ae, ce - ae));
    // ▌计算法线向量:
    // ▌  cross(v1, v2): 两个边向量的叉积，得到垂直于该面的向量
    // ▌  normalize(): 归一化为单位向量 (长度=1)
    // ▌
    // ▌具体来说:
    // ▌  be - ae = 从 a 指向 b 的边向量
    // ▌  ce - ae = 从 a 指向 c 的边向量
    // ▌  cross(边1, 边2) = 垂直于这两条边所在平面的向量
    // ▌
    // ▌注意 V8 对比 V7 的变化:
    // ▌  V7 用 cross(be-ae, de-ae)，V8 改用 cross(be-ae, ce-ae)。
    // ▌  这是因为 V8 调整了 emitQuad 中的顶点顺序以配合背面剔除。

    fNormal = n; fDepth = depth; fFault = fault;
    // ▌将法线、深度和断层标志传递给片段着色器。
    // ▌这三个值在整个四边形面上会被自动插值 (varying interpolation)，
    // ▌但因为同一个面上这些值相同，所以插值结果就是原值。

    gl_Position = uMVP * vec4(ae, 1.0); EmitVertex();
    // ▌将第 1 个顶点从世界坐标变换到裁剪空间 (clip space)，然后发射。
    // ▌uMVP * vec4(ae, 1.0): 4×4矩阵乘以齐次坐标 (w=1.0)
    // ▌EmitVertex(): GLSL 几何着色器内置函数，将当前设置的所有 out 变量
    // ▌(gl_Position, fNormal, fDepth, fFault) 打包为一个顶点输出。

    gl_Position = uMVP * vec4(be, 1.0); EmitVertex();
    // ▌发射第 2 个顶点。

    gl_Position = uMVP * vec4(ce, 1.0); EmitVertex();
    // ▌发射第 3 个顶点。此时 triangle_strip 已经可以形成第一个三角形 (a,b,c)。

    gl_Position = uMVP * vec4(de, 1.0); EmitVertex();
    // ▌发射第 4 个顶点。triangle_strip 自动形成第二个三角形 (b,c,d)。
    // ▌于是 4 个顶点产生 2 个三角形，拼合成一个四边形。

    EndPrimitive();
    // ▌结束当前三角带。后续的 EmitVertex() 会开始新的三角带。
    // ▌这确保了 6 个面各自独立，不会互相连接。
}

// ============================================================
// emitQuadInterior: 发射内表面四边形（封闭视觉空洞用）
// ============================================================
// 原理: 将顶点发射顺序翻转 (a,b,c,d → a,c,b,d)，
//       使三角形绕序从 CCW 变为 CW。
//       配合 glFrontFace(GL_CCW)：
//         - 翻转后的面从外部看是 CW（背面）→ 被剔除，外部看不见
//         - 从内部看是 CCW（正面）→ 正常渲染，内部可见
//       效果: 观察者进入油藏内部时，看到的是单元的内壁，
//             而不是空洞，形成视觉上封闭的实体感。
void emitQuadInterior(vec3 a, vec3 b, vec3 c, vec3 d, float depth, float fault) {
    vec3 ae=vex(a), be=vex(b), ce=vex(c), de=vex(d);

    // 法线方向取反（内表面法线朝内）
    vec3 n = normalize(cross(ce - ae, be - ae));
    fNormal = n; fDepth = depth; fFault = fault;

    // 翻转顶点顺序: a,c,b,d（将 CCW 外表面翻转为 CW）
    // triangle_strip(a,c,b,d) → 三角形 △acb 和 △cbd
    // 从内部看这两个三角形均为 CCW → 正面朝内 → 渲染可见
    gl_Position = uMVP * vec4(ae, 1.0); EmitVertex();
    gl_Position = uMVP * vec4(ce, 1.0); EmitVertex();
    gl_Position = uMVP * vec4(be, 1.0); EmitVertex();
    gl_Position = uMVP * vec4(de, 1.0); EmitVertex();
    EndPrimitive();
}

void main() {
// ▌几何着色器的入口函数。
// ▌对每个输入点 (= 每个网格单元) 执行一次。

    int cell = gl_PrimitiveIDIn;
    // ▌gl_PrimitiveIDIn: GLSL 内置变量，表示当前处理的图元在 draw call 中的编号。
    // ▌因为我们用 glDrawArrays(GL_POINTS, 0, ncells)，
    // ▌第 0 个点 → cell=0, 第 1 个点 → cell=1, ..., 以此类推。
    // ▌这个 cell 编号用来从 TBO 中读取对应单元的角点坐标。

    vec3 c0=getCorner(cell,0); vec3 c1=getCorner(cell,1);
    vec3 c2=getCorner(cell,2); vec3 c3=getCorner(cell,3);
    vec3 c4=getCorner(cell,4); vec3 c5=getCorner(cell,5);
    vec3 c6=getCorner(cell,6); vec3 c7=getCorner(cell,7);
    // ▌从 TBO 中读取当前网格单元的 8 个角点坐标。
    // ▌c0~c7 的编号与前面注释的 GRDECL corner layout 一致:
    // ▌  c0~c3 = 顶面4个角, c4~c7 = 底面4个角。

    float cx = (c0.x+c1.x+c2.x+c3.x+c4.x+c5.x+c6.x+c7.x)*0.125;
    // ▌计算 8 个角点的 X 坐标平均值 = 网格单元的中心 X 坐标。
    // ▌0.125 = 1/8。

    if (cx > uSliceX) return;
    // ▌X 轴切片: 如果该单元的中心 X 坐标超过切片阈值，
    // ▌直接 return，不输出任何顶点 → 该单元"消失" (被切掉)。
    // ▌这就是实时切片的核心机制: 在 GPU 端通过条件判断丢弃网格单元。
    // ▌不需要重新上传数据，只需改变 uSliceX 这个 uniform 值即可。

    float cz = (c0.z+c1.z+c2.z+c3.z+c4.z+c5.z+c6.z+c7.z)*0.125;
    if (cz > uSliceZ) return;
    // ▌Z 轴切片: 同理，超过 Z 阈值的单元也被切掉。
    // ▌这允许用户"剥开"上层地层，查看深层结构。

    // ── 计算单元中心点到摄像机的距离 ──────────────────────────────
    float cy = (c0.y+c1.y+c2.y+c3.y+c4.y+c5.y+c6.y+c7.y)*0.125;
    vec3  cellCenter = vec3(cx, cy, cz * uVExag);
    float dist = distance(cellCenter, uCamPos);

    float fault = texelFetch(uFaultFlag, cell).r;

    // ══════════════════════════════════════════════════════════
    // V11最终版: 球壳变形 (Shell Deformation) - 导师草图要求
    // ══════════════════════════════════════════════════════════
    // 设计目标：当观察者进入油藏内部时，靠近观察者的网格单元
    // 不仅要"可见"，更要被"几何变形"以构成一个围绕观察者的封闭球壳，
    // 实现"无论从多近的距离、从哪个角度看，模型都是封闭实体"。
    //
    // 算法核心：
    //   1) 远端单元 (dist > uShellRadius * 1.5)
    //      → 正常发射外表面 (常规渲染)
    //   2) 中等距离单元 (uShellRadius < dist <= uShellRadius * 1.5)
    //      → 同时发射外表面 + 内表面 (过渡区，避免视觉跳变)
    //   3) 近端单元 (dist <= uShellRadius)
    //      → "重新组织几何形态"：将该单元的8个角点投影到
    //        以观察者为中心、半径为 uShellRadius 的球面上，
    //        使这些单元共同组成一个围绕观察者的封闭球壳。

    // 距离比率：当 r=1.0 时位于球壳边界，r<1.0 在球壳内
    float shellR = max(uShellRadius, 1.0);  // 防止除零
    float r = dist / shellR;

    if (r > 1.5) {
        // ── 远端单元：标准外表面渲染 ─────────────────────────
        emitQuad(c0, c2, c1, c3, cz, fault); // top
        emitQuad(c4, c5, c6, c7, cz, fault); // bottom
        emitQuad(c0, c1, c4, c5, cz, fault); // front
        emitQuad(c3, c2, c7, c6, cz, fault); // back
        emitQuad(c2, c0, c6, c4, cz, fault); // left
        emitQuad(c1, c3, c5, c7, cz, fault); // right
    }
    else if (r > 1.0) {
        // ── 过渡区：同时发射内外表面（避免边界跳变）─────────
        emitQuad(c0, c2, c1, c3, cz, fault);
        emitQuad(c4, c5, c6, c7, cz, fault);
        emitQuad(c0, c1, c4, c5, cz, fault);
        emitQuad(c3, c2, c7, c6, cz, fault);
        emitQuad(c2, c0, c6, c4, cz, fault);
        emitQuad(c1, c3, c5, c7, cz, fault);

        emitQuadInterior(c0, c2, c1, c3, cz, fault);
        emitQuadInterior(c4, c5, c6, c7, cz, fault);
        emitQuadInterior(c0, c1, c4, c5, cz, fault);
        emitQuadInterior(c3, c2, c7, c6, cz, fault);
        emitQuadInterior(c2, c0, c6, c4, cz, fault);
        emitQuadInterior(c1, c3, c5, c7, cz, fault);
    }
    else {
        // ══════════════════════════════════════════════════════
        // 球壳变形区 - 导师草图核心要求实现
        // ══════════════════════════════════════════════════════
        // 把单元的8个角点"投影"到以观察者为中心的球面上：
        //   newCorner = uCamPos + normalize(corner - uCamPos) * shellR
        // 这样所有近端单元的角点都落在同一个球面上，
        // 它们的面拼合起来形成一个围绕观察者的封闭壳。
        //
        // 同时保留原始单元的深度颜色信息，让球壳显示出油藏的内部数据。

        vec3 nc0 = uCamPos + normalize(c0 - uCamPos) * shellR;
        vec3 nc1 = uCamPos + normalize(c1 - uCamPos) * shellR;
        vec3 nc2 = uCamPos + normalize(c2 - uCamPos) * shellR;
        vec3 nc3 = uCamPos + normalize(c3 - uCamPos) * shellR;
        vec3 nc4 = uCamPos + normalize(c4 - uCamPos) * shellR;
        vec3 nc5 = uCamPos + normalize(c5 - uCamPos) * shellR;
        vec3 nc6 = uCamPos + normalize(c6 - uCamPos) * shellR;
        vec3 nc7 = uCamPos + normalize(c7 - uCamPos) * shellR;

        // 发射变形后的单元（内表面，因为观察者在球壳内部看向外）
        emitQuadInterior(nc0, nc2, nc1, nc3, cz, fault);
        emitQuadInterior(nc4, nc5, nc6, nc7, cz, fault);
        emitQuadInterior(nc0, nc1, nc4, nc5, cz, fault);
        emitQuadInterior(nc3, nc2, nc7, nc6, cz, fault);
        emitQuadInterior(nc2, nc0, nc6, nc4, cz, fault);
        emitQuadInterior(nc1, nc3, nc5, nc7, cz, fault);
    }
}
"""

# ============================================================================
# 4.3 油藏网格片段着色器 (Reservoir Fragment Shader)
# ============================================================================

FRAGMENT_SHADER_SRC = """
#version 330 core

in  float fDepth;
// ▌从几何着色器接收的、经过光栅化插值的深度值。
// ▌用于映射到彩虹色带。

in  vec3  fNormal;
// ▌从几何着色器接收的法线向量 (已经过插值，但在平面上插值结果不变)。
// ▌用于光照计算。

in  float fFault;
// ▌从几何着色器接收的断层标志 (0.0 或 1.0)。

out vec4  FragColor;
// ▌片段着色器的最终输出: RGBA 颜色值。
// ▌这是写入帧缓冲 (frame buffer) 的颜色，最终显示在屏幕上。

uniform float uDepthMin;
// ▌数据集中所有单元的最小深度值 (世界坐标 Z)。
// ▌用于将深度归一化到 [0, 1] 范围。

uniform float uDepthRange;
// ▌深度范围 = max_depth - min_depth。
// ▌归一化公式: t = (depth - uDepthMin) / uDepthRange。

uniform int   uShowFaults;
// ▌是否显示断层高亮的开关 (1 = 显示, 0 = 不显示)。
// ▌由 CPU 端 show_faults 变量控制。

vec3 rainbow(float t) {
// ▌彩虹色带映射函数 (Rainbow Colormap)。
// ▌将 [0.0, 1.0] 的归一化值映射到颜色:
// ▌  0.00 → 蓝色 (0,0,1) —— 最浅层
// ▌  0.25 → 青色 (0,1,1)
// ▌  0.50 → 绿色 (0,1,0)
// ▌  0.75 → 黄色 (1,1,0)
// ▌  1.00 → 红色 (1,0,0) —— 最深层
// ▌
// ▌这种配色方案与商业油藏模拟软件 (Petrel, Eclipse Viewer, CMG) 的
// ▌深度色图一致，是石油工程领域的标准可视化方式。

    t = clamp(t, 0.0, 1.0);
    // ▌clamp: 将 t 限制在 [0, 1] 范围内，防止越界产生异常颜色。

    if      (t < 0.25) return mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), t*4.0);
    // ▌第一段 [0, 0.25]: 蓝 → 青。
    // ▌mix(a, b, factor): GLSL 线性插值函数，= a*(1-factor) + b*factor。
    // ▌t*4.0: 将 [0, 0.25] 映射到 [0, 1] 的插值因子。

    else if (t < 0.50) return mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (t-0.25)*4.0);
    // ▌第二段 [0.25, 0.50]: 青 → 绿。

    else if (t < 0.75) return mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (t-0.50)*4.0);
    // ▌第三段 [0.50, 0.75]: 绿 → 黄。

    else               return mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (t-0.75)*4.0);
    // ▌第四段 [0.75, 1.00]: 黄 → 红。
}

void main() {
// ▌片段着色器入口: 对每个像素 (fragment) 执行一次。

    float t   = clamp((fDepth - uDepthMin) / uDepthRange, 0.0, 1.0);
    // ▌深度归一化: 将实际深度值映射到 [0, 1]。
    // ▌最浅的单元 → t≈0 → 蓝色; 最深的单元 → t≈1 → 红色。

    vec3  col = rainbow(t);
    // ▌通过彩虹函数获得该深度对应的 RGB 颜色。

    if (uShowFaults == 1 && fFault > 0.5)
        col = mix(col, vec3(1.0, 1.0, 1.0), 0.6);
    // ▌断层高亮 (V8 新增):
    // ▌  如果断层显示已开启 (uShowFaults == 1)
    // ▌  且该片段属于断层单元 (fFault > 0.5, 即约等于 1.0):
    // ▌  将原色与白色 (1,1,1) 以 60% 的比例混合 → 产生浅白色调。
    // ▌  mix(col, white, 0.6) = col*0.4 + white*0.6，
    // ▌  效果: 断层区域呈现淡化的、偏白的半透明高亮效果。

    vec3  L1 = normalize(vec3(1.0, 2.0, 3.0));
    // ▌第一光源方向 (主光源)。
    // ▌normalize(1,2,3): 归一化后约为 (0.267, 0.535, 0.802)。
    // ▌这是一个从右上方前方照射的方向光 (directional light)。

    vec3  L2 = normalize(vec3(-1.0, 0.5, -1.0));
    // ▌第二光源方向 (补光/填充光, V8 新增)。
    // ▌从左后方上方照射，用于照亮主光源照不到的暗面，减少纯黑区域。
    // ▌V7 只有 L1 一个光源，V8 增加 L2 实现"三点光照"(含环境光)。

    vec3  n  = normalize(fNormal);
    // ▌重新归一化法线 (光栅化插值可能改变了长度)。

    float d  = 0.45 + 0.40*abs(dot(n,L1)) + 0.15*abs(dot(n,L2));
    // ▌光照强度计算 (Lambertian 漫反射模型的变体):
    // ▌
    // ▌  0.45 = 环境光 (ambient): 保证即使完全背光的面也不会全黑。
    // ▌  0.40 * abs(dot(n, L1)) = 主光源贡献:
    // ▌    dot(n, L1) = 法线与光源方向的点积 = cos(夹角)
    // ▌    abs(): 双面光照! 无论法线朝向光源还是背对光源，都取正值。
    // ▌    这消除了因法线方向不一致导致的黑色面块问题。
    // ▌    0.40 = 主光源强度权重。
    // ▌  0.15 * abs(dot(n, L2)) = 补光源贡献:
    // ▌    权重较小 (0.15)，只是轻微填充暗面。
    // ▌
    // ▌  最终光照系数 d 的范围: [0.45, 1.0]
    // ▌  (0.45 当两个光源的 dot 都=0时，1.0 当两个光源的 dot 都=1时)
    // ▌
    // ▌V7 的光照公式: d = max(abs(dot(n,L)), 0.15) * 1.2
    // ▌V8 改进: 用固定的三分量加权 (ambient + key + fill)，效果更柔和均匀。

    FragColor = vec4(col * d, 1.0);
    // ▌最终像素颜色 = 基础颜色 × 光照强度。
    // ▌alpha = 1.0 (完全不透明)。
    // ▌col * d: 按标量缩放 RGB 三个通道，d 越大越亮。
}
"""

# ============================================================================
# 4.4 井轨迹顶点着色器 (Well Trajectory Vertex Shader)
# ============================================================================

WELL_VERT_SRC = """
#version 330 core

layout(location = 0) in vec3 aPos;
// ▌井轨迹控制点的 3D 坐标 (已减去 origin 的归一化坐标)。
// ▌绑定到属性位置 0。

layout(location = 1) in vec3 aColor;
// ▌该井的颜色 (每口井一种固定颜色，如黄、青、橙等)。
// ▌绑定到属性位置 1。

out vec3 vColor;
// ▌传递给片段着色器的颜色。

uniform mat4  uMVP;
// ▌与油藏渲染共享同一个 MVP 矩阵 (同一摄像机视角)。

uniform float uVExag;
// ▌垂直夸大系数 (与油藏网格保持一致，确保井和网格对齐)。

void main() {
    vec3 p = vec3(aPos.x, aPos.y, aPos.z * uVExag);
    // ▌对 Z 坐标应用垂直夸大 (与油藏网格的 vex() 函数完全一致)。
    // ▌这保证井轨迹的 Z 放大倍数与油藏网格相同，井位看起来"穿透"在正确的地层中。

    gl_Position = uMVP * vec4(p, 1.0);
    // ▌MVP 变换到裁剪空间。

    vColor = aColor;
    // ▌直接传递颜色给片段着色器。
}
"""

# ============================================================================
# 4.5 井轨迹片段着色器 (Well Trajectory Fragment Shader)
# ============================================================================

WELL_FRAG_SRC = """
#version 330 core
in  vec3 vColor;
out vec4 FragColor;
void main() { FragColor = vec4(vColor, 1.0); }
// ▌极其简单: 直接使用传入的颜色，无光照处理。
// ▌alpha = 1.0 (不透明)。
// ▌井轨迹是线段 (GL_LINE_STRIP)，不需要光照也不需要法线。
"""

# ============================================================================
# 4.6 颜色图例覆盖层顶点着色器 (Overlay Vertex Shader)
# ============================================================================

OVERLAY_VERT_SRC = """
#version 330 core

layout(location = 0) in vec2 aPos;
// ▌2D 屏幕坐标 (NDC 空间: x,y 都在 [-1, +1] 范围内)。
// ▌(-1,-1) = 屏幕左下角, (+1,+1) = 屏幕右上角。

layout(location = 1) in vec3 aColor;
// ▌该顶点的 RGB 颜色 (对应彩虹色带的某个位置)。

out vec3 vColor;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    // ▌直接使用 2D 坐标作为裁剪空间坐标。
    // ▌z = 0.0 (在最前面，但因为渲染时关闭了深度测试，所以总是可见)。
    // ▌w = 1.0 (标准齐次坐标)。
    // ▌不需要 MVP 矩阵！这是直接在"屏幕空间"画的 2D 覆盖层。

    vColor = aColor;
}
"""

# ============================================================================
# 4.7 颜色图例覆盖层片段着色器 (Overlay Fragment Shader)
# ============================================================================

OVERLAY_FRAG_SRC = """
#version 330 core
in  vec3 vColor;
out vec4 FragColor;
void main() { FragColor = vec4(vColor, 1.0); }
// ▌同井轨迹一样简单: 直接使用传入颜色，无光照。
// ▌图例条的颜色由 CPU 端预计算并存入顶点数据。
"""

# ============================================================================
# 第五部分: 着色器辅助函数
# ============================================================================

def compile_shader(src, kind):
    """
    编译单个 GLSL 着色器。

    参数:
      src  (str):  GLSL 源代码字符串
      kind (GLenum): 着色器类型常量:
                     GL_VERTEX_SHADER   = 顶点着色器
                     GL_GEOMETRY_SHADER = 几何着色器
                     GL_FRAGMENT_SHADER = 片段着色器

    返回:
      int: 编译成功的着色器对象 ID (OpenGL handle)

    异常:
      RuntimeError: 编译失败时，包含 GLSL 编译错误信息
    """
    s = glCreateShader(kind)
    # ▌glCreateShader(): OpenGL 函数，创建一个空的着色器对象。
    # ▌返回一个整数 ID (handle)，后续所有操作都通过这个 ID 引用该着色器。

    glShaderSource(s, src)
    # ▌glShaderSource(): 将 GLSL 源代码字符串绑定到着色器对象。
    # ▌此时代码还没有被编译，只是"上传"了源码。

    glCompileShader(s)
    # ▌glCompileShader(): 编译着色器。
    # ▌GPU 驱动内置的 GLSL 编译器会将源代码编译为 GPU 可执行的机器码。
    # ▌这个过程可能失败 (语法错误、不支持的特性等)。

    if not glGetShaderiv(s, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(s).decode())
    # ▌glGetShaderiv(): 查询着色器编译状态。
    # ▌GL_COMPILE_STATUS: 如果编译成功返回 True (非零值)。
    # ▌如果失败，glGetShaderInfoLog() 返回详细错误日志 (字节串)，
    # ▌decode() 将其转为 Python 字符串后抛出异常。
    # ▌常见错误: 变量名拼写错误、类型不匹配、不支持的 GLSL 版本等。

    return s


def make_prog(vs_src, fs_src, gs_src=None):
    """
    编译并链接一个完整的着色器程序 (Shader Program)。

    参数:
      vs_src (str): 顶点着色器源代码
      fs_src (str): 片段着色器源代码
      gs_src (str, optional): 几何着色器源代码 (可选)。
                              井轨迹和图例不需要几何着色器，所以默认 None。

    返回:
      int: 链接成功的程序对象 ID

    异常:
      RuntimeError: 编译或链接失败

    说明:
      OpenGL 着色器程序 (Program) = 多个着色器的组合。
      最少需要: 顶点着色器 + 片段着色器。
      可选加入: 几何着色器 (在顶点和片段之间)。
      链接 (Link) 的过程会检查着色器之间的接口是否匹配
      (例如顶点着色器的 out 和片段着色器的 in 是否类型/名称一致)。
    """
    shaders = [compile_shader(vs_src, GL_VERTEX_SHADER)]
    # ▌编译顶点着色器 (必须有)。

    if gs_src:
        shaders.append(compile_shader(gs_src, GL_GEOMETRY_SHADER))
    # ▌如果提供了几何着色器源码，也编译它。
    # ▌只有油藏网格的渲染管线需要几何着色器，
    # ▌井轨迹和图例的管线不需要 (它们有自己独立的顶点数据)。

    shaders.append(compile_shader(fs_src, GL_FRAGMENT_SHADER))
    # ▌编译片段着色器 (必须有)。

    p = glCreateProgram()
    # ▌创建空的着色器程序对象。

    for sh in shaders:
        glAttachShader(p, sh)
        # ▌将编译好的着色器附加到程序上。

        glDeleteShader(sh)
        # ▌标记着色器对象为"可删除"。
        # ▌注意: 这不会立即销毁着色器——只有当它从所有程序中分离后才会真正删除。
        # ▌这是 OpenGL 的资源管理惯例: 编译后立即 delete，引用计数管理生命周期。

    glLinkProgram(p)
    # ▌链接程序: 将所有附加的着色器组合成一个可执行的管线。
    # ▌链接器会检查:
    # ▌  1) 顶点着色器的 out 与几何/片段着色器的 in 是否匹配
    # ▌  2) uniform 变量的声明是否一致
    # ▌  3) 是否满足 GPU 硬件限制 (如 max_vertices)

    if not glGetProgramiv(p, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(p).decode())
    # ▌检查链接是否成功，失败则抛出错误。

    return p


# ============================================================================
# 第六部分: 矩阵工具函数
# ============================================================================

def mat_perspective(fovy, aspect, near, far):
    """
    构建透视投影矩阵 (Perspective Projection Matrix)。

    参数:
      fovy   (float): 垂直视野角度 (Field of View, 单位: 度)。
                      45° 是常用值，接近人眼的自然视角。
      aspect (float): 窗口宽高比 = width / height。
                      1280/720 ≈ 1.778
      near   (float): 近裁剪面距离。比这更近的物体不会被渲染。
                      设为 sz*0.001 以避免裁掉近处物体。
      far    (float): 远裁剪面距离。比这更远的物体不会被渲染。
                      设为 sz*50.0 以确保整个场景可见。

    返回:
      np.array: 4×4 透视投影矩阵 (行主序, float32)。

    原理:
      透视投影模拟人眼/相机的锥形视野: 近大远小。
      它将视锥体 (frustum) 映射到 [-1, +1]³ 的标准化设备坐标 (NDC)。
      数学推导基于相似三角形和 Z 缓冲的非线性映射。
    """
    f = 1.0 / np.tan(np.radians(fovy) / 2.0)
    # ▌f = cot(fovy/2): 投影缩放因子。
    # ▌fovy 越大，视野越宽，物体在屏幕上越小 → f 越小。
    # ▌np.radians(): 将度转换为弧度 (OpenGL 三角函数需要弧度)。

    return np.array([
        [f/aspect, 0,  0,                     0],
        # ▌第一行: 控制 X 轴的投影。
        # ▌f/aspect: 水平方向的缩放。aspect > 1 时水平范围更大。

        [0,        f,  0,                     0],
        # ▌第二行: 控制 Y 轴的投影。f 直接作为垂直缩放因子。

        [0,        0,  (far+near)/(near-far),  2*far*near/(near-far)],
        # ▌第三行: Z 轴的非线性映射。
        # ▌(far+near)/(near-far): Z 缩放
        # ▌2*far*near/(near-far): Z 偏移
        # ▌这使得 Z 值在 near 到 far 之间被映射到 [-1, +1]，
        # ▌但映射是非线性的——近处精度高，远处精度低。

        [0,        0, -1,                     0],
        # ▌第四行: 透视除法。-1 将 Z 值复制到 W 分量 (的负值)，
        # ▌之后 GPU 做透视除法 (x/w, y/w, z/w) 实现近大远小的效果。
    ], dtype=np.float32)
    # ▌注意: 这个矩阵是行主序 (row-major) 写法。
    # ▌在传给 OpenGL 时，通过 glUniformMatrix4fv(..., GL_TRUE, ...) 的
    # ▌GL_TRUE 参数告诉 OpenGL 做转置，因为 OpenGL 期望列主序 (column-major)。


def mat_lookat(eye, target, up):
    """
    构建视图矩阵 (View Matrix) —— LookAt 矩阵。

    参数:
      eye    (np.array): 摄像机位置 (世界坐标), shape=(3,)
      target (np.array): 摄像机注视的目标点 (世界坐标), shape=(3,)
                         本程序中始终为原点 (0,0,0)。
      up     (np.array): 世界空间的"上"方向向量, shape=(3,)
                         通常为 (0,1,0) 表示 Y 轴朝上。

    返回:
      np.array: 4×4 视图矩阵 (行主序, float32)。

    原理:
      LookAt 矩阵将世界坐标系变换到摄像机坐标系。
      本质上是一个"反向平移 + 反向旋转": 将摄像机移到原点，朝向 -Z 方向。
      构建过程:
        1) f = normalize(target - eye): 摄像机的前方向
        2) r = normalize(cross(f, up)): 摄像机的右方向
        3) u = cross(r, f): 摄像机的上方向 (重新计算以确保正交)
        4) 旋转矩阵 = [r; u; -f] (3×3), 加上平移 = -dot(axis, eye)
    """
    f = target - eye;  f /= np.linalg.norm(f)
    # ▌前方向向量: 从眼睛指向目标，然后归一化。

    r = np.cross(f, up); r /= np.linalg.norm(r)
    # ▌右方向向量: 前方向与世界上方向的叉积，归一化。
    # ▌叉积的结果垂直于两个输入向量。

    u = np.cross(r, f)
    # ▌重新计算上方向: 右方向与前方向的叉积。
    # ▌不需要归一化，因为 r 和 f 都是正交的单位向量，叉积结果自动是单位向量。

    return np.array([
        [ r[0],  r[1],  r[2], -np.dot(r, eye)],
        # ▌第一行: 世界坐标投影到摄像机的右方向。
        # ▌-np.dot(r, eye): 平移分量 (eye 在右方向上的分量取反)。

        [ u[0],  u[1],  u[2], -np.dot(u, eye)],
        # ▌第二行: 投影到摄像机的上方向。

        [-f[0], -f[1], -f[2],  np.dot(f, eye)],
        # ▌第三行: 投影到摄像机的后方向 (-f)。
        # ▌OpenGL 约定: 摄像机朝 -Z 方向看，所以用 -f。

        [    0,     0,      0,              1 ],
        # ▌第四行: 齐次坐标的固定值。
    ], dtype=np.float32)


# ============================================================================
# 第七部分: GRDECL 数据解析器
# ============================================================================

class GRDECLParser:
    """
    GRDECL 格式角点网格数据解析器。

    GRDECL (Grid Corner-point Data in ECLipse format) 是 Schlumberger ECLIPSE
    油藏模拟器定义的文本文件格式，包含油藏网格的完整几何描述信息。

    核心概念:
      角点网格 (Corner-Point Grid, CPG) 是石油工程中最常用的网格类型。
      它由一组垂直/倾斜的"柱状线" (pillars) 和沿柱状线的深度值 (ZCORN) 定义。
      每个网格单元 (cell) 是一个不规则六面体 (hexahedron)，
      其 8 个角点坐标通过在柱状线上进行线性插值计算得到。

    文件中的关键数据块:
      SPECGRID:  网格维度 (nx, ny, nz)
      COORD:     柱状线端点坐标
      ZCORN:     角点深度值
      ACTNUM:    活跃单元标志
    """

    def __init__(self, fp):
        """
        初始化解析器。

        参数:
          fp (str): GRDECL 文件路径
        """
        self.filepath = fp
        # ▌存储文件路径

        self.nx = self.ny = self.nz = 0
        # ▌网格维度: (nx, ny, nz)
        # ▌nx = I 方向 (通常是东西方向) 的单元数
        # ▌ny = J 方向 (通常是南北方向) 的单元数
        # ▌nz = K 方向 (垂直/深度方向) 的层数
        # ▌Norne: 46×112×22 → 共 113,344 个单元 (含非活跃)

        self.coord = []; self.zcorn = []; self.actnum = []
        # ▌coord:  柱状线坐标数据 (float 列表)
        # ▌zcorn:  角点深度值 (float 列表)
        # ▌actnum: 活跃标志 (int 列表, 1=活跃, 0=非活跃)

        self.cells = []
        # ▌解析后的网格单元列表。
        # ▌每个元素 = (corners_normalized, avg_z, avg_x):
        # ▌  corners_normalized: 8 个角点坐标 (已减去 origin)，每个是 np.array(3)
        # ▌  avg_z: 8 个角点的平均 Z 坐标 (归一化后)，用于深度颜色映射
        # ▌  avg_x: 8 个角点的平均 X 坐标 (归一化后)，用于 X 轴切片

        self.fault_flags = []
        # ▌V8 新增: 断层标志列表，与 self.cells 一一对应。
        # ▌1.0 = 断层单元, 0.0 = 正常单元。

        self.min_b = np.full(3,  1e18)
        self.max_b = np.full(3, -1e18)
        # ▌归一化后坐标的包围盒 (bounding box):
        # ▌  min_b = [x_min, y_min, z_min]
        # ▌  max_b = [x_max, y_max, z_max]
        # ▌初始化为极值，在 _build() 中通过 np.minimum/maximum 逐步更新。

        self._real_origin = np.zeros(3)
        # ▌V8 新增: 保存坐标归一化时的原点 (原始坐标的中心)。
        # ▌井轨迹渲染时需要用同一个 origin 做坐标变换，确保井和网格对齐。


    def parse(self):
        """
        解析 GRDECL 文件。

        流程:
          1) 读取整个文件到内存
          2) 去除注释 (-- 开头的内容)
          3) 分词 (tokenize)
          4) 按关键字识别各数据段 (SPECGRID, COORD, ZCORN, ACTNUM)
          5) 调用 _build() 构建几何

        返回:
          bool: True 如果成功解析并生成了网格单元, False 否则
        """
        print(f"[Loader] Reading {self.filepath} ...")

        if not os.path.exists(self.filepath):
            print("[Error] File not found"); return False
        # ▌检查文件是否存在

        with open(self.filepath) as f:
            raw = f.read()
        # ▌一次性读取整个文件内容到字符串 raw。
        # ▌Norne ATW2013 文件约 100+ MB，包含数百万行数值。

        tokens = ' '.join(l.split('--')[0] for l in raw.split('\n'))
        # ▌预处理第一步: 去除注释。
        # ▌GRDECL 格式中，"--" 是行注释标志 (类似 Python 的 #)。
        # ▌l.split('--')[0]: 取 "--" 之前的部分，丢弃注释。
        # ▌' '.join(): 将所有行合并为一个长字符串，用空格分隔。

        tokens = tokens.replace('/', ' / ').split()
        # ▌预处理第二步: 分词。
        # ▌replace('/', ' / '): 将 "/" 与相邻内容分开
        #   (GRDECL 用 "/" 作为数据段的结束标志)。
        # ▌.split(): 按空白分割成 token 列表。
        # ▌结果: ['SPECGRID', '46', '112', '22', '1', 'F', '/', 'COORD', '456000.0', ...]

        it = iter(tokens)
        # ▌创建 token 迭代器，便于逐个消费。

        try:
            while True:
                t = next(it)
                # ▌读取下一个 token。

                if t == 'SPECGRID':
                    self.nx,self.ny,self.nz = int(next(it)),int(next(it)),int(next(it))
                    print(f"[Loader] Grid {self.nx}x{self.ny}x{self.nz}")
                    # ▌SPECGRID 关键字后面跟着 3 个整数: nx ny nz
                    # ▌(可能还有其他参数如 NUMRES 和 TYPE，但我们只取前 3 个)
                    # ▌Norne: 46 112 22

                elif t == 'COORD':
                    self.coord = self._rd(it, 6*(self.nx+1)*(self.ny+1))
                    # ▌COORD 数据: 柱状线 (pillar) 的顶底端点坐标。
                    # ▌角点网格有 (nx+1) × (ny+1) 根柱状线 (网格线交点)。
                    # ▌每根柱状线由 6 个 float 定义:
                    # ▌  x_top, y_top, z_top, x_bot, y_bot, z_bot
                    # ▌  (柱状线顶端点和底端点的 3D 坐标)
                    # ▌总数 = 6 × (46+1) × (112+1) = 6 × 47 × 113 = 31,866 个 float。

                elif t == 'ZCORN':
                    self.zcorn = self._rd(it, 8*self.nx*self.ny*self.nz)
                    # ▌ZCORN 数据: 每个网格单元 8 个角点的 Z (深度) 值。
                    # ▌总数 = 8 × 46 × 112 × 22 = 906,752 个 float。
                    # ▌这些值定义了每个角点沿柱状线的深度位置。
                    # ▌
                    # ▌ZCORN 数组的索引顺序遵循 ECLIPSE 约定:
                    # ▌  外层: 按 k 层 (从浅到深, k=0,1,...,nz-1)
                    # ▌  中层: 每层分为上半格 (hk=0) 和下半格 (hk=1)
                    # ▌  内层: 按 j 和 i 方向扫描

                elif t == 'ACTNUM':
                    self.actnum = self._rd(it, self.nx*self.ny*self.nz, True)
                    # ▌ACTNUM 数据: 活跃单元标志，每个单元一个整数。
                    # ▌1 = 活跃 (有流体、参与计算)
                    # ▌0 = 非活跃 (被地质排除、不含储层)
                    # ▌Norne 中约 44,431 个单元非活跃，剩余约 113,344 - 44,431 ≈ 68,913 活跃。
                    # ▌(实际活跃数约 11.3 万，取决于具体数据版本)

        except StopIteration:
            pass
        # ▌迭代器耗尽时，StopIteration 异常被捕获，解析结束。

        self._build()
        # ▌调用几何构建方法。

        return bool(self.cells)
        # ▌如果成功生成了至少一个网格单元，返回 True。


    def _rd(self, it, n, isint=False):
        """
        从 token 迭代器中读取 n 个数值。

        参数:
          it    (iterator): token 迭代器
          n     (int):      期望读取的数值数量
          isint (bool):     True = 解析为整数, False = 解析为浮点数

        返回:
          list: 包含 n 个数值 (int 或 float) 的列表

        说明:
          GRDECL 格式支持重复计数语法: "1000*0.0" 表示连续 1000 个 0.0。
          这在 ACTNUM 等大量重复数据中非常常见，大幅减少文件体积。
          例如: "10000*0 5000*1 8000*0 /" 表示 10000个0、5000个1、8000个0。
        """
        d = []
        # ▌收集解析出的数值

        while len(d) < n:
            try: v = next(it)
            except StopIteration: break
            # ▌逐个消费 token。

            if v == '/': break
            # ▌"/" 是 GRDECL 数据段的结束标志，遇到就停止。

            if '*' in v:
                a,b = v.split('*')
                try: d.extend([int(b) if isint else float(b)]*int(a))
                except: pass
                # ▌处理重复计数语法: "1000*0.0" → a="1000", b="0.0"
                # ▌[float("0.0")] * int("1000") = [0.0, 0.0, ..., 0.0] (1000个)
                # ▌extend: 一次性添加所有元素到列表末尾。
            else:
                try: d.append(int(v) if isint else float(v))
                except: pass
                # ▌普通数值: 直接转换为 int 或 float 后添加。
                # ▌try/except: 跳过无法解析的 token (如关键字名称)。

        return d


    def _build(self):
        """
        根据解析得到的 COORD, ZCORN, ACTNUM 数据构建网格几何。

        这是整个解析器最核心、最复杂的方法。

        流程:
          1) 将 COORD 重塑为柱状线数组 (ny+1, nx+1, 6)
          2) 遍历所有 (i, j, k) 单元:
             a) 跳过非活跃单元 (ACTNUM=0)
             b) 用 zidx() 计算 ZCORN 索引，获取 8 个角的 Z 值
             c) 用 pillar_pt() 在柱状线上线性插值，得到每个角的 (x, y, z)
          3) 计算坐标包围盒，求中心 origin
          4) 断层检测: 比较上下相邻层的深度跳变
          5) 坐标归一化 (减去 origin) 并存入 self.cells
        """
        print("[Loader] Building geometry ...")

        nx,ny,nz = self.nx,self.ny,self.nz
        # ▌取出网格维度的本地引用，方便后续使用。

        try:
            pillars = np.array(self.coord, dtype=np.float64).reshape(ny+1, nx+1, 6)
        except:
            print("[Error] COORD reshape failed"); return
        # ▌将 COORD 数据重塑为三维数组:
        # ▌  pillars[pj, pi, :] = 第 (pi, pj) 根柱状线的 6 个参数
        # ▌    [x_top, y_top, z_top, x_bot, y_bot, z_bot]
        # ▌
        # ▌为什么用 float64 (双精度)?
        # ▌  Norne 的原始坐标是 UTM Zone 32N:
        # ▌    x ≈ 456000 (6位数), y ≈ 7321000 (7位数)
        # ▌  float32 只有约 7 位有效数字，在这个量级下精度只到 1 米级别，
        # ▌  而网格单元大小可能只有几十米，会导致严重的几何误差。
        # ▌  float64 有约 15 位精度，足以处理这些大坐标。
        # ▌  (后面会将坐标减去 origin 归一化到零附近，才转为 float32 给 GPU)

        zc = np.array(self.zcorn, dtype=np.float64)
        # ▌ZCORN 数组转为 NumPy 数组 (float64)。

        def zidx(i,j,k,hk,hj,hi):
            """
            计算 ZCORN 数组中某个角点 Z 值的索引。

            这是 ECLIPSE 定义的 ZCORN 索引公式，理解这个公式是理解
            角点网格几何的关键。

            参数:
              i, j, k:     网格单元在 I, J, K 方向的索引 (0-based)
              hk:          垂直半格标志: 0=上半格(顶), 1=下半格(底)
              hj:          J方向半格标志: 0=j侧, 1=j+1侧
              hi:          I方向半格标志: 0=i侧, 1=i+1侧

            返回:
              int: ZCORN 数组中的一维索引

            公式解释:
              ZCORN 数组将三维网格"展平"为一维数组，索引顺序为:
                最外层: K方向 (层), 每层分上下 (hk=0,1)
                中间层: J方向 (行), 每行分前后 (hj=0,1)
                最内层: I方向 (列), 每列分左右 (hi=0,1)

              维度: (nz*2) × (ny*2) × (nx*2)
              总大小 = 8 * nx * ny * nz

              一维索引 = (k*2+hk) × (ny*2 × nx*2) + (j*2+hj) × (nx*2) + (i*2+hi)
            """
            return (k*2+hk)*(ny*2*nx*2) + (j*2+hj)*(nx*2) + (i*2+hi)

        def pillar_pt(pi, pj, z):
            """
            在柱状线 (pillar) 上进行线性插值，得到给定深度 z 处的 (x, y, z) 坐标。

            这是角点网格几何的核心算法。

            参数:
              pi, pj: 柱状线在 I, J 方向的索引
              z:      目标深度值 (来自 ZCORN)

            返回:
              np.array: [x, y, z] 三维坐标

            原理:
              柱状线是一条从顶面到底面的直线段:
                顶端 P_top = (x_top, y_top, z_top)
                底端 P_bot = (x_bot, y_bot, z_bot)

              已知目标 Z 值后，计算插值参数:
                t = (z - z_top) / (z_bot - z_top)

              然后线性插值 X 和 Y:
                x = x_top + t * (x_bot - x_top)
                y = y_top + t * (y_bot - y_top)

              如果柱状线是垂直的 (z_top ≈ z_bot)，则 t=0，直接取顶端坐标。
            """
            p = pillars[pj, pi]
            # ▌取出该柱状线的 6 个参数: [x_top, y_top, z_top, x_bot, y_bot, z_bot]

            dz = p[5]-p[2]
            # ▌dz = z_bot - z_top: 柱状线在 Z 方向的长度。

            t = 0.0 if abs(dz)<1e-9 else (z-p[2])/dz
            # ▌计算插值参数 t:
            # ▌  如果 dz ≈ 0 (柱状线退化为一个点)，t=0 避免除以零。
            # ▌  否则 t = (z - z_top) / (z_bot - z_top)。
            # ▌  t 的范围通常在 [0, 1]，但不严格限制。

            return np.array([p[0]+t*(p[3]-p[0]), p[1]+t*(p[4]-p[1]), z])
            # ▌线性插值:
            # ▌  x = x_top + t * (x_bot - x_top)
            # ▌  y = y_top + t * (y_bot - y_top)
            # ▌  z = 直接使用目标 z 值 (不需要插值，因为它就是已知量)

        defs = [(0,0,0,0,0),(1,0,0,0,1),(0,1,0,1,0),(1,1,0,1,1),
                (0,0,1,0,0),(1,0,1,0,1),(0,1,1,1,0),(1,1,1,1,1)]
        # ▌8 个角点的定义表。
        # ▌每个元组 (di, dj, hk, hj, hi) 表示:
        # ▌  di: I方向偏移 (0=当前柱, 1=i+1 柱)
        # ▌  dj: J方向偏移 (0=当前柱, 1=j+1 柱)
        # ▌  hk: K方向半格 (0=顶面, 1=底面)
        # ▌  hj: J方向半格 (ZCORN 索引用)
        # ▌  hi: I方向半格 (ZCORN 索引用)
        # ▌
        # ▌举例: defs[0] = (0,0,0,0,0) → 角点 c0:
        # ▌  柱状线 (i+0, j+0) 上的、顶面 (hk=0)、j前(hj=0)、i左(hi=0) 的深度。
        # ▌
        # ▌defs[5] = (1,0,1,0,1) → 角点 c5:
        # ▌  柱状线 (i+1, j+0) 上的、底面 (hk=1)、j前(hj=0)、i右(hi=1) 的深度。
        # ▌
        # ▌前 4 个 (defs[0]~defs[3]) = 顶面 (hk=0): c0, c1, c2, c3
        # ▌后 4 个 (defs[4]~defs[7]) = 底面 (hk=1): c4, c5, c6, c7

        cell_corners = {}
        # ▌V8 改进: 使用字典存储角点，键为 (k, j, i) 元组。
        # ▌V7 使用列表。字典的优势:
        # ▌  断层检测时需要查找 (k, j, i) 和 (k-1, j, i) 是否都存在，
        # ▌  字典的 "in" 查找是 O(1)，列表则需要额外的索引映射。

        mn = np.full(3, 1e18); mx = np.full(3,-1e18)
        # ▌坐标包围盒的临时变量 (原始坐标空间)。

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    # ▌三重循环遍历所有网格单元 (k, j, i)。
                    # ▌k = 层号 (0=最顶层, nz-1=最底层)
                    # ▌j = J方向行号, i = I方向列号

                    cid = k*nx*ny+j*nx+i
                    # ▌一维单元编号 (用于 ACTNUM 索引)。
                    # ▌按 k-j-i 顺序展平: cid = k*(nx*ny) + j*nx + i

                    if self.actnum and cid < len(self.actnum) and self.actnum[cid] == 0:
                        continue
                    # ▌跳过非活跃单元:
                    # ▌  如果有 ACTNUM 数据，且该单元的 ACTNUM = 0，跳过。
                    # ▌  非活跃单元通常是: 地层不存在的区域 (如被侵蚀/缺失)、
                    # ▌  不含石油的区域 (水层)、或网格边界外的填充区域。

                    cs = []; ok = True
                    for (di,dj,hk,hj,hi) in defs:
                        zi = zidx(i,j,k,hk,hj,hi)
                        # ▌计算该角点在 ZCORN 数组中的索引。

                        if zi >= len(zc): ok=False; break
                        # ▌索引越界检查。

                        cs.append(pillar_pt(i+di, j+dj, zc[zi]))
                        # ▌在柱状线 (i+di, j+dj) 上，以 ZCORN[zi] 为深度，
                        # ▌插值得到该角点的 (x, y, z) 坐标。

                    if ok:
                        for c in cs:
                            mn = np.minimum(mn, c)
                            mx = np.maximum(mx, c)
                        # ▌更新包围盒。

                        cell_corners[(k,j,i)] = cs
                        # ▌存入字典，键为 (k, j, i)。

        origin = (mn+mx)*0.5
        # ▌计算包围盒中心 = 所有坐标的大致中心点。
        # ▌Norne 的 origin 大约是: [456500, 7321500, 2650] (UTM 坐标)。

        self._real_origin = origin
        # ▌保存 origin，井轨迹渲染时需要用同一个 origin。

        print(f"[Loader] Origin: {origin}")

        # ========= 断层检测 (V8 新增) =========
        FAULT_THRESHOLD = 15.0
        # ▌断层检测的深度跳变阈值 (米)。
        # ▌如果上下相邻层的同一 (j, i) 位置的平均深度差超过 FAULT_THRESHOLD × 2 = 30 米，
        # ▌就认为这里存在断层。
        # ▌
        # ▌原理: 断层 (fault) 是地层的断裂面，导致同一位置的上下层出现明显的深度错位。
        # ▌正常的地层是连续的，相邻层之间的深度差约等于层厚 (通常几米到十几米)。
        # ▌如果差值异常大 (> 30 米)，很可能是断层造成的错位。

        fault_set = set()
        # ▌用集合 (set) 存储被标记为断层的单元的 (k, j, i) 键。
        # ▌集合的好处: 自动去重，且 "in" 查找是 O(1)。

        for k in range(1, nz):
            for j in range(ny):
                for i in range(nx):
                    if (k,j,i) in cell_corners and (k-1,j,i) in cell_corners:
                        # ▌如果当前层 k 和上一层 k-1 在 (j,i) 位置都有活跃单元:

                        z_cur  = np.mean([c[2] for c in cell_corners[(k,  j,i)]])
                        z_prev = np.mean([c[2] for c in cell_corners[(k-1,j,i)]])
                        # ▌分别计算两个单元 8 个角点 Z 值的平均值。

                        if abs(z_cur - z_prev) > FAULT_THRESHOLD * 2:
                            fault_set.add((k,  j,i))
                            fault_set.add((k-1,j,i))
                        # ▌如果深度差超过阈值，将两个单元都标记为断层。
                        # ▌标记上下两个单元是因为断层面通常影响断面两侧的单元。

        print(f"[Loader] Fault cells detected: {len(fault_set)}")

        # ========= 坐标归一化并存入 self.cells =========
        for key, cs in cell_corners.items():
            norm = [c - origin for c in cs]
            # ▌将 8 个角点坐标减去 origin:
            # ▌  原始: (456478, 7321504, 2578)
            # ▌  归一化: (456478-456500, 7321504-7321500, 2578-2650) ≈ (-22, 4, -72)
            # ▌
            # ▌这样做的关键目的:
            # ▌  1) 避免 float32 精度损失 (大数值相减时的灾难性抵消)
            # ▌  2) 让场景中心在 (0,0,0)，方便摄像机围绕原点旋转
            # ▌  3) 减小数值范围，提高 Z-buffer 精度

            avg_z = sum(c[2] for c in norm) / 8
            avg_x = sum(c[0] for c in norm) / 8
            # ▌计算归一化后的平均深度和平均 X 坐标 (用于色彩映射和切片)。

            self.cells.append((norm, avg_z, avg_x))
            # ▌存入 cells 列表。

            self.fault_flags.append(1.0 if key in fault_set else 0.0)
            # ▌记录该单元是否为断层 (1.0 或 0.0)。

        self.min_b = mn - origin
        self.max_b = mx - origin
        # ▌更新归一化后的包围盒。

        print(f"[Loader] {len(self.cells)} active cells")
        print(f"  X:[{self.min_b[0]:.0f}, {self.max_b[0]:.0f}]")
        print(f"  Y:[{self.min_b[1]:.0f}, {self.max_b[1]:.0f}]")
        print(f"  Z:[{self.min_b[2]:.0f}, {self.max_b[2]:.0f}]")


# ============================================================================
# 第八部分: GPU 缓冲区构建——油藏网格
# ============================================================================

def build_gpu(parser):
    """
    将解析后的油藏网格数据上传到 GPU。

    创建三个 GPU 资源:
      1) 角点坐标 TBO (Texture Buffer Object): 存储所有单元的 8×3 坐标
      2) 断层标志 TBO: 存储每个单元的断层标记 (0.0 或 1.0)
      3) 深度 VAO + VBO: 每个单元一个深度值 (作为顶点属性)

    参数:
      parser (GRDECLParser): 解析完成的解析器对象

    返回:
      tuple: (vao, tbo_corners_texture, tbo_faults_texture, ncells)
    """
    n = len(parser.cells)
    # ▌活跃网格单元总数 (约 11.3 万)。

    # -------- 1. 角点坐标 TBO --------
    cd = np.empty(n*24, dtype=np.float32)
    # ▌创建角点坐标的一维数组:
    # ▌  n 个单元 × 8 个角点 × 3 个坐标 = n×24 个 float32。
    # ▌  总大小 ≈ 113000 × 24 × 4 字节 ≈ 10.8 MB。

    for i,(cs,_,_) in enumerate(parser.cells):
        b = i*24
        for ci,c in enumerate(cs):
            cd[b+ci*3]=c[0]; cd[b+ci*3+1]=c[1]; cd[b+ci*3+2]=c[2]
    # ▌将所有角点坐标打包成连续的 float32 数组:
    # ▌  cd = [c0_x, c0_y, c0_z, c1_x, c1_y, c1_z, ..., c7_x, c7_y, c7_z,  ← 单元 0
    # ▌        c0_x, c0_y, c0_z, ...,                                         ← 单元 1
    # ▌        ...]

    tb = glGenBuffers(1)
    # ▌创建一个 OpenGL 缓冲区对象 (Buffer Object)。

    glBindBuffer(GL_TEXTURE_BUFFER, tb)
    # ▌将该缓冲区绑定为纹理缓冲类型。

    glBufferData(GL_TEXTURE_BUFFER, cd.nbytes, cd, GL_STATIC_DRAW)
    # ▌将 CPU 内存中的 cd 数组上传到 GPU 缓冲区。
    # ▌GL_STATIC_DRAW: 提示 GPU 这份数据上传后不会修改，可以优化存储位置。

    tt = glGenTextures(1)
    # ▌创建一个纹理对象 (用于绑定 TBO)。

    glBindTexture(GL_TEXTURE_BUFFER, tt)
    # ▌将纹理绑定为纹理缓冲类型。

    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, tb)
    # ▌关键函数: 将缓冲区 tb 关联到纹理 tt，格式为 GL_R32F。
    # ▌GL_R32F: 每个纹素 = 1 个 32 位浮点数 (存在红色通道)。
    # ▌这样在 GLSL 中就可以用 texelFetch(uCorners, index).r 来读取。

    # -------- 2. 断层标志 TBO (V8 新增) --------
    fd = np.array(parser.fault_flags, dtype=np.float32)
    # ▌断层标志数组: n 个 float32, 每个 = 0.0 或 1.0。

    ftb = glGenBuffers(1)
    glBindBuffer(GL_TEXTURE_BUFFER, ftb)
    glBufferData(GL_TEXTURE_BUFFER, fd.nbytes, fd, GL_STATIC_DRAW)
    ftt = glGenTextures(1)
    glBindTexture(GL_TEXTURE_BUFFER, ftt)
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, ftb)
    # ▌与角点 TBO 完全相同的创建流程:
    # ▌  缓冲区 → 纹理 → GL_R32F 格式关联。
    # ▌  GLSL 中通过 texelFetch(uFaultFlag, cell).r 读取。

    # -------- 3. 深度 VAO + VBO --------
    dd = np.array([c[1] for c in parser.cells], dtype=np.float32)
    # ▌每个单元的平均深度值数组: n 个 float32。
    # ▌c[1] = parser.cells[i] 元组的第二个元素 = avg_z。

    vao = glGenVertexArrays(1)
    # ▌创建顶点数组对象 (VAO)。
    # ▌VAO 是一个"容器"，记录了所有顶点属性的绑定状态。
    # ▌渲染时只需绑定 VAO，所有属性配置自动恢复。

    glBindVertexArray(vao)
    # ▌绑定 VAO (之后的属性配置都会被记录在这个 VAO 中)。

    vb = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vb)
    glBufferData(GL_ARRAY_BUFFER, dd.nbytes, dd, GL_STATIC_DRAW)
    # ▌创建 VBO 并上传深度数据。

    glVertexAttribPointer(0,1,GL_FLOAT,GL_FALSE,0,None)
    # ▌配置顶点属性 0 (对应 GLSL 中的 layout(location = 0)):
    # ▌  参数解释:
    # ▌    0:        属性位置 = 0
    # ▌    1:        每个属性包含 1 个分量 (float, 不是 vec2/vec3/vec4)
    # ▌    GL_FLOAT: 数据类型
    # ▌    GL_FALSE: 不需要归一化
    # ▌    0:        步长 = 0 (紧密排列, OpenGL 自动计算 = 1*sizeof(float) = 4 字节)
    # ▌    None:     偏移量 = 0 (从缓冲区起始位置开始)

    glEnableVertexAttribArray(0)
    # ▌启用属性位置 0。默认是禁用的。

    glBindVertexArray(0)
    # ▌解绑 VAO (防止后续操作意外修改它的状态)。

    return vao, tt, ftt, n
    # ▌返回: VAO、角点纹理、断层纹理、单元数量。


# ============================================================================
# 第九部分: GPU 缓冲区构建——井轨迹 (V8 新增)
# ============================================================================

def build_wells_gpu(parser):
    """
    将井轨迹数据上传到 GPU。

    每口井由若干 3D 控制点定义，渲染时用 GL_LINE_STRIP 连成线段。
    每个顶点包含 6 个 float: (x, y, z, r, g, b)。

    参数:
      parser (GRDECLParser): 解析器 (需要 _real_origin 做坐标归一化)

    返回:
      tuple: (vao, counts)
        vao:    VAO 对象
        counts: 每口井的顶点数列表，用于逐井绘制
    """
    origin = parser._real_origin
    # ▌使用与油藏网格相同的 origin 做坐标变换。

    well_colours = [
        [1.0,1.0,0.0],[0.0,1.0,1.0],[1.0,0.5,0.0],
        [1.0,0.0,1.0],[0.5,1.0,0.5],[1.0,0.8,0.8],
        [0.8,0.8,1.0],[1.0,1.0,0.5],[0.5,1.0,1.0],
        [1.0,0.6,0.6],[0.6,1.0,0.6],
    ]
    # ▌11 种预定义颜色 (亮色系，在深色背景上醒目):
    # ▌  黄、青、橙、紫、浅绿、浅粉、浅蓝、浅黄、浅青、浅红、浅绿。
    # ▌每口井循环使用一种颜色 (idx % len)。

    verts = []; counts = []
    # ▌verts: 所有井的顶点数据 (交错存储: x,y,z,r,g,b,x,y,z,r,g,b,...)
    # ▌counts: 每口井的顶点数 (用于确定 glDrawArrays 的绘制范围)

    for idx,(name,pts) in enumerate(NORNE_WELLS.items()):
        col = well_colours[idx % len(well_colours)]
        # ▌为每口井选择一种颜色。

        cnt = 0
        for (wx,wy,wz) in pts:
            verts += [wx-origin[0], wy-origin[1], wz-origin[2],
                      col[0], col[1], col[2]]
            cnt += 1
        # ▌将每个控制点的坐标减去 origin 后，加上颜色，打包到 verts 列表。

        counts.append(cnt)
        # ▌记录这口井有多少个顶点 (通常 = 2)。

    verts = np.array(verts, dtype=np.float32)
    # ▌转为 NumPy 数组。

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vb = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vb)
    glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)

    stride = 6*4
    # ▌步长 = 每个顶点的字节数 = 6 个 float × 4 字节/float = 24 字节。
    # ▌因为每个顶点包含: (x, y, z, r, g, b) = 6 个 float 交错存储。

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    # ▌属性 0 = 位置 (x, y, z):
    # ▌  3 个分量, 从偏移 0 字节开始, 步长 24 字节。

    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    # ▌属性 1 = 颜色 (r, g, b):
    # ▌  3 个分量, 从偏移 12 字节 (= 3 个 float 之后) 开始, 步长 24 字节。
    # ▌  ctypes.c_void_p(12): 将整数 12 转为 C 指针类型，OpenGL 函数需要这种类型。

    glEnableVertexAttribArray(1)
    glBindVertexArray(0)

    return vao, counts


# ============================================================================
# 第十部分: GPU 缓冲区构建——颜色图例条 (V8 新增)
# ============================================================================

def build_legend_gpu():
    """
    构建屏幕右侧的彩虹颜色图例条 (Colorbar Legend)。

    图例条是一个 2D 覆盖层，直接在屏幕坐标 (NDC) 中绘制。
    由 64 个小矩形纵向排列组成，从下到上颜色从蓝渐变到红。

    返回:
      tuple: (vao, total_vertices)
        vao:             VAO 对象
        total_vertices:  总顶点数 (= 64 × 4 = 256)
    """
    STEPS = 64
    # ▌图例条被分成 64 个小段 (segments), 每段一个小矩形。
    # ▌64 段足够产生平滑的渐变效果。

    x0, x1 = 0.88, 0.94
    # ▌图例条在 NDC 中的水平范围:
    # ▌  x0 = 0.88 (左边缘), x1 = 0.94 (右边缘)。
    # ▌  NDC 范围 [-1, +1], 所以 0.88~0.94 位于屏幕右侧约 94% 的位置。
    # ▌  宽度 = 0.06 (约屏幕宽度的 3%)。

    y0, y1 = -0.75, 0.75
    # ▌图例条在 NDC 中的垂直范围:
    # ▌  y0 = -0.75 (底部), y1 = 0.75 (顶部)。
    # ▌  高度 = 1.5 (约屏幕高度的 75%)。

    def rainbow_cpu(t):
        """
        CPU 端的彩虹色带函数 (与 GLSL 中的 rainbow() 完全对应)。

        参数: t (float): 归一化值 [0, 1]
        返回: (r, g, b) 元组

        注意: 这里的颜色顺序必须与 GLSL 版本完全一致！
        但仔细对比会发现，这个 CPU 版本的实现与 GLSL 版本略有不同:
          GLSL: 蓝→青→绿→黄→红 (RGB 通道变化不同)
          CPU:  相同的颜色过渡，但代码写法不同 (直接操作 RGB 分量而非 mix)
        最终的视觉效果是一致的。
        """
        t = max(0.0, min(1.0, t))
        if   t < 0.25: return 0.0,      t*4,          1.0
        # ▌[0, 0.25]: 蓝(0,0,1) → 青(0,1,1), 绿通道从0升到1

        elif t < 0.50: return 0.0,       1.0,          1-(t-0.25)*4
        # ▌[0.25, 0.50]: 青(0,1,1) → 绿(0,1,0), 蓝通道从1降到0

        elif t < 0.75: return (t-0.5)*4, 1.0,          0.0
        # ▌[0.50, 0.75]: 绿(0,1,0) → 黄(1,1,0), 红通道从0升到1

        else:          return 1.0,       1-(t-0.75)*4, 0.0
        # ▌[0.75, 1.00]: 黄(1,1,0) → 红(1,0,0), 绿通道从1降到0

    verts = []
    for i in range(STEPS):
        t0 = i/STEPS;       t1 = (i+1)/STEPS
        # ▌当前段的归一化范围: [t0, t1]

        yb = y0+t0*(y1-y0); yt = y0+t1*(y1-y0)
        # ▌当前段的 NDC 纵坐标: yb=底边, yt=顶边

        c0 = rainbow_cpu(t0); c1 = rainbow_cpu(t1)
        # ▌底边和顶边的颜色

        verts += [x0,yb,*c0, x1,yb,*c0, x0,yt,*c1, x1,yt,*c1]
        # ▌一个小矩形 = 4 个顶点 (用 triangle_strip 渲染):
        # ▌  左下 (x0,yb,c0), 右下 (x1,yb,c0), 左上 (x0,yt,c1), 右上 (x1,yt,c1)
        # ▌每个顶点 5 个值: (x, y, r, g, b)
        # ▌*c0 是 Python 的解包语法: (r,g,b) → r, g, b

    verts = np.array(verts, dtype=np.float32)

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vb = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vb)
    glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)

    stride = 5*4
    # ▌步长 = 5 个 float × 4 字节 = 20 字节 (x, y, r, g, b)

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    # ▌属性 0 = 2D 位置 (x, y), 偏移 0

    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
    # ▌属性 1 = 颜色 (r, g, b), 偏移 8 字节 (= 2 个 float 之后)

    glEnableVertexAttribArray(1)
    glBindVertexArray(0)

    return vao, STEPS*4
    # ▌返回: VAO 和总顶点数 (64段 × 4顶点/段 = 256)


# ============================================================================
# 第十一部分: 滚轮回调函数
# ============================================================================

def scroll_cb(win, x, y):
    """
    鼠标滚轮回调函数 (由 GLFW 在滚轮滚动时调用)。

    参数:
      win:  GLFW 窗口对象
      x:    水平滚动量 (通常为 0)
      y:    垂直滚动量 (正值 = 向上滚/放大, 负值 = 向下滚/缩小)
    """
    global cam_dist
    cam_dist = max(100.0, cam_dist*(0.85 if y>0 else 1.15))
    # ▌向上滚 (y > 0): 距离 × 0.85 → 拉近 (放大)
    # ▌向下滚 (y < 0): 距离 × 1.15 → 拉远 (缩小)
    # ▌max(100.0, ...): 限制最近距离为 100，防止穿透到模型内部。
    # ▌
    # ▌使用乘法而非加法的好处: 缩放是"比例式"的。
    # ▌在远处滚一下缩放很多，在近处滚一下缩放很少，手感更自然。


# ============================================================================
# 第十二部分: 主函数——初始化与渲染循环
# ============================================================================

def main():
    """
    程序入口: 初始化所有资源，进入渲染主循环。

    V10主循环每帧执行:
      1) 处理键盘/鼠标输入
      2) 获取VR头部姿态 (VR模式) 或计算普通摄像机矩阵
      3) 左眼渲染 → 右眼渲染 (VR模式) 或单次渲染 (普通模式)
      4) 提交帧给SteamVR compositor (VR模式)
      5) 交换缓冲区显示到桌面窗口
    """
    global cam_yaw,cam_pitch,cam_dist,last_x,last_y,mouse_pressed
    global slice_x,slice_z,vexag,show_faults,show_wells
    global vr_mode, vr_system, vr_compositor
    global proximity_radius
    global shell_radius
    # ▌声明要修改的全局变量 (Python 函数内修改全局变量需要 global 声明)。

    # ========= GLFW 初始化 =========
    if not glfw.init(): return
    # ▌初始化 GLFW 库。失败则退出。

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    # ▌请求 OpenGL 3.3 版本。
    # ▌这确保几何着色器 (3.2+) 和 texelFetch (3.0+) 可用。

    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    # ▌请求 Core Profile (不包含已弃用的固定管线功能)。

    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
    # ▌前向兼容模式 (macOS 需要这个才能用 Core Profile)。

    win = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT,
        "Norne Field v8  |  Q/A=X  E/D=Z  Z/X=VExag  F=Faults  W=Wells  ESC=Quit",
        None, None)
    # ▌创建窗口，标题显示操作说明。
    # ▌最后两个 None: 不全屏、不共享上下文。

    if not win: glfw.terminate(); return
    # ▌窗口创建失败则终止。

    glfw.make_context_current(win)
    # ▌将该窗口的 OpenGL 上下文设为当前线程的活跃上下文。
    # ▌之后所有 OpenGL 调用都会操作这个上下文。

    glfw.set_scroll_callback(win, scroll_cb)
    # ▌注册滚轮回调函数。

    print(f"[GL] {glGetString(GL_RENDERER).decode()}")
    print(f"[GL] {glGetString(GL_VERSION).decode()}")
    # ▌打印 GPU 型号和 OpenGL 版本 (调试用)。

    # ========= VR 初始化 (V10 新增) =========
    vr_render_width = WINDOW_WIDTH
    vr_render_height = WINDOW_HEIGHT
    vr_fbo_l = vr_fbo_r = None
    vr_tex_l = vr_tex_r = None

    if OPENVR_AVAILABLE and vr_mode:
        try:
            vr_system = openvr.init(openvr.VRApplication_Scene)
            # ▌初始化OpenVR，以"场景应用"模式运行。
            # ▌VRApplication_Scene = 完整的VR应用，可以渲染到头显。

            vr_compositor = openvr.VRCompositor()
            # ▌获取compositor接口，用于提交渲染帧。

            vr_render_width, vr_render_height = vr_system.getRecommendedRenderTargetSize()
            # ▌获取头显推荐的渲染分辨率（通常比屏幕分辨率高，用于抗锯齿）。

            print(f"[VR] 头显连接成功! 渲染分辨率: {vr_render_width}×{vr_render_height}")
            # 头显型号查询已跳过（新版openvr API已变更）

            # ---- 创建左眼FBO ----
            vr_fbo_l = glGenFramebuffers(1)
            vr_tex_l = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, vr_tex_l)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, vr_render_width, vr_render_height,
                         0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            vr_depth_l = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, vr_depth_l)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24,
                                  vr_render_width, vr_render_height)
            glBindFramebuffer(GL_FRAMEBUFFER, vr_fbo_l)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_2D, vr_tex_l, 0)
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                      GL_RENDERBUFFER, vr_depth_l)
            # ▌FBO = Framebuffer Object，离屏渲染目标。
            # ▌左眼场景渲染到vr_fbo_l，然后提交给compositor。

            # ---- 创建右眼FBO ----
            vr_fbo_r = glGenFramebuffers(1)
            vr_tex_r = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, vr_tex_r)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, vr_render_width, vr_render_height,
                         0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            vr_depth_r = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, vr_depth_r)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24,
                                  vr_render_width, vr_render_height)
            glBindFramebuffer(GL_FRAMEBUFFER, vr_fbo_r)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_2D, vr_tex_r, 0)
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                      GL_RENDERBUFFER, vr_depth_r)

            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            # ▌恢复默认帧缓冲。

        except Exception as e:
            # V16: 自动降级到桌面模式
            print(f"[VR] 启动时未检测到头显: {e}")
            print("[VR] 自动降级为桌面模式")
            print("[Mode] 提示：要使用 VR 请先启动 SteamVR 和头显，然后按 V 键切换")
            vr_mode = False
            vr_system = None
    else:
        # V16: openvr 库未安装时直接以桌面模式启动
        vr_mode = False
        if not OPENVR_AVAILABLE:
            print("[VR] openvr 库未安装，以桌面模式运行")
        else:
            print("[VR] 启动时未尝试 VR，以桌面模式运行")
        print("[Mode] 按 V 键可切换 VR / 桌面模式")

    # ▌辅助函数: 从OpenVR矩阵提取NumPy数组
    def ovr_mat34_to_np(m):
        """将OpenVR 3×4矩阵转换为4×4 NumPy矩阵"""
        r = np.eye(4, dtype=np.float32)
        for i in range(3):
            for j in range(4):
                r[i, j] = m[i][j]
        return r

    def ovr_proj_to_np(m):
        """将OpenVR投影矩阵转换为NumPy 4×4矩阵"""
        r = np.zeros((4,4), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                r[i, j] = m[i][j]
        return r

    def render_scene(mvp, width, height, cam_pos=None):
        """
        渲染完整场景（油藏+井+图例）到当前绑定的FBO。
        V10抽取为独立函数，供左右眼复用。
        cam_pos: 摄像机世界坐标，用于proximity culling。
        """
        glViewport(0, 0, width, height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # ==== 渲染油藏网格 ====
        glUseProgram(prog_res)
        glUniformMatrix4fv(uMVP_r,  1, GL_TRUE, mvp)
        glUniform1f(uDmin,   dmin)
        glUniform1f(uDrng,   drng)
        glUniform1f(uSliceX, xmin + slice_x*(xmax-xmin))
        glUniform1f(uSliceZ, zmin + slice_z*(zmax-zmin))
        glUniform1f(uVExag_r, vexag)
        glUniform1i(uShowF,  1 if show_faults else 0)
        # Proximity culling: 传入摄像机位置和半径
        if cam_pos is not None:
            glUniform3f(uCamPos, float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
        else:
            glUniform3f(uCamPos, 0.0, 0.0, 0.0)
        glUniform1f(uProxRad, float(proximity_radius))
        glUniform1f(uShellR, float(shell_radius))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_BUFFER, tbo_corners)
        glUniform1i(uCorns, 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_BUFFER, tbo_faults)
        glUniform1i(uFault, 1)
        glBindVertexArray(vao_res)
        glDrawArrays(GL_POINTS, 0, ncells)
        glBindVertexArray(0)

        # ==== 渲染井轨迹 ====
        if show_wells:
            glDisable(GL_CULL_FACE)
            glUseProgram(prog_well)
            glUniformMatrix4fv(uMVP_w, 1, GL_TRUE, mvp)
            glUniform1f(uVExag_w, vexag)
            glBindVertexArray(vao_well)
            offset = 0
            for cnt in well_counts:
                glDrawArrays(GL_LINE_STRIP, offset, cnt)
                offset += cnt
            glBindVertexArray(0)
            glEnable(GL_CULL_FACE)

        # ==== 渲染颜色图例 (仅普通模式，VR模式跳过) ====
        if not vr_mode:
            glDisable(GL_DEPTH_TEST)
            glDisable(GL_CULL_FACE)
            glUseProgram(prog_leg)
            glBindVertexArray(vao_leg)
            for i in range(leg_nverts // 4):
                glDrawArrays(GL_TRIANGLE_STRIP, i*4, 4)
            glBindVertexArray(0)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_CULL_FACE)

    # ========= 加载数据 =========
    p = GRDECLParser(FILENAME)
    if not p.parse(): glfw.terminate(); return
    # ▌解析 GRDECL 文件。如果失败 (文件不存在或数据无效)，终止。

    sz = float(np.max(p.max_b - p.min_b))
    # ▌场景尺寸 = 包围盒三个维度中最大的那个。
    # ▌用于设置摄像机距离和裁剪面。

    cam_dist = sz * 2.0
    # ▌初始摄像机距离 = 场景尺寸的 2 倍 → 确保整个模型可见。

    # ========= 编译着色器程序 =========
    try:
        prog_res  = make_prog(VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC, GEOMETRY_SHADER_SRC)
        # ▌油藏网格程序: 顶点 + 几何 + 片段 (3 个着色器)。

        prog_well = make_prog(WELL_VERT_SRC, WELL_FRAG_SRC)
        # ▌井轨迹程序: 顶点 + 片段 (2 个着色器，无几何着色器)。

        prog_leg  = make_prog(OVERLAY_VERT_SRC, OVERLAY_FRAG_SRC)
        # ▌图例覆盖层程序: 顶点 + 片段 (2 个着色器)。

    except RuntimeError as e:
        print(e); glfw.terminate(); return
    # ▌如果任何着色器编译/链接失败，打印错误并退出。

    # ========= 构建 GPU 资源 =========
    vao_res,  tbo_corners, tbo_faults, ncells = build_gpu(p)
    # ▌油藏网格: VAO + 角点TBO + 断层TBO + 单元数

    vao_well, well_counts  = build_wells_gpu(p)
    # ▌井轨迹: VAO + 每口井的顶点数列表

    vao_leg,  leg_nverts   = build_legend_gpu()
    # ▌图例条: VAO + 总顶点数

    # ========= 获取 Uniform 位置 =========
    uMVP_r    = glGetUniformLocation(prog_res,  "uMVP")
    uDmin     = glGetUniformLocation(prog_res,  "uDepthMin")
    uDrng     = glGetUniformLocation(prog_res,  "uDepthRange")
    uSliceX   = glGetUniformLocation(prog_res,  "uSliceX")
    uSliceZ   = glGetUniformLocation(prog_res,  "uSliceZ")
    uVExag_r  = glGetUniformLocation(prog_res,  "uVExag")
    uCorns    = glGetUniformLocation(prog_res,  "uCorners")
    uFault    = glGetUniformLocation(prog_res,  "uFaultFlag")
    uShowF    = glGetUniformLocation(prog_res,  "uShowFaults")
    uCamPos   = glGetUniformLocation(prog_res,  "uCamPos")
    uProxRad  = glGetUniformLocation(prog_res,  "uProximityRadius")
    uShellR   = glGetUniformLocation(prog_res,  "uShellRadius")
    # ▌V10新增: 摄像机位置和proximity culling半径的uniform位置。
    # ▌glGetUniformLocation() 返回一个整数 ID，
    # ▌之后通过 glUniform*() 函数使用这个 ID 来设置值。

    uMVP_w    = glGetUniformLocation(prog_well, "uMVP")
    uVExag_w  = glGetUniformLocation(prog_well, "uVExag")
    # ▌井轨迹着色器的 uniform 位置。

    # ========= 计算渲染参数 =========
    dmin = float(p.min_b[2])
    drng = float(p.max_b[2]-p.min_b[2]) or 1.0
    # ▌深度范围: [dmin, dmin + drng]。
    # ▌"or 1.0": 如果范围为零 (不可能但作防护)，设为 1.0 避免除零。

    xmin = float(p.min_b[0]); xmax = float(p.max_b[0])
    zmin = float(p.min_b[2]); zmax = float(p.max_b[2])
    # ▌X 和 Z 的坐标范围 (归一化后)，用于切片阈值映射。

    # 使用与工作版本v11相同的 near/far 值
    # 之前的 sz*0.00005 太小导致 VR 深度缓冲精度不足，渲染黑屏
    near = sz * 0.001
    far  = sz * 50.0
    # ▌近/远裁剪面:
    # ▌  near = 场景尺寸的 1/1000 → 足够近以避免裁掉模型前端
    # ▌  far  = 场景尺寸的 50 倍 → 足够远以包含整个场景
    # ▌  near/far 比值越小，Z-buffer 精度越低 (可能出现 z-fighting)。
    # ▌  这里 near/far = 0.001/50 = 2e-5, 在 24 位深度缓冲下勉强够用，
    # ▌  V8 加了 glPolygonOffset 来进一步缓解。

    # ========= OpenGL 全局状态设置 =========
    glEnable(GL_DEPTH_TEST)
    # ▌启用深度测试: 近处物体遮挡远处物体 (Z-buffer 算法)。

    glEnable(GL_CULL_FACE)
    # ▌V8 新增: 启用背面剔除。
    # ▌"背面"的三角形 (从摄像机看去顶点是顺时针排列的) 不会被渲染。
    # ▌这大幅减少了需要处理的三角形数量 (理论上减少约一半)。

    glCullFace(GL_BACK)
    # ▌指定剔除"背面" (而不是正面)。

    glFrontFace(GL_CCW)
    # ▌定义"正面"为逆时针 (Counter-Clockwise) 绕序。
    # ▌这是 OpenGL 的默认值，但显式设置更清晰。
    # ▌几何着色器中 emitQuad 的顶点顺序必须确保正面朝外。

    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(1.0, 1.0)
    # ▌V8 新增: 多边形偏移，减少 Z-fighting。
    # ▌Z-fighting: 当两个面几乎共面时 (如相邻网格单元的共享面)，
    # ▌  深度缓冲精度不足以区分它们，导致闪烁条纹。
    # ▌  glPolygonOffset(factor, units) 给每个多边形的深度值加一个小偏移:
    # ▌    offset = factor × slope + units × minimum_depth_step
    # ▌  (1.0, 1.0) 是常用值，给一个较小的偏移量。

    glClearColor(0.05, 0.07, 0.12, 1.0)
    # ▌背景色: 深海军蓝 (近黑色但不是纯黑)。
    # ▌这种暗色背景能让彩虹色图更加醒目。

    # ========= 打印操作说明 =========
    print("\n=== Controls ===")
    print("  Left-drag  : Rotate")
    print("  Scroll     : Zoom")
    print("  Q / A      : X cross-section (decrease / increase)")
    print("  E / D      : Z depth slice   (decrease / increase)")
    print("  Z / X      : Vertical exaggeration (decrease / increase, default 4x)")
    print("  F          : Toggle fault highlight (white = fault zone)")
    print("  W          : Toggle well trajectories")
    print("  ESC        : Quit")
    print("================")
    print("  Colorbar   : blue=shallow  →  red=deep")
    print("  White tint : fault zones\n")

    # ========= 按键消抖状态 =========
    prev_f = prev_w = prev_v = glfw.RELEASE
    frame_count = 0
    prev_i = glfw.RELEASE
    inside_mode = False
    saved_cam_dist = cam_dist
    hmd_init_inv = None  # V17: 首帧记录头显初始 pose 的逆矩阵，用于位置归零
    prev_r = glfw.RELEASE  # V17: R 键边沿检测，用于手动重新锚定
    # ▌记录 F、W、V 键的上一帧状态。
    # ▌用于检测"按下边沿" (edge trigger): 只在按键从松开变为按下的瞬间触发切换，
    # ▌而不是每帧都切换 (那样按住键会疯狂闪烁)。

    # ============================================================
    # ▌主渲染循环 (Main Render Loop)
    # ▌
    # ▌每帧执行一次迭代，直到用户关闭窗口。
    # ▌帧率取决于 GPU 性能和 V-Sync 设置，通常 45~60 FPS。
    # ============================================================
    while not glfw.window_should_close(win):

        # -------- 输入处理 --------
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(win, True)
        # ▌ESC 键 → 退出程序。

        # --- 鼠标旋转 ---
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            mx,my = glfw.get_cursor_pos(win)
            # ▌获取当前鼠标位置 (像素坐标)。

            if not mouse_pressed:
                last_x,last_y = mx,my; mouse_pressed=True
                # ▌首次按下: 记录起始位置，设置标志。
            else:
                cam_yaw   += (mx-last_x)*0.4
                cam_pitch += (my-last_y)*0.4
                # ▌计算鼠标位移量，乘以灵敏度系数 0.4 得到角度增量:
                # ▌  水平移动 → 偏航角变化
                # ▌  垂直移动 → 俯仰角变化
                # ▌0.4 表示鼠标每移动 1 像素，旋转 0.4°。

                cam_pitch  = float(np.clip(cam_pitch,-89,89))
                # ▌限制俯仰角在 [-89°, +89°]，防止万向锁。

                last_x,last_y = mx,my
                # ▌更新上一帧鼠标位置。
        else:
            mouse_pressed = False
        # ▌鼠标释放后重置标志。

        # --- 连续按键 (持续响应) ---
        step = 0.003
        # ▌每帧的切片增量。约 60 FPS 下，从 0% 到 100% 需要约 333 帧 ≈ 5.5 秒。

        if glfw.get_key(win, glfw.KEY_Q) == glfw.PRESS: slice_x = max(0.0, slice_x-step)
        # ▌Q 键: 减小 X 切面位置 (切掉更多)。

        if glfw.get_key(win, glfw.KEY_A) == glfw.PRESS: slice_x = min(1.0, slice_x+step)
        # ▌A 键: 增大 X 切面位置 (恢复显示)。

        if glfw.get_key(win, glfw.KEY_E) == glfw.PRESS: slice_z = max(0.0, slice_z-step)
        # ▌E 键: 减小 Z 切面。

        if glfw.get_key(win, glfw.KEY_D) == glfw.PRESS: slice_z = min(1.0, slice_z+step)
        # ▌D 键: 增大 Z 切面。

        if glfw.get_key(win, glfw.KEY_Z) == glfw.PRESS: vexag = max(1.0,  vexag-0.1)
        # ▌Z 键: 减小垂直夸大 (最小 1.0 = 真实比例)。

        if glfw.get_key(win, glfw.KEY_X) == glfw.PRESS: vexag = min(50.0, vexag+0.1)
        # ▌X 键: 增大垂直夸大 (最大 50 倍)。

        # B/N键: 调节Proximity Culling半径 (V10新增)
        prox_step = sz * 0.005
        if glfw.get_key(win, glfw.KEY_N) == glfw.PRESS:
            proximity_radius = min(sz * 2.0, proximity_radius + prox_step)
        # ▌N 键: 增大proximity半径（裁剪更大范围，"挖"出更大的空洞）。
        if glfw.get_key(win, glfw.KEY_B) == glfw.PRESS:
            proximity_radius = max(0.0, proximity_radius - prox_step)
        # ▌B 键: 减小proximity半径（恢复显示，0.0=关闭裁剪）。

        # --- 切换按键 (边沿触发, 消抖) ---
        cur_f = glfw.get_key(win, glfw.KEY_F)
        if cur_f == glfw.PRESS and prev_f == glfw.RELEASE:
            show_faults = not show_faults
        prev_f = cur_f

        cur_w = glfw.get_key(win, glfw.KEY_W)
        if cur_w == glfw.PRESS and prev_w == glfw.RELEASE:
            show_wells = not show_wells
        prev_w = cur_w

        # ════════════════════════════════════════════════════════════════
        # V17 新增: R 键重新锚定 VR 初始位置
        # ════════════════════════════════════════════════════════════════
        # 用途: 如果你戴头显时站歪了, 或者想换个起点重新走入模型,
        #       按 R 键会把当前位置作为新的"原点", 模型重新出现在你正前方.
        cur_r = glfw.get_key(win, glfw.KEY_R)
        if cur_r == glfw.PRESS and prev_r == glfw.RELEASE:
            hmd_init_inv = None  # 下一帧会重新锚定
            print("[VR] 已请求重新锚定, 下一帧生效")
        prev_r = cur_r

        # ════════════════════════════════════════════════════════════════
        # V14 修改：VR 模式下不允许键盘控制视角
        # ════════════════════════════════════════════════════════════════
        # 导师要求：VR 中通过身体走动控制位置，键盘不应改变 VR 视角。
        # 所有 [/] I 键的功能仅在桌面模式下生效。
        if not vr_mode:
            # 桌面模式下保留 I 键，方便桌面调试演示
            cur_i = glfw.get_key(win, glfw.KEY_I)
            if cur_i == glfw.PRESS and prev_i == glfw.RELEASE:
                if not inside_mode:
                    saved_cam_dist = cam_dist
                    cam_dist = 1.0
                    inside_mode = True
                    print("[演示] 桌面模式：已进入模型内部")
                else:
                    cam_dist = saved_cam_dist
                    inside_mode = False
                    print("[演示] 桌面模式：已退出模型内部")
            prev_i = cur_i

        cur_v = glfw.get_key(win, glfw.KEY_V)
        if cur_v == glfw.PRESS and prev_v == glfw.RELEASE:
            if not vr_mode:
                # 尝试实时初始化VR
                if OPENVR_AVAILABLE:
                    try:
                        if vr_system is None:
                            vr_system = openvr.init(openvr.VRApplication_Scene)
                            vr_render_width, vr_render_height = vr_system.getRecommendedRenderTargetSize()
                            print(f"[VR] 头显连接成功! 分辨率: {vr_render_width}×{vr_render_height}")

                            # 创建左眼FBO
                            vr_fbo_l = glGenFramebuffers(1)
                            vr_tex_l = glGenTextures(1)
                            glBindTexture(GL_TEXTURE_2D, vr_tex_l)
                            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, vr_render_width, vr_render_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                            vr_depth_l = glGenRenderbuffers(1)
                            glBindRenderbuffer(GL_RENDERBUFFER, vr_depth_l)
                            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, vr_render_width, vr_render_height)
                            glBindFramebuffer(GL_FRAMEBUFFER, vr_fbo_l)
                            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, vr_tex_l, 0)
                            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, vr_depth_l)

                            # 创建右眼FBO
                            vr_fbo_r = glGenFramebuffers(1)
                            vr_tex_r = glGenTextures(1)
                            glBindTexture(GL_TEXTURE_2D, vr_tex_r)
                            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, vr_render_width, vr_render_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                            vr_depth_r = glGenRenderbuffers(1)
                            glBindRenderbuffer(GL_RENDERBUFFER, vr_depth_r)
                            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, vr_render_width, vr_render_height)
                            glBindFramebuffer(GL_FRAMEBUFFER, vr_fbo_r)
                            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, vr_tex_r, 0)
                            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, vr_depth_r)
                            glBindFramebuffer(GL_FRAMEBUFFER, 0)

                        vr_mode = True
                        print("[VR] 已切换到VR模式")
                    except Exception as e:
                        print(f"[VR] 初始化失败: {e}")
                        print("[VR] 请确认SteamVR已启动且头显已连接")
                else:
                    print("[VR] openvr库未安装")
            else:
                vr_mode = False
                print("[VR] 已切换到普通窗口模式")
        prev_v = cur_v

        # -------- 摄像机计算 (轨道摄像机，作为VR基础视图) --------
        ry,rp = np.radians(cam_yaw), np.radians(cam_pitch)
        eye = np.array([
            cam_dist*np.cos(rp)*np.sin(ry),
            cam_dist*np.sin(rp),
            cam_dist*np.cos(rp)*np.cos(ry),
        ], dtype=np.float32)
        V  = mat_lookat(eye, np.zeros(3,dtype=np.float32), np.array([0,1,0],dtype=np.float32))
        Pr = mat_perspective(45.0, WINDOW_WIDTH/WINDOW_HEIGHT, near, far)
        mvp = Pr @ V

        # ════════════════════════════════════════════════════════════════
        # VR 模式渲染 (V14 重写) - 真实 6DoF 头显追踪
        # ════════════════════════════════════════════════════════════════
        # 导师要求:
        #   ① 模型必须静止在世界坐标的固定位置（不跟着头动）
        #   ② 相机 6DoF 完全由头显追踪决定（位置 + 朝向）
        #   ③ 不使用任何键盘控制 VR 视角
        #   ④ 用户通过身体走动来观察模型外部/内部
        #
        # 实现:
        #   模型放置位置:   世界原点 (0, 0, 0)，固定不动
        #   用户起始位置:   模型前方 VR_USER_OFFSET 米处
        #   位置缩放因子:   VR_WORLD_SCALE - 把现实1米映射到虚拟 N 米
        #                  让用户用脚走几步就能从外部走进模型内部
        if vr_mode and vr_system is not None and vr_fbo_l is not None:
            try:
                # 获取头部姿态
                render_poses = (openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount)()
                openvr.VRCompositor().waitGetPoses(render_poses, None)
                hmd_pose = render_poses[openvr.k_unTrackedDeviceIndex_Hmd]

                if hmd_pose.bPoseIsValid:
                    # hmd_mat 是头显在 SteamVR 追踪空间(米)中的姿态矩阵
                    hmd_mat = ovr_mat34_to_np(hmd_pose.mDeviceToAbsoluteTracking)

                    # ════════════════════════════════════════════════════════════
                    # V17 核心修复：首帧锚定 + 相对位姿计算
                    # ════════════════════════════════════════════════════════════
                    # 问题: hmd_mat[:3, 3] 是 SteamVR 追踪空间的绝对坐标,
                    #       包含头部高度 (Y≈1.6m), 乘以 20000 后会变成 32000m,
                    #       把用户传送到模型上方 32 公里. 而且每次启动追踪空间
                    #       绝对原点都有漂移, 导致每次运行结果不同.
                    #
                    # 修复: 首帧把当前 hmd_mat 作为"原点", 之后所有位姿都相对
                    #       于这个原点计算. 这样无论你戴头显时站哪里、朝向哪里,
                    #       模型一定在你正前方 VR_USER_OFFSET 米处.

                    # 首帧锚定（或按 R 键时重新锚定）
                    if hmd_init_inv is None:
                        hmd_init_inv = np.linalg.inv(hmd_mat).astype(np.float32)
                        print(f"[VR] ★ 已锚定初始位置 ★")
                        print(f"[VR]   头显物理位置 (SteamVR 坐标): ({hmd_mat[0,3]:.2f}, {hmd_mat[1,3]:.2f}, {hmd_mat[2,3]:.2f}) m")
                        print(f"[VR]   模型固定在你正前方 {VR_USER_OFFSET:.0f} 米处")
                        print(f"[VR]   走 {VR_USER_OFFSET/VR_WORLD_SCALE:.2f} 米 ≈ 走到模型中心")
                        print(f"[VR]   按 R 键可重新锚定（站歪了用）")

                    # 相对位姿: T_rel = inv(T_init) * T_current
                    # 这个矩阵表示"相对启动那一刻你移动/转动了多少"
                    T_relative = (hmd_init_inv @ hmd_mat).astype(np.float32)

                    # 提取相对旋转和相对位移
                    hmd_rot_rel = np.eye(4, dtype=np.float32)
                    hmd_rot_rel[:3, :3] = T_relative[:3, :3]

                    # 关键: 用相对位移而非绝对位置, 这样 Y 分量不会再爆炸
                    hmd_pos_phys_rel = T_relative[:3, 3]

                    # ════════════════════════════════════════════════════════════
                    # 颈部补偿 — 让"转头看四周"时模型不动, 只有"身体走动"才移动
                    # ════════════════════════════════════════════════════════════
                    # 问题: 头显追踪点在眼睛位置, 不在脖子转轴。你转头/点头时,
                    #   追踪点会绕脖子画一道弧 -> 产生假位移 -> ×scale 放大 ->
                    #   模型跟着头"轻微动"(这正是 5.21 演示时导师看到的问题)。
                    # 修复: 假设脖子转轴在追踪点后方0.10m、下方0.08m。若这一帧只是
                    #   纯旋转(没走路), 追踪点应落在 rot_only_pos; 把它减掉, 剩下的
                    #   才是真正的身体走动位移。转头时弧线位移被抵消 -> 模型不动;
                    #   前后走动时 -> 位移保留 -> 仍能进入模型。
                    neck_pivot = np.array([0.0, -0.08, -0.10], dtype=np.float32)
                    R_rel = T_relative[:3, :3]
                    rot_only_pos = R_rel @ (-neck_pivot) + neck_pivot
                    pure_walk = hmd_pos_phys_rel - rot_only_pos

                    # ════════════════════════════════════════════════════════════
                    # 关键修复: 颈部补偿只用于 X/Y(左右/上下), Z(前后)用原始位移
                    # ════════════════════════════════════════════════════════════
                    # 之前的问题: hmd_pos_world = pure_walk * scale, 连 Z 也用了补偿后
                    #   的值。但你往前走时, 头部不可避免的轻微俯仰让颈部补偿在 Z 上
                    #   产生一个反向分量, 把你的"前进"抵消掉 -> 虚拟位置 Z 几乎不变
                    #   -> 怎么走都进不去模型。
                    # 修复原理: 转头/点头造成的假位移主要在 X(左右弧)和 Y(上下弧),
                    #   Z(前后)方向的假位移很小。所以:
                    #     X/Y 用补偿后(pure_walk) -> 转头看四周时模型不晃
                    #     Z   用原始位移         -> 前进位移完整保留, 能走进模型
                    hmd_pos_world = np.array([
                        pure_walk[0] * VR_WORLD_SCALE,          # X: 补偿后, 转头不晃
                        pure_walk[1] * VR_WORLD_SCALE,          # Y: 补偿后, 点头不晃
                        hmd_pos_phys_rel[2] * VR_WORLD_SCALE    # Z: 原始位移, 保证能走进去
                    ], dtype=np.float32)

                    # 用户在世界坐标的最终位置 = 初始偏移 + 相对位移
                    user_world_pos = np.array([
                        hmd_pos_world[0],
                        hmd_pos_world[1],
                        hmd_pos_world[2] + VR_USER_OFFSET
                    ], dtype=np.float32)

                    # camera-to-world 矩阵: 旋转 + 平移
                    cam_to_world = hmd_rot_rel.copy()
                    cam_to_world[:3, 3] = user_world_pos

                    # 视图矩阵 V_vr = world-to-camera = inverse(camera-to-world)
                    V_vr = np.linalg.inv(cam_to_world).astype(np.float32)

                    # VR 相机的世界坐标（供 proximity culling 使用）
                    vr_cam_pos = user_world_pos

                    # V17: 重新开启调试输出, 每 60 帧打印一次
                    if frame_count % 60 == 0:
                        dist = float(np.linalg.norm(user_world_pos))
                        print(f"[VR追踪] 相对位移={hmd_pos_phys_rel[0]:+.2f},{hmd_pos_phys_rel[1]:+.2f},{hmd_pos_phys_rel[2]:+.2f}m "
                              f"虚拟位置=({user_world_pos[0]:+.0f},{user_world_pos[1]:+.0f},{user_world_pos[2]:+.0f}) "
                              f"距离={dist:.0f} (模型半径≈{VR_USER_OFFSET*0.5:.0f}, 球壳={shell_radius:.0f})")
                else:
                    # 头显姿态无效：使用安全的回退值
                    V_vr = mat_lookat(
                        np.array([0, 0, VR_USER_OFFSET], dtype=np.float32),
                        np.array([0, 0, 0], dtype=np.float32),
                        np.array([0, 1, 0], dtype=np.float32))
                    vr_cam_pos = np.array([0, 0, VR_USER_OFFSET], dtype=np.float32)

                # 左眼
                proj_l = ovr_proj_to_np(
                    vr_system.getProjectionMatrix(openvr.Eye_Left, near, far)).T
                e2h_l  = ovr_mat34_to_np(
                    vr_system.getEyeToHeadTransform(openvr.Eye_Left))
                mvp_l  = proj_l @ np.linalg.inv(e2h_l) @ V_vr

                # 右眼
                proj_r = ovr_proj_to_np(
                    vr_system.getProjectionMatrix(openvr.Eye_Right, near, far)).T
                e2h_r  = ovr_mat34_to_np(
                    vr_system.getEyeToHeadTransform(openvr.Eye_Right))
                mvp_r  = proj_r @ np.linalg.inv(e2h_r) @ V_vr

                # 渲染左眼到FBO（传入头显位置用于proximity culling）
                glBindFramebuffer(GL_FRAMEBUFFER, vr_fbo_l)
                status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
                if status != GL_FRAMEBUFFER_COMPLETE:
                    print(f"[VR] 左眼FBO不完整: {status}")
                glEnable(GL_DEPTH_TEST); glEnable(GL_CULL_FACE)
                render_scene(mvp_l.astype(np.float32), vr_render_width, vr_render_height, vr_cam_pos)

                # 渲染右眼到FBO
                glBindFramebuffer(GL_FRAMEBUFFER, vr_fbo_r)
                glEnable(GL_DEPTH_TEST); glEnable(GL_CULL_FACE)
                render_scene(mvp_r.astype(np.float32), vr_render_width, vr_render_height, vr_cam_pos)

                # ── 先 blit 镜像到桌面窗口（在 submit 之前，避免 OpenVR 状态污染）──
                glBindFramebuffer(GL_READ_FRAMEBUFFER, vr_fbo_l)
                glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
                glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
                glClearColor(0.0, 0.0, 0.0, 1.0)
                glClear(GL_COLOR_BUFFER_BIT)
                glBlitFramebuffer(
                    0, 0, vr_render_width, vr_render_height,
                    0, 0, WINDOW_WIDTH, WINDOW_HEIGHT,
                    GL_COLOR_BUFFER_BIT, GL_LINEAR)
                glBindFramebuffer(GL_FRAMEBUFFER, 0)

                # ── 然后提交给 SteamVR compositor ──
                tex_l = openvr.Texture_t()
                tex_l.handle   = int(vr_tex_l)
                tex_l.eType    = openvr.TextureType_OpenGL
                tex_l.eColorSpace = openvr.ColorSpace_Gamma
                tex_r = openvr.Texture_t()
                tex_r.handle   = int(vr_tex_r)
                tex_r.eType    = openvr.TextureType_OpenGL
                tex_r.eColorSpace = openvr.ColorSpace_Gamma
                openvr.VRCompositor().submit(openvr.Eye_Left,  tex_l)
                openvr.VRCompositor().submit(openvr.Eye_Right, tex_r)

            except Exception as e:
                import traceback
                print(f"[VR] 渲染错误: {e}")
                traceback.print_exc()
                # 不要立即关闭VR模式，让我们先看到错误

        else:
            # -------- 普通窗口模式渲染 --------
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glEnable(GL_DEPTH_TEST); glEnable(GL_CULL_FACE)
            render_scene(mvp, WINDOW_WIDTH, WINDOW_HEIGHT, eye)

        # -------- 更新窗口标题 --------
        mode_str = "VR" if vr_mode else "PC"
        glfw.set_window_title(win,
            f"Norne v18 [{mode_str}]  |  X:{slice_x*100:.0f}%  Z:{slice_z*100:.0f}%  "
            f"VExag:{vexag:.1f}x  "
            f"Faults:{'ON' if show_faults else 'OFF'}  "
            f"Wells:{'ON' if show_wells else 'OFF'}  "
            f"Prox:{proximity_radius:.0f}  Shell:{shell_radius:.0f}  " + ("[VR: 向前走→进入模型  R=重新锚定]" if vr_mode else f"[I]={'退出内部' if inside_mode else '进入内部'}  [V]=切换VR"))

        # -------- 帧结束 --------
        frame_count += 1
        glfw.swap_buffers(win)
        glfw.poll_events()

    # ========= 清理资源 =========
    if vr_system is not None:
        openvr.shutdown()
        # ▌关闭OpenVR运行时，释放头显连接。
    glfw.terminate()


# ============================================================================
# 第十三部分: 程序入口
# ============================================================================

if __name__ == "__main__":
    main()
    # ▌Python 标准入口判断:
    # ▌  当直接运行本脚本时 (python norne_viewer.py), __name__ == "__main__"
    # ▌  当作为模块导入时 (import norne_viewer), __name__ == "norne_viewer"
    # ▌  只有直接运行时才执行 main()。