# Author:Ach1n
# Data:2025/12/23 0:08
# Flag:I earned it.
# Author: Enhanced by Claude for Thesis Demonstration
# Date: 2025/12/23
# Purpose: GRDECL Corner Point Grid Visualization with Geometry Shader
# 这是为您的毕业论文准备的完善版本

# Author: Enhanced by Claude for Intel UHD Graphics 630 Compatibility
# Date: 2025/12/23
# Purpose: GRDECL Visualization with OpenGL 3.3 Compatibility
# 专门为 Intel 核显优化，兼容 OpenGL 3.3

# Author: Enhanced by Claude - Full Interactive Version
# Date: 2025/12/23
# Purpose: GRDECL Visualization with Multi-axis Slicing & Advanced Controls
# 功能增强版：三轴切片 + 属性切换 + 线框模式

import glfw
from OpenGL.GL import *
import numpy as np
import glm
import ctypes

# ==========================================
# 配置参数
# ==========================================
GRID_SIZE = 25
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# 全局变量 - 摄像机
cam_distance = 60.0
cam_yaw = 0.5
cam_pitch = 0.5
last_x, last_y = 0, 0
mouse_pressed = False

# 全局变量 - 切片参数（新增：支持三轴）
slice_x = 1.0  # X轴切片 (Q/A 控制)
slice_y = 1.0  # Y轴切片 (W/S 控制)
slice_z = 1.0  # Z轴切片 (E/D 控制)

# 全局变量 - 显示模式
wireframe_mode = False  # 线框模式开关 (F键)
attribute_mode = 0  # 属性显示模式 (1/2键切换)


# ==========================================
# 1. GRDECL 数据生成
# ==========================================
def generate_grdecl_corner_point_data(size):
    """生成 GRDECL 角点网格数据"""
    print(f"[DATA] 正在生成 {size}^3 GRDECL 种子数据...")

    data = []

    for x in range(size):
        for y in range(size):
            for z in range(size):
                cx = x - size / 2.0
                cy = y - size / 2.0

                # GRDECL 地质畸变
                distortion = np.sin(x * 0.25) * 2.5 + np.cos(y * 0.25) * 2.0
                cz = (z - size / 2.0) * 1.5 + distortion

                # 两种属性值（可以切换显示）
                attr1 = z / size  # 属性1：深度梯度
                attr2 = (np.sin(x * 0.3) + np.cos(y * 0.3) + 1.0) / 2.0  # 属性2：横向变化

                # 数据格式: [x, y, z, distortion, attr1, attr2]
                data.append([cx, cy, cz, distortion, attr1, attr2])

    result = np.array(data, dtype=np.float32)
    print(f"[DATA] 完成! 共 {len(result)} 个网格单元")
    return result


# ==========================================
# 2. 着色器代码 (增强版)
# ==========================================

VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in float aDistort;
layout (location = 2) in float aAttr1;
layout (location = 3) in float aAttr2;

out VS_OUT {
    vec3 center;      // 中心点坐标（用于三轴切片判断）
    float distortion;
    float attr1;
    float attr2;
} vs_out;

void main() {
    gl_Position = vec4(aPos, 1.0);
    vs_out.center = aPos;
    vs_out.distortion = aDistort;
    vs_out.attr1 = aAttr1;
    vs_out.attr2 = aAttr2;
}
"""

GEOMETRY_SHADER = """
#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 36) out;

in VS_OUT {
    vec3 center;
    float distortion;
    float attr1;
    float attr2;
} gs_in[];

out vec3 fNormal;
out float fAttr;

uniform mat4 mvp;
uniform vec3 uSlice;     // 三轴切片参数 (x, y, z)
uniform int uAttrMode;   // 属性模式选择

// 立方体8个角点
const vec3 corners[8] = vec3[](
    vec3(-0.5, -0.5, -0.5), vec3( 0.5, -0.5, -0.5),
    vec3( 0.5,  0.5, -0.5), vec3(-0.5,  0.5, -0.5),
    vec3(-0.5, -0.5,  0.5), vec3( 0.5, -0.5,  0.5),
    vec3( 0.5,  0.5,  0.5), vec3(-0.5,  0.5,  0.5)
);

// 6个面的索引（一维数组）
const int face_indices[24] = int[](
    0, 1, 2, 3,  // 底面
    4, 7, 6, 5,  // 顶面
    0, 4, 5, 1,  // 前面
    2, 6, 7, 3,  // 后面
    0, 3, 7, 4,  // 左面
    1, 5, 6, 2   // 右面
);

const vec3 normals[6] = vec3[](
    vec3( 0,  0, -1),
    vec3( 0,  0,  1),
    vec3( 0, -1,  0),
    vec3( 0,  1,  0),
    vec3(-1,  0,  0),
    vec3( 1,  0,  0)
);

void main() {
    vec4 center = gl_in[0].gl_Position;

    // === 三轴几何剔除（核心功能） ===
    // 归一化到 [0, 1] 范围
    vec3 normPos = (gs_in[0].center / 50.0) + 0.5;

    // 任意一轴超出切片范围就剔除
    if (normPos.x > uSlice.x + 0.02) return;
    if (normPos.y > uSlice.y + 0.02) return;
    if (normPos.z > uSlice.z + 0.02) return;

    // 计算8个角点
    vec4 actual_corners[8];
    for(int i = 0; i < 8; i++) {
        vec3 offset = corners[i];
        vec3 pos = center.xyz + offset;

        // GRDECL 变形
        if (pos.y > center.y) {
            pos.x += gs_in[0].distortion * 0.15;
            pos.z += gs_in[0].distortion * 0.08;
        }

        actual_corners[i] = vec4(pos, 1.0);
    }

    // 根据模式选择属性
    if (uAttrMode == 0) {
        fAttr = gs_in[0].attr1;
    } else {
        fAttr = gs_in[0].attr2;
    }

    // 构建6个面
    for(int face = 0; face < 6; face++) {
        fNormal = normals[face];

        int base = face * 4;
        vec4 v0 = actual_corners[face_indices[base + 0]];
        vec4 v1 = actual_corners[face_indices[base + 1]];
        vec4 v2 = actual_corners[face_indices[base + 2]];
        vec4 v3 = actual_corners[face_indices[base + 3]];

        // 三角形1
        gl_Position = mvp * v0; EmitVertex();
        gl_Position = mvp * v1; EmitVertex();
        gl_Position = mvp * v2; EmitVertex();
        EndPrimitive();

        // 三角形2
        gl_Position = mvp * v0; EmitVertex();
        gl_Position = mvp * v2; EmitVertex();
        gl_Position = mvp * v3; EmitVertex();
        EndPrimitive();
    }
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 fNormal;
in float fAttr;
out vec4 FragColor;

uniform int uAttrMode;

void main() {
    // 光照
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));
    float diff = max(dot(normalize(fNormal), lightDir), 0.0);
    float ambient = 0.35;

    // 根据模式选择颜色方案
    vec3 color;

    if (uAttrMode == 0) {
        // 模式1：深度梯度（蓝->青->黄->红）
        if (fAttr < 0.33) {
            color = mix(vec3(0.1, 0.1, 0.7), vec3(0.1, 0.7, 0.8), fAttr * 3.0);
        } else if (fAttr < 0.66) {
            color = mix(vec3(0.1, 0.7, 0.8), vec3(0.9, 0.8, 0.3), (fAttr - 0.33) * 3.0);
        } else {
            color = mix(vec3(0.9, 0.8, 0.3), vec3(0.9, 0.3, 0.2), (fAttr - 0.66) * 3.0);
        }
    } else {
        // 模式2：横向变化（绿->黄->橙）
        color = mix(vec3(0.2, 0.8, 0.3), vec3(0.9, 0.5, 0.1), fAttr);
    }

    vec3 finalColor = color * (diff * 0.7 + ambient);
    FragColor = vec4(finalColor, 1.0);
}
"""


# ==========================================
# 3. 着色器编译
# ==========================================
def create_shader_program():
    """编译着色器"""
    vs = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vs, VERTEX_SHADER)
    glCompileShader(vs)
    if not glGetShaderiv(vs, GL_COMPILE_STATUS):
        print(f"[ERROR] VS: {glGetShaderInfoLog(vs).decode()}")
        return None

    gs = glCreateShader(GL_GEOMETRY_SHADER)
    glShaderSource(gs, GEOMETRY_SHADER)
    glCompileShader(gs)
    if not glGetShaderiv(gs, GL_COMPILE_STATUS):
        print(f"[ERROR] GS: {glGetShaderInfoLog(gs).decode()}")
        return None

    fs = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fs, FRAGMENT_SHADER)
    glCompileShader(fs)
    if not glGetShaderiv(fs, GL_COMPILE_STATUS):
        print(f"[ERROR] FS: {glGetShaderInfoLog(fs).decode()}")
        return None

    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, gs)
    glAttachShader(program, fs)
    glLinkProgram(program)

    if not glGetProgramiv(program, GL_LINK_STATUS):
        print(f"[ERROR] Link: {glGetProgramInfoLog(program).decode()}")
        return None

    glDeleteShader(vs)
    glDeleteShader(gs)
    glDeleteShader(fs)

    print("[SHADER] ✅ 着色器编译成功!")
    return program


# ==========================================
# 4. 交互回调 (增强版)
# ==========================================
def mouse_button_callback(window, button, action, mods):
    global mouse_pressed
    if button == glfw.MOUSE_BUTTON_LEFT:
        mouse_pressed = (action == glfw.PRESS)


def cursor_position_callback(window, xpos, ypos):
    global last_x, last_y, cam_yaw, cam_pitch
    dx = xpos - last_x
    dy = ypos - last_y
    last_x = xpos
    last_y = ypos

    if mouse_pressed:
        cam_yaw -= dx * 0.005
        cam_pitch -= dy * 0.005
        cam_pitch = max(-1.5, min(1.5, cam_pitch))


def scroll_callback(window, xoffset, yoffset):
    global cam_distance
    cam_distance -= yoffset * 2.0
    cam_distance = max(10.0, min(150.0, cam_distance))


def key_callback(window, key, scancode, action, mods):
    """增强的键盘控制"""
    global slice_x, slice_y, slice_z, wireframe_mode, attribute_mode

    if action == glfw.PRESS or action == glfw.REPEAT:
        step = 0.03

        # === X轴切片 ===
        if key == glfw.KEY_Q:
            slice_x = max(0.0, slice_x - step)
        if key == glfw.KEY_A:
            slice_x = min(1.0, slice_x + step)

        # === Y轴切片 (新增) ===
        if key == glfw.KEY_W:
            slice_y = max(0.0, slice_y - step)
        if key == glfw.KEY_S:
            slice_y = min(1.0, slice_y + step)

        # === Z轴切片 (新增) ===
        if key == glfw.KEY_E:
            slice_z = max(0.0, slice_z - step)
        if key == glfw.KEY_D:
            slice_z = min(1.0, slice_z + step)

        # === 快捷操作 ===
        if key == glfw.KEY_R:
            # 重置所有切片
            slice_x = slice_y = slice_z = 1.0
            print("[RESET] 切片已重置")

        if key == glfw.KEY_X:
            # X轴剖面（只切X，保留YZ）
            slice_x = 0.5
            slice_y = slice_z = 1.0
            print("[VIEW] X轴剖面视图")

        if key == glfw.KEY_Y:
            # Y轴剖面
            slice_y = 0.5
            slice_x = slice_z = 1.0
            print("[VIEW] Y轴剖面视图")

        if key == glfw.KEY_Z:
            # Z轴剖面
            slice_z = 0.5
            slice_x = slice_y = 1.0
            print("[VIEW] Z轴剖面视图")

    # === 模式切换（按下立即生效）===
    if action == glfw.PRESS:
        if key == glfw.KEY_F:
            # 线框模式开关
            wireframe_mode = not wireframe_mode
            if wireframe_mode:
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                print("[MODE] 线框模式")
            else:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                print("[MODE] 填充模式")

        if key == glfw.KEY_1:
            attribute_mode = 0
            print("[ATTR] 属性模式1：深度梯度")

        if key == glfw.KEY_2:
            attribute_mode = 1
            print("[ATTR] 属性模式2：横向分布")

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)


# ==========================================
# 5. 主程序
# ==========================================
def main():
    global last_x, last_y

    if not glfw.init():
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(
        WINDOW_WIDTH, WINDOW_HEIGHT,
        "GRDECL Enhanced - Multi-Axis Slicing",
        None, None
    )

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    # 注册回调
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_key_callback(window, key_callback)
    last_x, last_y = glfw.get_cursor_pos(window)

    # 系统信息
    print("\n" + "=" * 70)
    print("GRDECL Enhanced Visualizer - Multi-Axis Interactive Edition")
    print("=" * 70)
    print(f"OpenGL: {glGetString(GL_VERSION).decode()}")
    print(f"显卡: {glGetString(GL_RENDERER).decode()}")
    print("=" * 70)

    # 生成数据
    grid_data = generate_grdecl_corner_point_data(GRID_SIZE)

    # 配置 VAO/VBO
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, grid_data.nbytes, grid_data, GL_STATIC_DRAW)

    stride = 6 * 4  # 6个float
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16))
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))
    glEnableVertexAttribArray(3)

    program = create_shader_program()
    if not program:
        glfw.terminate()
        return

    glEnable(GL_DEPTH_TEST)

    # 打印完整控制说明
    print("\n" + "=" * 70)
    print("📋 完整操作指南")
    print("=" * 70)
    print("【摄像机控制】")
    print("  鼠标左键拖拽 → 旋转视角")
    print("  鼠标滚轮     → 缩放距离")
    print()
    print("【三轴切片控制】")
    print("  Q / A → X轴切片 (左右)")
    print("  W / S → Y轴切片 (上下)")
    print("  E / D → Z轴切片 (前后)")
    print()
    print("【快捷视图】")
    print("  X → X轴剖面视图")
    print("  Y → Y轴剖面视图")
    print("  Z → Z轴剖面视图")
    print("  R → 重置所有切片")
    print()
    print("【显示模式】")
    print("  1 → 属性模式1 (深度梯度)")
    print("  2 → 属性模式2 (横向分布)")
    print("  F → 线框模式切换")
    print()
    print("  ESC → 退出程序")
    print("=" * 70)
    print("\n[INFO] 渲染中...\n")

    # 主循环
    frame_count = 0
    import time
    last_time = time.time()

    while not glfw.window_should_close(window):
        glClearColor(0.15, 0.15, 0.18, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(program)

        # 摄像机
        cam_x = np.sin(cam_yaw) * np.cos(cam_pitch) * cam_distance
        cam_y = np.sin(cam_pitch) * cam_distance
        cam_z = np.cos(cam_yaw) * np.cos(cam_pitch) * cam_distance

        view = glm.lookAt(
            glm.vec3(cam_x, cam_y, cam_z),
            glm.vec3(0, 0, 0),
            glm.vec3(0, 1, 0)
        )
        projection = glm.perspective(
            glm.radians(45.0),
            WINDOW_WIDTH / WINDOW_HEIGHT,
            0.1, 500.0
        )
        mvp = projection * view

        # 传递 Uniform
        glUniformMatrix4fv(
            glGetUniformLocation(program, "mvp"),
            1, GL_FALSE, glm.value_ptr(mvp)
        )
        glUniform3f(
            glGetUniformLocation(program, "uSlice"),
            slice_x, slice_y, slice_z
        )
        glUniform1i(
            glGetUniformLocation(program, "uAttrMode"),
            attribute_mode
        )

        # 绘制
        glBindVertexArray(VAO)
        glDrawArrays(GL_POINTS, 0, len(grid_data))

        glfw.swap_buffers(window)
        glfw.poll_events()

        # FPS 和状态显示
        frame_count += 1
        current_time = time.time()
        if current_time - last_time >= 1.0:
            fps = frame_count / (current_time - last_time)
            mode_str = "Depth" if attribute_mode == 0 else "Lateral"
            wire_str = "Wire" if wireframe_mode else "Fill"
            title = (f"GRDECL | FPS:{fps:.0f} | X:{slice_x:.2f} Y:{slice_y:.2f} Z:{slice_z:.2f} "
                     f"| {mode_str} | {wire_str}")
            glfw.set_window_title(window, title)
            frame_count = 0
            last_time = current_time

    # 清理
    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glDeleteProgram(program)
    glfw.terminate()
    print("\n[INFO] 程序正常退出")


if __name__ == "__main__":
    main()