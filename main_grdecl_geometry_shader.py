# Author:Ach1n
# Data:2025/12/21 18:38
# Flag:I earned it.
import glfw
from OpenGL.GL import *
import numpy as np
import glm
import ctypes

# --- 配置 ---
# 稍微降低一点分辨率以确保几何着色器性能
GRID_SIZE = 30
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# 全局变量
cam_distance = 60.0
cam_yaw = 0.5
cam_pitch = 0.5
last_x, last_y = 0, 0
mouse_pressed = False
slice_x = 1.0  # 切片参数


# ==========================================
# 1. 模拟 GRDECL (角点网格) 数据生成
# ==========================================
def generate_corner_point_grid(size):
    """
    生成仅仅包含【中心点】和【畸变参数】的数据。
    真正的几何体（六面体）将由显卡(Geometry Shader)实时生成。
    """
    print(f"Generating GRDECL-like seed points for {size}^3 grid...")

    data = []

    for x in range(size):
        for y in range(size):
            for z in range(size):
                # 基础坐标 (中心点)
                cx = x - size / 2.0
                cy = y - size / 2.0

                # 模拟地质起伏 (正弦波畸变)
                # 这会让生成的网格看起来像波浪一样扭曲，而不是平直的方块
                distortion = np.sin(x * 0.2) * 2.0 + np.cos(y * 0.2) * 2.0
                cz = (z - size / 2.0) * 1.5 + distortion

                # 属性值 (随深度变化，用于着色)
                attr = (z / size)

                # 数据格式: [x, y, z, distortion, attr]
                data.append([cx, cy, cz, distortion, attr])

    return np.array(data, dtype=np.float32)


# ==========================================
# 2. 着色器代码 (修复了数组定义)
# ==========================================

# Vertex Shader: 传递数据
VS_CODE = """
#version 330 core
layout (location = 0) in vec3 aPos;       // 中心点
layout (location = 1) in float aDistort;  // 畸变参数
layout (location = 2) in float aAttr;     // 属性

out VS_OUT {
    float distortion;
    float attr;
} vs_out;

void main() {
    gl_Position = vec4(aPos, 1.0);
    vs_out.distortion = aDistort;
    vs_out.attr = aAttr;
}
"""

# Geometry Shader: 核心逻辑
# 【修复点】：去掉了二维数组，改用一维数组
GS_CODE = """
#version 330 core
layout (points) in; // 输入：1个点
layout (triangle_strip, max_vertices = 24) out; // 输出：最多24个顶点(构成一个六面体)

in VS_OUT {
    float distortion;
    float attr;
} gs_in[];

out float fAttr; 
out vec3 fNormal;

uniform mat4 mvp;
uniform float uSliceX; 

// 立方体的8个角相对于中心的偏移
const vec3 cube_corners[8] = vec3[](
    vec3(-0.5, -0.5, -0.5), vec3( 0.5, -0.5, -0.5),
    vec3( 0.5,  0.5, -0.5), vec3(-0.5,  0.5, -0.5), 
    vec3(-0.5, -0.5,  0.5), vec3( 0.5, -0.5,  0.5),
    vec3( 0.5,  0.5,  0.5), vec3(-0.5,  0.5,  0.5)  
);

// 【修复】：使用一维数组定义面索引 (GLSL 3.3 兼容写法)
// 顺序：Bottom, Top, Front, Back, Left, Right
const int faces[24] = int[](
    0, 1, 3, 2, // Face 0
    4, 5, 7, 6, // Face 1
    0, 1, 4, 5, // Face 2
    2, 3, 6, 7, // Face 3
    0, 3, 4, 7, // Face 4
    1, 2, 5, 6  // Face 5
);

// 法线
const vec3 normals[6] = vec3[](
    vec3(0, 0, -1), vec3(0, 0, 1),
    vec3(0, -1, 0), vec3(0, 1, 0),
    vec3(-1, 0, 0), vec3(1, 0, 0)
);

void main() {
    vec4 center = gl_in[0].gl_Position;

    // --- 1. 几何剔除 (Slicing) ---
    // 这是导师要求的核心功能：在生成几何体之前就剔除它
    float normX = (center.x / 40.0) + 0.5; 
    if (normX > uSliceX) return; // 直接返回，不生成任何三角形！性能极高！

    // --- 2. 模拟 GRDECL 角点生成 ---
    vec4 corners[8];
    for(int i=0; i<8; i++) {
        vec3 offset = cube_corners[i];
        vec3 pos = center.xyz + offset; 

        // 【关键】：应用畸变。模拟地层被挤压、扭曲的效果
        // 这使得网格不再是完美的正方体，而是不规则六面体 (Corner Point Grid)
        if (pos.y > 0) {
            pos.x += gs_in[0].distortion * 0.2; 
            pos.z += gs_in[0].distortion * 0.1;
        }

        corners[i] = vec4(pos, 1.0);
    }

    fAttr = gs_in[0].attr;

    // --- 3. 发射顶点 (生成六面体) ---
    for(int f=0; f<6; f++) {
        fNormal = normals[f];

        // 手动计算索引：base + offset
        int base = f * 4;

        vec4 p0 = corners[faces[base + 0]];
        vec4 p1 = corners[faces[base + 1]];
        vec4 p2 = corners[faces[base + 2]];
        vec4 p3 = corners[faces[base + 3]];

        // 发射 Triangle Strip (4个点组成一个矩形面)
        gl_Position = mvp * p0; EmitVertex();
        gl_Position = mvp * p1; EmitVertex();
        gl_Position = mvp * p2; EmitVertex();
        gl_Position = mvp * p3; EmitVertex();

        EndPrimitive(); // 完成一个面
    }
}
"""

# Fragment Shader
FS_CODE = """
#version 330 core
in float fAttr;
in vec3 fNormal;
out vec4 FragColor;

void main() {
    // 简单的光照
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));
    float diff = max(dot(fNormal, lightDir), 0.0);
    float ambient = 0.4;

    // 颜色映射 (蓝 -> 红)
    vec3 color = mix(vec3(0.2, 0.2, 0.8), vec3(0.8, 0.2, 0.2), fAttr);

    vec3 finalColor = color * (diff + ambient);
    FragColor = vec4(finalColor, 1.0);
}
"""


def create_program():
    vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, VS_CODE);
    glCompileShader(vs)
    if not glGetShaderiv(vs, GL_COMPILE_STATUS): print(f"VS Err: {glGetShaderInfoLog(vs)}")

    gs = glCreateShader(GL_GEOMETRY_SHADER);
    glShaderSource(gs, GS_CODE);
    glCompileShader(gs)
    if not glGetShaderiv(gs, GL_COMPILE_STATUS): print(f"GS Err: {glGetShaderInfoLog(gs)}")

    fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, FS_CODE);
    glCompileShader(fs)
    if not glGetShaderiv(fs, GL_COMPILE_STATUS): print(f"FS Err: {glGetShaderInfoLog(fs)}")

    prog = glCreateProgram()
    glAttachShader(prog, vs);
    glAttachShader(prog, gs);
    glAttachShader(prog, fs)
    glLinkProgram(prog)
    if not glGetProgramiv(prog, GL_LINK_STATUS): print(f"Link Err: {glGetProgramInfoLog(prog)}")

    glDeleteShader(vs);
    glDeleteShader(gs);
    glDeleteShader(fs)
    return prog


# --- 交互 ---
def mouse_callback(window, button, action, mods):
    global mouse_pressed
    if button == glfw.MOUSE_BUTTON_LEFT: mouse_pressed = (action == glfw.PRESS)


def cursor_callback(window, x, y):
    global last_x, last_y, cam_yaw, cam_pitch
    dx = x - last_x;
    dy = y - last_y
    last_x = x;
    last_y = y
    if mouse_pressed:
        cam_yaw -= dx * 0.005
        cam_pitch -= dy * 0.005
        cam_pitch = max(-1.5, min(1.5, cam_pitch))


def scroll_callback(window, x, y):
    global cam_distance
    cam_distance -= y * 2.0
    cam_distance = max(5.0, min(200.0, cam_distance))


def key_callback(window, key, scancode, action, mods):
    global slice_x
    if action == glfw.PRESS or action == glfw.REPEAT:
        if key == glfw.KEY_Q: slice_x = max(0.0, slice_x - 0.05)
        if key == glfw.KEY_A: slice_x = min(1.0, slice_x + 0.05)


# --- Main ---
def main():
    global last_x, last_y
    if not glfw.init(): return
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, "Geometry Shader CPG Engine", None, None)
    glfw.make_context_current(window)

    glfw.set_mouse_button_callback(window, mouse_callback)
    glfw.set_cursor_pos_callback(window, cursor_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_key_callback(window, key_callback)
    last_x, last_y = glfw.get_cursor_pos(window)

    # 1. 生成种子点
    grid_data = generate_corner_point_grid(GRID_SIZE)

    # 2. VBO Setup
    VAO = glGenVertexArrays(1);
    glBindVertexArray(VAO)
    VBO = glGenBuffers(1);
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, grid_data.nbytes, grid_data, GL_STATIC_DRAW)

    stride = 5 * 4
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0));
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12));
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16));
    glEnableVertexAttribArray(2)

    program = create_program()
    glEnable(GL_DEPTH_TEST)

    print("--- Engine Started (v2) ---")
    print("Keys: [Q] Slice In  [A] Slice Out")

    while not glfw.window_should_close(window):
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(program)

        cam_x = np.sin(cam_yaw) * np.cos(cam_pitch) * cam_distance
        cam_y = np.sin(cam_pitch) * cam_distance
        cam_z = np.cos(cam_yaw) * np.cos(cam_pitch) * cam_distance
        view = glm.lookAt(glm.vec3(cam_x, cam_y, cam_z), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
        proj = glm.perspective(glm.radians(45.0), WINDOW_WIDTH / WINDOW_HEIGHT, 0.1, 500.0)

        glUniformMatrix4fv(glGetUniformLocation(program, "mvp"), 1, GL_FALSE, glm.value_ptr(proj * view))
        glUniform1f(glGetUniformLocation(program, "uSliceX"), slice_x)

        glBindVertexArray(VAO)
        # 【关键】：画的是点，由 Geometry Shader 生成体
        glDrawArrays(GL_POINTS, 0, len(grid_data))

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()