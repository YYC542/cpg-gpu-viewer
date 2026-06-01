# Author:Ach1n
# Data:2025/12/21 19:23
# Flag:I earned it.
import glfw
from OpenGL.GL import *
import numpy as np
import glm
import ctypes

# --- 配置 ---
GRID_SIZE = 20  # 網格大小 (20x20x20 = 8000個格子)
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# 全局變量
cam_distance = 50.0
cam_yaw = 0.5
cam_pitch = 0.5
last_x, last_y = 0, 0
mouse_pressed = False
slice_x = 1.0


# ==========================================
# 1. CPU 模擬 GRDECL 幾何生成 (核心算法)
# ==========================================
def generate_distorted_mesh(size):
    """
    在 CPU 端完成幾何著色器的工作：
    計算帶有地質畸變的六面體頂點。
    """
    print(f"CPU generating GRDECL geometry for {size}^3 grid...")

    vertices = []

    # 預定義立方體 8 個角的偏移 (0-7)
    base_corners = [
        (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),  # Bottom ring
        (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)  # Top ring
    ]

    # 定義 6 個面 (每個面 2 個三角形，共 6 個頂點)
    # 順序: Bottom, Top, Front, Back, Left, Right
    indices = [
        0, 1, 2, 0, 2, 3,  # Bottom
        4, 5, 6, 4, 6, 7,  # Top
        0, 1, 5, 0, 5, 4,  # Front
        2, 3, 7, 2, 7, 6,  # Back
        0, 3, 7, 0, 7, 4,  # Left
        1, 2, 6, 1, 6, 5  # Right
    ]

    for x in range(size):
        for y in range(size):
            for z in range(size):
                # 網格中心
                cx = x - size / 2.0
                cy = y - size / 2.0

                # --- 地質畸變算法 (模擬 GRDECL) ---
                # 讓地層像波浪一樣起伏
                distortion = np.sin(x * 0.3) * 2.5 + np.cos(y * 0.3) * 2.5
                cz = (z - size / 2.0) * 1.5 + distortion

                # 計算這個格子的 8 個具體角點坐標
                cell_corners = []
                for bx, by, bz in base_corners:
                    # 基礎位置
                    px = cx + bx
                    py = cy + by
                    pz = cz + bz

                    # 額外的非正交扭曲 (讓格子歪一點，像真實地層)
                    if py > cy:  # 如果是格子的上半部分
                        px += distortion * 0.1  # 隨著起伏發生剪切

                    cell_corners.append((px, py, pz))

                # 屬性值 (用於顏色)
                attr = z / size

                # 構建三角形頂點
                for idx in indices:
                    # 每個頂點: [x, y, z, attr, cell_center_x]
                    # cell_center_x 用於切片判斷
                    px, py, pz = cell_corners[idx]
                    vertices.extend([px, py, pz, attr, cx])

    return np.array(vertices, dtype=np.float32)


# ==========================================
# 2. 著色器 (極簡版 - 僅負責畫出來)
# ==========================================
VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in float aAttr;
layout (location = 2) in float aCenterX; // 用於切片

out float vAttr;
out float vCenterX;
uniform mat4 mvp;

void main() {
    gl_Position = mvp * vec4(aPos, 1.0);
    vAttr = aAttr;
    vCenterX = aCenterX;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in float vAttr;
in float vCenterX;
out vec4 FragColor;

uniform float uSliceX;

void main() {
    // 切片邏輯
    // 歸一化 CenterX (-15 到 15) 到 (0 到 1)
    float normX = (vCenterX / 30.0) + 0.5;
    if (normX > uSliceX) discard;

    // 顏色映射: 模擬地層深度顏色 (深藍 -> 淺青 -> 黃 -> 紅)
    vec3 color;
    if (vAttr < 0.5) {
        color = mix(vec3(0.1, 0.1, 0.6), vec3(0.1, 0.8, 0.8), vAttr * 2.0);
    } else {
        color = mix(vec3(0.1, 0.8, 0.8), vec3(0.8, 0.6, 0.1), (vAttr - 0.5) * 2.0);
    }

    // 簡單的光照模擬 (基於高度)
    float light = 0.8 + 0.2 * vAttr;

    FragColor = vec4(color * light, 1.0);
}
"""


def create_program():
    vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, VERTEX_SHADER);
    glCompileShader(vs)
    if not glGetShaderiv(vs, GL_COMPILE_STATUS): print(f"VS Err: {glGetShaderInfoLog(vs)}")
    fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, FRAGMENT_SHADER);
    glCompileShader(fs)
    if not glGetShaderiv(fs, GL_COMPILE_STATUS): print(f"FS Err: {glGetShaderInfoLog(fs)}")
    prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog)
    glDeleteShader(vs);
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


# --- 主程序 ---
def main():
    global last_x, last_y
    if not glfw.init(): return
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, "GRDECL Visualizer (Safe Mode)", None, None)
    glfw.make_context_current(window)

    glfw.set_mouse_button_callback(window, mouse_callback)
    glfw.set_cursor_pos_callback(window, cursor_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_key_callback(window, key_callback)
    last_x, last_y = glfw.get_cursor_pos(window)

    # 1. 生成幾何數據
    vertex_data = generate_distorted_mesh(GRID_SIZE)
    vertex_count = len(vertex_data) // 5  # 每個頂點5個float
    print(f"Geometry generated. Total vertices: {vertex_count}")

    # 2. VBO Setup
    VAO = glGenVertexArrays(1);
    glBindVertexArray(VAO)
    VBO = glGenBuffers(1);
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

    stride = 5 * 4
    # Pos (x,y,z)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0));
    glEnableVertexAttribArray(0)
    # Attr (attr)
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12));
    glEnableVertexAttribArray(1)
    # CenterX (for slicing)
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16));
    glEnableVertexAttribArray(2)

    program = create_program()
    glEnable(GL_DEPTH_TEST)

    print("--- GRDECL Visualizer Running ---")
    print("Use [Q] and [A] to slice the model.")

    while not glfw.window_should_close(window):
        glClearColor(0.2, 0.2, 0.2, 1.0)
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
        glDrawArrays(GL_TRIANGLES, 0, vertex_count)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()