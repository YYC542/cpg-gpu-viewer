# Author:Ach1n
# Data:2025/12/21 19:26
# Flag:I earned it.
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math

# --- 配置 ---
GRID_SIZE = 20  # 20x20x20
SPACING = 1.0

# 全局变量
cam_yaw = 45.0
cam_pitch = 30.0
cam_dist = 50.0
mouse_left_down = False
last_x, last_y = 0, 0
slice_x = 1.0  # 1.0 = 显示全部


def init():
    # 开启传统光照，让模型有立体感
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_DEPTH_TEST)

    # 光源位置
    glLightfv(GL_LIGHT0, GL_POSITION, [10, 20, 20, 1])

    # 深灰色背景
    glClearColor(0.2, 0.2, 0.2, 1.0)


def draw_distorted_cube(cx, cy, cz, distortion, attr):
    # 根据属性计算颜色 (蓝 -> 红)
    r = attr
    g = 0.2
    b = 1.0 - attr
    glColor3f(r, g, b)

    # 基础偏移
    # 我们不使用 glScale，而是手动计算8个角，为了模拟扭曲
    s = 0.45

    # 8个角的局部坐标
    # 0-3: Bottom, 4-7: Top
    offsets = [
        (-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s),
        (-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)
    ]

    # 计算世界坐标并应用畸变
    world_corners = []
    for ox, oy, oz in offsets:
        wx = cx + ox
        wy = cy + oy
        wz = cz + oz

        # --- 核心：这里模拟 GRDECL 的非正交特性 ---
        # 如果是格子的上半部分，让它随着起伏歪一点
        if oy > 0:
            wx += distortion * 0.2

        world_corners.append((wx, wy, wz))

    # 绘制 6 个面 (GL_QUADS)
    # 索引顺序
    faces = [
        (0, 1, 2, 3),  # Bottom
        (4, 5, 6, 7),  # Top
        (0, 1, 5, 4),  # Front
        (2, 3, 7, 6),  # Back
        (0, 3, 7, 4),  # Left
        (1, 2, 6, 5)  # Right
    ]

    glBegin(GL_QUADS)
    # 为了简化，我们只设置一个法线，这会让光照看起来稍微“硬”一点，但能看清
    glNormal3f(0, 1, 0)
    for face in faces:
        for idx in face:
            glVertex3f(*world_corners[idx])
    glEnd()


def main():
    global last_x, last_y, mouse_left_down, cam_yaw, cam_pitch, cam_dist, slice_x

    if not glfw.init(): return

    # 不设置 Core Profile，使用默认兼容模式
    window = glfw.create_window(1280, 720, "GRDECL Legacy Visualizer", None, None)
    if not window: glfw.terminate(); return
    glfw.make_context_current(window)

    init()

    print("GRDECL Visualizer (Legacy Mode)")
    print("Keys: [Q] Cut  [A] Restore")

    while not glfw.window_should_close(window):
        # 输入处理
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            curr_x, curr_y = glfw.get_cursor_pos(window)
            if not mouse_left_down:
                last_x, last_y = curr_x, curr_y
                mouse_left_down = True
            else:
                cam_yaw += (curr_x - last_x) * 0.5
                cam_pitch += (curr_y - last_y) * 0.5
                last_x, last_y = curr_x, curr_y
        else:
            mouse_left_down = False

        if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS: slice_x = max(0.0, slice_x - 0.02)
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS: slice_x = min(1.0, slice_x + 0.02)

        # 渲染
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        width, height = glfw.get_framebuffer_size(window)
        gluPerspective(45, width / height, 0.1, 1000.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # 摄像机
        rad_yaw = np.radians(cam_yaw)
        rad_pitch = np.radians(cam_pitch)
        eye_x = cam_dist * np.cos(rad_pitch) * np.sin(rad_yaw)
        eye_y = cam_dist * np.sin(rad_pitch)
        eye_z = cam_dist * np.cos(rad_pitch) * np.cos(rad_yaw)
        gluLookAt(eye_x, eye_y, eye_z, 0, 0, 0, 0, 1, 0)

        # --- 循环绘制所有格子 ---
        center_offset = GRID_SIZE / 2.0

        for x in range(GRID_SIZE):
            # 切片逻辑：简单的 CPU 剔除
            norm_x = x / GRID_SIZE
            if norm_x > slice_x: continue

            for y in range(GRID_SIZE):
                for z in range(GRID_SIZE):
                    # 坐标
                    cx = x - center_offset
                    cy = y - center_offset

                    # 地质起伏模拟 (Sin Wave)
                    distortion = math.sin(x * 0.2) * 2.0 + math.cos(y * 0.2) * 2.0
                    cz = (z - center_offset) * 1.5 + distortion

                    attr = z / GRID_SIZE

                    draw_distorted_cube(cx, cy, cz, distortion, attr)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()