# Author:Ach1n
# Data:2025/12/25 21:15
# Flag:I earned it.
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import os

# ==========================================
# 配置：指向您的真实油田数据文件
# ==========================================
FILENAME = "NORNE_ATW2013.GRDECL"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# 全局交互变量
# Norne 模型非常巨大（几千米），所以相机距离必须很大
cam_dist = 4000.0
cam_yaw = 45.0
cam_pitch = 30.0
last_x, last_y = 0, 0
mouse_pressed = False
slice_x = 1.0  # 切片参数 (0.0 - 1.0)


# -----------------------------------------------------------
# 1. GRDECL 解析器 (专门处理真实工业数据)
# -----------------------------------------------------------
class GRDECLParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.nx, self.ny, self.nz = 0, 0, 0
        self.coord = []
        self.zcorn = []
        self.actnum = []
        self.cells = []  # 存储解析后的六面体顶点

        # 初始化边界框
        self.center = np.array([0.0, 0.0, 0.0])
        self.min_b = np.array([float('inf')] * 3)
        self.max_b = np.array([float('-inf')] * 3)

    def parse(self):
        print(f"[Loader] 正在读取 {self.filepath} ...")
        if not os.path.exists(self.filepath):
            print(f"[Error] 文件未找到！请确认 {self.filepath} 在同目录下。")
            return False

        with open(self.filepath, 'r') as f:
            content = f.read()

        # 预处理：移除以 -- 开头的注释行（Norne数据包含很多注释）
        lines = content.split('\n')
        clean_lines = [line.split('--')[0] for line in lines]
        clean_content = ' '.join(clean_lines)

        # 简单的标记流解析
        tokens = clean_content.replace('/', ' / ').split()
        iterator = iter(tokens)

        try:
            while True:
                token = next(iterator)
                if token == 'SPECGRID':
                    self.nx = int(next(iterator))
                    self.ny = int(next(iterator))
                    self.nz = int(next(iterator))
                    print(f"[Loader] 网格尺寸: {self.nx} x {self.ny} x {self.nz}")
                elif token == 'COORD':
                    count = 6 * (self.nx + 1) * (self.ny + 1)
                    self.coord = self._read_data(iterator, count)
                elif token == 'ZCORN':
                    count = 8 * self.nx * self.ny * self.nz
                    self.zcorn = self._read_data(iterator, count)
                elif token == 'ACTNUM':
                    count = self.nx * self.ny * self.nz
                    self.actnum = self._read_data(iterator, count, is_int=True)

        except StopIteration:
            pass

        self._build_geometry()
        return True

    def _read_data(self, iterator, count, is_int=False):
        data = []
        while len(data) < count:
            try:
                val = next(iterator)
            except StopIteration:
                break

            if val == '/': break
            if '*' in val:  # 处理压缩格式 "100*0"
                num, v = val.split('*')
                try:
                    cnt = int(num)
                    val_conv = int(v) if is_int else float(v)
                    data.extend([val_conv] * cnt)
                except ValueError:
                    pass
            else:
                try:
                    val_conv = int(val) if is_int else float(val)
                    data.append(val_conv)
                except ValueError:
                    pass
        return data

    def _build_geometry(self):
        print("[Loader] 正在重建角点网格几何体 (Corner Point Geometry)...")
        # 将 COORD 重塑为 (Ny+1, Nx+1, 6)
        try:
            pillars = np.array(self.coord).reshape((self.ny + 1, self.nx + 1, 6))
        except ValueError:
            print("[Error] COORD 数据解析错误，可能文件不完整。")
            return

        z_idx = 0
        valid_cells = 0

        # 临时变量用于计算边界
        min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
        max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')

        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    cell_id = k * (self.nx * self.ny) + j * self.nx + i

                    # 获取该格子的 8 个深度值 (ZCORN)
                    if z_idx + 8 > len(self.zcorn): break
                    cell_z = self.zcorn[z_idx: z_idx + 8]
                    z_idx += 8

                    # 如果 ACTNUM 存在且为 0，则跳过（无效网格）
                    if self.actnum and cell_id < len(self.actnum) and self.actnum[cell_id] == 0:
                        continue

                    # 获取 4 根柱子索引
                    p_indices = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]
                    corners = []

                    # 简单插值计算 8 个角点的 (x, y)
                    for c_i in range(8):
                        z = cell_z[c_i]
                        p_idx = c_i % 4
                        pi, pj = p_indices[p_idx]

                        pil = pillars[pj, pi]  # [tx, ty, tz, bx, by, bz]
                        top = pil[0:3]
                        bot = pil[3:6]

                        # 线性插值因子
                        if bot[2] - top[2] == 0:
                            t = 0
                        else:
                            t = (z - top[2]) / (bot[2] - top[2])

                        x = top[0] + t * (bot[0] - top[0])
                        y = top[1] + t * (bot[1] - top[1])

                        corners.append((x, y, z))

                        # 更新边界
                        min_x = min(min_x, x);
                        max_x = max(max_x, x)
                        min_y = min(min_y, y);
                        max_y = max(max_y, y)
                        min_z = min(min_z, z);
                        max_z = max(max_z, z)

                    # 存储: (8个角点, 深度用于颜色, 中心X用于切片)
                    depth = corners[0][2]
                    center_x = (corners[0][0] + corners[7][0]) / 2.0
                    self.cells.append((corners, depth, center_x))
                    valid_cells += 1

        if valid_cells > 0:
            self.min_b = np.array([min_x, min_y, min_z])
            self.max_b = np.array([max_x, max_y, max_z])
            self.center = (self.min_b + self.max_b) / 2

        print(f"[Loader] 成功加载 {valid_cells} 个有效网格。")
        print(f"[Loader] 模型深度范围: {self.min_b[2]:.1f} 到 {self.max_b[2]:.1f}")


# -----------------------------------------------------------
# 2. 渲染器 (Legacy Mode - 绝对兼容)
# -----------------------------------------------------------
def init_gl():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glLightfv(GL_LIGHT0, GL_POSITION, [1000, 2000, 2000, 0])
    glClearColor(0.2, 0.2, 0.2, 1.0)


def draw_model(parser):
    # 使用动态计算的边界
    x_min, x_max = parser.min_b[0], parser.max_b[0]
    z_min, z_max = parser.min_b[2], parser.max_b[2]
    z_range = z_max - z_min if z_max != z_min else 1.0

    # 计算切片阈值
    x_threshold = x_min + slice_x * (x_max - x_min)

    glBegin(GL_QUADS)
    for corners, depth, center_x in parser.cells:
        # 切片逻辑
        if center_x > x_threshold: continue

        # 颜色：根据动态深度范围计算 (蓝->红)
        val = (depth - z_min) / z_range
        glColor3f(val, 0.5, 1.0 - val)

        # 绘制 6 个面 (拓扑顺序)
        # 顺序: Top(0132), Bot(4576), Front(0154), Back(2376), Left(0462), Right(1573)
        # Top
        glNormal3f(0, 1, 0)
        glVertex3f(*corners[0]);
        glVertex3f(*corners[1]);
        glVertex3f(*corners[3]);
        glVertex3f(*corners[2])
        # Bot
        glNormal3f(0, -1, 0)
        glVertex3f(*corners[4]);
        glVertex3f(*corners[5]);
        glVertex3f(*corners[7]);
        glVertex3f(*corners[6])
        # Front
        glNormal3f(0, 0, 1)
        glVertex3f(*corners[0]);
        glVertex3f(*corners[1]);
        glVertex3f(*corners[5]);
        glVertex3f(*corners[4])
        # Back
        glNormal3f(0, 0, -1)
        glVertex3f(*corners[2]);
        glVertex3f(*corners[3]);
        glVertex3f(*corners[7]);
        glVertex3f(*corners[6])
        # Left
        glNormal3f(-1, 0, 0)
        glVertex3f(*corners[0]);
        glVertex3f(*corners[4]);
        glVertex3f(*corners[6]);
        glVertex3f(*corners[2])
        # Right
        glNormal3f(1, 0, 0)
        glVertex3f(*corners[1]);
        glVertex3f(*corners[5]);
        glVertex3f(*corners[7]);
        glVertex3f(*corners[3])

    glEnd()


# -----------------------------------------------------------
# 3. 主循环
# -----------------------------------------------------------
def main():
    global cam_yaw, cam_pitch, cam_dist, last_x, last_y, mouse_pressed, slice_x

    if not glfw.init(): return

    # 兼容模式窗口
    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, "Norne Oilfield Viewer (Legacy)", None, None)
    if not window: glfw.terminate(); return
    glfw.make_context_current(window)

    init_gl()

    # 加载数据
    parser = GRDECLParser(FILENAME)
    if not parser.parse():
        glfw.terminate()
        return

    print("--- Viewer Started ---")
    print("Mouse Drag: Rotate | Scroll: Zoom | Q/A: Slice")

    while not glfw.window_should_close(window):
        # 交互
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            x, y = glfw.get_cursor_pos(window)
            if not mouse_pressed:
                last_x, last_y = x, y
                mouse_pressed = True
            else:
                dx, dy = x - last_x, y - last_y
                cam_yaw += dx * 0.5
                cam_pitch += dy * 0.5
                last_x, last_y = x, y
        else:
            mouse_pressed = False

        if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS: slice_x -= 0.02
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS: slice_x += 0.02
        slice_x = max(0.0, min(1.0, slice_x))

        # 渲染
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity()
        # Norne 油田很大，设置极大的远裁剪面 (100000.0)
        gluPerspective(45, WINDOW_WIDTH / WINDOW_HEIGHT, 10.0, 100000.0)

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity()

        cx, cy, cz = parser.center
        rad_y = np.radians(cam_yaw)
        rad_p = np.radians(cam_pitch)

        eye_x = cx + cam_dist * np.cos(rad_p) * np.sin(rad_y)
        eye_y = cy + cam_dist * np.sin(rad_p)
        eye_z = cz + cam_dist * np.cos(rad_p) * np.cos(rad_y)

        gluLookAt(eye_x, eye_y, eye_z, cx, cy, cz, 0, 1, 0)

        draw_model(parser)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()