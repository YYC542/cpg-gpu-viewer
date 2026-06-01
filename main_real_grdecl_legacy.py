# Author:Ach1n
# Data:2025/12/23 23:43
# Flag:I earned it.
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import os

# ==========================================
# 配置：已修改為真實 Norne 數據
# ==========================================
FILENAME = "NORNE_ATW2013.GRDECL"  # 【修改點 1】文件名
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# 全局交互變量
# 【修改點 2】距離調大，因為真實油田尺度很大
cam_dist = 4000.0
cam_yaw = 45.0
cam_pitch = 30.0
last_x, last_y = 0, 0
mouse_pressed = False
slice_x = 1.0  # 切片 (0.0 - 1.0)


# -----------------------------------------------------------
# 1. GRDECL 解析器 (處理真實工業數據)
# -----------------------------------------------------------
class GRDECLParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.nx, self.ny, self.nz = 0, 0, 0
        self.coord = []
        self.zcorn = []
        self.actnum = []
        self.cells = []  # 存儲解析後的六面體頂點
        # 【關鍵修復 1】強制初始化為 numpy 數組，防止後續索引錯誤
        self.center = np.array([0.0, 0.0, 0.0])
        self.min_b = np.array([0.0, 0.0, 0.0])
        self.max_b = np.array([0.0, 0.0, 0.0])

    def parse(self):
        print(f"[Loader] Reading {self.filepath} ...")
        if not os.path.exists(self.filepath):
            print(f"[Error] 文件未找到！請確認 {self.filepath} 在同目錄下。")
            return False

        with open(self.filepath, 'r') as f:
            content = f.read()

        # 簡單的標記流解析 (處理註釋和斜槓)
        # 移除 -- 開頭的註釋行 (簡單處理)
        lines = content.split('\n')
        clean_lines = [l for l in lines if not l.strip().startswith('--')]
        clean_content = ' '.join(clean_lines)

        tokens = clean_content.replace('/', ' / ').split()
        iterator = iter(tokens)

        try:
            while True:
                token = next(iterator)
                if token == 'SPECGRID':
                    self.nx = int(next(iterator))
                    self.ny = int(next(iterator))
                    self.nz = int(next(iterator))
                    print(f"[Loader] 網格尺寸: {self.nx} x {self.ny} x {self.nz}")
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

        # 處理幾何體
        self._build_geometry()
        return True

    def _read_data(self, iterator, count, is_int=False):
        data = []
        while len(data) < count:
            val = next(iterator)
            if val == '/': break
            if '*' in val:  # 處理壓縮格式 "100*0"
                num, v = val.split('*')
                data.extend([int(v) if is_int else float(v)] * int(num))
            else:
                try:
                    data.append(int(val) if is_int else float(val))
                except ValueError:
                    continue  # 跳過非數字字符
        return data

    def _build_geometry(self):
        print("[Loader] 正在重建角點網格幾何體 (Corner Point Geometry)...")
        # 將 COORD 重塑為 (Ny+1, Nx+1, 6)
        try:
            pillars = np.array(self.coord).reshape((self.ny + 1, self.nx + 1, 6))
        except ValueError:
            print("[Error] COORD 數據長度不匹配，可能文件讀取不完整。")
            return

        z_idx = 0
        valid_cells = 0

        # 計算中心點以便居中顯示
        self.min_b = np.array([999999.0] * 3)
        self.max_b = np.array([-999999.0] * 3)

        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    cell_id = k * (self.nx * self.ny) + j * self.nx + i

                    # 獲取該格子的 8 個深度值
                    cell_z = self.zcorn[z_idx: z_idx + 8]
                    z_idx += 8

                    # 如果 ACTNUM 存在且為 0，則跳過（無效網格）
                    if self.actnum and self.actnum[cell_id] == 0:
                        continue

                    # 獲取 4 根柱子
                    p_indices = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]
                    corners = []

                    # 簡單插值計算 8 個角點的 (x, y)
                    for c_i in range(8):
                        z = cell_z[c_i]
                        p_idx = c_i % 4
                        pi, pj = p_indices[p_idx]

                        pil = pillars[pj, pi]  # [tx, ty, tz, bx, by, bz]
                        top = pil[0:3]
                        bot = pil[3:6]

                        # 線性插值因子
                        if bot[2] - top[2] == 0:
                            t = 0
                        else:
                            t = (z - top[2]) / (bot[2] - top[2])

                        x = top[0] + t * (bot[0] - top[0])
                        y = top[1] + t * (bot[1] - top[1])

                        corners.append((x, y, z))

                        # 更新邊界框 (簡單過濾極端值)
                        if abs(x) < 100000 and abs(y) < 100000:
                            self.min_b = np.minimum(self.min_b, [x, y, z])
                            self.max_b = np.maximum(self.max_b, [x, y, z])

                    # 存儲這個格子的 8 個角點
                    # depth 用於顏色
                    depth = corners[0][2]
                    # 每個cell存儲: (8個角點, 深度, 中心X坐標)
                    center_x = (corners[0][0] + corners[7][0]) / 2.0
                    self.cells.append((corners, depth, center_x))
                    valid_cells += 1

        # 【關鍵修復 2】確保 center 是 numpy 數組且有效
        if valid_cells > 0:
            self.center = (self.min_b + self.max_b) / 2
        else:
            print("[Warning] 未找到有效網格，使用默認中心點。")
            self.center = np.array([0.0, 0.0, 0.0])

        print(f"[Loader] 成功加載 {valid_cells} 個有效網格。")
        print(f"[Loader] 模型中心: {self.center}")
        print(f"[Loader] X軸範圍: {self.min_b[0]} 到 {self.max_b[0]}")


# -----------------------------------------------------------
# 2. 渲染器 (Legacy Mode - 絕對兼容)
# -----------------------------------------------------------
def init_gl():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glLightfv(GL_LIGHT0, GL_POSITION, [1000, 2000, 2000, 0])
    glClearColor(0.2, 0.2, 0.2, 1.0)


def draw_model(parser):
    # 【關鍵修復 3】正確計算切片閾值
    # 使用解析器計算出的邊界框 min_b 和 max_b
    if not hasattr(parser, 'min_b') or not hasattr(parser, 'max_b'):
        return  # 數據未準備好

    x_min = parser.min_b[0]
    x_max = parser.max_b[0]

    # 計算當前的切片 X 坐標閾值
    # slice_x 是 0.0 到 1.0 的比例
    x_threshold = x_min + slice_x * (x_max - x_min)

    glBegin(GL_QUADS)
    for corners, depth, center_x in parser.cells:
        # --- 切片邏輯 (CPU端) ---
        # 如果該格子的中心 X 坐標大於閾值，就不畫它
        if center_x > x_threshold: continue

        # 顏色：根據深度 (Norne 大概在 2300m - 2700m)
        # 動態調整顏色範圍以適應不同模型
        z_min = parser.min_b[2]
        z_range = parser.max_b[2] - parser.min_b[2]
        if z_range == 0: z_range = 1.0

        val = (depth - z_min) / z_range
        val = max(0.0, min(1.0, val))  # 限制在 0-1

        # 簡單的熱力圖顏色 (藍 -> 紅)
        glColor3f(val, 0.5, 1.0 - val)

        # 繪製 6 個面 (拓撲順序)
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
        glVertex3f(*corners[2]);
        glVertex3f(*corners[6]);
        glVertex3f(*corners[4])
        # Right
        glNormal3f(1, 0, 0)
        glVertex3f(*corners[1]);
        glVertex3f(*corners[3]);
        glVertex3f(*corners[7]);
        glVertex3f(*corners[5])

    glEnd()


# -----------------------------------------------------------
# 3. 主循環
# -----------------------------------------------------------
def main():
    global cam_yaw, cam_pitch, cam_dist, last_x, last_y, mouse_pressed, slice_x

    if not glfw.init(): return

    # 兼容模式窗口
    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, "Real Norne Field Viewer", None, None)
    if not window: glfw.terminate(); return
    glfw.make_context_current(window)

    init_gl()

    # 加載數據
    parser = GRDECLParser(FILENAME)
    if not parser.parse():
        print("無法加載數據，程序退出。")
        glfw.terminate()
        return

    print("--- Viewer Started ---")
    print("Press [Q] to slice Left, [A] to slice Right")

    while not glfw.window_should_close(window):
        # 輸入處理
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

        # 切片控制
        if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
            slice_x -= 0.02
            print(f"Slicing X: {slice_x:.2f}")
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            slice_x += 0.02
            print(f"Slicing X: {slice_x:.2f}")

        slice_x = max(0.0, min(1.0, slice_x))

        # 渲染
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity()
        gluPerspective(45, WINDOW_WIDTH / WINDOW_HEIGHT, 10.0, 100000.0)  # 遠裁剪面設很大

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity()

        cx, cy, cz = parser.center
        rad_y = np.radians(cam_yaw)
        rad_p = np.radians(cam_pitch)

        # 攝像機
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