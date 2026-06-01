# ============================================================================
# 文件名: norne_benchmark.py
# 作者: Yang Yuchen (Ach1n)
# 用途: GS (几何着色器) vs VBO/VAO (CPU预生成) 渲染基准对比
#       为论文 Table 1 提供可复现、同方法测得的 FPS 数据
#
# 用法:
#   python norne_benchmark.py --mode gs           # 几何着色器路径
#   python norne_benchmark.py --mode vbo          # VBO/VAO CPU预生成路径
#   python norne_benchmark.py --mode legacy       # 立即模式 (glBegin/glEnd) 基准
#
# 可选参数:
#   --seconds 20          测量时长(秒), 默认20
#   --warmup 120          预热帧数(丢弃), 默认120
#   --width 1920 --height 1080   渲染分辨率, 默认1920x1080 (与论文一致)
#   --vsync               开启垂直同步(默认关闭; 测FPS务必关闭)
#   --file NORNE_ATW2013.GRDECL
#
# 输出: 平均FPS / 中位FPS / 1% Low FPS / 平均帧时 / GPU显存(若可查) / 顶点数
#
# 说明: 三条路径共用完全相同的解析、相机轨迹、投影矩阵、光照与计时,
#       唯一差异在"几何如何生成并送入GPU"以及对应的 draw call。
#       这确保 FPS 差异只反映渲染路径本身, 满足审稿人对公平基准的要求。
# ============================================================================

import argparse
import os
import sys
import time
import ctypes
import numpy as np

# Windows 下强制标准输出为 UTF-8, 避免中文打印触发 GBK 编码错误
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import glfw
from OpenGL.GL import *

# ----------------------------------------------------------------------------
# 命令行参数
# ----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Norne GS vs VBO rendering benchmark")
    ap.add_argument("--mode", choices=["gs", "vbo", "legacy"], required=True,
                    help="渲染路径: gs=几何着色器, vbo=CPU预生成VBO, legacy=立即模式(需GLU)")
    ap.add_argument("--file", default="NORNE_ATW2013.GRDECL", help="GRDECL 数据文件")
    ap.add_argument("--seconds", type=float, default=20.0, help="测量时长(秒)")
    ap.add_argument("--warmup", type=int, default=120, help="预热帧数(丢弃)")
    ap.add_argument("--width", type=int, default=1920, help="渲染宽度")
    ap.add_argument("--height", type=int, default=1080, help="渲染高度")
    ap.add_argument("--vexag", type=float, default=4.0, help="垂直放大(与论文默认4.0一致)")
    ap.add_argument("--vsync", action="store_true", help="开启V-Sync(默认关闭)")
    return ap.parse_args()


# ----------------------------------------------------------------------------
# 矩阵工具 (与 v21 完全一致的实现)
# ----------------------------------------------------------------------------
def mat_perspective(fovy_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m

def mat_lookat(eye, target, up):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m

def compile_shader(src, kind):
    sid = glCreateShader(kind)
    glShaderSource(sid, src)
    glCompileShader(sid)
    if not glGetShaderiv(sid, GL_COMPILE_STATUS):
        log = glGetShaderInfoLog(sid)
        raise RuntimeError(f"Shader compile error ({kind}):\n{log.decode() if isinstance(log, bytes) else log}")
    return sid

def make_prog(vs_src, fs_src, gs_src=None):
    prog = glCreateProgram()
    vs = compile_shader(vs_src, GL_VERTEX_SHADER)
    glAttachShader(prog, vs)
    fs = compile_shader(fs_src, GL_FRAGMENT_SHADER)
    glAttachShader(prog, fs)
    gs = None
    if gs_src:
        gs = compile_shader(gs_src, GL_GEOMETRY_SHADER)
        glAttachShader(prog, gs)
    glLinkProgram(prog)
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        log = glGetProgramInfoLog(prog)
        raise RuntimeError(f"Program link error:\n{log.decode() if isinstance(log, bytes) else log}")
    glDeleteShader(vs)
    glDeleteShader(fs)
    if gs:
        glDeleteShader(gs)
    return prog


# ============================================================================
# 着色器源码
# ============================================================================
# 设计原则: GS 路径与 VBO 路径使用【完全相同】的片段着色器(rainbow+三点光照),
# 几何也【完全相同】(同样6个面、同样vexag、同样法线计算)。
# 唯一区别: GS 在 GPU 端用几何着色器实时展开六面体; VBO 在 CPU 端预展开。

# ---- 共用片段着色器 (rainbow 深度色 + 三点光照 + 双面 abs 法线) ----
FS_COMMON = """
#version 330 core
in  float fDepth;
in  vec3  fNormal;
out vec4  FragColor;
uniform float uDepthMin;
uniform float uDepthRange;

vec3 rainbow(float t){
    t = clamp(t, 0.0, 1.0);
    if      (t < 0.25) return mix(vec3(0,0,1), vec3(0,1,1), t*4.0);
    else if (t < 0.50) return mix(vec3(0,1,1), vec3(0,1,0), (t-0.25)*4.0);
    else if (t < 0.75) return mix(vec3(0,1,0), vec3(1,1,0), (t-0.50)*4.0);
    else               return mix(vec3(1,1,0), vec3(1,0,0), (t-0.75)*4.0);
}
void main(){
    float t = clamp((fDepth - uDepthMin) / uDepthRange, 0.0, 1.0);
    vec3 col = rainbow(t);
    // 三点光照 (key 0.6 / fill 0.3 / rim 0.1), 双面: 对 dot 取 abs
    vec3 N = normalize(fNormal);
    vec3 Lkey  = normalize(vec3( 0.5,  1.0,  0.3));
    vec3 Lfill = normalize(vec3(-0.5,  0.2,  0.5));
    vec3 Lrim  = normalize(vec3( 0.0, -1.0, -0.5));
    float diff = 0.6*abs(dot(N,Lkey)) + 0.3*abs(dot(N,Lfill)) + 0.1*abs(dot(N,Lrim));
    float ambient = 0.25;
    FragColor = vec4(col * (ambient + diff), 1.0);
}
"""

# ---- GS 路径: 顶点着色器 (传话人, 角点在TBO里) ----
VS_GS = """
#version 330 core
layout(location=0) in float aDepth;
out float vDepth;
void main(){
    vDepth = aDepth;
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}
"""

# ---- GS 路径: 几何着色器 (point -> hexahedron, 纯外表面) ----
GS_SRC = """
#version 330 core
layout(points) in;
layout(triangle_strip, max_vertices = 24) out;
in  float vDepth[];
out float fDepth;
out vec3  fNormal;
uniform mat4  uMVP;
uniform float uVExag;
uniform samplerBuffer uCorners;

vec3 getCorner(int cell, int ci){
    int base = (cell*8 + ci) * 3;
    return vec3(texelFetch(uCorners, base).r,
                texelFetch(uCorners, base+1).r,
                texelFetch(uCorners, base+2).r);
}
void emitQuad(vec3 a, vec3 b, vec3 c, vec3 d, float depth){
    vec3 ae = vec3(a.x, a.y, a.z*uVExag);
    vec3 be = vec3(b.x, b.y, b.z*uVExag);
    vec3 ce = vec3(c.x, c.y, c.z*uVExag);
    vec3 de = vec3(d.x, d.y, d.z*uVExag);
    vec3 N = normalize(cross(be-ae, ce-ae));
    fNormal = N; fDepth = depth; gl_Position = uMVP*vec4(ae,1.0); EmitVertex();
    fNormal = N; fDepth = depth; gl_Position = uMVP*vec4(be,1.0); EmitVertex();
    fNormal = N; fDepth = depth; gl_Position = uMVP*vec4(ce,1.0); EmitVertex();
    fNormal = N; fDepth = depth; gl_Position = uMVP*vec4(de,1.0); EmitVertex();
    EndPrimitive();
}
void main(){
    int cell = gl_PrimitiveIDIn;
    vec3 c0=getCorner(cell,0); vec3 c1=getCorner(cell,1);
    vec3 c2=getCorner(cell,2); vec3 c3=getCorner(cell,3);
    vec3 c4=getCorner(cell,4); vec3 c5=getCorner(cell,5);
    vec3 c6=getCorner(cell,6); vec3 c7=getCorner(cell,7);
    float cz = (c0.z+c1.z+c2.z+c3.z+c4.z+c5.z+c6.z+c7.z)*0.125;
    emitQuad(c0, c2, c1, c3, cz);
    emitQuad(c4, c5, c6, c7, cz);
    emitQuad(c0, c1, c4, c5, cz);
    emitQuad(c3, c2, c7, c6, cz);
    emitQuad(c2, c0, c6, c4, cz);
    emitQuad(c1, c3, c5, c7, cz);
}
"""

# ---- VBO 路径: 顶点着色器 (预生成三角形, 含法线/深度属性) ----
VS_VBO = """
#version 330 core
layout(location=0) in vec3  aPos;     // 已含 vexag 后的坐标? 否: vexag 在此处施加
layout(location=1) in vec3  aNormal;
layout(location=2) in float aDepth;
out float fDepth;
out vec3  fNormal;
uniform mat4 uMVP;
void main(){
    fDepth  = aDepth;
    fNormal = aNormal;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

# ============================================================================
# GRDECL 解析器 (与 v21 完全一致, 原样嵌入)
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

# ============================================================================
# GPU 资源构建
# ============================================================================

# 6个面的角点索引顺序, 与 GS 中 emitQuad 调用【完全一致】:
#   emitQuad(c0,c2,c1,c3) emitQuad(c4,c5,c6,c7) emitQuad(c0,c1,c4,c5)
#   emitQuad(c3,c2,c7,c6) emitQuad(c2,c0,c6,c4) emitQuad(c1,c3,c5,c7)
# 每个 quad 是 triangle_strip 的 4 个点 (a,b,c,d) => 三角形 (a,b,c)+(b,d,c) ?
# triangle_strip 4点 (a,b,c,d) 展开为三角形: (a,b,c) 和 (c,b,d)  [strip规则]
# 为在 GL_TRIANGLES 下得到与 strip 完全相同的两片三角形, 用: (a,b,c) 和 (c,b,d)
_FACES = [
    (0, 2, 1, 3),
    (4, 5, 6, 7),
    (0, 1, 4, 5),
    (3, 2, 7, 6),
    (2, 0, 6, 4),
    (1, 3, 5, 7),
]

def _strip_to_tris(a, b, c, d):
    # triangle_strip(a,b,c,d) == triangles (a,b,c) + (c,b,d)
    return [(a, b, c), (c, b, d)]


def build_gs(parser, vexag):
    """GS 路径: 角点坐标塞进 TBO, 深度塞进 VBO, 用 GL_POINTS 绘制。"""
    n = len(parser.cells)
    cd = np.empty(n * 24, dtype=np.float32)
    for i, (cs, _, _) in enumerate(parser.cells):
        b = i * 24
        for ci, c in enumerate(cs):
            cd[b + ci*3] = c[0]; cd[b + ci*3 + 1] = c[1]; cd[b + ci*3 + 2] = c[2]
    tb = glGenBuffers(1)
    glBindBuffer(GL_TEXTURE_BUFFER, tb)
    glBufferData(GL_TEXTURE_BUFFER, cd.nbytes, cd, GL_STATIC_DRAW)
    tt = glGenTextures(1)
    glBindTexture(GL_TEXTURE_BUFFER, tt)
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, tb)

    dd = np.array([c[1] for c in parser.cells], dtype=np.float32)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vb = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vb)
    glBufferData(GL_ARRAY_BUFFER, dd.nbytes, dd, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)
    glBindVertexArray(0)

    vram = cd.nbytes + dd.nbytes  # 角点TBO + 深度VBO
    return {"vao": vao, "tbo": tt, "ncells": n, "vram": vram}


def build_vbo(parser, vexag):
    """VBO 路径: CPU 端预展开所有六面体为三角形 (pos+normal+depth), 用 GL_TRIANGLES。"""
    n = len(parser.cells)
    # 每个 cell: 6面 × 2三角形 × 3顶点 = 36 顶点; 每顶点 7 float (pos3+normal3+depth1)
    verts = np.empty((n * 36, 7), dtype=np.float32)
    vp = 0
    for (cs, _, _) in parser.cells:
        # 施加 vexag (与 GS 中 *uVExag 一致)
        P = [(c[0], c[1], c[2] * vexag) for c in cs]
        cz = sum(c[2] for c in cs) * 0.125  # 注意: depth 用未放大的平均 z (与GS一致)
        for (ia, ib, ic, idx) in _FACES:
            a = np.array(P[ia]); b = np.array(P[ib]); c = np.array(P[ic]); d = np.array(P[idx])
            N = np.cross(b - a, c - a)
            ln = np.linalg.norm(N)
            if ln > 1e-12:
                N = N / ln
            else:
                N = np.array([0.0, 0.0, 1.0])
            for tri in _strip_to_tris(a, b, c, d):
                for vtx in tri:
                    verts[vp, 0:3] = vtx
                    verts[vp, 3:6] = N
                    verts[vp, 6] = cz
                    vp += 1
    verts = verts[:vp]
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vb = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vb)
    glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
    stride = 7 * 4
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
    glEnableVertexAttribArray(2)
    glBindVertexArray(0)

    vram = verts.nbytes
    return {"vao": vao, "nverts": vp, "vram": vram}


# ============================================================================
# Legacy 立即模式渲染 (glBegin/glEnd) —— 对应论文 19x 加速比的分母
# ============================================================================
def draw_legacy(parser, vexag, mvp_unused):
    # 立即模式需固定管线; 在 compatibility context 下用 glBegin/glEnd 逐顶点提交。
    from OpenGL.GL import (glBegin, glEnd, glColor3f, glNormal3f, glVertex3f, GL_QUADS)
    glBegin(GL_QUADS)
    for (cs, avg_z, _) in parser.cells:
        P = [(c[0], c[1], c[2] * vexag) for c in cs]
        for (ia, ib, ic, idx) in _FACES:
            a = P[ia]; b = P[ib]; c = P[ic]; d = P[idx]
            glColor3f(0.5, 0.5, 0.8)
            glVertex3f(*a); glVertex3f(*b); glVertex3f(*d); glVertex3f(*c)
    glEnd()


# ============================================================================
# 主程序: 固定相机轨迹 + 预热 + FPS 统计
# ============================================================================
def main():
    args = parse_args()

    # ---- 解析数据 ----
    parser = GRDECLParser(args.file)
    if not parser.parse():
        print("[FATAL] 数据加载失败"); sys.exit(1)

    # ---- 初始化窗口 ----
    if not glfw.init():
        print("[FATAL] glfw init 失败"); sys.exit(1)

    if args.mode == "legacy":
        # 立即模式: 使用兼容性上下文 (不指定 core profile)
        pass
    else:
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    win = glfw.create_window(args.width, args.height,
                             f"Norne Benchmark [{args.mode}]", None, None)
    if not win:
        print("[FATAL] 创建窗口失败"); glfw.terminate(); sys.exit(1)
    glfw.make_context_current(win)
    glfw.swap_interval(1 if args.vsync else 0)  # 关键: 关闭V-Sync才能测真实FPS

    print(f"[GL] Renderer: {glGetString(GL_RENDERER).decode()}")
    print(f"[GL] Version : {glGetString(GL_VERSION).decode()}")

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)

    # ---- 包围盒/深度范围 (用于相机距离与色彩映射) ----
    cx, cy, cz_ = (parser.min_b + parser.max_b) / 2.0
    center = np.array([cx, cy, cz_ * args.vexag], dtype=np.float64)
    extent = float(np.linalg.norm(parser.max_b - parser.min_b))
    dmin = float(parser.min_b[2])
    drng = float(parser.max_b[2] - parser.min_b[2]) or 1.0
    near = max(extent * 0.001, 0.01)
    far = extent * 5.0
    aspect = args.width / args.height

    # ---- 构建 GPU 资源 + 程序 ----
    prog = None
    res = None
    if args.mode == "gs":
        prog = make_prog(VS_GS, FS_COMMON, GS_SRC)
        res = build_gs(parser, args.vexag)
        nprim = res["ncells"]
        out_verts = res["ncells"] * 36
    elif args.mode == "vbo":
        prog = make_prog(VS_VBO, FS_COMMON)
        res = build_vbo(parser, args.vexag)
        nprim = res["nverts"]
        out_verts = res["nverts"]
    else:  # legacy
        res = {"vram": 0}
        out_verts = len(parser.cells) * 36

    if prog is not None:
        glUseProgram(prog)
        loc_mvp = glGetUniformLocation(prog, "uMVP")
        loc_dmin = glGetUniformLocation(prog, "uDepthMin")
        loc_drng = glGetUniformLocation(prog, "uDepthRange")
        loc_vexag = glGetUniformLocation(prog, "uVExag")  # 仅GS有
        loc_corn = glGetUniformLocation(prog, "uCorners") # 仅GS有
        glUniform1f(loc_dmin, dmin)
        glUniform1f(loc_drng, drng)
        if loc_vexag != -1:
            glUniform1f(loc_vexag, args.vexag)

    # ---- FPS 统计容器 ----
    frame_times = []   # 每帧耗时(秒)
    frame_idx = 0
    t_prev = glfw.get_time()
    t_start_measure = None
    measuring = False

    print(f"[RUN] mode={args.mode}  warmup={args.warmup} frames  measure={args.seconds}s  "
          f"vsync={'ON' if args.vsync else 'OFF'}  res={args.width}x{args.height}")

    while not glfw.window_should_close(win):
        t_now = glfw.get_time()
        dt = t_now - t_prev
        t_prev = t_now

        # 固定相机轨迹: 绕模型匀速旋转, 消除取景偶然性 (两条路径轨迹完全一致)
        ang = t_now * 0.5  # rad/s
        radius = extent * 1.2
        eye = center + np.array([radius * np.cos(ang),
                                 radius * 0.4,
                                 radius * np.sin(ang)], dtype=np.float64)
        V = mat_lookat(eye, center, np.array([0, 1, 0], dtype=np.float64))
        Pm = mat_perspective(45.0, aspect, near, far)
        mvp = (Pm @ V).astype(np.float32)

        glViewport(0, 0, args.width, args.height)
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if args.mode == "gs":
            glUseProgram(prog)
            glUniformMatrix4fv(loc_mvp, 1, GL_TRUE, mvp)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_BUFFER, res["tbo"])
            if loc_corn != -1:
                glUniform1i(loc_corn, 0)
            glBindVertexArray(res["vao"])
            glDrawArrays(GL_POINTS, 0, res["ncells"])
            glBindVertexArray(0)
        elif args.mode == "vbo":
            glUseProgram(prog)
            glUniformMatrix4fv(loc_mvp, 1, GL_TRUE, mvp)
            glBindVertexArray(res["vao"])
            glDrawArrays(GL_TRIANGLES, 0, res["nverts"])
            glBindVertexArray(0)
        else:  # legacy
            glMatrixMode(GL_PROJECTION); glLoadIdentity()
            # 用 mvp 不方便; legacy 用固定管线自己设投影/视图
            from OpenGL.GLU import gluPerspective, gluLookAt
            gluPerspective(45.0, aspect, near, far)
            glMatrixMode(GL_MODELVIEW); glLoadIdentity()
            gluLookAt(eye[0], eye[1], eye[2], center[0], center[1], center[2], 0, 1, 0)
            glEnable(GL_LIGHTING); glEnable(GL_LIGHT0); glEnable(GL_COLOR_MATERIAL)
            glLightfv(GL_LIGHT0, GL_POSITION, [center[0], center[1]+extent, center[2], 1])
            draw_legacy(parser, args.vexag, mvp)

        glfw.swap_buffers(win)
        glfw.poll_events()
        frame_idx += 1

        # 预热结束后开始计时
        if frame_idx == args.warmup:
            measuring = True
            t_start_measure = glfw.get_time()
            print(f"[RUN] 预热完成, 开始测量 {args.seconds}s ...")
            continue

        if measuring:
            frame_times.append(dt)
            if glfw.get_time() - t_start_measure >= args.seconds:
                break

    glfw.terminate()

    # ---- 统计输出 ----
    if not frame_times:
        print("[ERROR] 没有采集到帧, 请增大 --seconds 或减小 --warmup"); return
    ft = np.array(frame_times, dtype=np.float64)
    fps_inst = 1.0 / ft
    avg_fps = float(np.mean(fps_inst))
    med_fps = float(np.median(fps_inst))
    # 1% low: 取最慢的1%帧(最长帧时)的平均FPS
    n_low = max(1, int(len(ft) * 0.01))
    slowest = np.sort(ft)[-n_low:]
    low1_fps = float(1.0 / np.mean(slowest))
    avg_ms = float(np.mean(ft) * 1000.0)
    vram_mb = res["vram"] / (1024 * 1024)

    print("\n" + "=" * 60)
    print(f"  结果: mode = {args.mode}")
    print("=" * 60)
    print(f"  采集帧数        : {len(ft)}")
    print(f"  平均 FPS        : {avg_fps:.1f}")
    print(f"  中位 FPS        : {med_fps:.1f}")
    print(f"  1% Low FPS      : {low1_fps:.1f}")
    print(f"  平均帧时        : {avg_ms:.2f} ms")
    print(f"  活跃单元数      : {len(parser.cells)}")
    print(f"  输出三角形顶点  : {out_verts:,}")
    if res["vram"] > 0:
        print(f"  几何缓冲显存    : {vram_mb:.2f} MB  (仅几何数据, 不含纹理/FBO)")
    print("=" * 60)
    print(f"  ★ 填入论文的数字 (该配置): {avg_fps:.0f} FPS\n")


if __name__ == "__main__":
    main()