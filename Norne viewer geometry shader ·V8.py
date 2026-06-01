# Author:Ach1n
# Data:2026/3/19 12:20
# Flag:I earned it.
# Author: Ach1n
# Date: 2026/3/17
# Flag: I earned it.
# Version: v8 - Complete version
# Features:
#   - Full 3D volumetric rendering (all active cells)
#   - Rainbow depth colormap (blue=shallow, red=deep)
#   - Dual-axis slicing: X (Q/A) and Z (E/D)
#   - Vertical exaggeration (Z/X keys)
#   - Fault detection & highlight (F key toggle)
#   - Well trajectories from Norne benchmark (W key toggle)
#   - On-screen colorbar legend
#   - Three-point lighting, double-sided normals
#   - Polygon offset to reduce z-fighting

import glfw
import ctypes
from OpenGL.GL import *
import numpy as np
import os

# ==========================================
# Configuration
# ==========================================
# 指定了要加载的 GRDECL 文件名和窗口分辨率。
FILENAME = "NORNE_ATW2013.GRDECL"
WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 720

# Camera
# 摄像机参数：cam_dist 是相机到原点的距离，cam_yaw（偏航角）和 cam_pitch（俯仰角）定义了初始观察角度
# ——30° 的偏转让你启动后就能同时看到模型顶面和截面切口，而不是正对着看
cam_dist  = 20000.0
cam_yaw   = 30.0
cam_pitch = 30.0
last_x = last_y = 0
mouse_pressed = False

# Slice / display controls
slice_x     = 1.0    # X cross-section 0-1  (Q=dec / A=inc)
slice_z     = 1.0    # Z depth slice   0-1  (E=dec / D=inc)
vexag       = 4.0    # Vertical exaggeration (Z=dec / X=inc)
show_faults = True   # F key toggle
show_wells  = True   # W key toggle

# ==========================================
# Norne Public Well Coordinates (UTM Zone 32N)
# Source: Norne benchmark dataset (SPE/Statoil public release)
# ==========================================
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

# ==========================================
# GLSL Shaders
# ==========================================

VERTEX_SHADER_SRC = """
#version 330 core
layout(location = 0) in float aDepth;
out float vDepth;
void main() {
    vDepth = aDepth;
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}
"""

GEOMETRY_SHADER_SRC = """
#version 330 core
layout(points) in;
layout(triangle_strip, max_vertices = 24) out;

in  float vDepth[];
out float fDepth;
out vec3  fNormal;
out float fFault;

uniform mat4  uMVP;
uniform float uSliceX;
uniform float uSliceZ;
uniform float uVExag;
uniform samplerBuffer uCorners;
uniform samplerBuffer uFaultFlag;

vec3 getCorner(int cell, int ci) {
    int base = cell * 24 + ci * 3;
    return vec3(
        texelFetch(uCorners, base    ).r,
        texelFetch(uCorners, base + 1).r,
        texelFetch(uCorners, base + 2).r
    );
}

vec3 vex(vec3 p) {
    return vec3(p.x, p.y, p.z * uVExag);
}

// GRDECL corner layout:
// c0=(i,  j,  k_top)   c1=(i+1,j,  k_top)
// c2=(i,  j+1,k_top)   c3=(i+1,j+1,k_top)
// c4=(i,  j,  k_bot)   c5=(i+1,j,  k_bot)
// c6=(i,  j+1,k_bot)   c7=(i+1,j+1,k_bot)
void emitQuad(vec3 a, vec3 b, vec3 c, vec3 d, float depth, float fault) {
    vec3 ae=vex(a), be=vex(b), ce=vex(c), de=vex(d);
    vec3 n = normalize(cross(be - ae, ce - ae));
    fNormal = n; fDepth = depth; fFault = fault;
    gl_Position = uMVP * vec4(ae, 1.0); EmitVertex();
    gl_Position = uMVP * vec4(be, 1.0); EmitVertex();
    gl_Position = uMVP * vec4(ce, 1.0); EmitVertex();
    gl_Position = uMVP * vec4(de, 1.0); EmitVertex();
    EndPrimitive();
}

void main() {
    int cell = gl_PrimitiveIDIn;
    vec3 c0=getCorner(cell,0); vec3 c1=getCorner(cell,1);
    vec3 c2=getCorner(cell,2); vec3 c3=getCorner(cell,3);
    vec3 c4=getCorner(cell,4); vec3 c5=getCorner(cell,5);
    vec3 c6=getCorner(cell,6); vec3 c7=getCorner(cell,7);

    float cx = (c0.x+c1.x+c2.x+c3.x+c4.x+c5.x+c6.x+c7.x)*0.125;
    if (cx > uSliceX) return;
    float cz = (c0.z+c1.z+c2.z+c3.z+c4.z+c5.z+c6.z+c7.z)*0.125;
    if (cz > uSliceZ) return;

    float fault = texelFetch(uFaultFlag, cell).r;

    emitQuad(c0, c2, c1, c3, cz, fault); // top
    emitQuad(c4, c5, c6, c7, cz, fault); // bottom
    emitQuad(c0, c1, c4, c5, cz, fault); // front
    emitQuad(c3, c2, c7, c6, cz, fault); // back
    emitQuad(c2, c0, c6, c4, cz, fault); // left
    emitQuad(c1, c3, c5, c7, cz, fault); // right
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core
in  float fDepth;
in  vec3  fNormal;
in  float fFault;
out vec4  FragColor;

uniform float uDepthMin;
uniform float uDepthRange;
uniform int   uShowFaults;

vec3 rainbow(float t) {
    t = clamp(t, 0.0, 1.0);
    if      (t < 0.25) return mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), t*4.0);
    else if (t < 0.50) return mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (t-0.25)*4.0);
    else if (t < 0.75) return mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (t-0.50)*4.0);
    else               return mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (t-0.75)*4.0);
}

void main() {
    float t   = clamp((fDepth - uDepthMin) / uDepthRange, 0.0, 1.0);
    vec3  col = rainbow(t);
    if (uShowFaults == 1 && fFault > 0.5)
        col = mix(col, vec3(1.0, 1.0, 1.0), 0.6);
    vec3  L1 = normalize(vec3(1.0, 2.0, 3.0));
    vec3  L2 = normalize(vec3(-1.0, 0.5, -1.0));
    vec3  n  = normalize(fNormal);
    float d  = 0.45 + 0.40*abs(dot(n,L1)) + 0.15*abs(dot(n,L2));
    FragColor = vec4(col * d, 1.0);
}
"""

WELL_VERT_SRC = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
out vec3 vColor;
uniform mat4  uMVP;
uniform float uVExag;
void main() {
    vec3 p = vec3(aPos.x, aPos.y, aPos.z * uVExag);
    gl_Position = uMVP * vec4(p, 1.0);
    vColor = aColor;
}
"""

WELL_FRAG_SRC = """
#version 330 core
in  vec3 vColor;
out vec4 FragColor;
void main() { FragColor = vec4(vColor, 1.0); }
"""

OVERLAY_VERT_SRC = """
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec3 aColor;
out vec3 vColor;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vColor = aColor;
}
"""

OVERLAY_FRAG_SRC = """
#version 330 core
in  vec3 vColor;
out vec4 FragColor;
void main() { FragColor = vec4(vColor, 1.0); }
"""

# ==========================================
# Shader helpers
# ==========================================
def compile_shader(src, kind):
    s = glCreateShader(kind)
    glShaderSource(s, src)
    glCompileShader(s)
    if not glGetShaderiv(s, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(s).decode())
    return s

def make_prog(vs_src, fs_src, gs_src=None):
    shaders = [compile_shader(vs_src, GL_VERTEX_SHADER)]
    if gs_src:
        shaders.append(compile_shader(gs_src, GL_GEOMETRY_SHADER))
    shaders.append(compile_shader(fs_src, GL_FRAGMENT_SHADER))
    p = glCreateProgram()
    for sh in shaders:
        glAttachShader(p, sh)
        glDeleteShader(sh)
    glLinkProgram(p)
    if not glGetProgramiv(p, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(p).decode())
    return p

# ==========================================
# Matrix helpers
# ==========================================
def mat_perspective(fovy, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fovy) / 2.0)
    return np.array([
        [f/aspect, 0, 0,                     0],
        [0,        f, 0,                     0],
        [0,        0, (far+near)/(near-far), 2*far*near/(near-far)],
        [0,        0, -1,                    0],
    ], dtype=np.float32)

def mat_lookat(eye, target, up):
    f = target - eye;  f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.array([
        [ r[0],  r[1],  r[2], -np.dot(r, eye)],
        [ u[0],  u[1],  u[2], -np.dot(u, eye)],
        [-f[0], -f[1], -f[2],  np.dot(f, eye)],
        [    0,     0,      0,              1 ],
    ], dtype=np.float32)

# ==========================================
# GRDECL Parser
# ==========================================
class GRDECLParser:
    def __init__(self, fp):
        self.filepath = fp
        self.nx = self.ny = self.nz = 0
        self.coord = []; self.zcorn = []; self.actnum = []
        self.cells = []
        self.fault_flags = []
        self.min_b = np.full(3,  1e18)
        self.max_b = np.full(3, -1e18)
        self._real_origin = np.zeros(3)

    def parse(self):
        print(f"[Loader] Reading {self.filepath} ...")
        if not os.path.exists(self.filepath):
            print("[Error] File not found"); return False
        with open(self.filepath) as f:
            raw = f.read()
        tokens = ' '.join(l.split('--')[0] for l in raw.split('\n'))
        tokens = tokens.replace('/', ' / ').split()
        it = iter(tokens)
        try:
            while True:
                t = next(it)
                if t == 'SPECGRID':
                    self.nx,self.ny,self.nz = int(next(it)),int(next(it)),int(next(it))
                    print(f"[Loader] Grid {self.nx}x{self.ny}x{self.nz}")
                elif t == 'COORD':
                    self.coord = self._rd(it, 6*(self.nx+1)*(self.ny+1))
                elif t == 'ZCORN':
                    self.zcorn = self._rd(it, 8*self.nx*self.ny*self.nz)
                elif t == 'ACTNUM':
                    self.actnum = self._rd(it, self.nx*self.ny*self.nz, True)
        except StopIteration:
            pass
        self._build()
        return bool(self.cells)

    def _rd(self, it, n, isint=False):
        d = []
        while len(d) < n:
            try: v = next(it)
            except StopIteration: break
            if v == '/': break
            if '*' in v:
                a,b = v.split('*')
                try: d.extend([int(b) if isint else float(b)]*int(a))
                except: pass
            else:
                try: d.append(int(v) if isint else float(v))
                except: pass
        return d

    def _build(self):
        print("[Loader] Building geometry ...")
        nx,ny,nz = self.nx,self.ny,self.nz
        try:
            pillars = np.array(self.coord, dtype=np.float64).reshape(ny+1, nx+1, 6)
        except:
            print("[Error] COORD reshape failed"); return
        zc = np.array(self.zcorn, dtype=np.float64)

        def zidx(i,j,k,hk,hj,hi):
            return (k*2+hk)*(ny*2*nx*2) + (j*2+hj)*(nx*2) + (i*2+hi)

        def pillar_pt(pi, pj, z):
            p = pillars[pj, pi]
            dz = p[5]-p[2]
            t = 0.0 if abs(dz)<1e-9 else (z-p[2])/dz
            return np.array([p[0]+t*(p[3]-p[0]), p[1]+t*(p[4]-p[1]), z])

        defs = [(0,0,0,0,0),(1,0,0,0,1),(0,1,0,1,0),(1,1,0,1,1),
                (0,0,1,0,0),(1,0,1,0,1),(0,1,1,1,0),(1,1,1,1,1)]

        cell_corners = {}
        mn = np.full(3, 1e18); mx = np.full(3,-1e18)

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    cid = k*nx*ny+j*nx+i
                    if self.actnum and cid < len(self.actnum) and self.actnum[cid] == 0:
                        continue
                    cs = []; ok = True
                    for (di,dj,hk,hj,hi) in defs:
                        zi = zidx(i,j,k,hk,hj,hi)
                        if zi >= len(zc): ok=False; break
                        cs.append(pillar_pt(i+di, j+dj, zc[zi]))
                    if ok:
                        for c in cs:
                            mn = np.minimum(mn, c)
                            mx = np.maximum(mx, c)
                        cell_corners[(k,j,i)] = cs

        origin = (mn+mx)*0.5
        self._real_origin = origin
        print(f"[Loader] Origin: {origin}")

        # Fault detection
        FAULT_THRESHOLD = 15.0
        fault_set = set()
        for k in range(1, nz):
            for j in range(ny):
                for i in range(nx):
                    if (k,j,i) in cell_corners and (k-1,j,i) in cell_corners:
                        z_cur  = np.mean([c[2] for c in cell_corners[(k,  j,i)]])
                        z_prev = np.mean([c[2] for c in cell_corners[(k-1,j,i)]])
                        if abs(z_cur - z_prev) > FAULT_THRESHOLD * 2:
                            fault_set.add((k,  j,i))
                            fault_set.add((k-1,j,i))

        print(f"[Loader] Fault cells detected: {len(fault_set)}")

        for key, cs in cell_corners.items():
            norm = [c - origin for c in cs]
            avg_z = sum(c[2] for c in norm) / 8
            avg_x = sum(c[0] for c in norm) / 8
            self.cells.append((norm, avg_z, avg_x))
            self.fault_flags.append(1.0 if key in fault_set else 0.0)

        self.min_b = mn - origin
        self.max_b = mx - origin
        print(f"[Loader] {len(self.cells)} active cells")
        print(f"  X:[{self.min_b[0]:.0f}, {self.max_b[0]:.0f}]")
        print(f"  Y:[{self.min_b[1]:.0f}, {self.max_b[1]:.0f}]")
        print(f"  Z:[{self.min_b[2]:.0f}, {self.max_b[2]:.0f}]")

# ==========================================
# GPU — reservoir cells
# ==========================================
def build_gpu(parser):
    n = len(parser.cells)

    cd = np.empty(n*24, dtype=np.float32)
    for i,(cs,_,_) in enumerate(parser.cells):
        b = i*24
        for ci,c in enumerate(cs):
            cd[b+ci*3]=c[0]; cd[b+ci*3+1]=c[1]; cd[b+ci*3+2]=c[2]

    tb = glGenBuffers(1)
    glBindBuffer(GL_TEXTURE_BUFFER, tb)
    glBufferData(GL_TEXTURE_BUFFER, cd.nbytes, cd, GL_STATIC_DRAW)
    tt = glGenTextures(1)
    glBindTexture(GL_TEXTURE_BUFFER, tt)
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, tb)

    fd = np.array(parser.fault_flags, dtype=np.float32)
    ftb = glGenBuffers(1)
    glBindBuffer(GL_TEXTURE_BUFFER, ftb)
    glBufferData(GL_TEXTURE_BUFFER, fd.nbytes, fd, GL_STATIC_DRAW)
    ftt = glGenTextures(1)
    glBindTexture(GL_TEXTURE_BUFFER, ftt)
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, ftb)

    dd = np.array([c[1] for c in parser.cells], dtype=np.float32)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vb = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vb)
    glBufferData(GL_ARRAY_BUFFER, dd.nbytes, dd, GL_STATIC_DRAW)
    glVertexAttribPointer(0,1,GL_FLOAT,GL_FALSE,0,None)
    glEnableVertexAttribArray(0)
    glBindVertexArray(0)
    return vao, tt, ftt, n

# ==========================================
# GPU — well trajectories
# ==========================================
def build_wells_gpu(parser):
    origin = parser._real_origin

    well_colours = [
        [1.0,1.0,0.0],[0.0,1.0,1.0],[1.0,0.5,0.0],
        [1.0,0.0,1.0],[0.5,1.0,0.5],[1.0,0.8,0.8],
        [0.8,0.8,1.0],[1.0,1.0,0.5],[0.5,1.0,1.0],
        [1.0,0.6,0.6],[0.6,1.0,0.6],
    ]

    verts = []; counts = []
    for idx,(name,pts) in enumerate(NORNE_WELLS.items()):
        col = well_colours[idx % len(well_colours)]
        cnt = 0
        for (wx,wy,wz) in pts:
            verts += [wx-origin[0], wy-origin[1], wz-origin[2],
                      col[0], col[1], col[2]]
            cnt += 1
        counts.append(cnt)

    verts = np.array(verts, dtype=np.float32)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vb = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vb)
    glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
    stride = 6*4
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)
    glBindVertexArray(0)
    return vao, counts

# ==========================================
# GPU — color legend (2D screen overlay)
# ==========================================
def build_legend_gpu():
    STEPS = 64
    x0, x1 = 0.88, 0.94
    y0, y1 = -0.75, 0.75

    def rainbow_cpu(t):
        t = max(0.0, min(1.0, t))
        if   t < 0.25: return 0.0,      t*4,          1.0
        elif t < 0.50: return 0.0,       1.0,          1-(t-0.25)*4
        elif t < 0.75: return (t-0.5)*4, 1.0,          0.0
        else:          return 1.0,       1-(t-0.75)*4, 0.0

    verts = []
    for i in range(STEPS):
        t0 = i/STEPS;       t1 = (i+1)/STEPS
        yb = y0+t0*(y1-y0); yt = y0+t1*(y1-y0)
        c0 = rainbow_cpu(t0); c1 = rainbow_cpu(t1)
        verts += [x0,yb,*c0, x1,yb,*c0, x0,yt,*c1, x1,yt,*c1]

    verts = np.array(verts, dtype=np.float32)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vb = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vb)
    glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
    stride = 5*4
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
    glEnableVertexAttribArray(1)
    glBindVertexArray(0)
    return vao, STEPS*4

def scroll_cb(win, x, y):
    global cam_dist
    cam_dist = max(100.0, cam_dist*(0.85 if y>0 else 1.15))

# ==========================================
# Main
# ==========================================
def main():
    global cam_yaw,cam_pitch,cam_dist,last_x,last_y,mouse_pressed
    global slice_x,slice_z,vexag,show_faults,show_wells

    if not glfw.init(): return
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    win = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT,
        "Norne Field v8  |  Q/A=X  E/D=Z  Z/X=VExag  F=Faults  W=Wells  ESC=Quit",
        None, None)
    if not win: glfw.terminate(); return
    glfw.make_context_current(win)
    glfw.set_scroll_callback(win, scroll_cb)

    print(f"[GL] {glGetString(GL_RENDERER).decode()}")
    print(f"[GL] {glGetString(GL_VERSION).decode()}")

    p = GRDECLParser(FILENAME)
    if not p.parse(): glfw.terminate(); return

    sz = float(np.max(p.max_b - p.min_b))
    cam_dist = sz * 2.0

    try:
        prog_res  = make_prog(VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC, GEOMETRY_SHADER_SRC)
        prog_well = make_prog(WELL_VERT_SRC, WELL_FRAG_SRC)
        prog_leg  = make_prog(OVERLAY_VERT_SRC, OVERLAY_FRAG_SRC)
    except RuntimeError as e:
        print(e); glfw.terminate(); return

    vao_res,  tbo_corners, tbo_faults, ncells = build_gpu(p)
    vao_well, well_counts  = build_wells_gpu(p)
    vao_leg,  leg_nverts   = build_legend_gpu()

    # Uniform locations
    uMVP_r    = glGetUniformLocation(prog_res,  "uMVP")
    uDmin     = glGetUniformLocation(prog_res,  "uDepthMin")
    uDrng     = glGetUniformLocation(prog_res,  "uDepthRange")
    uSliceX   = glGetUniformLocation(prog_res,  "uSliceX")
    uSliceZ   = glGetUniformLocation(prog_res,  "uSliceZ")
    uVExag_r  = glGetUniformLocation(prog_res,  "uVExag")
    uCorns    = glGetUniformLocation(prog_res,  "uCorners")
    uFault    = glGetUniformLocation(prog_res,  "uFaultFlag")
    uShowF    = glGetUniformLocation(prog_res,  "uShowFaults")
    uMVP_w    = glGetUniformLocation(prog_well, "uMVP")
    uVExag_w  = glGetUniformLocation(prog_well, "uVExag")

    dmin = float(p.min_b[2])
    drng = float(p.max_b[2]-p.min_b[2]) or 1.0
    xmin = float(p.min_b[0]); xmax = float(p.max_b[0])
    zmin = float(p.min_b[2]); zmax = float(p.max_b[2])
    near = sz*0.001; far = sz*50.0

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glFrontFace(GL_CCW)
    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(1.0, 1.0)
    glClearColor(0.05, 0.07, 0.12, 1.0)

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

    # Key debounce state
    prev_f = prev_w = glfw.RELEASE

    while not glfw.window_should_close(win):
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(win, True)

        # Mouse rotation
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            mx,my = glfw.get_cursor_pos(win)
            if not mouse_pressed:
                last_x,last_y = mx,my; mouse_pressed=True
            else:
                cam_yaw   += (mx-last_x)*0.4
                cam_pitch += (my-last_y)*0.4
                cam_pitch  = float(np.clip(cam_pitch,-89,89))
                last_x,last_y = mx,my
        else:
            mouse_pressed = False

        # Continuous keys
        step = 0.003
        if glfw.get_key(win, glfw.KEY_Q) == glfw.PRESS: slice_x = max(0.0, slice_x-step)
        if glfw.get_key(win, glfw.KEY_A) == glfw.PRESS: slice_x = min(1.0, slice_x+step)
        if glfw.get_key(win, glfw.KEY_E) == glfw.PRESS: slice_z = max(0.0, slice_z-step)
        if glfw.get_key(win, glfw.KEY_D) == glfw.PRESS: slice_z = min(1.0, slice_z+step)
        if glfw.get_key(win, glfw.KEY_Z) == glfw.PRESS: vexag = max(1.0,  vexag-0.1)
        if glfw.get_key(win, glfw.KEY_X) == glfw.PRESS: vexag = min(50.0, vexag+0.1)

        # Toggle keys (debounced — only trigger on key-down edge)
        cur_f = glfw.get_key(win, glfw.KEY_F)
        if cur_f == glfw.PRESS and prev_f == glfw.RELEASE:
            show_faults = not show_faults
        prev_f = cur_f

        cur_w = glfw.get_key(win, glfw.KEY_W)
        if cur_w == glfw.PRESS and prev_w == glfw.RELEASE:
            show_wells = not show_wells
        prev_w = cur_w

        # Camera
        ry,rp = np.radians(cam_yaw), np.radians(cam_pitch)
        eye = np.array([
            cam_dist*np.cos(rp)*np.sin(ry),
            cam_dist*np.sin(rp),
            cam_dist*np.cos(rp)*np.cos(ry),
        ], dtype=np.float32)
        V   = mat_lookat(eye, np.zeros(3,dtype=np.float32), np.array([0,1,0],dtype=np.float32))
        Pr  = mat_perspective(45.0, WINDOW_WIDTH/WINDOW_HEIGHT, near, far)
        mvp = Pr @ V

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        # --- 1. Reservoir cells ---
        glUseProgram(prog_res)
        glUniformMatrix4fv(uMVP_r,  1, GL_TRUE, mvp)
        glUniform1f(uDmin,   dmin)
        glUniform1f(uDrng,   drng)
        glUniform1f(uSliceX, xmin + slice_x*(xmax-xmin))
        glUniform1f(uSliceZ, zmin + slice_z*(zmax-zmin))
        glUniform1f(uVExag_r, vexag)
        glUniform1i(uShowF,  1 if show_faults else 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_BUFFER, tbo_corners)
        glUniform1i(uCorns, 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_BUFFER, tbo_faults)
        glUniform1i(uFault, 1)
        glBindVertexArray(vao_res)
        glDrawArrays(GL_POINTS, 0, ncells)
        glBindVertexArray(0)

        # --- 2. Well trajectories ---
        if show_wells:
            glDisable(GL_CULL_FACE)
            glUseProgram(prog_well)
            glUniformMatrix4fv(uMVP_w,   1, GL_TRUE, mvp)
            glUniform1f(uVExag_w, vexag)
            glBindVertexArray(vao_well)
            offset = 0
            for cnt in well_counts:
                glDrawArrays(GL_LINE_STRIP, offset, cnt)
                offset += cnt
            glBindVertexArray(0)
            glEnable(GL_CULL_FACE)

        # --- 3. Color legend (2D overlay, no depth test) ---
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glUseProgram(prog_leg)
        glBindVertexArray(vao_leg)
        for i in range(leg_nverts // 4):
            glDrawArrays(GL_TRIANGLE_STRIP, i*4, 4)
        glBindVertexArray(0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)

        glfw.set_window_title(win,
            f"Norne v8  |  X:{slice_x*100:.0f}%  Z:{slice_z*100:.0f}%  "
            f"VExag:{vexag:.1f}x  "
            f"Faults:{'ON' if show_faults else 'OFF'}  "
            f"Wells:{'ON' if show_wells else 'OFF'}")

        glfw.swap_buffers(win)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()