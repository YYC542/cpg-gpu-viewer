# Author:Ach1n
# Data:2026/3/17 13:05
# Flag:I earned it.
# Author: Ach1n
# Date: 2026/3/17
# Flag: I earned it.
# Version: v7 - Full 3D volumetric rendering with dual-axis slicing
# Description: Norne Oilfield GRDECL Visualizer
#              Modern OpenGL 3.3 Core Profile + Geometry Shader
#              Shows ALL layers (not just surface), with X and Z cross-section slicing

import glfw
from OpenGL.GL import *
import numpy as np
import os

# ==========================================
# Configuration
# ==========================================
FILENAME = "NORNE_ATW2013.GRDECL"
WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 720

# Camera — angled to show 3D structure + cross-section simultaneously
cam_dist  = 20000.0
cam_yaw   = 30.0        # slight rotation to show the cut face clearly
cam_pitch = 30.0        # see top surface AND the internal cross-section face
last_x = last_y = 0
mouse_pressed = False

# Dual slice controls
# Default: X slice at 60% → internal layers visible on startup (like reference image)
slice_x = 0.6           # X-axis cross-section (Q=decrease / A=increase)
slice_z = 1.0           # Z-axis (depth) cross-section (E=decrease / D=increase)
vexag  = 10.0          # Vertical exaggeration: Z key=decrease / X key=increase

# -----------------------------------------------------------
# GLSL
# -----------------------------------------------------------
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

uniform mat4  uMVP;
uniform float uSliceX;          // X cross-section threshold
uniform float uSliceZ;          // Z (depth) cross-section threshold
uniform float uVExag;           // Vertical exaggeration factor (1.0 = true scale)
uniform samplerBuffer uCorners;

vec3 getCorner(int cell, int ci) {
    int base = cell * 24 + ci * 3;
    return vec3(
        texelFetch(uCorners, base    ).r,
        texelFetch(uCorners, base + 1).r,
        texelFetch(uCorners, base + 2).r
    );
}

// Apply vertical exaggeration: scale Z axis only
vec3 vex(vec3 p) {
    return vec3(p.x, p.y, p.z * uVExag);
}

void emitFace(vec3 a, vec3 b, vec3 c, vec3 d, float depth) {
    // Exaggerate Z before computing normals → lighting correct in exaggerated space
    vec3 ae=vex(a), be=vex(b), ce=vex(c), de=vex(d);
    vec3 n = normalize(cross(be - ae, de - ae));
    fNormal = n;
    fDepth  = depth;
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

    // --- X-axis slice (in original unexaggerated space) ---
    float cx = (c0.x+c1.x+c2.x+c3.x+c4.x+c5.x+c6.x+c7.x)*0.125;
    if (cx > uSliceX) return;

    // --- Z-axis depth slice (in original unexaggerated space) ---
    float cz = (c0.z+c1.z+c2.z+c3.z+c4.z+c5.z+c6.z+c7.z)*0.125;
    if (cz > uSliceZ) return;

    float d = cz; // original depth for colour mapping

    // Emit all 6 faces (vex() applied inside emitFace)
    emitFace(c0, c1, c2, c3, d); // top
    emitFace(c4, c5, c6, c7, d); // bottom
    emitFace(c0, c1, c4, c5, d); // front
    emitFace(c2, c3, c6, c7, d); // back
    emitFace(c0, c2, c4, c6, d); // left
    emitFace(c1, c3, c5, c7, d); // right
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core
in  float fDepth;
in  vec3  fNormal;
out vec4  FragColor;

uniform float uDepthMin;
uniform float uDepthRange;

// Rainbow colour ramp: blue→cyan→green→yellow→red (like professional reservoir simulators)
vec3 rainbow(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c;
    if      (t < 0.25) c = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), t * 4.0);
    else if (t < 0.50) c = mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (t - 0.25) * 4.0);
    else if (t < 0.75) c = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (t - 0.50) * 4.0);
    else               c = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.75) * 4.0);
    return c;
}

void main() {
    float t   = clamp((fDepth - uDepthMin) / uDepthRange, 0.0, 1.0);
    vec3  col = rainbow(t);

    vec3  L = normalize(vec3(1.0, 2.0, 3.0));
    // abs() → both faces always lit, eliminates black patches from inverted normals
    float d = max(abs(dot(normalize(fNormal), L)), 0.15);
    FragColor = vec4(col * d * 1.2, 1.0);
}
"""

# -----------------------------------------------------------
# Shader helpers
# -----------------------------------------------------------
def compile_shader(src, kind):
    s = glCreateShader(kind)
    glShaderSource(s, src)
    glCompileShader(s)
    if not glGetShaderiv(s, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(s).decode())
    return s

def create_program(vs, gs, fs):
    p = glCreateProgram()
    for sh in [compile_shader(vs, GL_VERTEX_SHADER),
               compile_shader(gs, GL_GEOMETRY_SHADER),
               compile_shader(fs, GL_FRAGMENT_SHADER)]:
        glAttachShader(p, sh)
        glDeleteShader(sh)
    glLinkProgram(p)
    if not glGetProgramiv(p, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(p).decode())
    return p

# -----------------------------------------------------------
# Matrix helpers — row-major, passed with GL_TRUE (transpose)
# -----------------------------------------------------------
def mat_perspective(fovy, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fovy) / 2.0)
    return np.array([
        [f/aspect, 0,  0,                     0],
        [0,        f,  0,                     0],
        [0,        0,  (far+near)/(near-far),  2*far*near/(near-far)],
        [0,        0, -1,                     0],
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

# -----------------------------------------------------------
# GRDECL Parser (unchanged — robust pillar interpolation)
# -----------------------------------------------------------
class GRDECLParser:
    def __init__(self, fp):
        self.filepath = fp
        self.nx = self.ny = self.nz = 0
        self.coord = []; self.zcorn = []; self.actnum = []
        self.cells = []
        self.min_b = np.full(3,  1e18)
        self.max_b = np.full(3, -1e18)

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
            pillars = np.array(self.coord,dtype=np.float64).reshape(ny+1,nx+1,6)
        except:
            print("[Error] COORD reshape failed"); return
        zc = np.array(self.zcorn, dtype=np.float64)

        def zidx(i,j,k,hk,hj,hi):
            return (k*2+hk)*(ny*2*nx*2) + (j*2+hj)*(nx*2) + (i*2+hi)

        def pillar_pt(pi,pj,z):
            p = pillars[pj,pi]
            dz = p[5]-p[2]
            t = 0.0 if abs(dz)<1e-9 else (z-p[2])/dz
            return np.array([p[0]+t*(p[3]-p[0]), p[1]+t*(p[4]-p[1]), z])

        raw = []
        mn = np.full(3, 1e18); mx = np.full(3,-1e18)
        defs = [(0,0,0,0,0),(1,0,0,0,1),(0,1,0,1,0),(1,1,0,1,1),
                (0,0,1,0,0),(1,0,1,0,1),(0,1,1,1,0),(1,1,1,1,1)]

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    cid = k*nx*ny+j*nx+i
                    if self.actnum and cid<len(self.actnum) and self.actnum[cid]==0:
                        continue
                    cs = []
                    ok = True
                    for (di,dj,hk,hj,hi) in defs:
                        zi = zidx(i,j,k,hk,hj,hi)
                        if zi>=len(zc): ok=False; break
                        cs.append(pillar_pt(i+di,j+dj,zc[zi]))
                    if ok:
                        for c in cs:
                            mn = np.minimum(mn,c)
                            mx = np.maximum(mx,c)
                        raw.append(cs)

        # Centre at origin → float32-friendly values
        origin = (mn+mx)*0.5
        print(f"[Loader] Origin: {origin}")
        for cs in raw:
            norm = [c-origin for c in cs]
            avg_z = sum(c[2] for c in norm)/8
            avg_x = sum(c[0] for c in norm)/8
            self.cells.append((norm, avg_z, avg_x))
        self.min_b = mn-origin
        self.max_b = mx-origin
        print(f"[Loader] {len(self.cells)} active cells")
        print(f"  X:[{self.min_b[0]:.0f}, {self.max_b[0]:.0f}]")
        print(f"  Y:[{self.min_b[1]:.0f}, {self.max_b[1]:.0f}]")
        print(f"  Z:[{self.min_b[2]:.0f}, {self.max_b[2]:.0f}]")

# -----------------------------------------------------------
# GPU buffer setup
# -----------------------------------------------------------
def build_gpu(parser):
    n = len(parser.cells)
    # Corner buffer (8 corners × 3 floats per cell)
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

    # One depth value per cell → vertex attribute for geometry shader
    dd = np.array([c[1] for c in parser.cells], dtype=np.float32)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vb = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vb)
    glBufferData(GL_ARRAY_BUFFER, dd.nbytes, dd, GL_STATIC_DRAW)
    glVertexAttribPointer(0,1,GL_FLOAT,GL_FALSE,0,None)
    glEnableVertexAttribArray(0)
    glBindVertexArray(0)
    return vao, tt, n

def scroll_cb(win, x, y):
    global cam_dist
    cam_dist = max(100.0, cam_dist*(0.85 if y>0 else 1.15))

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    global cam_yaw,cam_pitch,cam_dist,last_x,last_y,mouse_pressed
    global slice_x, slice_z, vexag

    if not glfw.init(): return
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    win = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT,
        "Norne Field Viewer  |  X-slice: Q/A   Z-slice: E/D   Zoom: Scroll   Rotate: Drag",
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
    print(f"[Camera] scene size={sz:.0f}  initial dist={cam_dist:.0f}")

    try:
        prog = create_program(VERTEX_SHADER_SRC, GEOMETRY_SHADER_SRC, FRAGMENT_SHADER_SRC)
    except RuntimeError as e:
        print(e); glfw.terminate(); return

    vao, tbo, ncells = build_gpu(p)

    uMVP   = glGetUniformLocation(prog, "uMVP")
    uDmin  = glGetUniformLocation(prog, "uDepthMin")
    uDrng  = glGetUniformLocation(prog, "uDepthRange")
    uSliceX = glGetUniformLocation(prog, "uSliceX")
    uSliceZ = glGetUniformLocation(prog, "uSliceZ")
    uVExag = glGetUniformLocation(prog, "uVExag")
    uCorns = glGetUniformLocation(prog, "uCorners")

    dmin  = float(p.min_b[2])
    drng  = float(p.max_b[2]-p.min_b[2]) or 1.0
    xmin  = float(p.min_b[0])
    xmax  = float(p.max_b[0])
    zmin  = float(p.min_b[2])
    zmax  = float(p.max_b[2])
    near  = sz * 0.001
    far   = sz * 50.0

    glEnable(GL_DEPTH_TEST)
    glClearColor(0.05, 0.07, 0.12, 1.0)   # dark navy background

    print("\n=== Controls ===")
    print("  Left-drag : Rotate")
    print("  Scroll    : Zoom")
    print("  Q / A     : X cross-section  (decrease / increase)")
    print("  E / D     : Z depth slice    (decrease / increase)")
    print("  Z / X     : Vertical exaggeration (decrease / increase, default 10x)")
    print("  ESC       : Quit")
    print("================\n")

    # Key repeat state
    prev_q = prev_a = prev_e = prev_d = glfw.RELEASE

    while not glfw.window_should_close(win):
        if glfw.get_key(win, glfw.KEY_ESCAPE)==glfw.PRESS:
            glfw.set_window_should_close(win, True)

        # Mouse rotation
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT)==glfw.PRESS:
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

        # Slice controls (hold key for smooth sweep)
        step = 0.003
        if glfw.get_key(win, glfw.KEY_Q)==glfw.PRESS: slice_x = max(0.0, slice_x - step)
        if glfw.get_key(win, glfw.KEY_A)==glfw.PRESS: slice_x = min(1.0, slice_x + step)
        if glfw.get_key(win, glfw.KEY_E)==glfw.PRESS: slice_z = max(0.0, slice_z - step)
        if glfw.get_key(win, glfw.KEY_Z)==glfw.PRESS: vexag = max(1.0,  vexag - 0.5)   # Z = less exaggeration
        if glfw.get_key(win, glfw.KEY_X)==glfw.PRESS: vexag = min(50.0, vexag + 0.5)   # X = more exaggeration
        if glfw.get_key(win, glfw.KEY_D)==glfw.PRESS: slice_z = min(1.0, slice_z + step)

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
        glUseProgram(prog)

        glUniformMatrix4fv(uMVP,  1, GL_TRUE, mvp)
        glUniform1f(uDmin,  dmin)
        glUniform1f(uDrng,  drng)
        glUniform1f(uSliceX, xmin + slice_x*(xmax-xmin))
        glUniform1f(uSliceZ, zmin + slice_z*(zmax-zmin))
        glUniform1f(uVExag, vexag)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_BUFFER, tbo)
        glUniform1i(uCorns, 0)

        glBindVertexArray(vao)
        glDrawArrays(GL_POINTS, 0, ncells)
        glBindVertexArray(0)

        # Update title with current slice values
        glfw.set_window_title(win,
            f"Norne Field  |  X-slice: {slice_x*100:.0f}%  Z-slice: {slice_z*100:.0f}%  VExag: {vexag:.1f}x  "
            f"|  Q/A=X  E/D=Z  Scroll=Zoom  Drag=Rotate")

        glfw.swap_buffers(win)
        glfw.poll_events()

    glfw.terminate()

if __name__=="__main__":
    main()