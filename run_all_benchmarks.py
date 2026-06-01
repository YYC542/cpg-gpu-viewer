# ============================================================================
# 文件名: run_all_benchmarks.py
# 用途: 依次运行 gs / vbo / legacy 三条路径, 自动汇总成论文用的对比表。
#
# 用法 (在你本地 RTX 4070 机器上, 与 norne_benchmark.py 同目录):
#   python run_all_benchmarks.py
#
# 可选:
#   python run_all_benchmarks.py --seconds 30 --file NORNE_ATW2013.GRDECL
#
# 注意:
#   - 必须在【真实 GPU】上跑 (本机), 不能在无显卡环境跑。
#   - 跑前关闭其他占用 GPU 的程序 (浏览器视频/游戏等), 插电、性能模式。
#   - 每条路径会弹出一个渲染窗口并自动旋转模型, 跑完自动关闭。
# ============================================================================

import argparse
import subprocess
import sys
import re
import os

def run_one(mode, args):
    cmd = [sys.executable, "norne_benchmark.py",
           "--mode", mode,
           "--file", args.file,
           "--seconds", str(args.seconds),
           "--warmup", str(args.warmup),
           "--width", str(args.width),
           "--height", str(args.height),
           "--vexag", str(args.vexag)]
    print(f"\n>>> 运行 {mode} ...  命令: {' '.join(cmd)}")
    # 强制 UTF-8 解码子程序输出 (避免 Windows GBK 编码报错)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=args.seconds + 120, env=env)
    except subprocess.TimeoutExpired:
        print(f"[WARN] {mode} 超时")
        return None
    # 用 utf-8 解码, 解不了的字节用替代符号代替, 绝不崩溃
    stdout = out.stdout.decode("utf-8", errors="replace") if out.stdout else ""
    stderr = out.stderr.decode("utf-8", errors="replace") if out.stderr else ""
    txt = stdout + stderr
    # 解析关键数字
    res = {"mode": mode}
    for key, pat in [
        ("avg",   r"平均 FPS\s*:\s*([\d.]+)"),
        ("med",   r"中位 FPS\s*:\s*([\d.]+)"),
        ("low1",  r"1% Low FPS\s*:\s*([\d.]+)"),
        ("ms",    r"平均帧时\s*:\s*([\d.]+)"),
        ("cells", r"活跃单元数\s*:\s*([\d,]+)"),
        ("verts", r"输出三角形顶点\s*:\s*([\d,]+)"),
        ("vram",  r"几何缓冲显存\s*:\s*([\d.]+)"),
    ]:
        m = re.search(pat, txt)
        res[key] = m.group(1) if m else "-"
    if res["avg"] == "-":
        print(f"[WARN] 没解析到 {mode} 的结果, 原始输出:\n{txt[-800:]}")
        return None
    print(f"    {mode}: 平均 {res['avg']} FPS, 帧时 {res['ms']} ms")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="NORNE_ATW2013.GRDECL")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--warmup", type=int, default=120)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--vexag", type=float, default=4.0)
    ap.add_argument("--skip-legacy", action="store_true",
                    help="跳过立即模式 (若 GLU 不可用或太慢)")
    args = ap.parse_args()

    if not os.path.exists(args.file):
        print(f"[FATAL] 找不到数据文件 {args.file}, 请把它放在本目录, 或用 --file 指定路径")
        sys.exit(1)

    modes = ["gs", "vbo"] if args.skip_legacy else ["gs", "vbo", "legacy"]
    results = []
    for m in modes:
        r = run_one(m, args)
        if r:
            results.append(r)

    if not results:
        print("[FATAL] 没有任何结果"); sys.exit(1)

    # ---- 汇总表 ----
    print("\n" + "=" * 72)
    print("  汇总 (分辨率 {}x{}, 满模型, 关剔除)".format(args.width, args.height))
    print("=" * 72)
    print(f"  {'路径':<10}{'平均FPS':>10}{'中位FPS':>10}{'1%Low':>10}{'帧时(ms)':>12}{'几何显存(MB)':>14}")
    print("  " + "-" * 64)
    name_map = {"gs": "GS(几何着色)", "vbo": "VBO预生成", "legacy": "Legacy立即模式"}
    fps = {}
    for r in results:
        print(f"  {name_map[r['mode']]:<10}{r['avg']:>10}{r['med']:>10}{r['low1']:>10}{r['ms']:>12}{r['vram']:>14}")
        try:
            fps[r["mode"]] = float(r["avg"])
        except ValueError:
            pass
    print("=" * 72)

    # ---- 论文用结论 ----
    print("\n【填入论文的数字】")
    if "gs" in fps:
        print(f"  Table 1 行1 (GS 满模型)        : {fps['gs']:.0f} FPS")
    if "vbo" in fps:
        print(f"  Table 1 新行 [VBO_FPS]         : {fps['vbo']:.0f} FPS")
        print(f"  Table 1 新行 [VBO_MS]          : {1000.0/fps['vbo']:.1f} ms")
    if "legacy" in fps:
        print(f"  Table 1 行5 (Legacy)           : {fps['legacy']:.0f} FPS")
    if "gs" in fps and "legacy" in fps and fps["legacy"] > 0:
        print(f"  GS vs Legacy 加速比            : {fps['gs']/fps['legacy']:.0f}×  (替换论文里的 19×)")
    if "gs" in fps and "vbo" in fps:
        diff = (fps["gs"] - fps["vbo"]) / fps["vbo"] * 100
        print(f"  GS 相对 VBO 帧率差             : {diff:+.1f}%  (预期接近0; 论文§5.3据此微调)")
    print("\n  把上面这些数字发给我, 我帮你填进论文并相应调整正文措辞。")
    print("  显存对比仍用计算值: GS≈13MB vs VBO≈48.8MB (论文§5.4不变)。\n")


if __name__ == "__main__":
    main()