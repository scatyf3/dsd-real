"""Overview figure: the real e2e system, the bridge that fits the sim from it,
the simulator, one lockstep step on both sides, and the resulting error.

    python docs/figures/make_e2e_vs_sim.py      # -> e2e_vs_sim.{svg,pdf,png}

Stdlib only; PDF/PNG need rsvg-convert. MAPE cells are read from
results/table3_client/real_vs_sim_*.md, the step timings from
results/table3_client/calibration.pooled.json and the deployment p50s.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = ROOT / "results/table3_client"

W, H = 1000, 652
FONT = "TeX Gyre Heros, Helvetica, Arial, sans-serif"
MONO = "Adwaita Mono, Menlo, monospace"
INK, INK2, INK3 = "#1f1f1e", "#4f4f4a", "#8a8a84"
BLUE, BLUE_T, BLUE_L = "#2a78d6", "#edf4fd", "#9ec5f4"
ORNG, ORNG_T, ORNG_L = "#d95926", "#fdf0e9", "#f5b596"
GRAY, GRAY_T = "#6b6b66", "#f4f4f2"

out: list[str] = []


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=12.5, color=INK, weight=400, anchor="start", font=FONT, italic=False, rot=None):
    t = f' transform="rotate({rot} {x} {y})"' if rot else ""
    st = ' font-style="italic"' if italic else ""
    out.append(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" fill="{color}" '
               f'font-weight="{weight}" text-anchor="{anchor}"{st}{t}>{esc(s)}</text>')


def rect(x, y, w, h, fill="none", stroke="none", sw=1, r=6, dash=None, extra=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" '
               f'stroke="{stroke}" stroke-width="{sw}"{d}{extra}/>')


def box(x, y, w, h, title, lines, accent, fill="#ffffff", tsize=13, lsize=11.5, mono_lines=(), lh=14.5):
    rect(x, y, w, h, fill=fill, stroke=accent, sw=1.2, r=5)
    text(x + 8, y + 17, title, size=tsize, weight=650, color=INK)
    for i, ln in enumerate(lines):
        text(x + 8, y + 17 + 16 + i * lh, ln, size=lsize, color=INK2,
             font=MONO if i in mono_lines else FONT)


def arrow(pts, color=INK2, sw=1.3, dash=None, head="ah", both=False):
    p = " ".join(f"{x},{y}" for x, y in pts)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    s = f' marker-start="url(#{head}s)"' if both else ""
    out.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}"{d}'
               f' marker-end="url(#{head})"{s} stroke-linejoin="round"/>')


def hook(x, y, n, label, w):
    """A shim hook: numbered pill patched into dsdSim."""
    rect(x, y, w, 20, fill=ORNG_T, stroke=ORNG, sw=1.1, r=10)
    out.append(f'<circle cx="{x + 10}" cy="{y + 10}" r="7" fill="{ORNG}"/>')
    text(x + 10, y + 14, str(n), size=10.5, color="#ffffff", weight=700, anchor="middle")
    text(x + 22, y + 14.2, label, size=11.5, color=INK)


def lane(y, h, tint, accent, label):
    rect(20, y, 940, h, fill=tint, stroke="none", r=8)
    out.append(f'<rect x="20" y="{y}" width="22" height="{h}" fill="{accent}" rx="6"/>')
    out.append(f'<rect x="34" y="{y}" width="8" height="{h}" fill="{accent}"/>')
    text(35, y + h / 2, label, size=11, weight=700, color="#ffffff", anchor="middle", rot=-90)

# --------------------------------------------------------------------- data
def mape_rows():
    rows = [("pooled", "fit, pooled", "profile"), ("tiered", "fit, per-N", "profile"),
            ("replay", "fit, per-N", "replay"), ("vidur_raw_replay", "VIDUR (raw)", "replay"),
            ("vidur_oh_replay", "VIDUR + OH", "replay"), ("vidur_oh", "VIDUR + OH", "profile")]
    res = []
    for key, lat, acc in rows:
        md = (RES / f"real_vs_sim_{key}.md").read_text()
        allv = float(re.search(r"MAPE over \d+ cells: ([\d.]+)%", md).group(1))
        byn = [float(v) for v in re.findall(r"N=\d+ ([\d.]+)%", re.search(r"MAPE by N:.*", md).group(0))]
        res.append((lat, acc, byn, allv))
    return res


cal = json.loads((RES / "calibration.pooled.json").read_text())
tg, dr = cal["target"], cal["draft"]["vanilla"]
B, G, SEQS, RTT = 2, 4, 2, 10.0  # vanilla N=1: p50 batch 2, both requests on d0
sim_fwd = tg["verify_base_ms"] + tg["verify_ms_per_seq"] * B + tg["verify_ms_per_token"] * B * (G + 1)
sim_draft = dr["base_ms"] + dr["ms_per_token"] * G + dr["ms_per_token_per_seq"] * G * (SEQS - 1)
# docs/table3-results.md deployment table, vanilla N=1 (p50 of steps.jsonl)
real_fwd, real_draft, real_net = 25.9, 23.2, 11.0

# --------------------------------------------------------------------- canvas
out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
out.append(f"""<defs>
<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="{INK2}"/></marker>
<marker id="ahs" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="{INK2}"/></marker>
<marker id="ab" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="{BLUE}"/></marker>
<marker id="abs" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="{BLUE}"/></marker>
<marker id="ao" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="{ORNG}"/></marker>
<marker id="aos" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="{ORNG}"/></marker>
<pattern id="net" patternUnits="userSpaceOnUse" width="5" height="5" patternTransform="rotate(45)">
  <rect width="5" height="5" fill="#ffffff"/><line x1="0" y1="0" x2="0" y2="5" stroke="{INK3}" stroke-width="1.6"/></pattern>
</defs>""")
rect(0, 0, W, H, fill="#ffffff", r=0)

# ===================================================================== (a)
text(4, 16, "(a)", size=14, weight=700)

# ---- real lane
lane(20, 136, BLUE_T, BLUE, "REAL SYSTEM")
y0, hb = 44, 98
box(52, y0, 118, hb, "Client (driver)", ["gsm8k lengths", "Poisson, λ = N/s", "↑ prompt  +RTT/2", "↓ tokens  +RTT/2"], BLUE)
# target with its engine-step strip
box(184, y0, 282, hb, "Target · Llama-3.1-8B · A6000", ["vLLM 0.27.1, ≤64 seqs, continuous batching"], BLUE)
sx, sy = 190, y0 + 50
rect(sx, sy, 132, 22, fill=BLUE, r=3)
text(sx + 66, sy + 15, "forward: verify + prefill", size=11, color="#ffffff", weight=600, anchor="middle")
rect(sx + 134, sy, 136, 22, fill="#ffffff", stroke=BLUE, sw=1.2, r=3)
text(sx + 202, sy + 15, "propose()  — blocks", size=11, color=BLUE, weight=600, anchor="middle")
text(sx, sy + 38, "RemoteDraftProposer via custom_class", size=10.5, color=INK2, font=MONO)
# transport
box(480, y0, 110, hb, "ZMQ", ["netem 10 ms RTT", "+≈1 ms transport", "1 message per", "drafter per step"], BLUE)
# drafters: stacked cards
for k in (2, 1):
    rect(604 + 4 * k, y0 - 4 * k, 156, hb, fill="#ffffff", stroke=BLUE_L, sw=1, r=5)
box(604, y0, 156, hb, "Drafters d1 … dN", ["Llama-3.2-1B (own vLLM)", "or EAGLE3 head", "2 GPUs: T1 = 1 per GPU", "T2 = 2–4 per GPU (N≥4)"], BLUE)
# records
box(782, y0, 166, hb, "Per-round records", ["steps.jsonl: γ, accepted,", "t_send, t_recv, draft_ms", "dsd_metrics.json:", "TTFT · TPOT · Thr."], BLUE)
# real arrows
arrow([(170, sy + 11), (sx - 2, sy + 11)], color=BLUE, head="ab")
arrow([(460, sy + 11), (478, sy + 11)], color=BLUE, head="ab", both=True)
arrow([(590, sy + 11), (602, sy + 11)], color=BLUE, head="ab", both=True)
text(535, y0 + hb + 25, "γ drafts ⇄ confirmed prefix", size=10.5, color=INK2, anchor="middle", italic=True)
arrow([(466, y0 + 22), (473, y0 + 22), (473, y0 - 14), (865, y0 - 14), (865, y0 - 1)], color=BLUE, head="ab", dash="3 2.5")
text(820, y0 - 18, "logged every propose()", size=10.5, color=BLUE, anchor="end", italic=True)

# ---- bridge lane
yb = 170
lane(yb, 100, GRAY_T, GRAY, "BRIDGE")
y1, h1 = yb + 20, 72
box(52, y1, 174, h1, "Workload", ["same trace & prompt lengths", "same Poisson arrivals (seed)", "rounds × γ/L̄ (output scale)"], GRAY, lsize=11)
box(238, y1, 266, h1, "Latency A · fit (calibrate_from_run)",
    [f"verify = {tg['verify_base_ms']:.1f} + {tg['verify_ms_per_seq']:.2f}·B + {tg['verify_ms_per_token']:.3f}·Σ(γ+1)",
     f"draft  = {dr['base_ms']:.1f} + {dr['ms_per_token']:.1f}·γ + {dr['ms_per_token_per_seq']:.2f}·γ·(B_d−1)",
     "pooled over runs  |  per N (tier)"], GRAY, lsize=11, mono_lines=(0, 1))
box(516, y1, 214, h1, "Latency B · VIDUR (A6000)", ["operator profiles, sarathi kernels", "+ overhead, fit on N=1 only:", "4.5 ms/step; draft 3.8/call, 1.4/step"], GRAY, lsize=11)
box(742, y1, 206, h1, "Acceptance", ["profile: P(accept | pos, ctx),", "   from a γ=8 calibration run", "replay: real γ + accepted / round"], GRAY, lsize=11)
# records -> bridge: one drop, one bus
arrow([(865, y0 + hb), (865, y1 - 1)], color=GRAY)
bus = yb + 9
out.append(f'<line x1="121" y1="{bus}" x2="865" y2="{bus}" stroke="{GRAY}" stroke-width="1.3"/>')
for xc in (121, 371):
    arrow([(xc, bus), (xc, y1 - 1)], color=GRAY)

# ---- sim lane
ys = 284
lane(ys, 124, ORNG_T, ORNG, "SIMULATOR")
y2, h2 = ys + 14, 96
box(52, y2, 138, h2, "make_config.py", ["config.yaml +", "retimed trace", "1 virtual drafter", "per request (RR label)"], ORNG)
# dsdSim box with patched components and hooks
rect(204, y2, 548, h2, fill="#ffffff", stroke=ORNG, sw=1.2, r=5)
text(212, y2 + 17, "dsdSim (unmodified clone) · SimPy event loop", size=13, weight=650)
text(744, y2 + 17, "①–③ patched in by dsdsim_shim.py", size=10.5, color=ORNG, anchor="end", italic=True)
rect(212, y2 + 26, 330, 64, fill="#fbfbfa", stroke="#d8d8d4", sw=1, r=4)
text(220, y2 + 41, "TargetServer: batch window 0.01 ms, ≤64 jobs", size=11, color=INK2)
hook(220, y2 + 48, 1, "latency: fit | VIDUR (one joint batch)", 246)
hook(220, y2 + 69, 2, "lockstep: + RTT + draft(γ_max, busiest d)", 268)
rect(550, y2 + 26, 194, 64, fill="#fbfbfa", stroke="#d8d8d4", sw=1, r=4)
text(558, y2 + 41, "DraftServer: conversations", size=11, color=INK2)
hook(558, y2 + 48, 3, "acceptance: profile | replay", 180)
text(558, y2 + 82, "Metrics.summary()", size=10.5, color=INK2, font=MONO)
box(766, y2, 182, h2, "compare_real_sim.py", ["same metric rules", "(dsdSim convention)", "real / sim (err) per cell,", "MAPE → (c)"], INK2)


def badge(x, y, n):
    out.append(f'<circle cx="{x}" cy="{y}" r="6.5" fill="{ORNG}"/>')
    text(x, y + 3.8, str(n), size=10, color="#ffffff", weight=700, anchor="middle")


# bridge -> sim: arrows end on the dsdSim border, badge = the hook they feed
yo = y1 + h1
arrow([(121, yo), (121, y2 - 1)], color=GRAY)
arrow([(371, yo), (371, y2 - 1)], color=GRAY)
badge(382, y2 - 10, 1)
arrow([(623, yo), (623, yo + 13), (470, yo + 13), (470, y2 - 1)], color=GRAY)
badge(481, y2 - 10, 1)
arrow([(845, yo), (845, yo + 13), (660, yo + 13), (660, y2 - 1)], color=GRAY)
badge(671, y2 - 10, 3)
# sim flow
arrow([(190, y2 + 48), (202, y2 + 48)], color=ORNG, head="ao")
arrow([(752, y2 + 58), (764, y2 + 58)], color=ORNG, head="ao")
# real metrics -> compare (bypass on the right edge)
arrow([(948, y0 + 80), (978, y0 + 80), (978, y2 + 30), (950, y2 + 30)], color=BLUE, head="ab")
text(988, 220, "real metrics", size=10.5, color=BLUE, anchor="middle", italic=True, rot=90)
# ===================================================================== (b)
yB = 430
text(4, yB + 4, "(b)", size=14, weight=700)
text(32, yB + 4, "One lockstep engine step, vanilla N=1 (B=2, γ=4): drafting is a tail after the forward", size=12.5, weight=650)
x0, ppm = 72, 7.9  # px per ms
rows = [("Real", BLUE, [("forward", real_fwd, "f"), ("", real_net / 2, "n"), ("draft (slowest)", real_draft, "d"), ("", real_net / 2, "n")], "p50, steps.jsonl"),
        ("Sim", ORNG, [("provider(batch)", sim_fwd, "f"), ("", RTT / 2, "n"), ("draft(γ_max, 2 seqs)", sim_draft, "d"), ("", RTT / 2, "n")], "pooled fit")]
for r, (name, col, segs, note) in enumerate(rows):
    y = yB + 24 + r * 54
    text(x0 - 10, y + 17, name, size=12.5, weight=700, color=col, anchor="end")
    x, t = x0, 0.0
    for lab, ms, kind in segs:
        w = ms * ppm
        if kind == "f":
            rect(x, y, w - 2, 26, fill=col, r=3)
            text(x + w / 2, y + 17.5, f"{lab}  {ms:.1f}", size=11, color="#ffffff", weight=600, anchor="middle")
        elif kind == "d":
            rect(x, y, w - 2, 26, fill=BLUE_L if col == BLUE else ORNG_L, r=3)
            text(x + w / 2, y + 17.5, f"{lab}  {ms:.1f}", size=11, color=INK, weight=600, anchor="middle")
        else:
            rect(x, y, w - 2, 26, fill="url(#net)", stroke="#cfcfca", sw=0.8, r=3)
        x += w
        t += ms
    text(x + 6, y + 13, f"Σ {t:.1f} ms", size=11.5, weight=650, color=INK)
    text(x + 6, y + 26, note, size=10.5, color=INK3)
    if r == 1:
        xe = x0 + sim_fwd * ppm
        out.append(f'<path d="M{xe - 1},{y - 3} l-4,-6 h8 z" fill="{INK}"/>')
        text(xe - 2, y + 40, "forward ends: jobs complete; the target is held through the tail", size=10.5, color=INK2, italic=True)
        out.append(f'<line x1="{xe - 1}" y1="{y + 28}" x2="{xe - 1}" y2="{y + 31}" stroke="{INK2}"/>')
# legend + axis
ya = yB + 140
out.append(f'<line x1="{x0}" y1="{ya}" x2="{x0 + 65 * ppm}" y2="{ya}" stroke="{INK3}" stroke-width="0.8"/>')
for ms in range(0, 70, 10):
    xx = x0 + ms * ppm
    out.append(f'<line x1="{xx}" y1="{ya}" x2="{xx}" y2="{ya + 4}" stroke="{INK3}" stroke-width="0.8"/>')
    text(xx, ya + 15, f"{ms}", size=10.5, color=INK3, anchor="middle")
text(x0 + 65 * ppm + 6, ya + 4, "ms", size=10.5, color=INK3)
rect(x0, ya + 26, 18, 11, fill="url(#net)", stroke="#cfcfca", sw=0.8, r=2)
text(x0 + 24, ya + 35, "network, RTT/2 each way (real: injected 10 ms + transport)", size=10.5, color=INK2)
text(x0, ya + 54, "propose() fans out to all drafters at once; the step waits for the slowest one.", size=10.5, color=INK2)

# ===================================================================== (c)
xC = 636
text(xC - 14, yB + 4, "(c)", size=14, weight=700)
text(xC + 14, yB + 4, "Sim error vs. real, MAPE % (3 methods × 3 metrics)", size=12.5, weight=650)
data = mape_rows()
cw, ch, lx = 38, 22, xC + 14
cx0 = lx + 134
hy = yB + 26
text(lx, hy, "latency", size=10.5, color=INK3, weight=600)
text(lx + 74, hy, "accept.", size=10.5, color=INK3, weight=600)
for j, lab in enumerate(["N=1", "N=2", "N=4", "N=7", "all"]):
    text(cx0 + j * (cw + 2) + cw / 2 + (8 if j == 4 else 0), hy, lab, size=10.5, color=INK3, weight=600, anchor="middle")
# one-hue sequential ramp (orange, light -> dark), 0..55 %
ramp = ["#fdf0e9", "#fbd9c6", "#f7bb9c", "#f19a71", "#e8794a", "#d95926", "#b4441a", "#8d3311"]


def cell_color(v):
    i = min(len(ramp) - 1, int(v / 55 * (len(ramp) - 1) + 0.5))
    return ramp[i], ("#ffffff" if i >= 5 else INK)


best = min(range(len(data)), key=lambda i: data[i][3])
for i, (lat, acc, byn, allv) in enumerate(data):
    y = hy + 8 + i * (ch + 2)
    w8 = 700 if i == best else 400
    text(lx, y + 15, lat, size=11, color=INK, weight=w8)
    text(lx + 74, y + 15, acc, size=11, color=INK, weight=w8)
    for j, v in enumerate(byn + [allv]):
        xx = cx0 + j * (cw + 2) + (8 if j == 4 else 0)
        fc, tc = cell_color(v)
        rect(xx, y, cw, ch, fill=fc, r=2)
        text(xx + cw / 2, y + 15, f"{v:.1f}", size=11, color=tc, anchor="middle", weight=650 if j == 4 else 400)
ny = hy + 8 + len(data) * (ch + 2) + 12
text(lx, ny, "T1 (N≤2): each drafter owns a GPU. N≥4 shares GPUs (T2);", size=10.5, color=INK2)
text(lx, ny + 13, "N=7 saturates (vLLM admission queue, not modeled).", size=10.5, color=INK2)
text(lx, ny + 26, "OH = per-step overhead fitted on N=1 runs only.", size=10.5, color=INK2)

out.append("</svg>")
svg = HERE / "e2e_vs_sim.svg"
svg.write_text("\n".join(out))
if shutil.which("rsvg-convert"):
    subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(HERE / "e2e_vs_sim.pdf"), str(svg)], check=True)
    subprocess.run(["rsvg-convert", "-z", "2.5", "-o", str(HERE / "e2e_vs_sim.png"), str(svg)], check=True)
print(svg)
