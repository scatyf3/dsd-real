"""System figure, client-side scenario (dsdSim's): every request is born at
a drafter's client device and owned by that drafter; its prompt goes up to the
target and its tokens come back. Components, one engine step's data path, and
the dsdSim component each piece stands in for.

    python docs/figures/make_dsd_arch.py      # -> dsd_arch.{svg,pdf,png}

Stdlib only; PDF/PNG need rsvg-convert and a static font (TeX Gyre Heros).
"""
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
W, H = 1000, 332
FONT = "TeX Gyre Heros, Helvetica, Arial, sans-serif"
MONO = "Adwaita Mono, Menlo, monospace"
INK, INK2, INK3 = "#1f1f1e", "#4f4f4a", "#8a8a84"
BLUE, BLUE_T, BLUE_L = "#2a78d6", "#edf4fd", "#9ec5f4"
TEAL, TEAL_T, TEAL_L = "#14946a", "#e8f6f0", "#8fd3b9"
GRAY, GRAY_T, GRAY_L = "#6b6b66", "#f4f4f2", "#d6d6d1"
ORNG, ORNG_T = "#d95926", "#fdf0e9"

out: list[str] = []


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=12, color=INK, weight=400, anchor="start", font=FONT, italic=False, rot=None):
    t = f' transform="rotate({rot} {x} {y})"' if rot is not None else ""
    st = ' font-style="italic"' if italic else ""
    out.append(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" fill="{color}" '
               f'font-weight="{weight}" text-anchor="{anchor}"{st}{t}>{esc(s)}</text>')


def rect(x, y, w, h, fill="none", stroke="none", sw=1, r=6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" '
               f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def arrow(pts, color=INK2, sw=1.4, dash=None, mk="k", both=False):
    p = " ".join(f"{x},{y}" for x, y in pts)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    s = f' marker-start="url(#m{mk})"' if both else ""
    out.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}"{d}'
               f' marker-end="url(#m{mk})"{s} stroke-linejoin="round"/>')


def step(x, y, n):
    """Numbered step of one engine iteration."""
    out.append(f'<circle cx="{x}" cy="{y}" r="8" fill="{INK}"/>')
    text(x, y + 4, str(n), size=10.5, color="#ffffff", weight=700, anchor="middle")


def simtag(x, y, s, anchor="start"):
    """The dsdSim component this real component stands in for."""
    w = 7.0 * len(s) + 14
    x0 = x - w if anchor == "end" else (x - w / 2 if anchor == "middle" else x)
    rect(x0, y, w, 17, fill=ORNG_T, stroke=ORNG, sw=1, r=8.5, dash="3 2")
    text(x0 + w / 2, y + 12.5, s, size=10.5, color=ORNG, weight=600, anchor="middle")


def gpu(x, y, s, color):
    rect(x, y, 7.2 * len(s) + 10, 16, fill=color, r=3)
    text(x + 5, y + 12, s, size=10.5, color="#ffffff", weight=700)


def pill(x, y, w, s, sub, color, fill):
    rect(x, y, w, 36, fill=fill, stroke=color, sw=1.1, r=5)
    text(x + w / 2, y + 15, s, size=12, color=INK, weight=650, anchor="middle")
    text(x + w / 2, y + 29, sub, size=10.5, color=INK2, anchor="middle")


out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
mk = "".join(
    f'<marker id="m{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
    f'orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="{c}"/></marker>'
    for k, c in (("k", INK2), ("b", BLUE), ("t", TEAL), ("o", ORNG), ("g", GRAY)))
out.append(f"""<defs>{mk}
<pattern id="hatch" patternUnits="userSpaceOnUse" width="5" height="5" patternTransform="rotate(45)">
<rect width="5" height="5" fill="#ffffff"/><line x1="0" y1="0" x2="0" y2="5" stroke="{GRAY_L}" stroke-width="2"/></pattern>
</defs>""")
rect(0, 0, W, H, fill="#ffffff", r=0)

# ============================================================ client devices (left)
for k in (2, 1):
    rect(20 + 5 * k, 24 - 5 * k, 272, 294, fill="#ffffff", stroke=TEAL_L, sw=1, r=8)
rect(20, 24, 272, 294, fill=TEAL_T, stroke=TEAL, sw=1.3, r=8)
out.append(f'<text x="32" y="47" font-family="{FONT}" font-size="15" font-weight="700" fill="{INK}">'
           'Client device d<tspan dy="4" font-size="11">i</tspan><tspan dx="6" dy="-4">× N</tspan></text>')
gpu(228, 34, "GPU1–2", TEAL)
rect(32, 62, 248, 48, fill="#ffffff", stroke=TEAL_L, sw=1, r=5)
text(42, 81, "Client", size=13, weight=700)
text(42, 99, "owns its requests", size=11, color=INK2)
rect(32, 132, 248, 174, fill="#ffffff", stroke=TEAL_L, sw=1, r=5)
text(42, 152, "Draft server", size=13, weight=700)
text(270, 152, "ZMQ / HTTP", size=11, color=INK2, anchor="end")


def engine(y, tag, title, line):
    rect(40, y, 232, 54, fill=TEAL_T, stroke=TEAL_L, sw=1, r=4)
    rect(46, y + 7, 15, 15, fill=TEAL, r=3)
    text(53.5, y + 18.5, tag, size=10.5, color="#ffffff", weight=700, anchor="middle")
    text(68, y + 19, title, size=12, weight=700)
    out.append(f'<circle cx="54" cy="{y + 37}" r="6" fill="{INK}"/>')
    text(54, y + 40.5, "4", size=8.5, color="#ffffff", weight=700, anchor="middle")
    text(66, y + 41, line, size=11, color=INK2)


engine(164, "A", "1B drafter (vLLM)", "rollback: re-prefill confirmed prefix")
engine(224, "B", "EAGLE3 head (PyTorch)", "rollback: truncate KV, re-prefill")
step(50, 292, 5)
text(64, 296, "propose γ draft tokens", size=12, weight=650)

# ============================================================ network (middle)
# one link per client device: prompt/tokens and drafts share it, as in dsdSim
rect(316, 66, 80, 194, fill="url(#hatch)", stroke=GRAY_L, sw=1, r=4)
rect(320, 128, 72, 40, fill="#ffffff", r=3)
text(356, 145, "network", size=11.5, weight=700, anchor="middle")
text(356, 160, "RTT/2 each way", size=9.5, color=INK2, anchor="middle")

# ============================================================ target node
rect(410, 16, 576, 302, fill="none", stroke=GRAY, sw=1.2, r=10, dash="5 4")
text(422, 36, "Target node", size=12, color=GRAY, weight=700)
gpu(504, 24, "GPU0", BLUE)
tx, ty, tw, th = 596, 40, 380, 266
rect(tx, ty, tw, th, fill=BLUE_T, stroke=BLUE, sw=1.3, r=8)
text(tx + 12, ty + 22, "Target vLLM Server", size=15, weight=700)
text(tx + tw - 12, ty + 22, "Llama-3.1-8B", size=11.5, color=INK2, anchor="end")
rect(tx + 10, ty + 44, tw - 20, 36, fill="#ffffff", stroke=BLUE_L, sw=1, r=4)
text(tx + tw / 2, ty + 67, "Scheduler · continuous batching", size=12, weight=650, anchor="middle")
dx, dy, dw, dh = tx + 10, ty + 92, 164, 164
rect(dx, dy, dw, dh, fill="#ffffff", stroke=BLUE, sw=1.1, r=6)
text(dx + dw / 2, dy + 20, "Draft Runner", size=13, weight=700, anchor="middle")
rect(dx + 10, dy + 30, dw - 20, 70, fill=BLUE, r=5)
text(dx + dw / 2, dy + 52, "Plugin", size=13, weight=700, color="#ffffff", anchor="middle")
text(dx + dw / 2, dy + 70, "RemoteDraftProposer", size=10.5, color="#ffffff", anchor="middle", font=MONO)
text(dx + dw / 2, dy + 88, "custom_class", size=10.5, color="#dbe9fb", anchor="middle", font=MONO)
step(dx + 18, dy + 128, 3)
text(dx + 30, dy + 132, "propose() each step", size=11, color=INK2, italic=True)
rx, rw = tx + 204, 166
rect(rx, dy, rw, dh, fill="#ffffff", stroke=BLUE, sw=1.1, r=6)
text(rx + rw / 2, dy + 20, "Target Runner", size=13, weight=700, anchor="middle")
rect(rx + 10, dy + 30, rw - 20, 40, fill=BLUE_T, stroke=BLUE_L, sw=1, r=4)
text(rx + rw / 2, dy + 55, "verify + prefill", size=12, weight=650, anchor="middle")
text(rx + 10, dy + 92, "rejection sampler", size=11, color=INK2)
text(rx + 10, dy + 110, "hidden states (for B)", size=11, color=INK2)
step(rx + 18, dy + 128, 2)
text(rx + 30, dy + 132, "one forward per step", size=11, color=INK2, italic=True)
gx = (dx + dw + rx) / 2
arrow([(rx - 2, dy + 52), (dx + dw + 2, dy + 52)], color=BLUE, mk="b")
arrow([(dx + dw + 2, dy + 80), (rx - 2, dy + 80)], color=BLUE, mk="b")
arrow([(rx - 2, dy + 106), (dx + dw + 2, dy + 106)], color=BLUE, mk="b", dash="3 2")
for yy, lab in ((dy + 46, "tokens"), (dy + 96, "drafts"), (dy + 121, "hidden")):
    text(gx, yy, lab, size=9, color=BLUE, anchor="middle")

# ---- draft client
cx, cy, cw, chh = 422, 132, 162, 174
rect(cx, cy, cw, chh, fill="#ffffff", stroke=GRAY, sw=1.3, r=8)
text(cx + cw / 2, cy + 22, "Draft Client", size=14, weight=700, anchor="middle")
pill(cx + 10, cy + 34, cw - 20, "IPC", "in-process", GRAY, GRAY_T)
pill(cx + 10, cy + 78, cw - 20, "ZMQ", "msgpack", GRAY, GRAY_T)
pill(cx + 10, cy + 122, cw - 20, "HTTP", "JSON", GRAY, GRAY_T)

# ============================================================ data paths
ya = 92
arrow([(280, ya), (314, ya)], color=TEAL, mk="t")
arrow([(396, ya), (tx + 8, ya)], color=TEAL, mk="t")
step(426, ya - 13, 1)
text(438, ya - 9, "prompt (once per request)", size=11, color=INK2, italic=True)
# the verify result reaching the device is also the user's output
arrow([(160, 131), (160, 112)], color=TEAL, mk="t")
text(168, 125, "output tokens", size=10.5, color=INK2, italic=True)
yp, yq = dy + 48, dy + 84
arrow([(dx - 1, yp), (cx + cw + 1, yp)], color=INK2)
arrow([(cx + cw + 1, yq), (dx - 1, yq)], color=INK2)
ys_, yr_ = 200, 240
arrow([(cx, ys_), (398, ys_)], color=INK2)
arrow([(316, ys_), (282, ys_)], color=INK2)
arrow([(280, yr_), (314, yr_)], color=INK2)
arrow([(396, yr_), (cx - 1, yr_)], color=INK2)


def halo(x, y, s, size=10.5):
    out.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{INK2}" font-style="italic" '
               f'text-anchor="middle" stroke="#ffffff" stroke-width="4" paint-order="stroke" '
               f'stroke-linejoin="round">{s}</text>')


halo(356, ys_ - 7, "accepted + bonus")
halo(356, yr_ + 16, "γ drafts")
step(298, yr_ + 14, 6)

out.append("</svg>")
svg = HERE / "dsd_arch.svg"
svg.write_text("\n".join(out))
if shutil.which("rsvg-convert"):
    subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(HERE / "dsd_arch.pdf"), str(svg)], check=True)
    subprocess.run(["rsvg-convert", "-z", "2.5", "-o", str(HERE / "dsd_arch.png"), str(svg)], check=True)
print(svg)
