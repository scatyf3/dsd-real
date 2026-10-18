"""Minimal workflow figure: client devices (client + draft model) and one
target server; requests go up and tokens come back, and every round the
target sends the confirmed tokens to the request's drafter and verifies the
drafts it returns. The detailed version is make_dsd_arch.py.

    python docs/figures/make_dsd_workflow.py      # -> dsd_workflow.{svg,pdf,png}

Stdlib only; PDF/PNG need rsvg-convert and a static font (TeX Gyre Heros).
"""
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
W, H = 1000, 300
FONT = "TeX Gyre Heros, Helvetica, Arial, sans-serif"
INK, INK2 = "#1f1f1e", "#4f4f4a"
BLUE, BLUE_T, BLUE_L = "#2a78d6", "#edf4fd", "#9ec5f4"
TEAL, TEAL_T, TEAL_L = "#14946a", "#e8f6f0", "#8fd3b9"
GRAY_L = "#d6d6d1"

out: list[str] = []


def text(x, y, s, size=14, color=INK, weight=400, anchor="start", italic=False):
    st = ' font-style="italic"' if italic else ""
    out.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{color}" '
               f'font-weight="{weight}" text-anchor="{anchor}"{st}>{s}</text>')


def rect(x, y, w, h, fill="none", stroke="none", sw=1.3, r=8, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" '
               f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def arrow(x1, y, x2, color, label, above=True):
    out.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="1.8" '
               f'marker-end="url(#m{color[1:]})"/>')
    # white halo so the label reads over the network box's dashed edge
    out.append(f'<text x="{(x1 + x2) / 2}" y="{y - 9 if above else y + 20}" font-family="{FONT}" font-size="13.5" '
               f'fill="{INK2}" font-style="italic" text-anchor="middle" stroke="#ffffff" stroke-width="5" '
               f'paint-order="stroke" stroke-linejoin="round">{label}</text>')


def comp(x, y, w, h, title, sub, color, fill):
    rect(x, y, w, h, fill=fill, stroke=color)
    text(x + w / 2, y + h / 2 - 2, title, size=16, weight=700, anchor="middle")
    text(x + w / 2, y + h / 2 + 17, sub, size=12.5, color=INK2, anchor="middle")


out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
out.append("<defs>" + "".join(
    f'<marker id="m{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" '
    f'orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="{c}"/></marker>'
    for c in (TEAL, BLUE, INK2)) + "</defs>")
rect(0, 0, W, H, fill="#ffffff", r=0)

# client devices x N (stacked)
for k in (2, 1):
    rect(30 + 6 * k, 30 - 6 * k, 290, 240, fill="#ffffff", stroke=TEAL_L, sw=1)
rect(30, 30, 290, 240, fill=TEAL_T, stroke=TEAL)
text(46, 58, "Client device × N", size=17, weight=700)
comp(50, 76, 250, 66, "Client", "issues requests, receives tokens", TEAL, "#ffffff")
comp(50, 180, 250, 70, "Draft model", "proposes γ tokens per round", TEAL, "#ffffff")

# network
rect(452, 30, 96, 240, fill="none", stroke=GRAY_L, sw=1.2, r=6, dash="4 4")
text(500, 22, "network", size=13, color=INK2, anchor="middle")

# target server
rect(680, 30, 290, 240, fill=BLUE_T, stroke=BLUE)
text(696, 58, "Target server", size=17, weight=700)
comp(700, 76, 250, 66, "Serving engine", "batches all requests", BLUE, "#ffffff")
comp(700, 180, 250, 70, "Target model", "verifies all drafts in one pass", BLUE, "#ffffff")

# request path
arrow(300, 100, 698, TEAL, "prompt")
arrow(700, 124, 302, TEAL, "output tokens", above=False)
# speculation round
arrow(700, 204, 302, BLUE, "confirmed tokens")
arrow(300, 228, 698, BLUE, "γ draft tokens", above=False)
# one round, repeated
out.append(f'<path d="M 825 142 L 825 178" stroke="{INK2}" stroke-width="1.4" marker-end="url(#m{INK2[1:]})"/>')
text(833, 165, "admit", size=12, color=INK2, italic=True)
text(500, 292, "repeated every round: draft → send → verify → return accepted tokens",
     size=13, color=INK2, anchor="middle", italic=True)

out.append("</svg>")
svg = HERE / "dsd_workflow.svg"
svg.write_text("\n".join(out))
if shutil.which("rsvg-convert"):
    subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(HERE / "dsd_workflow.pdf"), str(svg)], check=True)
    subprocess.run(["rsvg-convert", "-z", "2.5", "-o", str(HERE / "dsd_workflow.png"), str(svg)], check=True)
print(svg)
