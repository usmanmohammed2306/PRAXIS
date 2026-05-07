#!/usr/bin/env python3
"""
Generate PRAXIS conference poster as a PDF.

Usage:  python3 scripts/generate_poster_pdf.py
Output: docs/PRAXIS_POSTER.pdf

Poster size: 48" × 36" landscape  (3456 × 2592 pt at 72 dpi).
Color scheme matches ASU Poster Template (#902346 maroon, #E8A020 gold).
"""
from __future__ import annotations
import io
import math
import urllib.request
from pathlib import Path

from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.utils import ImageReader

# ─────────────────────────────────────────────────────────────────────────────
# PAGE DIMENSIONS  (48" × 36" landscape)
# ─────────────────────────────────────────────────────────────────────────────
W = int(48 * inch)   # 3456 pt
H = int(36 * inch)   # 2592 pt

MARGIN  = 24          # outer page margin  (pt)
GAP     = 20          # gap between columns
HDR_H   = 235         # header height
BODY_T  = H - HDR_H - 14   # y of top of body
BODY_B  = MARGIN            # y of bottom of body
BODY_H  = BODY_T - BODY_B  # available body height

# three columns
C1_W = 1030;  C1_X = MARGIN
C2_W = 1090;  C2_X = C1_X + C1_W + GAP
C3_W = W - 2*MARGIN - 2*GAP - C1_W - C2_W;  C3_X = C2_X + C2_W + GAP

# section header bar
SH    = 46    # section header bar height
SH_FS = 18    # section header font size
SP    = 10    # section inner padding

# ─────────────────────────────────────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────────────────────────────────────
MAROON   = HexColor('#902346')
MAROON2  = HexColor('#801F3E')
GOLD     = HexColor('#E8A020')
TEAL     = HexColor('#2A9D8F')
LTEAL    = HexColor('#E0F5F0')
LGOLD    = HexColor('#FFF7E0')
LMAROON  = HexColor('#FAF0F3')
LGRAY    = HexColor('#F5F5F5')
DGRAY    = HexColor('#CCCCCC')
DARK     = HexColor('#2D2D2D')
GRAY     = HexColor('#777777')
RED      = HexColor('#C62828')
GREEN    = HexColor('#2E7D32')
LRED     = HexColor('#FFEBEE')
LGREEN   = HexColor('#E8F5E9')


# ─────────────────────────────────────────────────────────────────────────────
# PRIMITIVE DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def frect(c, x, y, w, h, fill, stroke=None, sw=1):
    c.saveState()
    c.setFillColor(fill)
    if stroke:
        c.setStrokeColor(stroke)
        c.setLineWidth(sw)
        c.rect(x, y, w, h, fill=1, stroke=1)
    else:
        c.rect(x, y, w, h, fill=1, stroke=0)
    c.restoreState()


def rrect(c, x, y, w, h, r, fill, stroke=None, sw=1):
    c.saveState()
    c.setFillColor(fill)
    if stroke:
        c.setStrokeColor(stroke)
        c.setLineWidth(sw)
        c.roundRect(x, y, w, h, r, fill=1, stroke=1)
    else:
        c.roundRect(x, y, w, h, r, fill=1, stroke=0)
    c.restoreState()


def line(c, x1, y1, x2, y2, color=MAROON, sw=1.5):
    c.saveState()
    c.setStrokeColor(color)
    c.setLineWidth(sw)
    c.line(x1, y1, x2, y2)
    c.restoreState()


def dline(c, x1, y1, x2, y2, color=DGRAY, sw=1, dash=(4, 3)):
    c.saveState()
    c.setStrokeColor(color)
    c.setLineWidth(sw)
    c.setDash(*dash)
    c.line(x1, y1, x2, y2)
    c.restoreState()


def arrow_down(c, cx, y_top, y_bot, color=MAROON, sw=1.5, hs=10):
    """Vertical downward arrow."""
    c.saveState()
    c.setFillColor(color)
    c.setStrokeColor(color)
    c.setLineWidth(sw)
    c.line(cx, y_top, cx, y_bot + hs)
    # arrowhead
    p = c.beginPath()
    p.moveTo(cx, y_bot)
    p.lineTo(cx - hs*0.45, y_bot + hs)
    p.lineTo(cx + hs*0.45, y_bot + hs)
    p.close()
    c.drawPath(p, fill=1, stroke=0)
    c.restoreState()


def arrow_right(c, x_left, cx_y, x_right, color=MAROON, sw=1.5, hs=10):
    """Horizontal rightward arrow."""
    c.saveState()
    c.setFillColor(color)
    c.setStrokeColor(color)
    c.setLineWidth(sw)
    c.line(x_left, cx_y, x_right - hs, cx_y)
    p = c.beginPath()
    p.moveTo(x_right, cx_y)
    p.lineTo(x_right - hs, cx_y + hs*0.45)
    p.lineTo(x_right - hs, cx_y - hs*0.45)
    p.close()
    c.drawPath(p, fill=1, stroke=0)
    c.restoreState()


def arrow_left(c, x_right, cx_y, x_left, color=MAROON, sw=1.5, hs=10):
    """Horizontal leftward arrow."""
    c.saveState()
    c.setFillColor(color)
    c.setStrokeColor(color)
    c.setLineWidth(sw)
    c.line(x_right, cx_y, x_left + hs, cx_y)
    p = c.beginPath()
    p.moveTo(x_left, cx_y)
    p.lineTo(x_left + hs, cx_y + hs*0.45)
    p.lineTo(x_left + hs, cx_y - hs*0.45)
    p.close()
    c.drawPath(p, fill=1, stroke=0)
    c.restoreState()


def arrow_generic(c, x1, y1, x2, y2, color=MAROON, sw=1.5, hs=10):
    """Generic arrow from (x1,y1) to (x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L < 1:
        return
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    c.saveState()
    c.setFillColor(color)
    c.setStrokeColor(color)
    c.setLineWidth(sw)
    c.line(x1, y1, x2 - ux*hs*0.7, y2 - uy*hs*0.7)
    p = c.beginPath()
    p.moveTo(x2, y2)
    p.lineTo(x2 - ux*hs + px*hs*0.42, y2 - uy*hs + py*hs*0.42)
    p.lineTo(x2 - ux*hs - px*hs*0.42, y2 - uy*hs - py*hs*0.42)
    p.close()
    c.drawPath(p, fill=1, stroke=0)
    c.restoreState()


def label_c(c, cx, y, text, fs, color=DARK, bold=False):
    """Center-aligned text."""
    font = "Helvetica-Bold" if bold else "Helvetica"
    c.saveState()
    c.setFont(font, fs)
    c.setFillColor(color)
    c.drawCentredString(cx, y, text)
    c.restoreState()


def label_l(c, x, y, text, fs, color=DARK, bold=False):
    """Left-aligned text."""
    font = "Helvetica-Bold" if bold else "Helvetica"
    c.saveState()
    c.setFont(font, fs)
    c.setFillColor(color)
    c.drawString(x, y, text)
    c.restoreState()


def wrapped(c, x, y_top, w, text, fs, color=DARK, bold=False, leading=None):
    """Simple word-wrap. Returns y below last line."""
    font = "Helvetica-Bold" if bold else "Helvetica"
    if leading is None:
        leading = fs * 1.35
    c.saveState()
    c.setFont(font, fs)
    c.setFillColor(color)
    words = text.split()
    lines = []
    cur = ""
    for word in words:
        test = (cur + " " + word).strip()
        if c.stringWidth(test, font, fs) <= w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    cy = y_top - fs
    for ln in lines:
        c.drawString(x, cy, ln)
        cy -= leading
    c.restoreState()
    return cy


def person_icon(c, cx, y_bottom, h, initial, color=MAROON, label_text="", sub_text="",
                fs_label=14, fs_sub=11):
    """Person icon (circle head + body rectangle). Returns y_top of icon."""
    head_r = h * 0.27
    body_h = h * 0.45
    body_w = h * 0.50
    # Body (draw first, behind head)
    c.saveState()
    c.setFillColor(color)
    c.roundRect(cx - body_w/2, y_bottom, body_w, body_h, 5, fill=1, stroke=0)
    # Head
    c.setFillColor(color)
    c.setStrokeColor(white)
    c.setLineWidth(2)
    c.circle(cx, y_bottom + body_h + head_r * 0.9, head_r, fill=1, stroke=1)
    # Initial
    c.setFont("Helvetica-Bold", int(head_r * 1.1))
    c.setFillColor(white)
    c.drawCentredString(cx, y_bottom + body_h + head_r * 0.9 - head_r * 0.38, initial)
    c.restoreState()
    icon_top = y_bottom + body_h + 2 * head_r * 1.0
    # Labels below
    cy_lbl = y_bottom - 4
    if label_text:
        label_c(c, cx, cy_lbl - fs_label, label_text, fs_label, color, bold=True)
    if sub_text:
        label_c(c, cx, cy_lbl - fs_label - fs_sub - 2, sub_text, fs_sub, GRAY)
    return icon_top


def flow_box(c, x, y_top, w, h, lines, fill=LGRAY, stroke=MAROON, sw=1.5, r=6):
    """Rounded box with multiple text lines (list of (fs, text, bold, color)).
    Returns (center_x, y_bottom_of_box, y_top_of_box)."""
    y_bot = y_top - h
    rrect(c, x, y_bot, w, h, r, fill, stroke, sw)
    cx = x + w / 2
    # vertical center the lines
    total_h = sum(fs * 1.35 for fs, _, _, _ in lines)
    cy = y_bot + h/2 + total_h/2
    for fs, text, bold, col in lines:
        cy -= fs
        label_c(c, cx, cy, text, fs, col, bold)
        cy -= fs * 0.35
    return (cx, y_bot, y_top)


def sec_hdr(c, x, y_top, w, title):
    """Draw section header bar. Returns y below bar."""
    frect(c, x, y_top - SH, w, SH, MAROON)
    label_l(c, x + 12, y_top - SH + 14, title, SH_FS, white, bold=True)
    return y_top - SH - SP


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

def draw_header(c):
    # Dark maroon base
    frect(c, 0, H - HDR_H, W, HDR_H, MAROON2)
    # Lighter maroon top half
    frect(c, 0, H - HDR_H//2, W, HDR_H//2, MAROON)
    # Gold accent strip
    frect(c, 228, H - HDR_H + 12, 6, HDR_H - 24, GOLD)

    # ASU Logo (download; fall back to text)
    logo_x, logo_y, logo_w, logo_h = 14, H - HDR_H + 16, 208, 195
    logo_loaded = False
    try:
        url = "https://ires.ubc.ca/files/2019/10/ASU-logo-white-background.png"
        with urllib.request.urlopen(url, timeout=8) as resp:
            img_data = ImageReader(io.BytesIO(resp.read()))
        c.drawImage(img_data, logo_x, logo_y, width=logo_w, height=logo_h,
                    preserveAspectRatio=True, anchor='sw', mask='auto')
        logo_loaded = True
    except Exception:
        # Fallback: styled text logo
        rrect(c, logo_x, logo_y + 30, logo_w, 135, 6, HexColor('#6B0020'))
        label_c(c, logo_x + logo_w/2, logo_y + 100, "ASU", 52, white, bold=True)
        label_c(c, logo_x + logo_w/2, logo_y + 58,  "Arizona State", 14, GOLD)
        label_c(c, logo_x + logo_w/2, logo_y + 42,  "University",    14, GOLD)

    # PRAXIS title
    c.saveState()
    c.setFont("Helvetica-Bold", 78)
    c.setFillColor(white)
    c.drawString(242, H - 66, "PRAXIS")
    c.restoreState()

    # Full name
    label_l(c, 242, H - 103, "Procedural Retrieval-Augmented eXperience-Informed System",
            26, white)
    # Tagline
    label_l(c, 242, H - 134, "Continual Procedural Memory for Tool-Calling LLM Agents",
            21, GOLD, bold=False)
    # Authors
    label_l(c, 242, H - 163, "Usman Mohammed  ·  ASU School of Computing & Augmented Intelligence",
            17, white)
    # Badges
    badges = [
        ("tau-bench retail", GOLD, 1.5),
        ("tau-bench airline", GOLD, 1.5),
        ("BFCL V4", GOLD, 1.5),
        ("Qwen2.5-14B · vLLM · 1×A100", TEAL, 1.5),
        ("Zero Fine-tuning", white, 1),
    ]
    bx = 242
    by = H - 202
    for btxt, bstroke, bsw in badges:
        bw = len(btxt) * 9 + 18
        rrect(c, bx, by - 4, bw, 28, 5, MAROON2, bstroke, bsw)
        label_c(c, bx + bw/2, by + 3, btxt, 13, white, bold=True)
        bx += bw + 8


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN 1  —  The Analogy  ·  Key Numbers  ·  Future Work
# ─────────────────────────────────────────────────────────────────────────────

def draw_c1(c):
    x, w = C1_X, C1_W
    y = BODY_T        # current y, decreases as we draw downward

    # ── THE CORE IDEA ────────────────────────────────────────────────────────
    y = sec_hdr(c, x, y, w, "The Core Idea — The Analogy")

    # Quote box
    q_h = 52
    rrect(c, x, y - q_h, w, q_h, 7, LGOLD, GOLD, 2)
    label_c(c, x + w/2, y - q_h + 17,
            '"An agent that learns from every interaction — not just its own."',
            13, DARK, bold=False)
    y -= q_h + 14

    # ── 4 worker icons ────────────────────────────────────────────────────────
    icon_h   = 155       # height of icon shape
    icon_lbl = 32        # space for labels below icon
    icon_tot = icon_h + icon_lbl
    slot_w   = w // 4

    data = [
        ("A", "Act",        "baseline", HexColor('#666666')),
        ("R", "ReAct",      "baseline", HexColor('#666666')),
        ("V", "Vanilla TC", "baseline", HexColor('#666666')),
        ("P", "PRAXIS",     "learns &\nobserves",  MAROON),
    ]
    worker_cx = []
    for i, (init, name, sub, col) in enumerate(data):
        cx = x + i * slot_w + slot_w // 2
        worker_cx.append(cx)
        person_icon(c, cx, y - icon_tot, icon_h, init, col,
                    label_text=name, sub_text=sub, fs_label=14, fs_sub=11)

    # Dashed separator between baseline workers and PRAXIS
    sep_x = x + 3 * slot_w
    dline(c, sep_x, y - 4, sep_x, y - icon_tot - 4, DGRAY, 1.5, (6, 4))

    y -= icon_tot + 8

    # ── Flow diagram (baselines → traj → distiller → memory) ────────────────
    flow_w  = 3 * slot_w - 10   # spans the 3 baseline workers
    flow_cx = x + flow_w / 2 + 5

    # trajectories.jsonl
    box_h = 52
    traj_bot = y - box_h
    flow_box(c, x + 5, y, flow_w, box_h,
             [(13, "trajectories.jsonl", True,  DARK),
              (10, "outputs per benchmark run", False, GRAY)],
             fill=LGRAY, stroke=MAROON, sw=1.5)

    # Arrows from 3 workers down to traj box
    for i in range(3):
        wx = x + i * slot_w + slot_w // 2
        arrow_generic(c, wx, y + 8, flow_cx - flow_w/2*0.6 + i*flow_w*0.6/2, y,
                      HexColor('#999999'), 1.2, 8)
    y -= box_h + 4

    arrow_down(c, flow_cx, y, y - 14, MAROON, 1.5, 9)
    y -= 16

    # DISTILLER
    box_h = 62
    flow_box(c, x + 5, y, flow_w, box_h,
             [(14, "DISTILLER", True,  MAROON),
              (11, "strips PII · extracts procedure", False, DARK),
              (10, "tool order · failure patterns · anti-patterns", False, GRAY)],
             fill=LMAROON, stroke=MAROON, sw=1.5)
    dist_bot = y - box_h
    y -= box_h + 4

    arrow_down(c, flow_cx, y, y - 14, MAROON, 1.5, 9)
    y -= 16

    # MEMORY BANK
    box_h = 65
    mem_top = y
    mem_bot = y - box_h
    mem_mid_y = (mem_top + mem_bot) / 2
    flow_box(c, x + 5, y, flow_w, box_h,
             [(14, "MEMORY BANK", True,  TEAL),
              (11, "retail · airline · bfcl  ·  max 4,096 cards", False, DARK),
              (10, "grows every run · deduplicated · quality-scored", False, GRAY)],
             fill=LTEAL, stroke=TEAL, sw=2)
    y -= box_h

    # PRAXIS box (right of memory bank, same row)
    p_cx = x + 3 * slot_w + slot_w // 2
    p_w  = slot_w - 20
    p_h  = box_h + 14
    p_x  = p_cx - p_w // 2
    p_bot = mem_mid_y - p_h / 2
    rrect(c, p_x, p_bot, p_w, p_h, 7, LMAROON, MAROON, 2)
    label_c(c, p_cx, p_bot + p_h * 0.68, "PRAXIS", 14, MAROON, bold=True)
    label_c(c, p_cx, p_bot + p_h * 0.47, "AGENT",  12, DARK,   bold=True)
    label_c(c, p_cx, p_bot + p_h * 0.26, "top-5 cards", 10, GRAY)
    label_c(c, p_cx, p_bot + p_h * 0.12, "every 2 steps", 10, GRAY)

    # Arrow: memory → PRAXIS
    arrow_right(c, x + 5 + flow_w, mem_mid_y, p_x, TEAL, 2, 9)
    # Arrow: PRAXIS → memory (feedback, above)
    arrow_left(c, p_x, p_bot + p_h + 10, x + 5 + flow_w - 10,
               GOLD, 1.5, 8)

    y -= 14
    y -= 4

    # Caption
    label_l(c, x, y - 16,
            "After each run, new cards are distilled back — every run starts smarter.",
            11, GRAY, bold=False)
    y -= 26

    # ── KEY NUMBERS ──────────────────────────────────────────────────────────
    y -= 12
    y = sec_hdr(c, x, y, w, "Key Numbers")

    nums = [
        ("0",      "model fine-tuning steps"),
        ("4",      "agents compared"),
        ("5",      "cards retrieved per query"),
        ("2",      "steps between memory refreshes"),
        ("2,400",  "max chars in TacticalPlaybook"),
        ("4,096",  "max memory cards per domain"),
    ]
    box_w = (w - GAP) // 3 - 2
    box_h = 100
    for i, (num, lbl) in enumerate(nums):
        col_i = i % 3
        row_i = i // 3
        bx = x + col_i * (box_w + 9)
        by = y - row_i * (box_h + 8) - box_h
        rrect(c, bx, by, box_w, box_h, 7, LGRAY, MAROON, 1)
        label_c(c, bx + box_w/2, by + box_h * 0.54, num,  32, MAROON, bold=True)
        # wrapped label
        c.saveState()
        c.setFont("Helvetica", 11)
        c.setFillColor(DARK)
        lw = c.stringWidth(lbl, "Helvetica", 11)
        if lw <= box_w - 12:
            c.drawCentredString(bx + box_w/2, by + 16, lbl)
        else:
            words = lbl.split()
            mid = len(words) // 2
            l1 = " ".join(words[:mid])
            l2 = " ".join(words[mid:])
            c.drawCentredString(bx + box_w/2, by + 24, l1)
            c.drawCentredString(bx + box_w/2, by + 11, l2)
        c.restoreState()
    y -= 2 * (box_h + 8) + 10

    # ── FUTURE WORK ──────────────────────────────────────────────────────────
    y -= 10
    y = sec_hdr(c, x, y, w, "Future Work")

    fw_items = [
        ("1", "Cross-Domain Transfer",
         "Can retail lessons about 'verify before mutating' help an airline agent? "
         "Memory is currently siloed per domain."),
        ("2", "Active Memory Querying",
         "Let PRAXIS explicitly request past failures: 'retrieve cancel_reservation failures' "
         "— agent-initiated recall at any point."),
        ("3", "Mid-Task Online Distillation",
         "Distill lessons within a trajectory. Hit failure at step 5, "
         "learn and adjust by step 8 without waiting for run to complete."),
        ("4", "Multi-Agent Shared Memory",
         "Multiple PRAXIS instances share a federated memory bank "
         "while keeping private task data completely local."),
    ]
    item_h = 100
    for num, title, body in fw_items:
        # Number badge
        frect(c, x, y - item_h, 40, item_h, MAROON)
        label_c(c, x + 20, y - item_h/2 - 8, num, 20, white, bold=True)
        # Content box
        rrect(c, x + 44, y - item_h, w - 44, item_h, 5, LGRAY, DGRAY, 1)
        label_l(c, x + 54, y - 18, title, 14, MAROON, bold=True)
        wrapped(c, x + 54, y - 34, w - 64, body, 11, DARK)
        y -= item_h + 7


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN 2  —  System Architecture  ·  Mutation Guard  ·  Novel Contributions
# ─────────────────────────────────────────────────────────────────────────────

def draw_c2(c):
    x, w = C2_X, C2_W
    y = BODY_T
    cx_col = x + w / 2   # horizontal center of this column

    # ── SYSTEM ARCHITECTURE ──────────────────────────────────────────────────
    y = sec_hdr(c, x, y, w, "System Architecture")

    # OFFLINE sub-header
    sub_s_h = 30
    frect(c, x, y - sub_s_h, w, sub_s_h, MAROON2)
    label_c(c, cx_col, y - sub_s_h + 8, "OFFLINE PIPELINE  ·  After Each Run", 13, white)
    y -= sub_s_h + 8

    # trajectories.jsonl
    bh = 50
    bw = w - 60
    bx = x + 30
    _, traj_bot, traj_top = flow_box(c, bx, y, bw, bh,
                                     [(13, "trajectories.jsonl",  True,  DARK),
                                      (10, "one JSONL per benchmark run", False, GRAY)],
                                     fill=LGRAY, stroke=MAROON, sw=1.5)
    traj_cx = bx + bw / 2
    y -= bh + 4

    arrow_down(c, traj_cx, y, y - 14, MAROON, 1.5, 9)
    y -= 16

    # DISTILLER
    bh = 76
    _, dist_bot, _ = flow_box(c, bx, y, bw, bh,
                               [(14, "DISTILLER", True, MAROON),
                                (11, "strips PII (IDs, emails, prices, dates)", False, DARK),
                                (11, "extracts tool order · failure patterns · anti-patterns", False, DARK),
                                (10, "scores confidence · leakage audit", False, GRAY)],
                               fill=LMAROON, stroke=MAROON, sw=2)
    y -= bh + 4

    arrow_down(c, traj_cx, y, y - 14, MAROON, 1.5, 9)
    y -= 16

    # MEMORY BANK
    bh = 80
    _, mem_bot, mem_top = flow_box(c, bx, y, bw, bh,
                                   [(14, "MEMORY BANK", True, TEAL),
                                    (11, "retail.jsonl · airline.jsonl · bfcl.jsonl", False, DARK),
                                    (11, "deduplicated by content signature", False, DARK),
                                    (10, "max 4,096 cards per domain · quality decay after 30 days", False, GRAY)],
                                   fill=LTEAL, stroke=TEAL, sw=2)
    mem_mid_y = (mem_top + mem_bot) / 2
    y -= bh + 18

    # ONLINE sub-header
    frect(c, x, y - sub_s_h, w, sub_s_h, MAROON2)
    label_c(c, cx_col, y - sub_s_h + 8, "ONLINE PIPELINE  ·  Per Task", 13, white)
    y -= sub_s_h + 8

    # Horizontal flow: Task → Retriever → Playbook
    task_w   = 180;  task_h = 55
    retr_w   = 320;  retr_h = 90
    play_w   = 280;  play_h = 90
    h_gap    = 22
    total_w  = task_w + retr_w + play_w + 2*h_gap
    start_x  = x + (w - total_w) / 2

    task_x = start_x
    task_bot = y - task_h
    rrect(c, task_x, task_bot, task_w, task_h, 5, LGRAY, MAROON, 1.5)
    label_c(c, task_x + task_w/2, task_bot + task_h*0.56, "User Task", 13, DARK, bold=True)
    label_c(c, task_x + task_w/2, task_bot + task_h*0.24, "arrives", 10, GRAY)

    retr_x   = task_x + task_w + h_gap
    retr_bot = y - retr_h
    rrect(c, retr_x, retr_bot, retr_w, retr_h, 5, LMAROON, MAROON, 2)
    label_c(c, retr_x + retr_w/2, retr_bot + retr_h*0.74, "HybridRetriever", 13, MAROON, bold=True)
    label_c(c, retr_x + retr_w/2, retr_bot + retr_h*0.53, "BM25 × 0.60 + TF-IDF × 0.40", 10, DARK)
    label_c(c, retr_x + retr_w/2, retr_bot + retr_h*0.33, "domain boost + quality boost", 10, DARK)
    label_c(c, retr_x + retr_w/2, retr_bot + retr_h*0.13, "diversity cap: 3 per task type", 10, GRAY)

    play_x   = retr_x + retr_w + h_gap
    play_bot = y - play_h
    rrect(c, play_x, play_bot, play_w, play_h, 5, LGOLD, GOLD, 2)
    label_c(c, play_x + play_w/2, play_bot + play_h*0.74, "TacticalPlaybook", 13, DARK, bold=True)
    label_c(c, play_x + play_w/2, play_bot + play_h*0.54, "≤ 2,400 chars · ≤ 36 lines", 10, DARK)
    label_c(c, play_x + play_w/2, play_bot + play_h*0.34, "next actions · verify steps", 10, DARK)
    label_c(c, play_x + play_w/2, play_bot + play_h*0.14, "recovery hints · forbidden behaviors", 10, GRAY)

    # Arrows between horizontal boxes
    task_mid_y  = task_bot + task_h / 2
    retr_mid_y  = retr_bot + retr_h / 2
    play_mid_y  = play_bot + play_h / 2
    arrow_right(c, task_x + task_w, task_mid_y, retr_x,         MAROON, 1.5, 9)
    arrow_right(c, retr_x + retr_w, retr_mid_y, play_x,         GOLD,   2,   9)

    y -= max(task_h, retr_h, play_h) + 8

    arrow_down(c, play_x + play_w/2, y, y - 16, MAROON, 1.5, 9)
    y -= 18

    # PRAXIS AGENT box (full width)
    bh = 92
    agent_cx = cx_col
    _, agent_bot, agent_top = flow_box(
        c, x + 10, y, w - 20, bh,
        [(15, "PRAXIS AGENT", True, MAROON),
         (11, "system prompt = base policy + domain wiki + TacticalPlaybook + startup analysis", False, DARK),
         (11, "SABER reflection gate on all WRITE tool calls", False, DARK),
         (10, "playbook refreshed every 2 effective steps as task evolves", False, GRAY)],
        fill=LMAROON, stroke=MAROON, sw=3)
    y -= bh + 8

    # Memory → Agent dotted arrow (seed cards at startup)
    # Draw a curved dotted path from memory bank down through the gap to the agent
    dline(c, traj_cx, mem_bot - 4, traj_cx, agent_top + 4, TEAL, 1.5, (5, 4))
    label_c(c, traj_cx - 56, (mem_bot + agent_top)/2, "seed cards", 9, TEAL)
    label_c(c, traj_cx - 56, (mem_bot + agent_top)/2 - 11, "at startup", 9, TEAL)

    # Tool outcome boxes (3 boxes below agent)
    tool_bh = 48
    tool_bw  = (w - 20 - 2*10) // 3
    tools = [
        ("READ tools", "execute directly",  LGRAY,   GRAY,   DARK),
        ("WRITE tools", "reflection guard", LMAROON, MAROON, MAROON),
        ("RESPOND",    "end task",          LGRAY,   GRAY,   DARK),
    ]
    for i, (t, s, fill_c, str_c, txt_c) in enumerate(tools):
        tx = x + 10 + i * (tool_bw + 10)
        rrect(c, tx, y - tool_bh, tool_bw, tool_bh, 4, fill_c, str_c, 1.5)
        label_c(c, tx + tool_bw/2, y - tool_bh + tool_bh*0.60, t,  12, txt_c, bold=True)
        label_c(c, tx + tool_bw/2, y - tool_bh + tool_bh*0.26, s,  10, GRAY)
        # Arrow from agent to tool box
        tcx = tx + tool_bw / 2
        arrow_down(c, tcx, y, y - tool_bh + 2, MAROON if i == 1 else GRAY, 1.2, 7)

    y -= tool_bh + 8

    # Feedback label
    label_c(c, cx_col, y - 14,
            "↑  completed trajectories distilled back into MEMORY BANK after every run",
            10, TEAL)
    y -= 22

    # ── MUTATION REFLECTION GUARD ─────────────────────────────────────────────
    y -= 12
    y = sec_hdr(c, x, y, w, "Mutation Reflection Guard (SABER-style)")

    # Simple 3-column layout: READ | CHECK | WRITE
    col_w3 = (w - 2*10) // 3
    fc_h   = 115

    # READ column (left)
    rx = x
    rrect(c, rx, y - fc_h, col_w3, fc_h, 6, LGRAY, GRAY, 1)
    frect(c, rx, y - 32, col_w3, 32, HexColor('#AAAAAA'))
    label_c(c, rx + col_w3/2, y - 20, "READ TOOLS", 13, white, bold=True)
    label_c(c, rx + col_w3/2, y - fc_h + fc_h*0.54, "get_order_details", 11, DARK)
    label_c(c, rx + col_w3/2, y - fc_h + fc_h*0.38, "search_flights", 11, DARK)
    label_c(c, rx + col_w3/2, y - fc_h + fc_h*0.22, "find_product", 11, DARK)
    label_c(c, rx + col_w3/2, y - fc_h + fc_h*0.10, "→ Execute immediately", 10, GRAY)

    # CHECK column (center)
    check_x = x + col_w3 + 10
    rrect(c, check_x, y - fc_h, col_w3, fc_h, 6, LMAROON, MAROON, 2)
    frect(c, check_x, y - 32, col_w3, 32, MAROON)
    label_c(c, check_x + col_w3/2, y - 20, "REFLECTION CHECK", 12, white, bold=True)
    label_c(c, check_x + col_w3/2, y - fc_h + fc_h*0.66,
            "Same model asked:", 11, DARK)
    c.saveState()
    c.setFont("Helvetica-Oblique", 10)
    c.setFillColor(MAROON)
    c.drawCentredString(check_x + col_w3/2, y - fc_h + fc_h*0.52,
                        '"Do observations confirm this')
    c.drawCentredString(check_x + col_w3/2, y - fc_h + fc_h*0.39,
                        'action is safe & requested?"')
    c.restoreState()
    # ALLOW / BLOCK badges
    rrect(c, check_x + 12, y - fc_h + 14, col_w3//2 - 18, 22, 4, LGREEN, GREEN, 1.5)
    label_c(c, check_x + col_w3//4, y - fc_h + 26, "ALLOW", 11, GREEN, bold=True)
    rrect(c, check_x + col_w3//2 + 6, y - fc_h + 14, col_w3//2 - 18, 22, 4, LRED, RED, 1.5)
    label_c(c, check_x + col_w3*3//4, y - fc_h + 26, "BLOCK", 11, RED, bold=True)

    # WRITE column (right)
    write_x = x + 2 * (col_w3 + 10)
    rrect(c, write_x, y - fc_h, col_w3, fc_h, 6, LMAROON, MAROON, 2)
    frect(c, write_x, y - 32, col_w3, 32, MAROON2)
    label_c(c, write_x + col_w3/2, y - 20, "WRITE TOOLS", 13, white, bold=True)
    label_c(c, write_x + col_w3/2, y - fc_h + fc_h*0.54, "cancel_pending_order", 11, DARK)
    label_c(c, write_x + col_w3/2, y - fc_h + fc_h*0.38, "modify_reservation", 11, DARK)
    label_c(c, write_x + col_w3/2, y - fc_h + fc_h*0.22, "return_items", 11, DARK)
    label_c(c, write_x + col_w3/2, y - fc_h + fc_h*0.10, "→ Check first", 10, MAROON)

    # Arrows: from CHECK → ALLOW/BLOCK (implicit), and READ → execute arrow
    arrow_right(c, rx + col_w3, y - fc_h/2, check_x, MAROON, 1.5, 8)
    arrow_right(c, check_x + col_w3, y - fc_h/2, write_x, MAROON, 1.5, 8)
    y -= fc_h + 8

    # ── NOVEL CONTRIBUTIONS ───────────────────────────────────────────────────
    y -= 12
    y = sec_hdr(c, x, y, w, "Novel Contributions")

    contribs = [
        ("Continual Procedural Distillation",
         "First system to continuously distill agent trajectories into "
         "structured, leakage-audited procedural memory without model retraining."),
        ("Leakage-Safe Memory Architecture",
         "Raw trajectories are never stored. Multi-layer PII audit. "
         "Test-split blocking prevents answer contamination."),
        ("Experience from ALL Agents",
         "Memory is distilled from ALL controllers (Baseline, Act, ReAct, PRAXIS). "
         "Failures become negative-example lessons."),
    ]
    item_h = 88
    for title, body in contribs:
        # Left maroon accent bar
        frect(c, x, y - item_h, 8, item_h, MAROON)
        rrect(c, x + 10, y - item_h, w - 10, item_h, 5, LGRAY, DGRAY, 1)
        label_l(c, x + 20, y - 18, title, 13, MAROON, bold=True)
        wrapped(c, x + 20, y - 34, w - 30, body, 11, DARK)
        y -= item_h + 6


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN 3  —  Memory Card  ·  Retrieval  ·  Results  ·  What is New
# ─────────────────────────────────────────────────────────────────────────────

def draw_c3(c):
    x, w = C3_X, C3_W
    y = BODY_T
    cx_col = x + w / 2

    # ── PROCESSMEMORYCARD ─────────────────────────────────────────────────────
    y = sec_hdr(c, x, y, w, "ProcessMemoryCard — What PRAXIS Remembers")

    card_h = 765
    rrect(c, x, y - card_h, w, card_h, 8, white, MAROON, 2.5)
    # Card header row
    frect(c, x, y - 44, w, 44, MAROON)
    label_l(c, x + 12, y - 29, "ProcessMemoryCard", 14, white, bold=True)
    label_c(c, x + w - 140, y - 29, "[domain: retail  · outcome: ✓]", 11, GOLD)

    def card_row(y_top, lbl, val, alt=False):
        rh = 48
        fill = LGRAY if alt else white
        frect(c, x, y_top - rh, w, rh, fill)
        # divider
        line(c, x, y_top - rh, x + w, y_top - rh, DGRAY, 0.5)
        lw = 185
        frect(c, x, y_top - rh, lw, rh, HexColor('#F0ECF0') if not alt else HexColor('#EAE6EA'))
        label_l(c, x + 8, y_top - rh + 15, lbl, 11, MAROON, bold=True)
        c.saveState()
        c.setFont("Helvetica", 11)
        c.setFillColor(DARK)
        # simple word wrap inside value column
        val_x = x + lw + 8
        val_w = w - lw - 12
        words = val.split()
        cur_line = ""
        cur_y = y_top - 14
        for word in words:
            test = (cur_line + " " + word).strip()
            if c.stringWidth(test, "Helvetica", 11) <= val_w:
                cur_line = test
            else:
                c.drawString(val_x, cur_y, cur_line)
                cur_y -= 13
                cur_line = word
        if cur_line:
            c.drawString(val_x, cur_y, cur_line)
        c.restoreState()
        return y_top - rh

    cy = y - 44
    cy = card_row(cy, "Task type",   "exchange_delivered_order_items", alt=False)
    cy = card_row(cy, "Outcome",     "successful  (reward = 1.0)",     alt=True)
    cy = card_row(cy, "Lesson",
                  "After item lookup: get_order_details → get_product_details → exchange_delivered_order_items",
                  alt=False)
    cy = card_row(cy, "Tool order",
                  "[ ground_identity, get_order_details, get_product_details, exchange_delivered_order_items ]",
                  alt=True)
    cy = card_row(cy, "Don't do",    "copy IDs, emails, or payment methods from prior examples", alt=False)
    cy = card_row(cy, "Verify",      "fetch live order state from get_ tool before any mutating call", alt=True)
    cy = card_row(cy, "Confirm",     "ask user to confirm exact target, option, and irreversible effect", alt=False)
    cy = card_row(cy, "Confidence",  "0.87  ·  Used: 12×  ·  Successes: 10  ·  Failures: 2", alt=True)

    # "NOT stored" strip at bottom of card
    not_y = y - card_h + 1
    frect(c, x, not_y, w, 44, LRED)
    line(c, x, not_y + 44, x + w, not_y + 44, RED, 0.5)
    label_c(c, cx_col, not_y + 28, "× NO order IDs     × NO emails     × NO payment data     × NO prices     × NO raw transcripts",
            11, RED, bold=True)
    label_c(c, cx_col, not_y + 13, "Only the procedure.  Only the lesson.", 11, MAROON)

    y -= card_h + 12

    # ── HYBRID RETRIEVAL ──────────────────────────────────────────────────────
    y = sec_hdr(c, x, y, w, "Hybrid Retrieval — Finding the Right Lesson")

    # Formula box
    f_h = 50
    rrect(c, x, y - f_h, w, f_h, 6, LGOLD, GOLD, 2)
    label_c(c, cx_col, y - f_h + 30, "score  =  0.60 × BM25_norm  +  0.40 × TF-IDF_cosine  +  δ_domain  +  δ_quality", 12, DARK, bold=True)
    label_c(c, cx_col, y - f_h + 14, "diversity cap: max 3 cards per task_category  ·  top-5 returned per query", 10, GRAY)
    y -= f_h + 8

    # 3-step flow (horizontal)
    step_bh = 82
    step_bw = (w - 2*12) // 3
    steps = [
        ("Current State",       ["task description", "+ last tool called", "+ failed tools  + step #"],  LGRAY,   MAROON, DARK),
        ("HybridRetriever",     ["BM25 × 0.60", "+ TF-IDF × 0.40", "→ top-5 cards"],                   LMAROON, MAROON, MAROON),
        ("TacticalPlaybook",    ["≤ 2,400 chars", "next actions · verify", "recovery · forbidden"],      LGOLD,   GOLD,   DARK),
    ]
    for i, (title, lines_t, fill_c, str_c, col_t) in enumerate(steps):
        sx = x + i * (step_bw + 12)
        rrect(c, sx, y - step_bh, step_bw, step_bh, 6, fill_c, str_c, 2)
        label_c(c, sx + step_bw/2, y - 16, title, 12, col_t, bold=True)
        for j, ln in enumerate(lines_t):
            label_c(c, sx + step_bw/2, y - 31 - j*16, ln, 10, DARK)
        if i < 2:
            arrow_right(c, sx + step_bw, y - step_bh/2, sx + step_bw + 12,
                        str_c, 2, 8)
    y -= step_bh + 6

    label_c(c, cx_col, y - 14,
            "Playbook refreshes every 2 steps — guidance evolves as the task progresses",
            10, GRAY)
    y -= 22

    # ── EXPERIMENTAL RESULTS ──────────────────────────────────────────────────
    y -= 10
    y = sec_hdr(c, x, y, w, "Experimental Results")

    # Table
    cols = ["Benchmark",         "Vanilla TC", "Act",  "ReAct", "PRAXIS ★",  "Δ"]
    cws  = [w - 640,              128,           90,     110,     180,          90]
    th   = 38
    # Header row
    col_start = x
    for ci, (cn, cw_) in enumerate(zip(cols, cws)):
        frect(c, col_start, y - th, cw_, th, MAROON)
        line(c, col_start, y - th, col_start, y, MAROON2, 0.5)
        label_c(c, col_start + cw_/2, y - th + 13, cn, 12, white, bold=True)
        col_start += cw_

    rows = [
        ("tau-bench retail",  "—", "—", "—", "—", "+  pp"),
        ("tau-bench airline", "—", "—", "—", "—", "+  pp"),
        ("BFCL V4",           "—", "—", "—", "—", "+  pp"),
    ]
    row_h = 44
    for ri, row in enumerate(rows):
        col_start = x
        fill_row = LGRAY if ri % 2 == 0 else white
        for ci, (val, cw_) in enumerate(zip(row, cws)):
            cell_fill = LGOLD   if ci == 4 else fill_row
            cell_col  = DARK
            cell_bold = False
            if ci == 4:
                cell_col  = DARK
                cell_bold = True
            if ci == 5:
                cell_col  = GREEN
                cell_bold = True
            rrect(c, col_start, y - th - (ri+1)*row_h, cw_, row_h, 0, cell_fill, DGRAY, 0.5)
            label_c(c, col_start + cw_/2, y - th - (ri+1)*row_h + 16, val, 12, cell_col, bold=cell_bold)
            col_start += cw_
    y -= th + len(rows) * row_h + 6

    label_c(c, cx_col, y - 14,
            "Same model  ·  same tools  ·  same tasks  ·  temperature 0.0  ·  zero fine-tuning",
            10, GRAY)
    y -= 22

    # ── WHAT IS NEW ───────────────────────────────────────────────────────────
    y -= 8
    y = sec_hdr(c, x, y, w, "What Makes PRAXIS Different")

    comparisons = [
        (MAROON, "Standard RAG",
         "retrieves documents to answer questions",
         "retrieves procedures to execute multi-step tasks"),
        (TEAL,   "In-Context Learning",
         "static examples in every prompt — expensive, fixed",
         "growing compact lessons accumulated across all runs"),
        (GOLD,   "Fine-tuning",
         "expensive, slow, risks catastrophic forgetting",
         "zero training  ·  works on any frozen OpenAI-compatible model"),
    ]
    item_h = 118
    for bar_col, other_name, other_desc, praxis_desc in comparisons:
        frect(c, x, y - item_h, 8, item_h, bar_col)
        rrect(c, x + 10, y - item_h, w - 10, item_h, 5, LGRAY, DGRAY, 1)
        half_w = (w - 30) // 2

        # Left: other approach
        label_l(c, x + 18, y - 18, other_name, 13, DARK, bold=True)
        wrapped(c, x + 18, y - 34, half_w - 10, other_desc, 11, GRAY)

        # Center arrow
        ax = x + 18 + half_w
        label_c(c, ax, y - item_h/2 - 6, "→", 20, bar_col, bold=True)

        # Right: PRAXIS
        label_l(c, ax + 18, y - 18, "PRAXIS", 13, MAROON, bold=True)
        wrapped(c, ax + 18, y - 34, half_w - 10, praxis_desc, 11, DARK)

        y -= item_h + 6


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def build():
    out_dir = Path(__file__).parents[1] / "docs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "PRAXIS_POSTER.pdf"

    c = canvas.Canvas(str(out_path), pagesize=(W, H))
    c.setTitle("PRAXIS — Procedural Retrieval-Augmented eXperience-Informed System")

    # White background
    frect(c, 0, 0, W, H, white)

    draw_header(c)
    draw_c1(c)
    draw_c2(c)
    draw_c3(c)

    # Column dividers (subtle)
    for div_x in [C2_X - GAP//2, C3_X - GAP//2]:
        dline(c, div_x, BODY_T - 10, div_x, BODY_B + 10, DGRAY, 0.5, (2, 4))

    c.save()
    return out_path


if __name__ == "__main__":
    path = build()
    size = path.stat().st_size / 1024
    print(f"PDF written to: {path}")
    print(f"File size:      {size:.1f} KB")
    print(f"Page size:      48\" × 36\" landscape")
    print(f"Open with any PDF viewer.")
