#!/usr/bin/env python3
"""
Generate PRAXIS conference poster as a draw.io (.drawio) file.

Run:   python scripts/generate_poster.py
Output: docs/PRAXIS_POSTER.drawio

Opens in https://app.diagrams.net  or draw.io desktop.
Poster size: 3300 × 4400 px  (≈ 36 × 48 inches at 91.67 DPI, portrait).
"""
from __future__ import annotations
import textwrap
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
PW, PH   = 3300, 4400      # page width / height
HDR_H    = 310             # header height
BODY_Y   = HDR_H + 15      # body start y
OPT      = 25              # outer padding
GAP      = 12              # gap between sections within a column
SEC_H    = 55              # section header bar height
C1_X, C1_W = OPT, 960
C2_X, C2_W = 1010, 960
C3_X, C3_W = 1995, 1280

# ─────────────────────────────────────────────────────────────────────────────
# COLORS  (matching template palette)
# ─────────────────────────────────────────────────────────────────────────────
MAROON   = "#902346"
MAROON2  = "#801F3E"
GOLD     = "#E8A020"
TEAL     = "#2A9D8F"
WHITE    = "#FFFFFF"
LGRAY    = "#f5f5f5"
LTEAL    = "#E8F5F0"
LGOLD    = "#FFF8E8"
LMAROON  = "#FBF0F4"
DARK     = "#333333"
GRAY     = "#888888"

# ─────────────────────────────────────────────────────────────────────────────
# XML HELPERS
# ─────────────────────────────────────────────────────────────────────────────
_cid = 2
_cells: list[str] = []


def _nid() -> int:
    global _cid
    v = _cid; _cid += 1; return v


def xe(s: str) -> str:
    """Escape a Python string for use inside an XML attribute."""
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;"))


def he(s: str) -> str:
    """Escape a string for embedding *inside* HTML that is itself an XML attr value."""
    # Called when building the html string that will go inside value=""
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;"))


def _cell(cid, value, style, x, y, w, h, vertex=True, src=None, tgt=None):
    if vertex:
        _cells.append(
            f'<mxCell id="{cid}" value="{value}" style="{xe(style)}" '
            f'vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'
            f'</mxCell>'
        )
    else:
        _cells.append(
            f'<mxCell id="{cid}" value="{value}" style="{xe(style)}" '
            f'edge="1" source="{src}" target="{tgt}" parent="1">'
            f'<mxGeometry relative="1" as="geometry"/>'
            f'</mxCell>'
        )


def box(x, y, w, h, style, value=""):
    """Plain rectangle, value is raw text (XML-escaped internally)."""
    cid = _nid()
    _cell(cid, xe(value), style, x, y, w, h)
    return cid


def hbox(x, y, w, h, style, html):
    """Rectangle whose value is pre-escaped HTML markup string.

    ``html`` may contain &lt; &gt; &amp; already escaped, but raw
    double-quotes inside HTML attribute values (e.g. style="...") must
    still be escaped to &quot; so the outer XML value="" attribute stays
    well-formed.
    """
    cid = _nid()
    # Escape residual raw double-quotes that are inside HTML attributes
    # (they would break the surrounding XML value="..." delimiter).
    safe = html.replace('"', '&quot;')
    _cell(cid, safe, style, x, y, w, h)
    return cid


def arrow(src, tgt, label="", col=MAROON, dashed=False, exit_side="bottom",
          entry_side="top"):
    """Directed arrow between two cell IDs."""
    cid = _nid()
    ex = {"bottom": "exitX=0.5;exitY=1;exitDx=0;exitDy=0;",
          "right":  "exitX=1;exitY=0.5;exitDx=0;exitDy=0;",
          "left":   "exitX=0;exitY=0.5;exitDx=0;exitDy=0;"}[exit_side]
    en = {"top":   "entryX=0.5;entryY=0;entryDx=0;entryDy=0;",
          "left":  "entryX=0;entryY=0.5;entryDx=0;entryDy=0;",
          "right": "entryX=1;entryY=0.5;entryDx=0;entryDy=0;"}[entry_side]
    dash = "dashed=1;" if dashed else ""
    style = (f"rounded=1;orthogonalLoop=1;jettySize=auto;{ex}{en}"
             f"strokeColor={col};strokeWidth=2;endArrow=block;endFill=1;"
             f"fontColor={col};fontSize=18;{dash}")
    _cells.append(
        f'<mxCell id="{cid}" value="{xe(label)}" style="{xe(style)}" '
        f'edge="1" source="{src}" target="{tgt}" parent="1">'
        f'<mxGeometry relative="1" as="geometry"/>'
        f'</mxCell>'
    )
    return cid


# ─────────────────────────────────────────────────────────────────────────────
# REUSABLE SHAPE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def sec_header(x, y, w, title):
    """Maroon section header bar."""
    s = (f"rounded=0;whiteSpace=wrap;html=1;fillColor={MAROON};"
         f"strokeColor={MAROON};fontColor={WHITE};fontSize=28;fontStyle=1;"
         f"align=left;spacingLeft=14;verticalAlign=middle;")
    return box(x, y, w, SEC_H, s, title)


def content_bg(x, y, w, h, fill=WHITE, stroke=MAROON):
    s = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};"
         f"strokeColor={stroke};strokeWidth=1;")
    return box(x, y, w, h, s)


def flow_rect(x, y, w, h, fill=WHITE, stroke=MAROON, sw=2, rounded=True):
    r = "1" if rounded else "0"
    s = (f"rounded={r};whiteSpace=wrap;html=1;fillColor={fill};"
         f"strokeColor={stroke};strokeWidth={sw};")
    return box(x, y, w, h, s)


def big_number(x, y, w, h, number, label):
    """Gold number + maroon label, side by side."""
    html = (f'&lt;div style="text-align:center;"&gt;'
            f'&lt;span style="font-size:56px;font-weight:bold;color:{MAROON}"&gt;{he(number)}&lt;/span&gt;'
            f'&lt;br/&gt;'
            f'&lt;span style="font-size:20px;color:{DARK}"&gt;{he(label)}&lt;/span&gt;'
            f'&lt;/div&gt;')
    s = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={LGRAY};"
         f"strokeColor={MAROON};strokeWidth=1;align=center;verticalAlign=middle;")
    return hbox(x, y, w, h, s, html)


def person_box(x, y, w, h, name, sub, color=MAROON, fill=LMAROON):
    """Person card with colored header and label."""
    s = (f"shape=mxgraph.business.man;whiteSpace=wrap;html=1;"
         f"fillColor={color};strokeColor={color};fontColor={WHITE};"
         f"fontSize=24;fontStyle=1;verticalLabelPosition=bottom;"
         f"verticalAlign=top;labelPosition=center;align=center;"
         f"labelBackgroundColor=none;")
    icon_h = int(h * 0.65)
    icon_id = box(x + (w - icon_h // 2) // 2, y, icon_h // 2, icon_h, s)
    html = (f'&lt;div style="text-align:center"&gt;'
            f'&lt;span style="font-size:22px;font-weight:bold;color:{color}"&gt;{he(name)}&lt;/span&gt;'
            f'&lt;br/&gt;'
            f'&lt;span style="font-size:18px;color:{GRAY}"&gt;{he(sub)}&lt;/span&gt;'
            f'&lt;/div&gt;')
    ts = "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=top;"
    hbox(x, y + icon_h + 4, w, h - icon_h - 4, ts, html)
    return icon_id


def card_row(x, y, w, h, content_html, fill=WHITE, stroke="#DDCCCC"):
    s = (f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};"
         f"strokeColor={stroke};strokeWidth=1;")
    return hbox(x, y, w, h, s, content_html)


def label(x, y, w, h, html, align="left", valign="top"):
    s = (f"text;html=1;strokeColor=none;fillColor=none;"
         f"align={align};verticalAlign={valign};whiteSpace=wrap;")
    return hbox(x, y, w, h, s, html)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
def build_header():
    # Background gradient (2 layers)
    box(0, 0, PW, HDR_H,
        f"rounded=0;fillColor={MAROON2};strokeColor={MAROON2};")
    box(0, 0, PW, HDR_H // 2,
        f"rounded=0;fillColor={MAROON};strokeColor={MAROON};")

    # Gold accent strip
    box(248, 10, 6, HDR_H - 20,
        f"rounded=0;fillColor={GOLD};strokeColor={GOLD};")

    # ASU Logo
    box(20, 12, 215, 175,
        "shape=image;verticalLabelPosition=bottom;labelBackgroundColor=default;"
        "verticalAlign=top;aspect=fixed;imageAspect=0;"
        "image=https://ires.ubc.ca/files/2019/10/ASU-logo-white-background.png;"
        "clipPath=inset(10% 1% 5% 3%);")

    # PRAXIS title
    title_html = (f'&lt;span style="font-size:88px;font-weight:bold;color:{WHITE}"&gt;'
                  f'PRAXIS&lt;/span&gt;')
    label(260, 8, 2100, 108, title_html, "left", "middle")

    # Full name
    full_html = (f'&lt;span style="font-size:36px;color:{WHITE}"&gt;'
                 f'Procedural Retrieval-Augmented eXperience-Informed System'
                 f'&lt;/span&gt;')
    label(260, 115, 2800, 52, full_html, "left", "middle")

    # Tagline
    tag_html = (f'&lt;span style="font-size:28px;font-style:italic;color:{GOLD}"&gt;'
                f'Continual Procedural Memory for Tool-Calling LLM Agents'
                f'&lt;/span&gt;')
    label(260, 170, 2800, 44, tag_html, "left", "middle")

    # Author
    auth_html = (f'&lt;span style="font-size:25px;color:{WHITE}"&gt;'
                 f'Usman Mohammed &amp;nbsp;&amp;middot;&amp;nbsp; '
                 f'ASU School of Computing &amp; Augmented Intelligence'
                 f'&lt;/span&gt;')
    label(260, 218, 2800, 40, auth_html, "left", "middle")

    # Badges
    bs = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={MAROON2};"
          f"strokeColor={GOLD};strokeWidth=2;fontColor={WHITE};fontSize=19;fontStyle=1;")
    box(260, 263, 240, 38, bs, "tau-bench retail")
    box(510, 263, 258, 38, bs, "tau-bench airline")
    box(778, 263, 140, 38, bs, "BFCL V4")
    bt = bs.replace(f"strokeColor={GOLD}", f"strokeColor={TEAL}")
    box(928, 263, 400, 38, bt, "Qwen2.5-14B · vLLM · 1×A100")
    bw = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={MAROON2};"
          f"strokeColor={WHITE};strokeWidth=1;fontColor={GOLD};fontSize=19;fontStyle=1;")
    box(1338, 263, 240, 38, bw, "Zero Fine-tuning")


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN 1  —  Analogy · Key Numbers · Future Work
# ─────────────────────────────────────────────────────────────────────────────
def build_c1():
    cx, cw = C1_X, C1_W
    y = BODY_Y

    # ── Section: The Analogy ─────────────────────────────────────────────────
    sec_header(cx, y, cw, "The Core Idea")
    y += SEC_H

    # Quote box
    q_html = (f'&lt;span style="font-size:23px;font-style:italic;color:{MAROON}"&gt;'
              f'&amp;ldquo;An agent that learns from every interaction &amp;mdash; not just its own.&amp;rdquo;'
              f'&lt;/span&gt;')
    hbox(cx, y, cw, 62,
         f"rounded=1;whiteSpace=wrap;html=1;fillColor={LGOLD};"
         f"strokeColor={GOLD};strokeWidth=2;align=center;verticalAlign=middle;",
         q_html)
    y += 68

    # 4 person icons row  (3 baselines on left, PRAXIS on far right)
    icon_w, icon_h = 185, 155
    gap3 = (cw - 4 * icon_w - 30) // 3  # gap between first 3
    px = [cx + OPT, cx + OPT + icon_w + gap3,
          cx + OPT + 2 * (icon_w + gap3),
          cx + cw - OPT - icon_w]  # PRAXIS far right

    names = [("Act", "baseline"), ("ReAct", "baseline"), ("Vanilla TC", "baseline")]
    worker_ids = []
    for i, (nm, sub) in enumerate(names):
        wid = person_box(px[i], y, icon_w, icon_h, nm, sub,
                         color="#666666", fill=LGRAY)
        worker_ids.append(wid)

    praxis_id = person_box(px[3], y, icon_w, icon_h, "PRAXIS", "observes & learns",
                           color=MAROON, fill=LMAROON)

    # "observes →" divider label
    label(cx + OPT + 3 * icon_w + 3 * gap3 + 8, y + icon_h // 2 - 20, 70, 40,
          f'&lt;span style="font-size:22px;color:{GOLD};font-weight:bold"&gt;&rarr;&lt;/span&gt;',
          "center", "middle")
    y += icon_h + 5

    # Labels under icons
    for i, (nm, _) in enumerate(names):
        lhtml = f'&lt;span style="font-size:20px;color:#666"&gt;200 tasks&lt;/span&gt;'
        label(px[i], y, icon_w, 30, lhtml, "center", "middle")
    y += 36

    # Vertical flow: traj → distiller → memory
    fw = cw - 2 * OPT - icon_w - 30  # width of flow boxes (leaves room for PRAXIS column)
    fx = cx + OPT  # left edge of flow column

    traj_id = flow_rect(fx, y, fw, 52, fill=LGRAY, stroke=MAROON)
    traj_html = (f'&lt;span style="font-size:22px;font-weight:bold;color:{DARK}"&gt;'
                 f'trajectories.jsonl&lt;/span&gt;'
                 f'&lt;span style="font-size:18px;color:{GRAY}"&gt; &amp;middot; outputs per run&lt;/span&gt;')
    label(fx + 10, y + 10, fw - 20, 32, traj_html, "center", "middle")
    for wid in worker_ids:
        arrow(wid, traj_id, col=GRAY)
    y += 52 + 5

    dist_id = flow_rect(fx, y, fw, 80, fill=LMAROON, stroke=MAROON)
    dist_html = (f'&lt;b style="font-size:23px;color:{MAROON}"&gt;DISTILLER&lt;/b&gt;'
                 f'&lt;br/&gt;&lt;span style="font-size:18px;color:{DARK}"&gt;'
                 f'strips PII &amp;middot; extracts procedure &amp;middot; audits leakage&lt;/span&gt;')
    label(fx + 10, y + 8, fw - 20, 64, dist_html, "center", "middle")
    arrow(traj_id, dist_id)
    y += 80 + 5

    mem_id = flow_rect(fx, y, fw, 90, fill=LTEAL, stroke=TEAL, sw=2)
    mem_html = (f'&lt;b style="font-size:23px;color:{TEAL}"&gt;MEMORY BANK&lt;/b&gt;'
                f'&lt;br/&gt;&lt;span style="font-size:18px;color:{DARK}"&gt;'
                f'retail.jsonl &amp;middot; airline.jsonl &amp;middot; bfcl.jsonl&lt;/span&gt;'
                f'&lt;br/&gt;&lt;span style="font-size:16px;color:{TEAL}"&gt;'
                f'grows every run &amp;middot; deduplicated &amp;middot; max 4,096 cards&lt;/span&gt;')
    label(fx + 10, y + 8, fw - 20, 74, mem_html, "center", "middle")
    arrow(dist_id, mem_id)
    y += 90 + 5

    # Memory → PRAXIS arrow (right side)
    arrow(mem_id, praxis_id,
          label="top-5 cards\nevery 2 steps",
          col=TEAL, exit_side="right", entry_side="left")

    # Note text below flow
    note_html = (f'&lt;span style="font-size:19px;color:{DARK}"&gt;'
                 f'After each run, PRAXIS distills new cards and adds them back &amp;mdash; '
                 f'the memory grows smarter with every execution.'
                 f'&lt;/span&gt;')
    label(cx, y, cw, 58, note_html, "left", "middle")
    y += 64

    # ── Section: Key Numbers ─────────────────────────────────────────────────
    sec_header(cx, y, cw, "Key Numbers")
    y += SEC_H + 5

    nums = [
        ("0",     "model fine-tuning steps"),
        ("4",     "agents compared"),
        ("5",     "cards retrieved per query"),
        ("2",     "steps between memory refreshes"),
        ("2 400", "max chars in TacticalPlaybook"),
        ("4 096", "max memory cards per domain"),
    ]
    nw = (cw - OPT) // 3 - OPT // 3
    nh = 88
    for i, (num, lbl) in enumerate(nums):
        col_idx = i % 3
        row_idx = i // 3
        nx = cx + col_idx * (nw + OPT // 2)
        ny = y + row_idx * (nh + 8)
        big_number(nx, ny, nw, nh, num, lbl)
    y += 2 * (nh + 8) + GAP

    # ── Section: Future Work ─────────────────────────────────────────────────
    sec_header(cx, y, cw, "Future Work")
    y += SEC_H + 5

    fw_items = [
        ("1", "Cross-Domain Transfer",
         "Can retail lessons about 'verify before mutating' help an airline agent? "
         "Memory is currently siloed per domain."),
        ("2", "Active Memory Querying",
         "Let PRAXIS explicitly request past failures: "
         "'retrieve all cancel_reservation failures' — agent-initiated recall."),
        ("3", "Mid-Task Online Distillation",
         "Distill lessons within a single trajectory. Hit failure at step 5, "
         "learn and adjust by step 8 — without waiting for the run to complete."),
        ("4", "Multi-Agent Shared Memory",
         "Multiple PRAXIS instances contribute to a federated memory bank "
         "while keeping private task data completely local."),
    ]
    item_h = 148
    for num, title, body in fw_items:
        # Number badge
        box(cx, y, 46, item_h,
            f"rounded=1;whiteSpace=wrap;html=1;fillColor={MAROON};strokeColor={MAROON};",
            "")
        n_html = f'&lt;b style="font-size:26px;color:{WHITE}"&gt;{he(num)}&lt;/b&gt;'
        label(cx, y + item_h // 2 - 16, 46, 32, n_html, "center", "middle")

        # Content area
        content_bg(cx + 50, y, cw - 50, item_h, fill=LGRAY)
        t_html = (f'&lt;b style="font-size:22px;color:{MAROON}"&gt;{he(title)}&lt;/b&gt;'
                  f'&lt;br/&gt;&lt;span style="font-size:19px;color:{DARK}"&gt;{he(body)}&lt;/span&gt;')
        label(cx + 60, y + 8, cw - 70, item_h - 16, t_html, "left", "top")
        y += item_h + 8


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN 2  —  System Architecture · Mutation Guard · Novel Contributions
# ─────────────────────────────────────────────────────────────────────────────
def build_c2():
    cx, cw = C2_X, C2_W
    y = BODY_Y

    # ── Section: System Architecture ─────────────────────────────────────────
    sec_header(cx, y, cw, "System Architecture")
    y += SEC_H + 5

    # OFFLINE sub-label
    sub_s = (f"rounded=0;whiteSpace=wrap;html=1;fillColor={MAROON2};"
             f"strokeColor={MAROON2};fontColor={WHITE};fontSize=21;fontStyle=1;"
             f"align=center;")
    box(cx, y, cw, 34, sub_s, "OFFLINE PIPELINE  ·  After Each Run")
    y += 34 + 8

    # Trajectory box
    tr = flow_rect(cx + 30, y, cw - 60, 55, fill=LGRAY, stroke=MAROON)
    t_html = (f'&lt;b style="font-size:21px;color:{DARK}"&gt;trajectories.jsonl&lt;/b&gt;'
              f'&lt;span style="font-size:17px;color:{GRAY}"&gt;&amp;nbsp;&amp;middot; '
              f'one file per benchmark run&lt;/span&gt;')
    label(cx + 40, y + 8, cw - 80, 39, t_html, "center", "middle")
    y += 55 + 4

    # Distiller
    di = flow_rect(cx + 30, y, cw - 60, 95, fill=LMAROON, stroke=MAROON)
    d_html = (f'&lt;b style="font-size:22px;color:{MAROON}"&gt;DISTILLER&lt;/b&gt;'
              f'&lt;br/&gt;&lt;span style="font-size:17px;color:{DARK}"&gt;'
              f'&amp;bull; strips PII (IDs, emails, prices, dates)&lt;br/&gt;'
              f'&amp;bull; extracts tool order, failure patterns, anti-patterns&lt;br/&gt;'
              f'&amp;bull; scores confidence &amp;middot; leakage audit&lt;/span&gt;')
    label(cx + 40, y + 6, cw - 80, 83, d_html, "left", "top")
    arrow(tr, di)
    y += 95 + 4

    # Memory bank
    mi = flow_rect(cx + 30, y, cw - 60, 105, fill=LTEAL, stroke=TEAL, sw=2)
    m_html = (f'&lt;b style="font-size:22px;color:{TEAL}"&gt;MEMORY BANK&lt;/b&gt;'
              f'&lt;br/&gt;&lt;span style="font-size:17px;color:{DARK}"&gt;'
              f'&amp;bull; retail.jsonl &amp;middot; airline.jsonl &amp;middot; bfcl.jsonl&lt;br/&gt;'
              f'&amp;bull; deduplicated by content signature&lt;br/&gt;'
              f'&amp;bull; max 4,096 cards per domain &amp;middot; quality decay after 30 days&lt;/span&gt;')
    label(cx + 40, y + 6, cw - 80, 93, m_html, "left", "top")
    arrow(di, mi)
    y += 105 + 14

    # ONLINE sub-label
    box(cx, y, cw, 34, sub_s, "ONLINE PIPELINE  ·  Per Task")
    y += 34 + 8

    # Task + Retriever (horizontal)
    task_id = flow_rect(cx + 10, y, 190, 60, fill=LGRAY, stroke=MAROON)
    task_html = f'&lt;b style="font-size:20px;color:{DARK}"&gt;User Task&lt;/b&gt;'
    label(cx + 10, y + 12, 190, 36, task_html, "center", "middle")

    retr_id = flow_rect(cx + 220, y, 340, 110, fill=LMAROON, stroke=MAROON)
    r_html = (f'&lt;b style="font-size:20px;color:{MAROON}"&gt;HybridRetriever&lt;/b&gt;'
              f'&lt;br/&gt;&lt;span style="font-size:16px;color:{DARK}"&gt;'
              f'BM25 &amp;times;0.60 + TF-IDF &amp;times;0.40&lt;br/&gt;'
              f'domain boost + quality boost&lt;br/&gt;'
              f'diversity cap: 3 per task type&lt;/span&gt;')
    label(cx + 230, y + 6, 320, 98, r_html, "left", "top")
    arrow(task_id, retr_id, exit_side="right", entry_side="left")

    play_id = flow_rect(cx + 580, y, 360, 110, fill=LGOLD, stroke=GOLD, sw=2)
    p_html = (f'&lt;b style="font-size:20px;color:{DARK}"&gt;TacticalPlaybook&lt;/b&gt;'
              f'&lt;br/&gt;&lt;span style="font-size:16px;color:{DARK}"&gt;'
              f'&amp;leq;2,400 chars &amp;middot; &amp;leq;36 lines&lt;br/&gt;'
              f'next actions &amp;middot; verify steps&lt;br/&gt;'
              f'recovery hints &amp;middot; forbidden behaviors&lt;/span&gt;')
    label(cx + 590, y + 6, 340, 98, p_html, "left", "top")
    arrow(retr_id, play_id, exit_side="right", entry_side="left",
          label="top-5 cards", col=GOLD)
    y += 110 + 8

    # PRAXIS Agent box
    agent_id = flow_rect(cx + 30, y, cw - 60, 110, fill=LMAROON, stroke=MAROON, sw=3)
    ag_html = (f'&lt;b style="font-size:24px;color:{MAROON}"&gt;PRAXIS AGENT&lt;/b&gt;'
               f'&lt;br/&gt;&lt;span style="font-size:17px;color:{DARK}"&gt;'
               f'system prompt = base policy + domain wiki + TacticalPlaybook + startup analysis&lt;br/&gt;'
               f'SABER reflection gate on all WRITE tool calls&lt;br/&gt;'
               f'playbook refreshed every 2 effective steps&lt;/span&gt;')
    label(cx + 40, y + 6, cw - 80, 98, ag_html, "left", "top")
    # Arrows into agent
    arrow(play_id, agent_id)
    # Memory → Agent (dotted, labeled)
    arrow(mi, agent_id, label="seed cards\nat startup",
          col=TEAL, dashed=True)
    y += 110 + 8

    # 3 tool outcome boxes
    tw = (cw - 60 - 2 * 12) // 3
    read_id = flow_rect(cx + 30, y, tw, 60, fill=LGRAY, stroke="#888888")
    box_html = f'&lt;b style="font-size:19px;color:{DARK}"&gt;READ tools&lt;br/&gt;&lt;/b&gt;&lt;span style="font-size:16px;color:{GRAY}"&gt;execute directly&lt;/span&gt;'
    label(cx + 40, y + 8, tw - 20, 44, box_html, "center", "middle")

    write_id = flow_rect(cx + 30 + tw + 12, y, tw, 60, fill=LMAROON, stroke=MAROON)
    wbox_html = f'&lt;b style="font-size:19px;color:{MAROON}"&gt;WRITE tools&lt;br/&gt;&lt;/b&gt;&lt;span style="font-size:16px;color:{DARK}"&gt;reflection guard first&lt;/span&gt;'
    label(cx + 40 + tw + 12, y + 8, tw - 20, 44, wbox_html, "center", "middle")

    done_id = flow_rect(cx + 30 + 2 * (tw + 12), y, tw, 60, fill=LGRAY, stroke=GRAY)
    dbox_html = f'&lt;b style="font-size:19px;color:{DARK}"&gt;RESPOND&lt;br/&gt;&lt;/b&gt;&lt;span style="font-size:16px;color:{GRAY}"&gt;end task&lt;/span&gt;'
    label(cx + 40 + 2 * (tw + 12), y + 8, tw - 20, 44, dbox_html, "center", "middle")

    arrow(agent_id, read_id, exit_side="bottom", entry_side="top", col=GRAY)
    arrow(agent_id, write_id)
    arrow(agent_id, done_id, exit_side="bottom", entry_side="top", col=GRAY)
    y += 60 + GAP

    # "results distilled back" label
    loop_html = (f'&lt;span style="font-size:17px;font-style:italic;color:{TEAL}"&gt;'
                 f'&amp;uarr; completed trajectories distilled back into MEMORY BANK after every run'
                 f'&lt;/span&gt;')
    label(cx, y, cw, 38, loop_html, "center", "middle")
    y += 44

    # ── Section: Mutation Reflection Guard ───────────────────────────────────
    sec_header(cx, y, cw, "Mutation Reflection Guard")
    y += SEC_H + 8

    # Flowchart
    qa = flow_rect(cx + cw // 2 - 200, y, 400, 54, fill=LGRAY, stroke=MAROON)
    qa_html = f'&lt;b style="font-size:20px;color:{DARK}"&gt;Tool Call Requested&lt;/b&gt;'
    label(cx + cw // 2 - 190, y + 12, 380, 30, qa_html, "center", "middle")
    y += 54 + 4

    # Diamond (read vs write decision)
    diag_html = f'&lt;b style="font-size:19px;color:{DARK}"&gt;Write tool?&lt;/b&gt;'
    diag_id = hbox(cx + cw // 2 - 140, y, 280, 60,
                   f"rhombus;whiteSpace=wrap;html=1;fillColor={LMAROON};"
                   f"strokeColor={MAROON};strokeWidth=2;align=center;verticalAlign=middle;",
                   diag_html)
    arrow(qa, diag_id)
    y += 60 + 4

    # NO branch (left): execute
    no_id = flow_rect(cx + 20, y, 280, 54, fill=LGRAY, stroke=GRAY)
    no_html = f'&lt;span style="font-size:20px;color:{DARK}"&gt;Execute directly&lt;/span&gt;'
    label(cx + 30, y + 12, 260, 30, no_html, "center", "middle")
    arrow(diag_id, no_id, label="NO (read)", exit_side="left", entry_side="right", col=GRAY)

    # YES branch (right): reflection
    ref_id = flow_rect(cx + cw - 360, y, 340, 85, fill=LMAROON, stroke=MAROON, sw=2)
    ref_html = (f'&lt;b style="font-size:19px;color:{MAROON}"&gt;Reflection Check&lt;/b&gt;'
                f'&lt;br/&gt;&lt;span style="font-size:16px;color:{DARK}"&gt;'
                f'Same model asked: &amp;ldquo;Do observations confirm&lt;br/&gt;'
                f'this action is safe and requested?&amp;rdquo;&lt;/span&gt;')
    label(cx + cw - 350, y + 6, 320, 73, ref_html, "left", "top")
    arrow(diag_id, ref_id, label="YES (write)", exit_side="right", entry_side="left")
    y += 85 + 4

    # ALLOW / BLOCK
    allow_id = flow_rect(cx + cw - 360, y, 150, 50,
                         fill="#E8F5E9", stroke="#4CAF50", sw=2)
    allow_html = f'&lt;b style="font-size:20px;color:#2E7D32"&gt;ALLOW&lt;/b&gt;'
    label(cx + cw - 350, y + 10, 130, 30, allow_html, "center", "middle")

    block_id = flow_rect(cx + cw - 180, y, 160, 50,
                         fill="#FFEBEE", stroke="#C62828", sw=2)
    block_html = f'&lt;b style="font-size:20px;color:#C62828"&gt;BLOCK&lt;/b&gt;'
    label(cx + cw - 170, y + 10, 140, 30, block_html, "center", "middle")

    arrow(ref_id, allow_id, label="evidence\nconfirmed", exit_side="bottom", entry_side="top", col="#4CAF50")
    arrow(ref_id, block_id, label="policy\nviolation", exit_side="bottom", entry_side="top", col="#C62828")
    y += 50 + GAP

    # ── Section: Novel Contributions ─────────────────────────────────────────
    sec_header(cx, y, cw, "Novel Contributions")
    y += SEC_H + 5

    contribs = [
        ("Continual Procedural Distillation",
         "First system to continuously distill agent trajectories into "
         "structured, leakage-audited procedural memory without retraining."),
        ("Leakage-Safe Memory Architecture",
         "Strict separation of raw trajectories vs. distilled knowledge. "
         "Multi-layer PII audit. Test-split blocking prevents data contamination."),
        ("Experience from All Agents",
         "Memory is distilled from ALL controllers (Baseline, Act, ReAct, PRAXIS). "
         "PRAXIS benefits from their failures as negative examples."),
    ]
    item_h = 100
    for title, body in contribs:
        content_bg(cx, y, cw, item_h, fill=LGRAY)
        bar_s = f"rounded=0;fillColor={MAROON};strokeColor={MAROON};"
        box(cx, y, 8, item_h, bar_s)
        ch = (f'&lt;b style="font-size:20px;color:{MAROON}"&gt;{he(title)}&lt;/b&gt;'
              f'&lt;br/&gt;&lt;span style="font-size:17px;color:{DARK}"&gt;{he(body)}&lt;/span&gt;')
        label(cx + 16, y + 8, cw - 20, item_h - 16, ch, "left", "top")
        y += item_h + 6


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN 3  —  Memory Card · Retrieval · Results · What is New
# ─────────────────────────────────────────────────────────────────────────────
def build_c3():
    cx, cw = C3_X, C3_W
    y = BODY_Y

    # ── Section: Memory Card ─────────────────────────────────────────────────
    sec_header(cx, y, cw, "ProcessMemoryCard")
    y += SEC_H

    # Card outer frame
    content_bg(cx, y, cw, 845, fill=WHITE, stroke=MAROON)

    # Card header (maroon)
    ch_s = (f"rounded=0;whiteSpace=wrap;html=1;fillColor={MAROON};"
            f"strokeColor={MAROON};fontColor={WHITE};align=left;spacingLeft=12;"
            f"fontSize=21;fontStyle=1;verticalAlign=middle;")
    box(cx, y, cw, 50, ch_s,
        "ProcessMemoryCard                  [domain: retail  ·  outcome: ✓ successful]")
    cy = y + 50

    def crow(label_s, val_s, h=48, fill=WHITE):
        nonlocal cy
        s = (f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};"
             f"strokeColor=#DDCCCC;strokeWidth=1;align=left;")
        lw = 220
        box(cx, cy, lw, h, s, "")
        lh = (f'&lt;b style="font-size:18px;color:{MAROON}"&gt;{he(label_s)}&lt;/b&gt;')
        label(cx + 8, cy + 4, lw - 16, h - 8, lh, "left", "middle")
        box(cx + lw, cy, cw - lw, h,
            f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};"
            f"strokeColor=#DDCCCC;strokeWidth=1;", "")
        vh = f'&lt;span style="font-size:18px;color:{DARK}"&gt;{he(val_s)}&lt;/span&gt;'
        label(cx + lw + 8, cy + 4, cw - lw - 16, h - 8, vh, "left", "middle")
        cy += h

    crow("Task type", "exchange_delivered_order_items", fill=LGRAY)
    crow("Outcome",   "successful  (reward = 1.0)")
    crow("Lesson",
         "After item lookup, recover via: get_order_details → get_product_details → exchange_delivered_order_items",
         h=60, fill=LGRAY)
    crow("Tool order",
         "[ ground_identity, get_order_details, get_product_details, exchange_delivered_order_items ]",
         h=54)
    crow("Don't do",  "copy IDs, emails, or payment methods from prior examples", fill=LGRAY)
    crow("Verify",    "fetch live order state before any mutating call")
    crow("Confirm",   "ask user to confirm exact target, option, and irreversible effect", fill=LGRAY)
    crow("Confidence","0.87  ·  Used: 12×  ·  Successes: 10  ·  Failures: 2")

    y += 845 + 5

    # What is NOT stored
    not_html = (f'&lt;span style="font-size:19px;color:{DARK}"&gt;'
                f'&lt;b style="color:#C62828"&gt;&amp;times;&lt;/b&gt; NO order IDs'
                f'&amp;emsp;&lt;b style="color:#C62828"&gt;&amp;times;&lt;/b&gt; NO emails'
                f'&amp;emsp;&lt;b style="color:#C62828"&gt;&amp;times;&lt;/b&gt; NO payment data'
                f'&amp;emsp;&lt;b style="color:#C62828"&gt;&amp;times;&lt;/b&gt; NO prices'
                f'&amp;emsp;&lt;b style="color:#C62828"&gt;&amp;times;&lt;/b&gt; NO raw transcripts'
                f'&lt;/span&gt;')
    hbox(cx, y, cw, 42,
         f"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFEBEE;"
         f"strokeColor=#C62828;strokeWidth=1;align=center;verticalAlign=middle;",
         not_html)
    y += 48

    # ── Section: Hybrid Retrieval ─────────────────────────────────────────────
    sec_header(cx, y, cw, "Hybrid Retrieval")
    y += SEC_H + 5

    # Score formula box
    formula_html = (f'&lt;div style="text-align:center"&gt;'
                    f'&lt;span style="font-size:22px;font-family:monospace;color:{DARK}"&gt;'
                    f'score&lt;sub&gt;i&lt;/sub&gt; = 0.60 &amp;times; BM25&lt;sub&gt;norm&lt;/sub&gt;'
                    f'&amp;nbsp;+&amp;nbsp; 0.40 &amp;times; TF-IDF&lt;sub&gt;cosine&lt;/sub&gt;'
                    f'&amp;nbsp;+&amp;nbsp; &amp;delta;&lt;sub&gt;domain&lt;/sub&gt;'
                    f'&amp;nbsp;+&amp;nbsp; &amp;delta;&lt;sub&gt;quality&lt;/sub&gt;'
                    f'&lt;/span&gt;&lt;/div&gt;')
    hbox(cx, y, cw, 52,
         f"rounded=1;whiteSpace=wrap;html=1;fillColor={LGOLD};"
         f"strokeColor={GOLD};strokeWidth=2;align=center;verticalAlign=middle;",
         formula_html)
    y += 58

    # 3-step flow (Query → Retriever → Playbook)
    step_w = (cw - 2 * 20) // 3
    step_ids = []
    steps = [
        (f"Current State", f"task desc + last tool\n+ failed tools + step #"),
        (f"HybridRetriever", f"BM25 + TF-IDF\ntop-5, diversity cap 3"),
        (f"TacticalPlaybook", f"≤2,400 chars\ninjected into system prompt"),
    ]
    fills = [LGRAY, LMAROON, LGOLD]
    strokes = [MAROON, MAROON, GOLD]
    for i, ((title, body), fill, stroke) in enumerate(zip(steps, fills, strokes)):
        sx = cx + i * (step_w + 10)
        sid = flow_rect(sx, y, step_w, 95, fill=fill, stroke=stroke, sw=2)
        s_html = (f'&lt;b style="font-size:19px;color:{DARK}"&gt;{he(title)}&lt;/b&gt;'
                  f'&lt;br/&gt;&lt;span style="font-size:16px;color:{DARK}"&gt;{he(body)}&lt;/span&gt;')
        label(sx + 8, y + 8, step_w - 16, 79, s_html, "center", "middle")
        if i > 0:
            arrow(step_ids[-1], sid, exit_side="right", entry_side="left")
        step_ids.append(sid)
    y += 95 + 5

    retr_note = (f'&lt;span style="font-size:17px;font-style:italic;color:{GRAY}"&gt;'
                 f'Playbook refreshes every 2 effective steps — guidance evolves as the task progresses&lt;/span&gt;')
    label(cx, y, cw, 38, retr_note, "center", "middle")
    y += 44

    # ── Section: Results ─────────────────────────────────────────────────────
    sec_header(cx, y, cw, "Experimental Results")
    y += SEC_H + 5

    # Table header
    cols = ["Benchmark", "Vanilla TC", "Act", "ReAct", "PRAXIS ★", "Δ"]
    col_ws = [320, 170, 120, 140, 250, 100]
    assert sum(col_ws) <= cw
    th_s = (f"rounded=0;whiteSpace=wrap;html=1;fillColor={MAROON};"
            f"strokeColor={MAROON};fontColor={WHITE};fontSize=18;fontStyle=1;"
            f"align=center;verticalAlign=middle;")
    tx = cx
    for cname, cw_ in zip(cols, col_ws):
        box(tx, y, cw_, 42, th_s, cname)
        tx += cw_
    y += 42

    # Table rows
    rows = [
        ("tau-bench retail",  "—", "—", "—", "—", "+  pp"),
        ("tau-bench airline", "—", "—", "—", "—", "+  pp"),
        ("BFCL V4",           "—", "—", "—", "—", "+  pp"),
    ]
    for ri, row in enumerate(rows):
        fill = LGRAY if ri % 2 == 0 else WHITE
        tx = cx
        for ci, (cell_val, cw_) in enumerate(zip(row, col_ws)):
            cs = (f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};"
                  f"strokeColor=#DDCCCC;strokeWidth=1;fontColor={DARK};"
                  f"fontSize=18;align=center;verticalAlign=middle;")
            if ci == 4:  # PRAXIS column
                cs = cs.replace(f"fillColor={fill}", f"fillColor={LGOLD}")
                cs = cs.replace(f"fontColor={DARK}", f"fontColor={DARK};fontStyle=1")
            if ci == 5:  # Delta column
                cs = cs.replace(f"fontColor={DARK}", f"fontColor=#2E7D32;fontStyle=1")
            box(tx, y, cw_, 48, cs, cell_val)
            tx += cw_
        y += 48

    note_html = (f'&lt;span style="font-size:17px;font-style:italic;color:{GRAY}"&gt;'
                 f'Same model &amp;middot; same tools &amp;middot; same tasks &amp;middot; '
                 f'zero fine-tuning. Only axis: prompting strategy + memory.&lt;/span&gt;')
    label(cx, y, cw, 36, note_html, "center", "middle")
    y += 42

    # ── Section: What is New ─────────────────────────────────────────────────
    sec_header(cx, y, cw, "What Makes PRAXIS Different")
    y += SEC_H + 5

    comparisons = [
        (MAROON, "Standard RAG",
         "retrieves documents to answer questions",
         "retrieves procedures to execute multi-step tasks"),
        (TEAL, "In-Context Learning",
         "static examples in every prompt — expensive, fixed",
         "growing compact lessons accumulated across all runs"),
        (GOLD, "Fine-tuning",
         "expensive, slow, risks catastrophic forgetting",
         "zero training, works on any frozen OpenAI-compatible model"),
    ]
    item_h2 = 138
    for bar_col, other, other_desc, praxis_desc in comparisons:
        content_bg(cx, y, cw, item_h2, fill=LGRAY)
        box(cx, y, 8, item_h2,
            f"rounded=0;fillColor={bar_col};strokeColor={bar_col};")

        half = (cw - 28) // 2
        oh = (f'&lt;b style="font-size:18px;color:{DARK}"&gt;{he(other)}&lt;/b&gt;'
              f'&lt;br/&gt;&lt;span style="font-size:16px;color:{GRAY}"&gt;{he(other_desc)}&lt;/span&gt;')
        label(cx + 16, y + 8, half - 8, item_h2 - 16, oh, "left", "top")

        # Arrow
        arrow_html = f'&lt;b style="font-size:24px;color:{bar_col}"&gt;&rarr;&lt;/b&gt;'
        label(cx + 16 + half - 8, y + item_h2 // 2 - 18, 28, 36,
              arrow_html, "center", "middle")

        ph = (f'&lt;b style="font-size:18px;color:{MAROON}"&gt;PRAXIS&lt;/b&gt;'
              f'&lt;br/&gt;&lt;span style="font-size:16px;color:{DARK}"&gt;{he(praxis_desc)}&lt;/span&gt;')
        label(cx + 16 + half + 24, y + 8, half - 20, item_h2 - 16, ph, "left", "top")
        y += item_h2 + 6


# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLE AND WRITE
# ─────────────────────────────────────────────────────────────────────────────
def build_poster():
    build_header()
    build_c1()
    build_c2()
    build_c3()

    cells_xml = "\n        ".join(_cells)
    return textwrap.dedent(f"""\
        <mxfile host="app.diagrams.net" modified="2025-01-01T00:00:00.000Z"
                agent="PRAXIS poster generator" version="29.3.6" pages="1">
          <diagram id="praxis-poster" name="PRAXIS Poster">
            <mxGraphModel grid="0" page="0" pageScale="1"
                          pageWidth="{PW}" pageHeight="{PH}"
                          math="0" shadow="0">
              <root>
                <mxCell id="0"/>
                <mxCell id="1" parent="0"/>
                {cells_xml}
              </root>
            </mxGraphModel>
          </diagram>
        </mxfile>
        """)


# ─────────────────────────────────────────────────────────────────────────────
# HTML ENTITY → UNICODE  (make output valid strict XML)
# ─────────────────────────────────────────────────────────────────────────────
_HTML_ENTITIES = {
    "&middot;":  "·",
    "&mdash;":   "—",
    "&ndash;":   "–",
    "&ldquo;":   "“",
    "&rdquo;":   "”",
    "&lsquo;":   "‘",
    "&rsquo;":   "’",
    "&bull;":    "•",
    "&rarr;":    "→",
    "&larr;":    "←",
    "&uarr;":    "↑",
    "&darr;":    "↓",
    "&emsp;":    " ",
    "&ensp;":    " ",
    "&nbsp;":    " ",
    "&times;":   "×",
    "&divide;":  "÷",
    "&leq;":     "≤",
    "&geq;":     "≥",
    "&ne;":      "≠",
    "&approx;":  "≈",
    "&infin;":   "∞",
    "&alpha;":   "α",
    "&beta;":    "β",
    "&delta;":   "δ",
    "&Delta;":   "Δ",
    "&sub;":     "₀",  # fallback — won't appear
    "&plusmn;":  "±",
    "&deg;":     "°",
    "&copy;":    "©",
    "&reg;":     "®",
    "&trade;":   "™",
    "&hellip;":  "…",
    "&laquo;":   "«",
    "&raquo;":   "»",
}


def _fix_entities(xml_str: str) -> str:
    """Replace HTML named entities with their Unicode equivalents so the
    output is valid XML (the standard XML set only includes &amp; &lt;
    &gt; &apos; &quot;).  The draw.io HTML renderer still shows them
    correctly because they are just Unicode code-points at that point."""
    for entity, uchar in _HTML_ENTITIES.items():
        xml_str = xml_str.replace(entity, uchar)
    return xml_str


if __name__ == "__main__":
    out_dir = Path(__file__).parents[1] / "docs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "PRAXIS_POSTER.drawio"
    poster_xml = _fix_entities(build_poster())
    out_path.write_text(poster_xml, encoding="utf-8")
    print(f"Poster written to: {out_path}")
    print(f"Cells generated:   {_cid - 2}")
    print(f"File size:         {out_path.stat().st_size / 1024:.1f} KB")
    print("Open with: https://app.diagrams.net  or  draw.io desktop app")
