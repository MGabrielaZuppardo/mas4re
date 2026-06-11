"""
Gera o arquivo excalidraw da Figura 1 do artigo SBCARS 2026:
Pipeline vs Baseline — side-by-side architectural comparison.

Uso:
    python docs/gen_figure.py
Saída:
    docs/fig1_architecture.excalidraw
"""

import json
import random
import time

random.seed(42)
TS = int(time.time() * 1000)


def uid():
    return f"{random.randint(0, 0xFFFFFFFF):08x}"


def rect(
    id_,
    x,
    y,
    w,
    h,
    fill,
    stroke,
    stroke_w=2,
    stroke_style="solid",
    roughness=0,
    radius=True,
    dash=False,
):
    return {
        "id": id_,
        "type": "rectangle",
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "angle": 0,
        "strokeColor": stroke,
        "backgroundColor": fill,
        "fillStyle": "solid",
        "strokeWidth": stroke_w,
        "strokeStyle": "dashed" if dash else stroke_style,
        "roughness": roughness,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": {"type": 3} if radius else {"type": 1},
        "seed": random.randint(1, 99999),
        "version": 1,
        "versionNonce": random.randint(1, 99999),
        "isDeleted": False,
        "boundElements": [],
        "updated": TS,
        "link": None,
        "locked": False,
    }


def text(id_, x, y, w, h, content, size=16, color="#1e293b", align="center", bold=False, family=2):
    return {
        "id": id_,
        "type": "text",
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "angle": 0,
        "strokeColor": color,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": None,
        "seed": random.randint(1, 99999),
        "version": 1,
        "versionNonce": random.randint(1, 99999),
        "isDeleted": False,
        "boundElements": [],
        "updated": TS,
        "link": None,
        "locked": False,
        "text": content,
        "fontSize": size,
        "fontFamily": family,
        "textAlign": align,
        "verticalAlign": "top",
        "containerId": None,
        "originalText": content,
        "lineHeight": 1.25,
        "autoResize": True,
    }


def arrow(id_, x, y, dx, dy, color="#475569", stroke_w=2, label=None, stroke_style="solid"):
    el = {
        "id": id_,
        "type": "arrow",
        "x": x,
        "y": y,
        "width": abs(dx),
        "height": abs(dy),
        "angle": 0,
        "strokeColor": color,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": stroke_w,
        "strokeStyle": stroke_style,
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": {"type": 2},
        "seed": random.randint(1, 99999),
        "version": 1,
        "versionNonce": random.randint(1, 99999),
        "isDeleted": False,
        "boundElements": [],
        "updated": TS,
        "link": None,
        "locked": False,
        "points": [[0, 0], [dx, dy]],
        "lastCommittedPoint": None,
        "startBinding": None,
        "endBinding": None,
        "startArrowhead": None,
        "endArrowhead": "arrow",
    }
    return el


# ── Paleta ─────────────────────────────────────────────────────────────────────
C_REQ = "#dbeafe"  # input/output: light blue
C_REQ_STR = "#2563eb"
C_BASE = "#fef3c7"  # baseline agent: light amber
C_BASE_STR = "#d97706"
C_PIPE = "#dcfce7"  # pipeline agents: light green
C_PIPE_STR = "#16a34a"
C_STATE = "#f8fafc"  # StateGraph bg
C_STATE_ST = "#94a3b8"  # StateGraph border
C_CONF = "#fef9c3"  # confidence flag: light yellow
C_CONF_STR = "#ca8a04"
C_BG_L = "#fffbeb"  # left section bg
C_BG_R = "#f0fdf4"  # right section bg
C_DIV = "#cbd5e1"  # divider
C_DARK = "#0f172a"
C_MID = "#475569"
C_FAINT = "#94a3b8"

# ── Coordenadas ────────────────────────────────────────────────────────────────
# Left section: x=20 .. 600   center=310
# Right section: x=640 .. 1240  center=940
LX = 20
LW = 580
LCX = LX + LW // 2  # 310
RX = 640
RW = 600
RCX = RX + RW // 2  # 940
TOT_H = 860

elems = []

# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUNDS + HEADERS
# ══════════════════════════════════════════════════════════════════════════════
elems.append(rect("bg_l", LX, 10, LW, TOT_H, C_BG_L, C_DIV, 1, radius=False))
elems.append(rect("bg_r", RX, 10, RW, TOT_H, C_BG_R, C_DIV, 1, radius=False))

elems.append(text("t_baseline", LX, 22, LW, 36, "BASELINE", 24, C_DARK, "center", bold=True))
elems.append(text("t_bl_sub", LX, 60, LW, 22, "(single-agent)", 15, C_MID, "center"))
elems.append(
    text("t_pipeline", RX, 22, RW, 36, "PIPELINE  (MAS4RE)", 24, C_DARK, "center", bold=True)
)
elems.append(
    text(
        "t_pl_sub",
        RX,
        60,
        RW,
        22,
        "(LangGraph StateGraph · two specialised agents)",
        13,
        C_MID,
        "center",
    )
)

# Divider
elems.append(
    {
        "id": "divider",
        "type": "line",
        "x": 625,
        "y": 10,
        "width": 0,
        "height": TOT_H,
        "angle": 0,
        "strokeColor": C_DIV,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": None,
        "seed": 1,
        "version": 1,
        "versionNonce": 1,
        "isDeleted": False,
        "boundElements": [],
        "updated": TS,
        "link": None,
        "locked": False,
        "points": [[0, 0], [0, TOT_H]],
        "lastCommittedPoint": None,
        "startBinding": None,
        "endBinding": None,
        "startArrowhead": None,
        "endArrowhead": None,
    }
)

# ══════════════════════════════════════════════════════════════════════════════
# LEFT — BASELINE
# ══════════════════════════════════════════════════════════════════════════════
# Input
elems.append(rect("bl_req", LCX - 155, 100, 310, 55, C_REQ, C_REQ_STR, 2))
elems.append(
    text("bl_req_t", LCX - 155, 119, 310, 24, "Requirement", 17, C_REQ_STR, "center", bold=True)
)

# Arrow ↓
elems.append(arrow("bl_arr1", LCX, 155, 0, 48, C_MID))

# BaselineAgent box
elems.append(rect("bl_agent", LX + 30, 205, LW - 60, 255, C_BASE, C_BASE_STR, 2))
elems.append(
    text(
        "bl_agent_t", LX + 30, 215, LW - 60, 26, "BaselineAgent", 17, "#92400e", "center", bold=True
    )
)

# Inner "Single LLM Call" box
elems.append(rect("bl_llm", LX + 70, 248, LW - 140, 50, "#fffbeb", C_BASE_STR, 1))
elems.append(
    text(
        "bl_llm_t",
        LX + 70,
        262,
        LW - 140,
        22,
        "Single LLM Call",
        14,
        "#b45309",
        "center",
        bold=True,
    )
)

# Task lines
elems.append(
    text(
        "bl_t1",
        LX + 55,
        308,
        LW - 110,
        24,
        "① Classify: F / NF  ·  NFR category",
        13,
        "#78350f",
        "center",
    )
)
elems.append(
    text(
        "bl_t2",
        LX + 55,
        332,
        LW - 110,
        24,
        "② Prioritize: MoSCoW  ·  score  ·  rank",
        13,
        "#78350f",
        "center",
    )
)
elems.append(
    text(
        "bl_note",
        LX + 30,
        362,
        LW - 60,
        20,
        "joint context window — no intermediate typed state",
        12,
        C_FAINT,
        "center",
    )
)

# Bracket showing both tasks are joint
elems.append(
    text(
        "bl_joint",
        LX + 30,
        390,
        LW - 60,
        18,
        "─────── tasks share same generation step ───────",
        11,
        "#d97706",
        "center",
    )
)

# Comparison note: ✗
elems.append(rect("bl_warn", LX + 50, 427, LW - 100, 26, "#fee2e2", "#f87171", 1))
elems.append(
    text(
        "bl_warn_t",
        LX + 50,
        433,
        LW - 100,
        18,
        "✗  no typed inter-agent signal",
        12,
        "#dc2626",
        "center",
    )
)

# Arrow ↓
elems.append(arrow("bl_arr2", LCX, 465, 0, 45, C_MID))

# Output box
elems.append(rect("bl_out", LX + 30, 512, LW - 60, 90, C_REQ, C_REQ_STR, 2))
elems.append(
    text(
        "bl_out_t",
        LX + 30,
        524,
        LW - 60,
        24,
        "PrioritizedRequirement",
        16,
        C_REQ_STR,
        "center",
        bold=True,
    )
)
elems.append(
    text(
        "bl_out_f",
        LX + 30,
        554,
        LW - 60,
        18,
        "type · nfr_category · confidence · priority · score",
        12,
        "#3b82f6",
        "center",
    )
)

# Architecture label at bottom
elems.append(
    text(
        "bl_arch",
        LX,
        625,
        LW,
        20,
        "1 agent  ·  1 prompt  ·  1 LLM call per requirement",
        12,
        C_FAINT,
        "center",
    )
)

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — PIPELINE (MAS4RE)
# ══════════════════════════════════════════════════════════════════════════════
# Input
elems.append(rect("pl_req", RCX - 155, 100, 310, 55, C_REQ, C_REQ_STR, 2))
elems.append(
    text("pl_req_t", RCX - 155, 119, 310, 24, "Requirement", 17, C_REQ_STR, "center", bold=True)
)

# Arrow ↓
elems.append(arrow("pl_arr0", RCX, 155, 0, 48, C_MID))

# LangGraph StateGraph dashed container
elems.append(
    rect("pl_sg", RX + 20, 205, RW - 40, 500, C_STATE, C_STATE_ST, 2, dash=True, radius=False)
)
elems.append(
    text(
        "pl_sg_t",
        RX + 20,
        212,
        RW - 40,
        18,
        "LangGraph  StateGraph  (PipelineState)",
        12,
        C_STATE_ST,
        "center",
    )
)

# ── ClassificationAgent ───────────────────────────────────────────────────────
elems.append(rect("pl_cls", RX + 50, 238, RW - 100, 145, C_PIPE, C_PIPE_STR, 2))
elems.append(
    text(
        "pl_cls_t",
        RX + 50,
        250,
        RW - 100,
        26,
        "ClassificationAgent",
        17,
        "#14532d",
        "center",
        bold=True,
    )
)

elems.append(rect("pl_cls_llm", RX + 90, 282, RW - 180, 42, "#f0fdf4", C_PIPE_STR, 1))
elems.append(
    text(
        "pl_cls_llm_t",
        RX + 90,
        295,
        RW - 180,
        20,
        "LLM Call #1 — classify only",
        13,
        "#166534",
        "center",
        bold=True,
    )
)

elems.append(
    text(
        "pl_cls_out",
        RX + 50,
        332,
        RW - 100,
        20,
        "→  type  ·  nfr_category  ·  confidence ∈ [0, 1]",
        12,
        "#15803d",
        "center",
    )
)
elems.append(
    text(
        "pl_cls_note",
        RX + 50,
        354,
        RW - 100,
        18,
        "writes ClassifiedRequirement to PipelineState",
        11,
        C_FAINT,
        "center",
    )
)

# Arrow between agents (inside StateGraph)
elems.append(arrow("pl_arr1", RCX, 383, 0, 30, C_MID))

# ── Confidence flag ───────────────────────────────────────────────────────────
elems.append(rect("pl_conf", RX + 80, 415, RW - 160, 42, C_CONF, C_CONF_STR, 1))
elems.append(
    text(
        "pl_conf_t",
        RX + 80,
        428,
        RW - 160,
        20,
        "conf < 0.70  →  ⚠  uncertainty flag  →  prioritizer uses conservative strategy",
        11,
        "#92400e",
        "center",
    )
)

# Arrow to PrioritizationAgent
elems.append(arrow("pl_arr2", RCX, 457, 0, 30, C_MID))

# ── PrioritizationAgent ───────────────────────────────────────────────────────
elems.append(rect("pl_pri", RX + 50, 490, RW - 100, 145, C_PIPE, C_PIPE_STR, 2))
elems.append(
    text(
        "pl_pri_t",
        RX + 50,
        502,
        RW - 100,
        26,
        "PrioritizationAgent",
        17,
        "#14532d",
        "center",
        bold=True,
    )
)

elems.append(rect("pl_pri_llm", RX + 90, 534, RW - 180, 42, "#f0fdf4", C_PIPE_STR, 1))
elems.append(
    text(
        "pl_pri_llm_t",
        RX + 90,
        547,
        RW - 180,
        20,
        "LLM Call #2 — prioritize only",
        13,
        "#166534",
        "center",
        bold=True,
    )
)

elems.append(
    text(
        "pl_pri_in",
        RX + 50,
        584,
        RW - 100,
        20,
        "←  receives type + nfr_category + confidence  (typed)",
        12,
        "#15803d",
        "center",
    )
)
elems.append(
    text(
        "pl_pri_note",
        RX + 50,
        606,
        RW - 100,
        18,
        "→  priority  ·  priority_score  ·  priority_rank",
        11,
        C_FAINT,
        "center",
    )
)

# Check mark
elems.append(rect("pl_ok", RX + 80, 643, RW - 160, 26, "#dcfce7", "#16a34a", 1))
elems.append(
    text(
        "pl_ok_t",
        RX + 80,
        649,
        RW - 160,
        18,
        "✓  typed inter-agent signal via PipelineState",
        12,
        "#15803d",
        "center",
    )
)

# Arrow ↓ out of StateGraph
elems.append(arrow("pl_arr3", RCX, 710, 0, 42, C_MID))

# Output box
elems.append(rect("pl_out", RX + 20, 754, RW - 40, 90, C_REQ, C_REQ_STR, 2))
elems.append(
    text(
        "pl_out_t",
        RX + 20,
        766,
        RW - 40,
        24,
        "PrioritizedRequirement",
        16,
        C_REQ_STR,
        "center",
        bold=True,
    )
)
elems.append(
    text(
        "pl_out_f",
        RX + 20,
        796,
        RW - 40,
        18,
        "type · nfr_category · confidence · priority · score",
        12,
        "#3b82f6",
        "center",
    )
)

# Architecture label at bottom
elems.append(
    text(
        "pl_arch",
        RX,
        860,
        RW,
        20,
        "2 agents  ·  2 task-specific prompts  ·  2 LLM calls  ·  typed PipelineState",
        12,
        C_FAINT,
        "center",
    )
)

# ══════════════════════════════════════════════════════════════════════════════
# BUILD & SAVE
# ══════════════════════════════════════════════════════════════════════════════
scene = {
    "type": "excalidraw",
    "version": 2,
    "source": "https://excalidraw.com",
    "elements": elems,
    "appState": {
        "gridSize": None,
        "viewBackgroundColor": "#ffffff",
        "currentItemFontFamily": 2,
    },
    "files": {},
}

out = "docs/fig1_architecture.excalidraw"
with open(out, "w", encoding="utf-8") as f:
    json.dump(scene, f, indent=2, ensure_ascii=False)

print(f"Salvo: {out}")
print(f"Elementos: {len(elems)}")
print("Abra em: https://excalidraw.com  (drag-and-drop do arquivo)")
