"""把 reports 里的结构化数据画成可直接发专栏的两张图。

用法：
    .venv\\Scripts\\python.exe charts/make_charts.py

读 battlelines.json / gestures.json，输出：
    charts/battlelines.png   四条争论的战线
    charts/gestures.png      六种「姿势」表
    charts/*.csv             同一份数据的表格版（Excel 可直接打开，带 BOM）

想改文字或增删条目，只改两个 JSON，再跑一次这个脚本即可——不用重画。
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent

W = 1280
MARGIN = 52

BG = "#FBF9F4"
INK = "#23211E"
BODY = "#3A3733"
SUB = "#6B665E"
FAINT = "#8E897F"
LINE = "#DFD8C9"
DIVIDER = "#EDE7DA"

LEFT_BG, LEFT_BD, LEFT_INK = "#E9F0F4", "#AEC6D5", "#2F5A75"
RIGHT_BG, RIGHT_BD, RIGHT_INK = "#F8EEE3", "#DCBF9F", "#8A5320"
YES_BG, YES_INK = "#E7F0E4", "#3D6B33"
NO_BG, NO_INK = "#EFECE5", "#6B665E"

FONT_R = "C:/Windows/Fonts/msyh.ttc"
FONT_B = "C:/Windows/Fonts/msyhbd.ttc"
_fonts: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(FONT_B if bold else FONT_R, size)
    return _fonts[key]


def wrap(d: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    """按像素宽度折行；中文没有空格，逐字符折。"""
    out: list[str] = []
    for para in str(text).split("\n"):
        cur = ""
        for ch in para:
            if d.textlength(cur + ch, font=f) <= max_w:
                cur += ch
            elif ch in "。，、；：？！）】》」』”’%" and cur:
                cur += ch          # 行首禁则：标点不单独起一行
            else:
                out.append(cur)
                cur = ch
        out.append(cur)
    return out


def chip(d: ImageDraw.ImageDraw, x: int, y: int, text: str, *,
         size: int = 14, fg: str = SUB, bg: str = "#FFFFFF",
         bd: str = LINE, pad_x: int = 8, h: int = 24) -> int:
    """画一个小标签，返回占用宽度。"""
    f = font(size)
    w = int(d.textlength(text, font=f)) + pad_x * 2
    d.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=bg, outline=bd, width=1)
    d.text((x + pad_x, y + (h - size) // 2 - 2), text, font=f, fill=fg)
    return w


def badge(d: ImageDraw.ImageDraw, x: int, y: int, text: str, fg: str, bg: str) -> int:
    f = font(15, True)
    w = int(d.textlength(text, font=f)) + 20
    d.rounded_rectangle([x, y, x + w, y + 26], radius=7, fill=bg)
    d.text((x + 10, y + 4), text, font=f, fill=fg)
    return w


# ───────────────────────── 图一：四条战线 ─────────────────────────

def measure_card(d, data, cw: int) -> int:
    pts = wrap(d, data["points"], font(18), cw - 44)
    impl = wrap(d, data["implicit"], font(15), cw - 44) if data.get("implicit") else []
    h = 18 + 27 + 8 + 27 * len(pts) + 12 + 26 + 16
    if impl:
        h += 26 * len(impl) + 10
    return h


def draw_card(d, x, y, cw, h, data, *, bg, bd, ink, right=False):
    d.rounded_rectangle([x, y, x + cw, y + h], radius=16, fill=bg, outline=bd, width=2)
    ix = x + 22
    d.text((ix, y + 18), data["stance"], font=font(19, True), fill=ink)
    yy = y + 18 + 27 + 8
    for ln in wrap(d, data["points"], font(18), cw - 44):
        d.text((ix, yy), ln, font=font(18), fill=BODY)
        yy += 27
    if data.get("implicit"):
        yy += 4
        for ln in wrap(d, data["implicit"], font(15), cw - 44):
            d.text((ix, yy), ln, font=font(15), fill=FAINT)
            yy += 26
    cy = y + h - 16 - 26
    cx = ix
    if data["clusters"]:
        for c in data["clusters"]:
            cx += chip(d, cx, cy, c, size=15, fg=ink, bd=bd) + 8
    else:
        cx += chip(d, cx, cy, "隐含立场", size=14, fg=FAINT, bd=bd) + 8


def chart_battlelines() -> Path:
    data = json.loads((HERE / "battlelines.json").read_text(encoding="utf-8"))
    meta, lines, footer = data["meta"], data["lines"], data["footer"]

    H = 2600
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = 52

    d.text((MARGIN, y), meta["headline"], font=font(34, True), fill=INK)
    y += 62
    d.text((MARGIN, y), meta["subheadline"], font=font(18), fill=SUB)
    y += 42
    d.line([MARGIN, y, W - MARGIN, y], fill=LINE, width=2)
    y += 36

    gap = 60
    cw = (W - 2 * MARGIN - gap) // 2

    for i, ln in enumerate(lines):
        # 序号 + 问题 + 层次标签
        d.text((MARGIN, y - 2), ln["no"], font=font(23, True), fill=LEFT_INK)
        ql = wrap(d, ln["question"], font(25, True), W - 2 * MARGIN - 60)
        qy = y
        for q in ql:
            d.text((MARGIN + 38, qy), q, font=font(22, True), fill=INK)
            qy += 36
        # 层次标签右对齐
        lv = " / ".join(ln["level"])
        lw = int(d.textlength(lv, font=font(14))) + 18
        chip(d, W - MARGIN - lw, y + 6, lv, size=14, fg=SUB)
        y = qy + 12

        hl = measure_card(d, ln["left"], cw)
        hr = measure_card(d, ln["right"], cw)
        ch = max(hl, hr)
        draw_card(d, MARGIN, y, cw, ch, ln["left"], bg=LEFT_BG, bd=LEFT_BD, ink=LEFT_INK)
        draw_card(d, W - MARGIN - cw, y, cw, ch, ln["right"], bg=RIGHT_BG, bd=RIGHT_BD, ink=RIGHT_INK)

        # 中缝：分歧
        cx = W // 2
        cy = y + ch // 2
        label = "分歧"
        lw2 = int(d.textlength(label, font=font(15, True))) + 22
        d.line([MARGIN + cw + 8, cy, cx - lw2 // 2 - 10, cy], fill=LINE, width=2)
        d.line([cx + lw2 // 2 + 10, cy, W - MARGIN - cw - 8, cy], fill=LINE, width=2)
        d.rounded_rectangle([cx - lw2 // 2, cy - 15, cx + lw2 // 2, cy + 15],
                            radius=15, fill="#FFFFFF", outline=LINE, width=1)
        d.text((cx - lw2 // 2 + 11, cy - 11), label, font=font(15, True), fill=SUB)

        y += ch + 34
        if i < len(lines) - 1:
            d.line([MARGIN, y - 17, W - MARGIN, y - 17], fill=DIVIDER, width=1)

    # 底部事实带
    y += 6
    facts = footer["facts"]
    note = footer["note"]
    fl = wrap(d, note, font(18), W - 2 * MARGIN - 56)
    band_h = 26 + 34 + 10 + 27 * len(fl) + 22
    d.rounded_rectangle([MARGIN, y, W - MARGIN, y + band_h], radius=14, fill="#F3EEE3")
    d.text((MARGIN + 28, y + 22), facts, font=font(19, True), fill=INK)
    yy = y + 22 + 40
    for ln_ in fl:
        d.text((MARGIN + 28, yy), ln_, font=font(18), fill=SUB)
        yy += 27

    y += band_h + 26
    d.text((MARGIN, y), footer["source"], font=font(15), fill=FAINT)
    y += 40

    img = img.crop((0, 0, W, min(y, H)))
    out = HERE / "battlelines.png"
    img.save(out, "PNG")
    return out


# ───────────────────────── 图二：六种姿势 ─────────────────────────

def chart_gestures() -> Path:
    data = json.loads((HERE / "gestures.json").read_text(encoding="utf-8"))
    meta, rows, footer = data["meta"], data["rows"], data["footer"]

    c_name, w_name = MARGIN, 250
    c_does, w_does = MARGIN + 268, 530
    c_kern = c_does + w_does + 28
    w_kern = W - MARGIN - c_kern

    H = 2400
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = 52

    d.text((MARGIN, y), meta["headline"], font=font(32, True), fill=INK)
    y += 58
    for ln in wrap(d, meta["subheadline"], font(19), W - 2 * MARGIN):
        d.text((MARGIN, y), ln, font=font(17), fill=SUB)
        y += 30
    y += 14

    # 表头
    d.line([MARGIN, y, W - MARGIN, y], fill=LINE, width=2)
    y += 14
    d.text((c_name, y), "姿势", font=font(16, True), fill=FAINT)
    d.text((c_does, y), "在争论里干了什么", font=font(16, True), fill=FAINT)
    d.text((c_kern, y), "有没有被埋掉的内核", font=font(16, True), fill=FAINT)
    y += 34
    d.line([MARGIN, y, W - MARGIN, y], fill=LINE, width=2)
    y += 26

    for r in rows:
        top = y
        # 左列：名称 + 类型 + 出现次数
        nl = wrap(d, r["name"], font(20, True), w_name)
        yy = y
        for s in nl:
            d.text((c_name, yy), s, font=font(19, True), fill=INK)
            yy += 30
        yy += 6
        cx = c_name
        for k in r["kinds"]:
            cx += chip(d, cx, yy, k, size=13, fg=SUB, h=23) + 6
        d.text((c_name, yy + 34), f"出现 {r['seen']} 处", font=font(15), fill=FAINT)
        left_h = yy + 60 - top

        # 中列：字面 + 作用
        yy = y
        for s in wrap(d, "字面：" + r["literal"], font(16), w_does):
            d.text((c_does, yy), s, font=font(16), fill=FAINT)
            yy += 25
        yy += 6
        for s in wrap(d, r["does"], font(18), w_does):
            d.text((c_does, yy), s, font=font(18), fill=BODY)
            yy += 28
        mid_h = yy - top

        # 右列：内核
        k = r["kernel"]
        yy = y
        bw = badge(d, c_kern, yy, "有内核" if k["has"] else "无",
                   YES_INK if k["has"] else NO_INK, YES_BG if k["has"] else NO_BG)
        for s in wrap(d, k["text"], font(17), w_kern):
            d.text((c_kern, yy + 34), s, font=font(17),
                   fill=YES_INK if k["has"] else NO_INK)
            yy += 26
        # 让首行与徽标同行排布更紧凑：把文字整体上移一行
        kern_h = yy + 34 - top

        row_h = max(left_h, mid_h, kern_h) + 22
        y = top + row_h
        d.line([MARGIN, y, W - MARGIN, y], fill=DIVIDER, width=1)
        y += 24

    # 底部
    facts, note = footer["facts"], footer["note"]
    fl = wrap(d, note, font(17), W - 2 * MARGIN - 56)
    band_h = 24 + 32 + 8 + 26 * len(fl) + 20
    d.rounded_rectangle([MARGIN, y, W - MARGIN, y + band_h], radius=14, fill="#F3EEE3")
    d.text((MARGIN + 28, y + 20), facts, font=font(19, True), fill=INK)
    yy = y + 20 + 34
    for s in fl:
        d.text((MARGIN + 28, yy), s, font=font(17), fill=SUB)
        yy += 26

    y += band_h + 24
    d.text((MARGIN, y), footer["source"], font=font(15), fill=FAINT)
    y += 40

    img = img.crop((0, 0, W, min(y, H)))
    out = HERE / "gestures.png"
    img.save(out, "PNG")
    return out


# ───────────────────────── 表格版（Excel 可直接开） ─────────────────────────

def export_csv() -> None:
    import csv

    bl = json.loads((HERE / "battlelines.json").read_text(encoding="utf-8"))
    with open(HERE / "battlelines.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["战线", "问题", "层次", "左侧立场", "左侧要点", "左侧观点簇",
                    "右侧立场", "右侧要点", "右侧观点簇", "备注"])
        for ln in bl["lines"]:
            w.writerow([ln["no"], ln["question"], " / ".join(ln["level"]),
                        ln["left"]["stance"], ln["left"]["points"], " ".join(ln["left"]["clusters"]),
                        ln["right"]["stance"], ln["right"]["points"],
                        " ".join(ln["right"]["clusters"]), ln["right"].get("implicit", "")])

    gs = json.loads((HERE / "gestures.json").read_text(encoding="utf-8"))
    with open(HERE / "gestures.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["姿势", "类型", "出现次数", "字面意思", "在争论里做了什么",
                    "有内核", "内核内容"])
        for r in gs["rows"]:
            w.writerow([r["name"], " ".join(r["kinds"]), r["seen"], r["literal"],
                        r["does"], "有" if r["kernel"]["has"] else "无", r["kernel"]["text"]])


if __name__ == "__main__":
    for p in (chart_battlelines(), chart_gestures()):
        print("saved", p, f"{p.stat().st_size/1024:.0f} KB")
    export_csv()
    print("saved csv")
