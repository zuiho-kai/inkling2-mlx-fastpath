# Replay real per-token timestamps from trace.json frame by frame: stock left, fast path right, live tok/s; ffmpeg to mp4.
import sys, json, subprocess, os, re, shutil
from PIL import Image, ImageDraw, ImageFont
tr = json.load(open(sys.argv[1])); outmp4 = sys.argv[2]; fps = 30
W, H = 1920, 1080; PAD = 40; COLW = (W - 3 * PAD) // 2
mono = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 19)
big = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 60)
mid = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 32)
small = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 26)
BG, PANEL, FG, DIM = (14, 16, 22), (24, 27, 36), (225, 228, 235), (130, 136, 150)
COL = {"base": (150, 155, 170), "ours": (255, 166, 64)}
TOP = 210; LINE_H = 25; MAXL = (H - TOP - 150) // LINE_H
def state(ev, t):
    done = [e for e in ev if e[0] <= t]
    text = re.sub(r"<\|[a-z_]+\|>", "", "".join(e[2] for e in done)).lstrip("\n")
    n = len(done)
    if n >= 2:
        # decode speed counted from the first token (prefill excluded)
        tps = (n - 1) / max(done[-1][0] - ev[0][0], 1e-6) if t < ev[-1][0] else (len(ev) - 1) / (ev[-1][0] - ev[0][0])
    else:
        tps = 0.0
    return text, n, tps, t >= ev[-1][0]
def wrap(text, width_chars):
    lines = []
    for raw in text.split("\n"):
        while len(raw) > width_chars: lines.append(raw[:width_chars]); raw = raw[width_chars:]
        lines.append(raw)
    return lines
cw = mono.getlength("M"); WCH = int((COLW - 40) // cw)
tend = max(tr["base"]["events"][-1][0], tr["ours"]["events"][-1][0]) + 2.5
frames = int(tend * fps)
tmp = os.path.splitext(outmp4)[0] + "_frames"; os.makedirs(tmp, exist_ok=True)
for f in range(frames):
    t = f / fps
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.text((PAD, 28), "Inkling 2 Mini on one Mac: same prompt, same 4-bit weights", font=mid, fill=FG)
    d.text((PAD, 72), "Apple M5 Max · MLX · greedy decoding · real per-token timestamps, played back at 1x", font=small, fill=DIM)
    for i, (k, title) in enumerate((("base", "Stock mlx-lm"), ("ours", "Our fused decode path"))):
        x0 = PAD + i * (COLW + PAD)
        d.rounded_rectangle([x0, 120, x0 + COLW, H - 90], 16, fill=PANEL)
        text, n, tps, fin = state(tr[k]["events"], t)
        d.text((x0 + 24, 134), title, font=mid, fill=COL[k])
        s = f"{tps:5.1f} tok/s"; d.text((x0 + COLW - 24 - big.getlength(s), 124), s, font=big, fill=COL[k] if k == "ours" else FG)
        lines = wrap(text, WCH)[-MAXL:]
        for j, ln in enumerate(lines):
            d.text((x0 + 20, TOP + j * LINE_H), ln, font=mono, fill=FG)
        st = f"{n} tokens" + ("  ·  done" if fin else "")
        d.text((x0 + 24, H - 132), st, font=small, fill=DIM)
    sp = tr["ours"]["decode_tps"] / tr["base"]["decode_tps"]
    foot = f"Token-for-token identical output: {'yes' if tr['same_tokens'] else 'NO'}   ·   decode speed {sp:.2f}x"
    d.text((PAD, H - 66), foot, font=small, fill=FG)
    im.save(f"{tmp}/{f:05d}.png")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{tmp}/%05d.png", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-crf", "18", outmp4], check=True)
shutil.rmtree(tmp); print(outmp4, frames)
