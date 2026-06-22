# -*- coding: utf-8 -*-
"""Edit ZaminAI_Officer_Demo.mp4
- Cuts dead loading screens
- Covers "5000 elements" error text with clean overlay
- Speeds up 1.5×
- Adds title card, bridge cards, captions, closing, background music
- Target: ~85 seconds
"""
import sys, io, math, os
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
from moviepy import (VideoFileClip, CompositeVideoClip, ImageClip,
                     concatenate_videoclips, AudioFileClip)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageFont
import scipy.io.wavfile as wav

HERE = Path(__file__).parent
SRC  = HERE / "ZaminAI_Officer_Demo.mp4"
OUT  = Path(r"C:\Users\Eigenaar\Downloads\ZaminAI_Demo_FINAL.mp4")
MUS  = HERE / "audio_edit" / "music_demo.wav"
(HERE / "audio_edit").mkdir(exist_ok=True)

FPS   = 30
SPEED = 1.5

# ── Source video dimensions (set dynamically in build()) ─────────────────────
W, H = 1280, 720

# ── Error text region to cover (sidebar, approx position on 1280x720) ────────
ERR_X, ERR_Y, ERR_W, ERR_H = 0, 485, 315, 45   # x,y,w,h of error text area


# ── Generate background music ─────────────────────────────────────────────────
def gen_music(duration=90, sr=44100):
    if MUS.exists():
        print("  Music cached.")
        return
    print("  Generating background music…")
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    rng = np.random.default_rng(42)
    # A-minor ambient pad — same clean approach as edit_officer_video.py
    layers = [(110.00,.28,.000),(130.81,.20,+.08),(164.81,.16,-.06),
              (196.00,.10,+.04),(220.00,.08,-.03),(261.63,.05,+.02)]
    pad = np.zeros_like(t)
    for freq, vol, detune in layers:
        lfo_rate  = 0.06 + rng.uniform(-.01,.01)
        lfo_phase = rng.uniform(0, 2*np.pi)
        lfo = 1.0 + 0.004*np.sin(2*np.pi*lfo_rate*t + lfo_phase)
        pad += vol * np.sin(2*np.pi*(freq+detune)*t) * lfo
    k = np.ones(int(sr*.004)) / int(sr*.004)
    pad = np.convolve(pad, k, mode='same')
    fi, fo = int(4*sr), int(5*sr)
    env = np.ones_like(t)
    env[:fi] = np.linspace(0,1,fi)**2
    env[-fo:] = np.linspace(1,0,fo)**2
    pad *= env
    peak = np.max(np.abs(pad))
    if peak > 0: pad = pad/peak*0.09
    wav.write(str(MUS), sr, (pad*32767).astype(np.int16))
    print(f"  Music: {MUS.name}")


# ── Generate a title/bridge card ─────────────────────────────────────────────
def make_card(title, subtitle="", duration=3.0, accent="#27ae60"):
    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
    fig.patch.set_facecolor("#080f1e")
    ax.set_facecolor("#080f1e")
    ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis('off')

    # Grid lines (subtle)
    for y in range(0, H, 60):
        ax.plot([0,W],[y,y],color="#163352",lw=.4,alpha=.4)
    for x in range(0, W, 80):
        ax.plot([x,x],[0,H],color="#163352",lw=.4,alpha=.4)

    # Accent line
    ax.plot([W*.15,W*.85],[H*.52,H*.52],color=accent,lw=2,alpha=.8)
    ax.plot([W*.15,W*.15],[H*.52,H*.52+6],color=accent,lw=3)

    # ZaminAI logo top-left
    ax.text(30,H-30,"● ZaminAI",fontsize=11,color=accent,
            fontfamily='monospace',va='top',fontweight='bold')

    # Main title
    ax.text(W/2, H/2+22, title, fontsize=28, color='white',
            ha='center', va='center', fontweight='bold',
            fontfamily='monospace')

    # Subtitle
    if subtitle:
        ax.text(W/2, H/2-26, subtitle, fontsize=13, color="#5a8fc0",
                ha='center', va='center', fontfamily='monospace')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = np.array(Image.open(buf).resize((W,H)).convert('RGB'))
    return ImageClip(img).with_duration(duration)


# ── Cover error text on every frame ──────────────────────────────────────────
def cover_error(frame):
    """Replace the red error text region with a clean dark patch."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Dark rectangle to cover error
    draw.rectangle([ERR_X, ERR_Y, ERR_X+ERR_W, ERR_Y+ERR_H],
                   fill=(8, 15, 30))  # matches dashboard background

    # Optional: write clean status text
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()

    draw.text((ERR_X+6, ERR_Y+14), "🛰️ Satellite data ready",
              fill=(39, 174, 96), font=font)

    return np.array(img)


# ── Captions ─────────────────────────────────────────────────────────────────
def add_caption(clip, text, start=0, dur=None):
    """Overlay a bottom-left caption pill on a clip segment."""
    dur = dur or clip.duration
    W2, H2 = clip.size

    fig, ax = plt.subplots(figsize=(W2/100, H2/100), dpi=100)
    fig.patch.set_alpha(0); ax.set_facecolor((0,0,0,0))
    ax.set_xlim(0,W2); ax.set_ylim(0,H2); ax.axis('off')

    pad = 10
    ax.add_patch(mpatches.FancyBboxPatch(
        (20, 18), W2*.55, 38,
        boxstyle="round,pad=5",
        facecolor=(8/255, 15/255, 30/255, .85),
        edgecolor=(39/255, 174/255, 96/255, .7),
        linewidth=1.2, transform=ax.transData
    ))
    ax.text(32, 37, text, fontsize=11.5, color='white',
            fontfamily='monospace', va='center', fontweight='bold')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                transparent=True, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    ov = ImageClip(np.array(Image.open(buf).resize((W2,H2))),
                   is_mask=False).with_duration(dur).with_start(start)
    return CompositeVideoClip([clip, ov])


# ── Main build ────────────────────────────────────────────────────────────────
def build():
    global W, H
    print(f"Loading source: {SRC.name}")
    src = VideoFileClip(str(SRC))
    W, H = src.w, src.h
    print(f"  {W}x{H}  {src.duration:.1f}s  FPS={src.fps}")

    gen_music(duration=120)

    print("Cutting segments…")

    # ── Segment A: t=10-43s — Dashboard loads, Afghanistan, Kunduz selected ──
    # Clean — no error yet
    seg_a = (src.subclipped(10, 43)
                .with_fps(FPS)
                .with_duration(33))
    # seg_a is clean — no error to cover

    # ── Bridge 1 — replace 100s loading screen ───────────────────────────────
    bridge1 = make_card("Querying Live Satellite Data",
                        "Sentinel-2 · SAR Radar · MODIS · CHIRPS", 2.5)

    # ── Segment B: t=148-163s — "Detecting fields…" spinner ─────────────────
    seg_b = (src.subclipped(148, 163)
                .with_fps(FPS))

    # ── Bridge 2 ──────────────────────────────────────────────────────────────
    bridge2 = make_card("Auto-detect Fields from Satellite",
                        "Dynamic World V1 · 10 m resolution", 2.5)

    # ── Segment C: t=163-222s — Results with captions (cover error) ──────────
    seg_c_raw = (src.subclipped(163, 222)
                    .with_fps(FPS))
    # Cover the error text region on every frame
    seg_c = seg_c_raw.image_transform(cover_error)

    # ── Speed up all recording segments ───────────────────────────────────────
    def speed(clip):
        new_dur = clip.duration / SPEED
        return clip.with_duration(new_dur).with_fps(FPS)

    seg_a   = speed(seg_a)
    seg_b   = speed(seg_b)
    seg_c   = speed(seg_c)

    # ── Captions on sped clips ────────────────────────────────────────────────
    seg_a = add_caption(seg_a,
        "Select any country, province, district or village — worldwide",
        start=0, dur=min(8, seg_a.duration))
    seg_b = add_caption(seg_b,
        "Auto-detect fields from satellite — any region on Earth",
        start=0, dur=seg_b.duration)
    seg_c = add_caption(seg_c,
        "Full satellite analysis: NDVI · Water · Rainfall · Land Cover",
        start=0, dur=min(8, seg_c.duration))

    # ── Title card ────────────────────────────────────────────────────────────
    title = make_card("ZaminAI",
                      "Satellite Intelligence for Field Officers · zaminai.org", 5.5,
                      accent="#27ae60")

    # ── Closing card ─────────────────────────────────────────────────────────
    closing = make_card("zaminai.onrender.com/officer.html",
                        "Free · Open · Works for 190+ countries", 5.0,
                        accent="#f39c12")

    # ── Assemble ──────────────────────────────────────────────────────────────
    parts = [title, seg_a, bridge1, seg_b, bridge2, seg_c, closing]
    total = sum(p.duration for p in parts)
    print(f"Total assembled: {total:.1f}s ({total/60:.1f} min)")

    final = concatenate_videoclips(parts, method="compose")

    # ── Add music ─────────────────────────────────────────────────────────────
    from moviepy import afx
    music = AudioFileClip(str(MUS)).subclipped(0, final.duration)
    music = music.with_effects([afx.AudioFadeIn(2), afx.AudioFadeOut(3)])
    final = final.with_audio(music)

    # ── Render ────────────────────────────────────────────────────────────────
    print(f"Rendering → {OUT.name}  (~{total:.0f}s)…")
    final.write_videofile(str(OUT), fps=FPS, codec="libx264",
                          audio_codec="aac", preset="fast",
                          ffmpeg_params=["-crf","22"], logger=None)

    src.close(); final.close(); music.close()

    size_mb = OUT.stat().st_size / 1024 / 1024
    print(f"Done! {OUT.name}  {size_mb:.1f}MB  {total:.0f}s")


if __name__ == "__main__":
    build()
