#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, io as _io
sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
ZaminAI Officer Dashboard — 60-second Demo Video  (v2)
Natural voice · Ken Burns zoom · Audio-driven timing · Smooth crossfades
"""

import asyncio, os, io, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image as PILImage
from moviepy import VideoClip, AudioFileClip, concatenate_videoclips
import edge_tts

# ── Settings ──────────────────────────────────────────────────────────────────
W, H, FPS, DPI = 1280, 720, 30, 96
SHOT_DIR  = Path("screenshots_officer")
AUDIO_DIR = Path("audio_officer_v2")
OUT_FILE  = "ZaminAI_Officer_Demo.mp4"
APP_URL   = "https://zaminai.onrender.com/officer.html"

TTS_VOICE = "en-US-AriaNeural"   # Warm, natural American female neural voice
TTS_RATE  = "-3%"                 # Slightly slower = more natural, easier to follow

AUDIO_DIR.mkdir(exist_ok=True)
SHOT_DIR.mkdir(exist_ok=True)

# ── Brand colours ─────────────────────────────────────────────────────────────
BG    = "#080f1e"
GREEN = "#27ae60"
GRN2  = "#2ecc71"
BLUE  = "#3498db"
WHITE = "#dde4f0"
DIM   = "#5a7a9a"
GOLD  = "#f39c12"
RED   = "#e74c3c"

# ── Scene definitions (screenshot, zoom_dir, narration, title, subtitle) ──────
# Durations are driven by audio length — no hardcoding needed.
SCENES = [
    (
        "title",      # special: no screenshot, animated slide
        "none",
        "",           # title slides are silent (musical pause)
        "ZaminAI",
        "Officer Dashboard  ·  Satellite Intelligence for Field Officers"
    ),
    (
        "dashboard_loaded",
        "in",
        "Welcome to ZaminAI. "
        "The officer dashboard puts satellite intelligence in the hands of every field team — "
        "for any province or district, anywhere in the world. "
        "Just type a country, choose a region, and ZaminAI handles the rest.",
        "WORLDWIDE COVERAGE",
        "190+ countries  ·  GADM administrative boundaries  ·  Sentinel-2 + Landsat"
    ),
    (
        "kunduz_selected",
        "pan-right",
        "We select Afghanistan, then choose Kunduz province — "
        "one of the country's main agricultural regions. "
        "The map zooms in, draws the provincial boundary, "
        "and loads district data automatically.",
        "SELECT ANY REGION",
        "Province  ·  District  ·  Village  ·  any admin level worldwide"
    ),
    (
        "analysis_results",
        "out",
        "In under two minutes, ZaminAI pulls live satellite data for the entire province. "
        "Vegetation health from Sentinel-2, "
        "water availability from radar, "
        "annual rainfall from CHIRPS, "
        "and land surface temperature from MODIS. "
        "All satellite. All real. All free.",
        "FULL SATELLITE ANALYSIS",
        "NDVI · Water Index · SAR Radar · Annual Rainfall · MODIS Temperature"
    ),
    (
        "land_cover",
        "in",
        "The land cover breakdown uses Google Dynamic World — "
        "a ten-metre resolution global classification updated in near real-time. "
        "For Kunduz, we see fifty percent cropland, "
        "thirty-seven percent bare ground, and twelve percent urban settlements — "
        "with exact hectare figures for every class.",
        "LAND COVER BREAKDOWN",
        "9 classes  ·  Dynamic World V1  ·  10 m resolution  ·  exact ha per class"
    ),
    (
        "layer_panel",
        "out",
        "The layer panel adds on-demand satellite overlays. "
        "Toggle NDVI health to see where crops are thriving or stressed. "
        "Switch to water index to spot irrigation gaps. "
        "Bare soil highlights erosion risk. "
        "Each layer is computed live from Google Earth Engine "
        "and takes about thirty to sixty seconds to load.",
        "SATELLITE LAYER PANEL",
        "NDVI Health · Water Index · Bare Soil · Crop Type · Forest · on demand"
    ),
    (
        "detect_fields",
        "pan-left",
        "Auto-detect maps individual agricultural fields from satellite data. "
        "Each field is coloured by NDVI health — "
        "deep green for healthy, yellow for moderate, red for stressed. "
        "Officers can see exactly where to focus before they set foot on the ground.",
        "AUTO-DETECT FIELDS",
        "Satellite field mapping  ·  NDVI health colours  ·  any district worldwide"
    ),
    (
        "analysis_results",   # reuse — show NDVI + recommendations
        "pan-left",
        "Recommendations update automatically based on satellite data. "
        "Low rainfall and water index triggers an irrigation alert. "
        "Frost risk from MODIS triggers a crop-variety warning. "
        "SAR radar soil moisture informs irrigation scheduling — "
        "all without the officer leaving the dashboard.",
        "ACTIONABLE RECOMMENDATIONS",
        "Irrigation alerts · frost risk · soil moisture · crop type advice"
    ),
    (
        "land_cover",         # reuse — show land cover detail
        "in",
        "Every region tells a different story. "
        "Kunduz shows farmland expanding toward the Kunduz river. "
        "Kabul shows urban growth replacing agricultural land. "
        "Kenya shows smallholder mosaic fields interrupted by forest. "
        "ZaminAI reads the landscape the same way everywhere.",
        "EVERY REGION IS DIFFERENT",
        "Afghanistan · Kenya · India · Netherlands · any country worldwide"
    ),
    (
        "title_end",
        "none",
        "",
        "ZaminAI",
        "Free  ·  Open  ·  For every field officer on Earth  ·  zaminai.org"
    ),
]

# ── Step 1: Capture screenshots ───────────────────────────────────────────────
async def capture_screenshots():
    # Skip if all shots already exist
    needed = [s for s in ("dashboard_loaded","kunduz_selected","analysis_results",
                           "land_cover","layer_panel","detect_fields")
              if not (SHOT_DIR / f"{s}.png").exists()]
    if not needed:
        print("  All screenshots already exist — skipping capture.")
        return

    from playwright.async_api import async_playwright
    print(f"  Capturing: {needed}")
    async with async_playwright() as p:
        br   = await p.chromium.launch(headless=True)
        page = await br.new_page(viewport={"width": W, "height": H})
        print(f"  Loading {APP_URL}…")
        await page.goto(APP_URL, wait_until="networkidle", timeout=90000)
        await page.wait_for_timeout(3000)

        if "dashboard_loaded" in needed:
            await page.screenshot(path=str(SHOT_DIR / "dashboard_loaded.png"))
            print("  OK dashboard_loaded.png")

        # Layer panel open
        try:
            await page.click("#layer-panel-hdr")
            await page.wait_for_timeout(1000)
        except: pass
        if "layer_panel" in needed:
            await page.screenshot(path=str(SHOT_DIR / "layer_panel.png"))
            print("  OK layer_panel.png")
        try:
            await page.click("#layer-panel-hdr")
        except: pass

        # Afghanistan / Kunduz
        print("  Selecting Afghanistan / Kunduz…")
        await page.fill("#country-in", "Afghanistan")
        await page.press("#country-in", "Enter")
        await page.wait_for_timeout(6000)
        opts = await page.locator("#prov-sel option").count()
        if opts > 1:
            await page.select_option("#prov-sel", label="Kunduz")
            await page.wait_for_timeout(2000)
        if "kunduz_selected" in needed:
            await page.screenshot(path=str(SHOT_DIR / "kunduz_selected.png"))
            print("  OK kunduz_selected.png")

        # Run analysis
        print("  Running analysis (1-3 min)…")
        await page.click("#analyse-btn")
        try:
            await page.wait_for_selector("#results.open", timeout=240000)
            await page.wait_for_timeout(1500)
        except: pass
        if "analysis_results" in needed:
            await page.screenshot(path=str(SHOT_DIR / "analysis_results.png"))
            print("  OK analysis_results.png")

        # Scroll to land cover
        try:
            await page.evaluate("document.getElementById('res-body').scrollTop = 700")
            await page.wait_for_timeout(700)
        except: pass
        if "land_cover" in needed:
            await page.screenshot(path=str(SHOT_DIR / "land_cover.png"))
            print("  OK land_cover.png")

        # Auto-detect fields
        print("  Auto-detecting fields…")
        try:
            await page.click("#detect-btn")
            await page.wait_for_timeout(10000)
        except: pass
        if "detect_fields" in needed:
            await page.screenshot(path=str(SHOT_DIR / "detect_fields.png"))
            print("  OK detect_fields.png")

        await br.close()

# ── Step 2: Generate narration in parallel ────────────────────────────────────
async def gen_audio(idx, narr):
    path = AUDIO_DIR / f"s{idx:02d}.mp3"
    if path.exists():
        return
    c = edge_tts.Communicate(narr, TTS_VOICE, rate=TTS_RATE)
    await c.save(str(path))

async def generate_all_audio():
    tasks = [gen_audio(i, sc[2]) for i, sc in enumerate(SCENES) if sc[2]]
    await asyncio.gather(*tasks)

# ── Step 3: Visual helpers ────────────────────────────────────────────────────
def ease(t, dur=1.0):
    x = float(np.clip(t / max(dur, 0.001), 0, 1))
    return float(np.clip(x * x * (3 - 2 * x), 0, 1))

def load_shot(name):
    path = SHOT_DIR / f"{name}.png"
    img  = PILImage.open(path).convert("RGB")
    # Resize to 1440x900 to give room for pan
    return np.array(img.resize((1440, 900), PILImage.LANCZOS))

def ken_burns(img_arr, t, dur, direction="in"):
    ih, iw = img_arr.shape[:2]
    prog = ease(t, dur)
    pan_frac = 0.0
    if direction == "in":
        zoom = 1.0 + 0.07 * prog
    elif direction == "out":
        zoom = 1.07 - 0.07 * prog
    elif direction == "pan-right":
        zoom, pan_frac = 1.05, prog * 0.4
    elif direction == "pan-left":
        zoom, pan_frac = 1.05, (1 - prog) * 0.4
    else:
        zoom = 1.0
    cw = int(W / zoom); ch = int(H / zoom)
    x0 = int((iw - cw) * (0.5 + pan_frac * 0.3))
    y0 = int((ih - ch) * 0.5)
    x0 = max(0, min(x0, iw - cw)); y0 = max(0, min(y0, ih - ch))
    crop = img_arr[y0:y0+ch, x0:x0+cw]
    return np.array(PILImage.fromarray(crop).resize((W, H), PILImage.LANCZOS))

def add_overlay(frame, title, subtitle, t, dur):
    fade_a = min(1.0, t / 0.5)
    if t > dur - 0.5:
        fade_a = min(fade_a, (dur - t) / 0.5)
    if fade_a <= 0.02:
        return frame

    fig = plt.figure(figsize=(W/DPI, H/DPI), dpi=DPI)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.set_facecolor("none")

    a = fade_a

    # Top bar
    ax.add_patch(plt.Rectangle((0, H-46), W, 46, fc=BG, alpha=a*0.88, zorder=5))
    # Green accent left
    ax.add_patch(plt.Rectangle((0, H-46), 5, 46, fc=GREEN, alpha=a, zorder=6))
    ax.text(16, H-23, title, fontsize=13, fontweight="bold",
            color=GREEN, va="center", alpha=a, zorder=7)
    ax.text(W-14, H-23, "ZaminAI Officer", fontsize=10,
            color=DIM, va="center", ha="right", alpha=a*0.8, zorder=7)

    # Bottom bar
    ax.add_patch(plt.Rectangle((0, 0), W, 36, fc=BG, alpha=a*0.88, zorder=5))
    ax.text(W//2, 18, subtitle, fontsize=10.5, color=WHITE,
            va="center", ha="center", alpha=a*0.9, zorder=7)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, transparent=True,
                bbox_inches="tight", pad_inches=0)
    buf.seek(0); plt.close(fig)
    overlay = PILImage.open(buf).convert("RGBA").resize((W, H), PILImage.LANCZOS)
    base = PILImage.fromarray(frame.astype(np.uint8)).convert("RGBA")
    return np.array(PILImage.alpha_composite(base, overlay).convert("RGB"))

def title_frame(t, dur, title_text, subtitle_text):
    a = min(1.0, t / 0.7)
    if t > dur - 0.6:
        a = min(a, (dur - t) / 0.6)

    fig = plt.figure(figsize=(W/DPI, H/DPI), dpi=DPI, facecolor=BG)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.set_facecolor(BG)

    # Subtle grid background
    for x in range(0, W, 52):
        ax.axvline(x, color=GREEN, alpha=0.04, lw=0.5)
    for y in range(0, H, 52):
        ax.axhline(y, color=GREEN, alpha=0.04, lw=0.5)

    # Green accent bar
    bar_w = W * 0.55 * a
    ax.add_patch(plt.Rectangle((W//2 - bar_w/2, H//2 - 80), bar_w, 3, fc=GREEN, alpha=a))

    # Satellite dot
    ax.plot(W//2 - bar_w/2 - 12, H//2 - 78.5, 'o', color=GREEN, markersize=6, alpha=a)

    # Main title
    ax.text(W//2, H//2 - 40, title_text, fontsize=72, fontweight="bold",
            color=WHITE, va="center", ha="center", alpha=a,
            fontfamily="monospace")

    # Subtitle
    ax.text(W//2, H//2 + 30, subtitle_text, fontsize=16,
            color=DIM, va="center", ha="center", alpha=a*0.9)

    # Tag line
    tag = "LIVE  ·  WORLDWIDE  ·  FREE  ·  zaminai.org"
    ax.text(W//2, H//2 + 66, tag, fontsize=11,
            color=GREEN, va="center", ha="center", alpha=a*0.7, fontfamily="monospace")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=BG, bbox_inches="tight", pad_inches=0)
    buf.seek(0); plt.close(fig)
    return np.array(PILImage.open(buf).convert("RGB").resize((W, H), PILImage.LANCZOS))

# ── Step 4: Build clips ───────────────────────────────────────────────────────
def build_video():
    clips = []
    FADE = 0.45   # crossfade duration in seconds

    for i, (key, zoom_dir, narr, title, sub) in enumerate(SCENES):
        # Determine duration from audio (or fixed for silent slides)
        audio_path = AUDIO_DIR / f"s{i:02d}.mp3"
        if narr and audio_path.exists():
            aud  = AudioFileClip(str(audio_path))
            dur  = aud.duration + 0.4   # small tail
        else:
            aud  = None
            dur  = 4.5 if "end" in key else 5.5   # title slides

        is_title = (key in ("title", "title_end"))

        if is_title:
            def make_frame(t, _dur=dur, _ttl=title, _sub=sub):
                return title_frame(t, _dur, _ttl, _sub)
        else:
            try:
                img_arr = load_shot(key)
            except Exception as e:
                print(f"  WARNING: Could not load {key}.png — using blank frame ({e})")
                img_arr = np.zeros((900, 1440, 3), dtype=np.uint8)

            def make_frame(t, _arr=img_arr, _zd=zoom_dir, _dur=dur, _ttl=title, _sub=sub):
                base = ken_burns(_arr, t, _dur, _zd)
                return add_overlay(base, _ttl, _sub, t, _dur)

        clip = VideoClip(make_frame, duration=dur).with_fps(FPS)

        # Add crossfade in/out
        clip = clip.with_effects([
            __import__('moviepy.video.fx', fromlist=['CrossFadeIn']).CrossFadeIn(FADE),
            __import__('moviepy.video.fx', fromlist=['CrossFadeOut']).CrossFadeOut(FADE),
        ])

        if aud is not None:
            clip = clip.with_audio(aud)

        clips.append(clip)
        print(f"  Scene {i+1:02d}: {key:20s} {dur:.1f}s  '{narr[:45]}…'" if narr else
              f"  Scene {i+1:02d}: {key:20s} {dur:.1f}s  [title slide]")

    final = concatenate_videoclips(clips, method="compose",
                                    padding=-FADE)    # overlap = crossfade
    total = sum(c.duration for c in clips) - FADE * (len(clips) - 1)
    print(f"\nRendering {OUT_FILE}  (~{total:.0f}s)…")
    final.write_videofile(OUT_FILE, fps=FPS, codec="libx264",
                          audio_codec="aac", logger="bar",
                          temp_audiofile="temp_officer_v2.m4a",
                          remove_temp=True)
    return OUT_FILE

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    print("=" * 62)
    print("  ZaminAI Officer Dashboard — 1-Minute Demo Video  v2")
    print("=" * 62)

    print("\n[1/3] Screenshots…")
    await capture_screenshots()

    print("\n[2/3] Voice narration (en-US-AriaNeural)…")
    await generate_all_audio()

    # Print audio durations
    total_audio = 0
    for i, (key, _, narr, *_) in enumerate(SCENES):
        p = AUDIO_DIR / f"s{i:02d}.mp3"
        if p.exists():
            from moviepy import AudioFileClip as _AC
            d = _AC(str(p)).duration
            total_audio += d
            print(f"  s{i:02d} {key:20s} {d:.1f}s")
    print(f"  Total narration: {total_audio:.1f}s")

    print("\n[3/3] Assembling video…")
    out = build_video()

    print("\n" + "=" * 62)
    print(f"  Done!  ->  {out}")
    print("=" * 62)

if __name__ == "__main__":
    asyncio.run(main())
