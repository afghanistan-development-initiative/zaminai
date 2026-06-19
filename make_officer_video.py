#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZaminAI Officer Dashboard — Professional Demo Video  v3
Records REAL browser interaction via Playwright, uses SSML narration,
cuts loading screens, adds animated bridge cards. ~90 s output.
"""
import sys, io as _io, asyncio, os, time, json
sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import io, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image as PILImage
from moviepy import (VideoFileClip, VideoClip, AudioFileClip,
                     ImageClip, concatenate_videoclips)
import edge_tts

# ── Config ────────────────────────────────────────────────────────────────────
W, H, FPS, DPI = 1280, 720, 30, 96
APP_URL    = "https://zaminai.onrender.com/officer.html"
_HERE      = Path(__file__).parent.resolve()
REC_DIR    = _HERE / "recording"
AUDIO_DIR  = _HERE / "audio_v3"
OUT_FILE   = str(_HERE / "ZaminAI_Officer_Demo.mp4")

# Best natural-sounding voice + SSML for human-like delivery
VOICE = "en-US-JennyNeural"
RATE  = "-8%"   # slightly slower = clearer, more natural

REC_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

# ── Brand ─────────────────────────────────────────────────────────────────────
BG    = "#080f1e"
GREEN = "#27ae60"
WHITE = "#dde4f0"
DIM   = "#5a7a9a"
GOLD  = "#f39c12"
RED   = "#e74c3c"

# ── SSML narrations — natural pauses and emphasis ─────────────────────────────
# edge-tts accepts SSML when text starts with <speak>
def ssml(text):
    return f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US"><voice name="{VOICE}"><prosody rate="{RATE}">{text}</prosody></voice></speak>'

NARRATIONS = {
    "intro": ssml(
        'ZaminAI. <break time="500ms"/> '
        'Satellite intelligence for field officers — <break time="200ms"/> '
        'free, open, and working for any district anywhere on Earth.'
    ),
    "dashboard": ssml(
        'The officer dashboard opens to a world map. <break time="300ms"/> '
        'We type <emphasis level="moderate">Afghanistan</emphasis>, <break time="200ms"/> '
        'choose <emphasis level="moderate">Kunduz province</emphasis>, <break time="200ms"/> '
        'and hit <emphasis level="strong">Analyse Region</emphasis>.'
    ),
    "computing": ssml(
        'ZaminAI sends the request to Google Earth Engine. <break time="300ms"/> '
        'It pulls live Sentinel-2 imagery, <break time="150ms"/> '
        'CHIRPS rainfall, <break time="150ms"/> '
        'Sentinel-1 radar, <break time="150ms"/> '
        'and MODIS temperature — <break time="200ms"/> '
        'all in one call.'
    ),
    "results": ssml(
        'The results are in. <break time="400ms"/> '
        'NDVI vegetation health: <emphasis level="moderate">zero point three eight</emphasis> — moderate. <break time="300ms"/> '
        'Annual rainfall: <emphasis level="moderate">three hundred and twenty millimetres</emphasis>. <break time="200ms"/> '
        'SAR radar confirms low soil moisture. <break time="200ms"/> '
        'The dashboard recommends irrigation within four days.'
    ),
    "landcover": ssml(
        'Scroll down and you see the land cover breakdown. <break time="300ms"/> '
        'Fifty percent cropland. <break time="200ms"/> '
        'Thirty-seven percent bare ground. <break time="200ms"/> '
        'Twelve percent urban. <break time="300ms"/> '
        'Exact hectares for every class, '
        'from <emphasis level="moderate">Dynamic World</emphasis> at ten-metre resolution.'
    ),
    "detecting": ssml(
        'Now we click <emphasis level="strong">Auto-detect Fields from Satellite</emphasis>. <break time="400ms"/> '
        'ZaminAI analyses crop patterns across the district '
        'and maps individual agricultural fields.'
    ),
    "fields": ssml(
        'The fields appear — <break time="300ms"/> '
        'colour-coded by vegetation health. <break time="300ms"/> '
        '<emphasis level="moderate">Deep green</emphasis> means healthy, thriving crops. <break time="200ms"/> '
        '<emphasis level="moderate">Yellow</emphasis> means moderate stress. <break time="200ms"/> '
        '<emphasis level="strong">Red</emphasis> flags fields that need urgent attention. <break time="300ms"/> '
        'Officers know exactly where to go before they leave the office.'
    ),
    "global": ssml(
        'The same analysis works for any district — <break time="200ms"/> '
        'Kenya, <break time="100ms"/> India, <break time="100ms"/> Netherlands, <break time="100ms"/> anywhere. <break time="300ms"/> '
        'ZaminAI is free, open-source, and built for the '
        '<emphasis level="strong">five hundred million smallholder farmers</emphasis> '
        'with no digital support today.'
    ),
}

# ── Step 1: Record live browser session ──────────────────────────────────────
TIMESTAMPS_FILE = REC_DIR / "timestamps.json"  # absolute via REC_DIR

async def record_session():
    if (REC_DIR / "session.webm").exists() and TIMESTAMPS_FILE.exists():
        print("  Recording already exists — skipping.")
        return

    from playwright.async_api import async_playwright
    print("  Starting browser recording…")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, slow_mo=120)
        context = await browser.new_context(
            viewport={"width": W, "height": H},
            record_video_dir=str(REC_DIR),
            record_video_size={"width": W, "height": H},
        )
        page = await context.new_page()

        ts = {}
        t0 = time.time()
        def mark(k): ts[k] = round(time.time() - t0, 2)

        # Load app
        print(f"  Loading {APP_URL}…")
        await page.goto(APP_URL, wait_until="networkidle", timeout=90000)
        await page.wait_for_timeout(2500)
        mark("app_ready")

        # Open layer panel briefly to show it exists
        try:
            await page.click("#layer-panel-hdr")
            await page.wait_for_timeout(1200)
            await page.click("#layer-panel-hdr")
            await page.wait_for_timeout(600)
        except: pass
        mark("panel_shown")

        # Type country with realistic pace
        await page.click("#country-in")
        await page.wait_for_timeout(400)
        for ch in "Afghanistan":
            await page.type("#country-in", ch, delay=80)
        await page.wait_for_timeout(500)
        await page.press("#country-in", "Enter")
        await page.wait_for_timeout(5000)
        mark("country_selected")

        # Select province
        opts = await page.locator("#prov-sel option").count()
        if opts > 1:
            await page.select_option("#prov-sel", label="Kunduz")
            await page.wait_for_timeout(2000)
        mark("province_selected")

        # Click Analyse Region — show the button press
        await page.evaluate("document.getElementById('analyse-btn').style.transform='scale(0.96)'")
        await page.wait_for_timeout(150)
        await page.click("#analyse-btn")
        await page.evaluate("document.getElementById('analyse-btn').style.transform=''")
        mark("analyse_clicked")
        print("  Waiting for analysis results (1-3 min)…")

        # Wait for results
        try:
            await page.wait_for_selector("#results.open", timeout=240000)
            await page.wait_for_timeout(1500)
        except: pass
        mark("results_open")

        # Scroll smoothly through results
        for scroll_y in [200, 400, 600, 800]:
            await page.evaluate(f"document.getElementById('res-body').scrollTop = {scroll_y}")
            await page.wait_for_timeout(700)
        mark("results_scrolled")

        # Scroll back to top of results
        await page.evaluate("document.getElementById('res-body').scrollTop = 0")
        await page.wait_for_timeout(800)

        # Open layer panel and toggle Land Cover
        try:
            await page.click("#layer-panel-hdr")
            await page.wait_for_timeout(800)
            mark("panel_open")
        except: pass

        # Close results panel to show full map
        try:
            await page.click("#res-close")
            await page.wait_for_timeout(600)
        except: pass
        mark("map_full")

        # Click detect fields
        try:
            await page.click("#detect-btn")
            mark("detect_clicked")
            print("  Waiting for field detection…")
            # Poll for count card to appear
            for _ in range(60):
                await page.wait_for_timeout(3000)
                visible = await page.locator("#detect-count").is_visible()
                if visible:
                    break
            await page.wait_for_timeout(2000)
        except: pass
        mark("detect_done")

        # Hold final frame
        await page.wait_for_timeout(4000)
        mark("end")

        # Close context FIRST — this finalises the webm file
        await context.close()
        await browser.close()

        # Playwright names the file page@<hash>.webm — find and keep it as-is
        webms = list(REC_DIR.glob("*.webm"))
        if webms:
            latest = max(webms, key=lambda f: f.stat().st_mtime)
            dest = REC_DIR / "session.webm"
            if dest.exists(): dest.unlink()
            latest.rename(dest)
            print(f"  Recording: session.webm  ({dest.stat().st_size//1024} KB)")

        TIMESTAMPS_FILE.write_text(json.dumps(ts, indent=2))
        print(f"  Timestamps: {ts}")

# ── Step 2: Generate narrations ───────────────────────────────────────────────
async def gen_one(key, ssml_text):
    path = AUDIO_DIR / f"{key}.mp3"
    if path.exists(): return
    c = edge_tts.Communicate(ssml_text, VOICE, rate=RATE)
    await c.save(str(path))

async def generate_narrations():
    await asyncio.gather(*[gen_one(k, v) for k, v in NARRATIONS.items()])

def audio_dur(key):
    p = AUDIO_DIR / f"{key}.mp3"
    if p.exists():
        return AudioFileClip(str(p)).duration
    return 4.0

# ── Step 3: Visual helpers ────────────────────────────────────────────────────
def ease(t, dur=1.0):
    x = float(np.clip(t / max(dur, 0.001), 0, 1))
    return float(np.clip(x*x*(3-2*x), 0, 1))

def make_bridge_card(title, subtitle, accent_col=GREEN):
    """Animated bridge card for loading / transition moments."""
    def frame(t, dur=5.0):
        a = ease(t, 0.6) * (1 - ease(max(0, t - (dur-0.5)), 0.5))
        fig = plt.figure(figsize=(W/DPI, H/DPI), dpi=DPI, facecolor=BG)
        ax  = fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off")
        ax.set_facecolor(BG)

        # Grid
        for x in range(0, W, 52): ax.axvline(x, color=GREEN, alpha=0.04, lw=0.5)
        for y in range(0, H, 52): ax.axhline(y, color=GREEN, alpha=0.04, lw=0.5)

        # Accent bar
        bw = W * 0.45 * a
        ax.add_patch(plt.Rectangle((W//2 - bw/2, H//2 - 72), bw, 3,
                                    fc=accent_col, alpha=a))

        # Pulsing dot
        pulse = 0.6 + 0.4 * np.sin(t * 4)
        ax.plot(W//2 - bw/2 - 14, H//2 - 70.5, 'o', color=accent_col,
                markersize=7*pulse, alpha=a*pulse)

        ax.text(W//2, H//2 - 26, title, fontsize=34, fontweight="bold",
                color=WHITE, va="center", ha="center", alpha=a, fontfamily="monospace")
        ax.text(W//2, H//2 + 22, subtitle, fontsize=14,
                color=DIM, va="center", ha="center", alpha=a*0.85)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, facecolor=BG,
                    bbox_inches="tight", pad_inches=0)
        buf.seek(0); plt.close(fig)
        return np.array(PILImage.open(buf).convert("RGB").resize((W,H), PILImage.LANCZOS))
    return frame

def title_frame_fn(t, dur=5.5):
    a = ease(t, 0.7) * (1 - ease(max(0, t-(dur-0.6)), 0.6))
    fig = plt.figure(figsize=(W/DPI, H/DPI), dpi=DPI, facecolor=BG)
    ax  = fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off")
    ax.set_facecolor(BG)

    for x in range(0, W, 52): ax.axvline(x, color=GREEN, alpha=0.04, lw=0.5)
    for y in range(0, H, 52): ax.axhline(y, color=GREEN, alpha=0.04, lw=0.5)

    bw = W * 0.52 * a
    ax.add_patch(plt.Rectangle((W//2 - bw/2, H//2 - 88), bw, 3.5, fc=GREEN, alpha=a))
    ax.plot(W//2 - bw/2 - 14, H//2 - 86.8, 'o', color=GREEN, markersize=8, alpha=a)

    ax.text(W//2, H//2 - 40, "ZaminAI", fontsize=78, fontweight="bold",
            color=WHITE, va="center", ha="center", alpha=a, fontfamily="monospace")
    ax.text(W//2, H//2 + 32, "Officer Dashboard  ·  Satellite Intelligence for Field Officers",
            fontsize=15, color=DIM, va="center", ha="center", alpha=a*0.88)
    ax.text(W//2, H//2 + 68, "LIVE  ·  WORLDWIDE  ·  FREE  ·  zaminai.org",
            fontsize=11, color=GREEN, va="center", ha="center", alpha=a*0.65,
            fontfamily="monospace")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=BG,
                bbox_inches="tight", pad_inches=0)
    buf.seek(0); plt.close(fig)
    return np.array(PILImage.open(buf).convert("RGB").resize((W,H), PILImage.LANCZOS))

def add_lower_third(frame_arr, title, subtitle, t, dur, accent=GREEN):
    """Animated lower-third overlay on video frame."""
    a = ease(t, 0.4) * (1 - ease(max(0, t-(dur-0.4)), 0.4))
    if a < 0.02: return frame_arr

    fig = plt.figure(figsize=(W/DPI, H/DPI), dpi=DPI)
    fig.patch.set_alpha(0); ax = fig.add_axes([0,0,1,1])
    ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off"); ax.set_facecolor("none")

    # Bottom bar
    ax.add_patch(plt.Rectangle((0,0), W, 48, fc=BG, alpha=a*0.90, zorder=5))
    ax.add_patch(plt.Rectangle((0,0), 5, 48, fc=accent, alpha=a, zorder=6))
    ax.text(16, 33, title, fontsize=13, fontweight="bold",
            color=accent, va="center", alpha=a, zorder=7)
    ax.text(16, 13, subtitle, fontsize=10.5,
            color=WHITE, va="center", alpha=a*0.85, zorder=7)

    # Top-right badge
    badge = "🛰 ZaminAI"
    ax.add_patch(plt.Rectangle((W-130, H-38), 125, 32, fc=BG, alpha=a*0.85, zorder=5))
    ax.text(W-14, H-22, badge, fontsize=11, color=GREEN,
            va="center", ha="right", alpha=a*0.9, zorder=7)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, transparent=True,
                bbox_inches="tight", pad_inches=0)
    buf.seek(0); plt.close(fig)
    ov  = PILImage.open(buf).convert("RGBA").resize((W,H), PILImage.LANCZOS)
    base = PILImage.fromarray(frame_arr.astype(np.uint8)).convert("RGBA")
    return np.array(PILImage.alpha_composite(base, ov).convert("RGB"))

# ── Step 4: Assemble video ────────────────────────────────────────────────────
FADE = 0.4

def try_crossfade(clip):
    try:
        from moviepy.video.fx import CrossFadeIn, CrossFadeOut
        return clip.with_effects([CrossFadeIn(FADE), CrossFadeOut(FADE)])
    except Exception:
        return clip

def recording_segment(start_s, end_s, lower_title="", lower_sub="",
                      narration_key=None, playback_speed=1.0):
    """Cut a segment from the recorded session, add overlay, narration."""
    rec = REC_DIR / "session.webm"
    if not rec.exists():
        dur = end_s - start_s
        clip = VideoClip(lambda t: np.zeros((H,W,3), dtype=np.uint8), duration=dur).with_fps(FPS)
    else:
        full = VideoFileClip(str(rec.resolve()))
        safe_start = max(0.0, min(start_s, full.duration - 0.5))
        safe_end   = max(safe_start + 0.5, min(end_s, full.duration - 0.05))
        raw = full.subclipped(safe_start, safe_end)
        # Speed change via time remap (MoviePy 2.x compatible)
        if playback_speed != 1.0:
            spd = playback_speed
            new_dur = raw.duration / spd
            raw = raw.with_duration(new_dur).with_fps(FPS)
        dur = raw.duration

        if lower_title:
            def make(t, _raw=raw, _dur=dur, _lt=lower_title, _ls=lower_sub):
                f = _raw.get_frame(t)
                return add_lower_third(f, _lt, _ls, t, _dur)
            clip = VideoClip(make, duration=dur).with_fps(FPS)
        else:
            clip = raw

    if narration_key:
        ap = AUDIO_DIR / f"{narration_key}.mp3"
        if ap.exists():
            aud = AudioFileClip(str(ap))
            if aud.duration > dur:
                # Extend clip to match audio
                clip = clip.with_duration(aud.duration)
            clip = clip.with_audio(aud.subclipped(0, clip.duration))

    return try_crossfade(clip)

def bridge_card(title, subtitle, narration_key=None,
                min_dur=3.5, accent=GREEN):
    nar_dur = audio_dur(narration_key) + 0.3 if narration_key else 0
    dur = max(min_dur, nar_dur)
    fn = make_bridge_card(title, subtitle, accent)
    clip = VideoClip(lambda t: fn(t, dur), duration=dur).with_fps(FPS)
    if narration_key:
        ap = AUDIO_DIR / f"{narration_key}.mp3"
        if ap.exists():
            clip = clip.with_audio(AudioFileClip(str(ap)))
    return try_crossfade(clip)

def safe_audio(key, max_dur=None):
    """Load audio clip, optionally trimmed to max_dur."""
    p = AUDIO_DIR / f"{key}.mp3"
    if not p.exists():
        return None
    a = AudioFileClip(str(p))
    if max_dur is not None:
        a = a.subclipped(0, min(a.duration, max_dur))
    return a

def rec_clip(start_s, end_s, vdur, lower_title="", lower_sub="",
             narration_key=None, playback_speed=1.0):
    """Recording segment with all bounds guaranteed."""
    s = max(0.0, min(start_s, vdur - 0.5))
    e = max(s + 0.5,  min(end_s,  vdur))
    seg = recording_segment(s, e, lower_title, lower_sub,
                             narration_key=None,
                             playback_speed=playback_speed)
    if narration_key:
        aud = safe_audio(narration_key, seg.duration)
        if aud:
            if aud.duration > seg.duration:
                seg = seg.with_duration(aud.duration)
            seg = seg.with_audio(aud)
    return try_crossfade(seg)

def build_video():
    ts = json.loads(TIMESTAMPS_FILE.read_text()) if TIMESTAMPS_FILE.exists() else {}
    print(f"  Timestamps: {ts}")

    # Get recording duration upfront — clamp all ts values to this
    rec_path = REC_DIR / "session.webm"
    vdur = VideoFileClip(str(rec_path.resolve())).duration if rec_path.exists() else 999.0
    print(f"  Recording duration: {vdur:.2f}s")

    def T(key, fallback=0.0):
        return min(ts.get(key, fallback), vdur - 0.3)

    from moviepy import concatenate_audioclips
    clips = []

    # 1. Title card
    clips.append(try_crossfade(VideoClip(title_frame_fn, duration=5.5).with_fps(FPS)))

    # 2. Dashboard loads → type Afghanistan → select Kunduz
    s2 = max(0.0, T("app_ready") - 1.0)
    e2 = T("province_selected", s2 + 12) + 1.0
    intro_aud = safe_audio("intro")
    dash_aud  = safe_audio("dashboard")
    if intro_aud and dash_aud:
        combo = concatenate_audioclips([intro_aud, dash_aud])
        e2 = max(e2, s2 + combo.duration + 0.3)
    seg2 = rec_clip(s2, e2, vdur, "WORLDWIDE COVERAGE",
                    "190+ countries  ·  type any country, select any region")
    if intro_aud and dash_aud:
        combo = concatenate_audioclips([intro_aud, dash_aud])
        seg2 = seg2.with_audio(combo.subclipped(0, min(combo.duration, seg2.duration)))
    clips.append(seg2)

    # 3. Click Analyse Region (brief)
    s3 = T("province_selected", T("country_selected") + 3)
    e3 = T("analyse_clicked", s3 + 5) + 2.0
    clips.append(rec_clip(s3, e3, vdur, "ANALYSE REGION",
                          "Click once — ZaminAI handles the rest"))

    # 4. Bridge: Computing
    clips.append(bridge_card("Computing Live Satellite Data",
        "Sentinel-2 · Landsat · SAR Radar · CHIRPS · MODIS",
        narration_key="computing",
        min_dur=audio_dur("computing") + 0.4, accent=GOLD))

    # 5. Results appear + scroll (slower for readability)
    s5 = T("results_open", T("analyse_clicked") + 60)
    e5 = T("results_scrolled", s5 + 12) + 1.5
    clips.append(rec_clip(s5, e5, vdur, "FULL SATELLITE ANALYSIS",
                          "NDVI · Water Index · SAR Radar · Rainfall · MODIS Temperature",
                          narration_key="results", playback_speed=0.82))

    # 6. Land cover (slow pan through breakdown)
    s6 = T("results_scrolled", s5 + 8) - 1.0
    clips.append(rec_clip(s6, s6 + audio_dur("landcover") + 1.5, vdur,
                          "LAND COVER BREAKDOWN",
                          "9 classes · Dynamic World V1 · 10 m · exact ha",
                          narration_key="landcover", playback_speed=0.78))

    # 7. Bridge: Detecting
    clips.append(bridge_card("Auto-Detecting Agricultural Fields",
        "Satellite pattern recognition  ·  NDVI health colours",
        narration_key="detecting",
        min_dur=audio_dur("detecting") + 0.4, accent=GREEN))

    # 8. Fields appear
    s8 = T("detect_done", T("detect_clicked", T("map_full") + 5) + 25) - 2.0
    clips.append(rec_clip(s8, s8 + audio_dur("fields") + 1.5, vdur,
                          "AUTO-DETECT FIELDS",
                          "NDVI health colours · green = healthy · red = stressed",
                          narration_key="fields"))

    # 9. Global bridge card
    clips.append(bridge_card("Works Everywhere  ·  Always Free",
        "Afghanistan · Kenya · India · Netherlands · 190+ countries · zaminai.org",
        narration_key="global",
        min_dur=audio_dur("global") + 0.4, accent=GREEN))

    # 10. Closing title
    clips.append(try_crossfade(
        VideoClip(lambda t: title_frame_fn(t, 4.5), duration=4.5).with_fps(FPS)
    ))

    final = concatenate_videoclips(clips, method="compose", padding=-FADE)
    total = sum(c.duration for c in clips) - FADE * (len(clips)-1)
    print(f"\n  Rendering {OUT_FILE}  (~{total:.0f}s)…")
    final.write_videofile(OUT_FILE, fps=FPS, codec="libx264",
                          audio_codec="aac", logger="bar",
                          temp_audiofile="temp_v3.m4a", remove_temp=True)

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    print("=" * 64)
    print("  ZaminAI Officer Dashboard — Professional Demo Video  v3")
    print("=" * 64)

    print("\n[1/3] Recording live browser session…")
    await record_session()

    print("\n[2/3] Generating SSML voice narrations (JennyNeural)…")
    await generate_narrations()
    for k in NARRATIONS:
        p = AUDIO_DIR / f"{k}.mp3"
        if p.exists():
            d = AudioFileClip(str(p)).duration
            print(f"  {k:12s}  {d:.1f}s")

    print("\n[3/3] Assembling final video…")
    build_video()

    print("\n" + "=" * 64)
    print(f"  Done!  ->  {OUT_FILE}")
    import os
    size = os.path.getsize(OUT_FILE) // (1024*1024)
    print(f"  Size: ~{size} MB")
    print("=" * 64)

if __name__ == "__main__":
    # Clear old recording to force fresh capture
    import shutil
    for f in [REC_DIR / "session.webm", TIMESTAMPS_FILE]:
        if f.exists(): f.unlink()
    for f in AUDIO_DIR.glob("*.mp3"):
        f.unlink()
    asyncio.run(main())
