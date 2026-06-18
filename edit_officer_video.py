#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZaminAI Officer Dashboard — Clean Video Edit
Style: cinematic title → sped-up recording + music + text captions → closing card
Voice: one short opening line only (optional)
Total: ~85 seconds
"""
import sys, io as _io, asyncio
sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import io, numpy as np, scipy.io.wavfile as wav, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image as PILImage
from moviepy import VideoFileClip, VideoClip, AudioFileClip, concatenate_videoclips
import edge_tts

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).parent.resolve()
INPUT_VID = Path(r"C:\Users\Eigenaar\Downloads\officer_dashboard.mp4")
OUTPUT    = Path(r"C:\Users\Eigenaar\Downloads\officer_dashboard_FINAL.mp4")
AUDIO_DIR = HERE / "audio_edit"
MUSIC_WAV = HERE / "audio_edit" / "music_v2.wav"
AUDIO_DIR.mkdir(exist_ok=True)

# ── Settings ──────────────────────────────────────────────────────────────────
VIDEO_SPEED = 1.25   # 89.9s → 71.9s — just enough faster to feel polished
TARGET_DUR  = 72.0   # trim to this after speed-up
FPS         = 30

# ── Brand ─────────────────────────────────────────────────────────────────────
BG    = "#080f1e"
GREEN = "#27ae60"
WHITE = "#dde4f0"
DIM   = "#5a7a9a"
DPI   = 96

# ── Text caption schedule (shown on screen only — NO voice reading) ───────────
# (time_in_sped_video, duration, line1, line2)
CAPTIONS = [
    ( 3.0,  8.0, "Select any country",          "Province · District · Village"),
    (18.0,  9.0, "Full satellite analysis",      "NDVI · Water · Rainfall · Radar · Temperature"),
    (35.0,  9.0, "Land cover breakdown",         "9 classes · Dynamic World · exact hectares"),
    (52.0,  8.0, "Satellite layer panel",        "Toggle on demand · live from Earth Engine"),
    (64.0,  8.0, "Auto-detect fields",           "NDVI health colours · green = healthy · red = stressed"),
]

# ── Step 1: Optional short opening voice line ─────────────────────────────────
INTRO_TEXT = "ZaminAI. Satellite intelligence for field officers — free, worldwide."
INTRO_MP3  = AUDIO_DIR / "intro_short.mp3"

async def gen_intro():
    if INTRO_MP3.exists(): return
    c = edge_tts.Communicate(INTRO_TEXT, "en-US-JennyNeural", rate="-5%")
    await c.save(str(INTRO_MP3))

# ── Step 2: Generate background music ────────────────────────────────────────
def gen_music(duration=90, sr=44100):
    if MUSIC_WAV.exists():
        print("  Music cached.")
        return
    print("  Generating background music…")
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Ambient minor pad — A minor chord (A2, C3, E3, A3)
    pad = np.zeros_like(t)
    for f, vol, det in [(110,.35,.0),(130.81,.25,+.15),(164.81,.20,-.10),(220,.15,+.05),(261.63,.08,-.08)]:
        lfo  = 1 + .006 * np.sin(2*np.pi*.12*t + np.random.rand()*6.28)
        pad += vol * np.sin(2*np.pi*(f+det)*t) * lfo

    # Subtle texture — gentle arpeggiated notes (very low volume)
    bpm = 72
    beat = bpm / 60
    arp_notes = [220, 261.63, 329.63, 261.63]   # A3 C4 E4 C4
    arp = np.zeros_like(t)
    note_dur = 1 / beat / 2
    for step in range(int(duration * beat * 2)):
        s = step / (beat * 2)
        e = s + note_dur * .7
        note = arp_notes[step % len(arp_notes)]
        mask = ((t >= s) & (t < e)).astype(float)
        env  = np.where(mask > 0, np.exp(-3 * np.mod(t - s, note_dur)), 0)
        arp += .04 * env * np.sin(2*np.pi*note*t)

    music = pad + arp

    # Fade in 2s / fade out 4s
    fi = int(2*sr); fo = int(4*sr)
    env = np.ones_like(t)
    env[:fi]  = np.linspace(0, 1, fi)
    env[-fo:] = np.linspace(1, 0, fo)
    music *= env

    # Normalise to -18 dBFS (subtle background level)
    peak = np.max(np.abs(music))
    if peak > 0: music = music / peak * 0.12

    wav.write(str(MUSIC_WAV), sr, (music * 32767).astype(np.int16))
    print(f"  Music: {MUSIC_WAV.name}")

# ── Step 3: Build title/closing cards ────────────────────────────────────────
W, H = 1910, 906

def ease(t, dur=1.0):
    x = np.clip(t/max(dur,.001), 0, 1)
    return float(np.clip(x*x*(3-2*x), 0, 1))

def title_card(t, dur, main, sub, is_closing=False):
    a_in  = ease(t, 0.8)
    a_out = 1 - ease(max(0, t-(dur-.7)), .7)
    a     = min(a_in, a_out)

    fig = plt.figure(figsize=(W/DPI, H/DPI), dpi=DPI, facecolor=BG)
    ax  = fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H)
    ax.axis("off"); ax.set_facecolor(BG)

    # Grid
    for x in range(0, W, 56): ax.axvline(x, color=GREEN, alpha=.035, lw=.5)
    for y in range(0, H, 56): ax.axhline(y, color=GREEN, alpha=.035, lw=.5)

    # Animated accent line
    bw = W * .48 * a
    ax.add_patch(plt.Rectangle((W//2 - bw/2, H//2 - (90 if not is_closing else 70)), bw, 3, fc=GREEN, alpha=a))
    ax.plot(W//2 - bw/2 - 15, H//2 - (88.5 if not is_closing else 68.5), 'o',
            color=GREEN, markersize=8, alpha=a)

    # Main text
    ax.text(W//2, H//2 - (38 if not is_closing else 20), main,
            fontsize=76 if not is_closing else 68, fontweight="bold",
            color=WHITE, va="center", ha="center", alpha=a, fontfamily="monospace")
    # Sub text
    ax.text(W//2, H//2 + (28 if not is_closing else 26), sub,
            fontsize=15, color=DIM, va="center", ha="center", alpha=a*.88)

    if is_closing:
        ax.text(W//2, H//2 + 66, "zaminai.org",
                fontsize=13, color=GREEN, va="center", ha="center", alpha=a*.7, fontfamily="monospace")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=BG, bbox_inches="tight", pad_inches=0)
    buf.seek(0); plt.close(fig)
    return np.array(PILImage.open(buf).convert("RGB").resize((W,H), PILImage.LANCZOS))

def add_caption(frame_arr, line1, line2, t_rel, dur):
    """Minimal bottom-left caption — fade in/hold/fade out."""
    a_in  = ease(t_rel, .35)
    a_out = 1 - ease(max(0, t_rel-(dur-.35)), .35)
    a = min(a_in, a_out)
    if a < .02: return frame_arr

    fig = plt.figure(figsize=(W/DPI, H/DPI), dpi=DPI)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H)
    ax.axis("off"); ax.set_facecolor("none")

    # Frosted glass pill — bottom left
    bx, by, bw, bh = 20, 12, max(len(line1),len(line2))*8+40, 50
    ax.add_patch(mpatches.FancyBboxPatch((bx, by), bw, bh,
        boxstyle="round,pad=4", fc=BG, ec=GREEN, lw=1.5, alpha=a*.88, zorder=5))
    ax.add_patch(plt.Rectangle((bx, by+bh-3), bw, 3, fc=GREEN, alpha=a, zorder=6))

    ax.text(bx+14, by+bh*.68, line1, fontsize=13, fontweight="bold",
            color=WHITE, va="center", alpha=a, zorder=7)
    ax.text(bx+14, by+bh*.28, line2, fontsize=10,
            color=DIM, va="center", alpha=a*.85, zorder=7)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, transparent=True,
                bbox_inches="tight", pad_inches=0)
    buf.seek(0); plt.close(fig)
    ov   = PILImage.open(buf).convert("RGBA").resize((W,H), PILImage.LANCZOS)
    base = PILImage.fromarray(frame_arr.astype(np.uint8)).convert("RGBA")
    return np.array(PILImage.alpha_composite(base, ov).convert("RGB"))

def colour_grade(frame):
    """Subtle cool professional grade."""
    f = frame.astype(np.float32)
    f = (f - 128) * 1.05 + 130          # slight contrast lift
    f[:,:,0] = np.clip(f[:,:,0]*.97, 0, 255)   # pull red
    f[:,:,2] = np.clip(f[:,:,2]*1.02, 0, 255)   # push blue
    return np.clip(f, 0, 255).astype(np.uint8)

# ── Step 4: Assemble ──────────────────────────────────────────────────────────
FADE = 0.5

def try_fx(clip):
    try:
        from moviepy.video.fx import CrossFadeIn, CrossFadeOut
        return clip.with_effects([CrossFadeIn(FADE), CrossFadeOut(FADE)])
    except: return clip

def build():
    print("  Loading source video…")
    src = VideoFileClip(str(INPUT_VID))
    src_dur = src.duration
    print(f"  Source: {src.w}x{src.h}  {src_dur:.1f}s → speed {VIDEO_SPEED}x → {src_dur/VIDEO_SPEED:.1f}s")

    # ── CLIP 1: Title card (with brief intro voice if available) ──────────────
    intro_aud = None
    if INTRO_MP3.exists():
        intro_aud = AudioFileClip(str(INTRO_MP3))
    title_dur = max(6.0, (intro_aud.duration + 1.0) if intro_aud else 6.0)

    def title_fn(t): return title_card(t, title_dur, "ZaminAI", "Officer Dashboard  ·  Satellite Intelligence")
    title_clip = VideoClip(title_fn, duration=title_dur).with_fps(FPS)
    if intro_aud:
        title_clip = title_clip.with_audio(intro_aud.subclipped(0, min(intro_aud.duration, title_dur)))
    title_clip = try_fx(title_clip)

    # ── CLIP 2: Main recording — sped up, graded, captions ───────────────────
    sped_dur = min(src_dur / VIDEO_SPEED, TARGET_DUR)

    def main_fn(t):
        src_t = min(t * VIDEO_SPEED, src_dur - 0.05)
        frame = colour_grade(src.get_frame(src_t))

        # Apply active caption
        for (cap_t, cap_dur, l1, l2) in CAPTIONS:
            if cap_t <= t <= cap_t + cap_dur:
                frame = add_caption(frame, l1, l2, t - cap_t, cap_dur)
                break
        return frame

    main_clip = VideoClip(main_fn, duration=sped_dur).with_fps(FPS)

    # Load music and mix with original audio (muted) — just music
    sr_m, m_data = wav.read(str(MUSIC_WAV))
    if m_data.ndim == 2: m_data = m_data.mean(axis=1)
    m_float = m_data.astype(np.float32) / 32767.0

    # Build music array for main clip length
    total_samp = int(sped_dur * sr_m)
    music_arr  = np.zeros(total_samp, dtype=np.float32)
    music_arr[:min(len(m_float), total_samp)] = m_float[:min(len(m_float), total_samp)]

    # Save music clip for main segment
    music_wav_clip = AUDIO_DIR / "main_music.wav"
    wav.write(str(music_wav_clip), sr_m, (music_arr * 32767).astype(np.int16))
    main_clip = main_clip.with_audio(AudioFileClip(str(music_wav_clip)))
    main_clip = try_fx(main_clip)

    # ── CLIP 3: Closing card ──────────────────────────────────────────────────
    close_dur = 5.5
    def close_fn(t): return title_card(t, close_dur, "ZaminAI", "Free  ·  Open  ·  Worldwide", is_closing=True)
    close_clip = VideoClip(close_fn, duration=close_dur).with_fps(FPS)
    close_clip = try_fx(close_clip)

    # ── Assemble ───────────────────────────────────────────────────────────────
    clips = [title_clip, main_clip, close_clip]
    final = concatenate_videoclips(clips, method="compose", padding=-FADE)
    total = sum(c.duration for c in clips) - FADE * (len(clips)-1)
    print(f"\n  Rendering {OUTPUT.name}  (~{total:.0f}s)…")
    final.write_videofile(str(OUTPUT), fps=FPS, codec="libx264",
                          audio_codec="aac", bitrate="5000k", logger="bar",
                          temp_audiofile=str(AUDIO_DIR/"tmp_final.m4a"),
                          remove_temp=True)
    import os
    print(f"\n  Done!  Size: {os.path.getsize(OUTPUT) // (1024*1024)} MB  Duration: ~{total:.0f}s")

async def main():
    print("="*60)
    print("  ZaminAI — Clean Video Edit")
    print("="*60)

    print("\n[1/3] Generating short intro voice…")
    await gen_intro()
    d = AudioFileClip(str(INTRO_MP3)).duration
    print(f"  Intro: {d:.1f}s — \"{INTRO_TEXT[:50]}\"")

    print("\n[2/3] Generating background music…")
    gen_music(duration=90)

    print("\n[3/3] Assembling video…")
    build()

if __name__ == "__main__":
    # Clear old files for a clean run
    for f in [INTRO_MP3, MUSIC_WAV, AUDIO_DIR/"main_music.wav"]:
        if f.exists(): f.unlink()
    asyncio.run(main())
