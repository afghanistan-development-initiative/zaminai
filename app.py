"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ZaminAI API  —  v7.0                                                        ║
║  Satellite Farming Intelligence for Afghan Smallholders                      ║
║                                                                              ║
║  Author : Maiwand Jan Alamzoi                                                ║
║  Org    : Afghanistan Development Initiative (ADI)                           ║
║  Collab : Wageningen University & Research (WUR) + FAO                       ║
║                                                                              ║
║  NEW in v7.0:                                                                ║
║    Supabase database — farmer profiles, saved fields, analysis history       ║
║                                                                              ║
║  Endpoints:                                                                  ║
║    GET  /health          — service status                                    ║
║    POST /analyse         — full field analysis                               ║
║    POST /ask             — AI question answering                             ║
║    POST /ndvi_tile       — NDVI thumbnail URL                                ║
║    POST /crop_detect     — crop type detection                               ║
║    POST /monthly_rain    — monthly rainfall breakdown                        ║
║    POST /soil            — soil analysis                                     ║
║    POST /db/farmer       — register or get farmer by phone                  ║
║    POST /db/field/save   — save a drawn field polygon                       ║
║    GET  /db/fields/<id>  — get all fields for a farmer                      ║
║    POST /db/analysis/save— save analysis result                             ║
║    POST /db/chat/save    — save AI conversation                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, json, math, logging, requests, threading, uuid, base64, time
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
CORS(app, origins="*")

# ── Environment variables ─────────────────────────────────────────────────────
GEMINI_KEY      = os.environ.get("GEMINI_API_KEY", "")
ANTHROPIC_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
GEE_SA          = os.environ.get("GEE_SERVICE_ACCOUNT", "")
GEE_KEY         = os.environ.get("GEE_PRIVATE_KEY", "").replace("\\n", "\n")
SUPABASE_URL    = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY    = os.environ.get("SUPABASE_KEY", "")
TELEGRAM_TOKEN  = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# ── Supabase client ───────────────────────────────────────────────────────────
sb = None
sb_ok = False
try:
    if SUPABASE_URL and SUPABASE_KEY:
        from supabase import create_client
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        sb_ok = True
        log.info("✓ Supabase connected")
    else:
        log.warning("Supabase credentials missing — database disabled")
except Exception as e:
    log.error(f"Supabase init failed: {e}")

# ── Google Earth Engine init ──────────────────────────────────────────────────
gee_ok = False
try:
    import ee
    if GEE_SA and GEE_KEY:
        ee.Initialize(ee.ServiceAccountCredentials(GEE_SA, key_data=GEE_KEY))
        gee_ok = True
        log.info("✓ GEE initialized")
    else:
        log.warning("GEE credentials missing — using regional database")
except Exception as e:
    log.error(f"GEE init failed: {e}")

log.info(f"AI: {'Gemini' if GEMINI_KEY else 'Smart fallback only'}")
log.info(f"DB: {'Supabase connected' if sb_ok else 'disabled'}")

# ════════════════════════════════════════════════════════════════════════════════
# RAG / VECTOR DATABASE  (Supabase pgvector)
# Run /rag/setup to get the one-time SQL migration you need to paste into
# your Supabase SQL editor before the first use.
# ════════════════════════════════════════════════════════════════════════════════

EMBED_MODEL = "gemini-embedding-2"
EMBED_DIM   = 768

rag_ok = False
try:
    if sb_ok:
        sb.table("knowledge_chunks").select("id").limit(1).execute()
        rag_ok = True
        log.info("✓ RAG / pgvector ready")
except Exception:
    log.warning("RAG not ready — call GET /rag/setup for the SQL migration")

def _auto_seed_rag():
    """Seed (or top-up) knowledge base when DB has fewer chunks than the seed list."""
    try:
        res   = sb.table("knowledge_chunks").select("id", count="exact").execute()
        count = res.count or 0
        if count >= len(_RAG_SEED_DOCS) or not GEMINI_KEY:
            return
        log.info(f"RAG has {count} chunks, seed list has {len(_RAG_SEED_DOCS)} — seeding new chunks...")
        # Re-seed all; rag_store is append-only so duplicates may occur for existing chunks,
        # but this is the simplest approach for small seed lists.
        stored = sum(
            1 for doc in _RAG_SEED_DOCS
            if rag_store(doc, source="seed_knowledge",
                         metadata={"type": "domain_knowledge"})
        )
        log.info(f"✓ Auto-seeded {stored}/{len(_RAG_SEED_DOCS)} knowledge chunks")
    except Exception as e:
        log.warning(f"Auto-seed RAG: {e}")


def embed_text(text: str) -> list | None:
    """Embed text using gemini-embedding-2 truncated to 768 dims (fits pgvector limit)."""
    if not GEMINI_KEY or not text.strip():
        return None
    try:
        url = (f"https://generativelanguage.googleapis.com/v1beta"
               f"/models/{EMBED_MODEL}:embedContent?key={GEMINI_KEY}")
        resp = requests.post(url, json={
            "model":            f"models/{EMBED_MODEL}",
            "content":          {"parts": [{"text": text[:8000]}]},
            "outputDimensionality": EMBED_DIM
        }, timeout=15)
        if resp.status_code == 200:
            return resp.json().get("embedding", {}).get("values")
        log.warning(f"embed_text HTTP {resp.status_code}")
    except Exception as e:
        log.warning(f"embed_text: {e}")
    return None


def rag_store(text: str, source: str = "manual", metadata: dict | None = None) -> bool:
    """Embed and store one knowledge chunk. Returns True on success."""
    if not sb_ok or not rag_ok or not GEMINI_KEY:
        return False
    text = text.strip()
    if not text:
        return False
    embedding = embed_text(text)
    if not embedding:
        return False
    try:
        sb.table("knowledge_chunks").insert({
            "content":    text[:4000],
            "embedding":  embedding,
            "source":     source,
            "metadata":   metadata or {},
            "created_at": datetime.utcnow().isoformat()
        }).execute()
        return True
    except Exception as e:
        log.warning(f"rag_store: {e}")
        return False


def _cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb + 1e-10)


def _parse_vec(raw) -> list | None:
    """Parse a pgvector string '[0.1,0.2,...]' or list into a Python list."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            import json
            return json.loads(raw)
        except Exception:
            return None
    return None


def rag_retrieve(question: str, top_k: int = 4, threshold: float = 0.50) -> list[str]:
    """Return top-k most similar knowledge chunks using in-Python cosine similarity."""
    if not sb_ok or not rag_ok or not GEMINI_KEY:
        return []
    q_emb = embed_text(question)
    if not q_emb:
        return []
    try:
        rows = sb.table("knowledge_chunks").select("content,embedding,source").execute().data or []
        scored = []
        for row in rows:
            vec = _parse_vec(row.get("embedding"))
            if not vec:
                continue
            sim = _cosine_similarity(q_emb, vec)
            if sim >= threshold:
                scored.append((sim, row["content"], row.get("source", "")))
        scored.sort(reverse=True)
        # Format as "content [source]" so Gemini can cite the origin
        return [
            f"{c} [{s}]" if s and s not in ("manual", "analysis", "conversation")
            else c
            for _, c, s in scored[:top_k]
        ]
    except Exception as e:
        log.warning(f"rag_retrieve: {e}")
        return []


# ════════════════════════════════════════════════════════════════════════════════
# DATABASE HELPERS
# All Supabase operations wrapped in try/except
# App works fully even if database is down
# ════════════════════════════════════════════════════════════════════════════════

def db_get_or_create_farmer(phone, language="en", province="Afghanistan"):
    """
    Get existing farmer by phone number, or create new one.
    Returns: farmer dict with id, phone, language, province
    """
    if not sb_ok or not phone:
        return None
    try:
        # Check if farmer exists
        res = sb.table("farmers").select("*").eq("phone", phone).execute()
        if res.data:
            # Update last seen
            sb.table("farmers").update({
                "last_seen": datetime.utcnow().isoformat(),
                "language": language
            }).eq("phone", phone).execute()
            log.info(f"✓ Farmer found: {phone}")
            return res.data[0]
        # Create new farmer
        new = sb.table("farmers").insert({
            "phone":    phone,
            "language": language,
            "province": province,
            "joined_at": datetime.utcnow().isoformat(),
            "last_seen": datetime.utcnow().isoformat()
        }).execute()
        log.info(f"✓ New farmer registered: {phone}")
        return new.data[0] if new.data else None
    except Exception as e:
        log.error(f"db_get_or_create_farmer: {e}")
        return None


def db_save_field(farmer_id, coords, label, province, area_ha, area_jereb):
    """
    Save a drawn field polygon to the database.
    Returns: field dict with id
    """
    if not sb_ok or not farmer_id:
        return None
    try:
        res = sb.table("fields").insert({
            "farmer_id":  farmer_id,
            "label":      label or "My Field",
            "coords":     json.dumps(coords),
            "province":   province,
            "area_ha":    area_ha,
            "area_jereb": area_jereb,
            "created_at": datetime.utcnow().isoformat()
        }).execute()
        log.info(f"✓ Field saved for farmer {farmer_id}")
        return res.data[0] if res.data else None
    except Exception as e:
        log.error(f"db_save_field: {e}")
        return None


def db_get_farmer_fields(farmer_id):
    if not sb_ok or not farmer_id:
        return []
    try:
        res = (sb.table("fields")
                 .select("*")
                 .eq("farmer_id", farmer_id)
                 .order("created_at", desc=True)
                 .execute())
        fields = res.data or []
        # Attach latest analysis (all real index values) to each field
        for f in fields:
            try:
                a = (sb.table("analyses")
                       .select("*")
                       .eq("field_id", f["id"])
                       .order("analysed_at", desc=True)
                       .limit(1)
                       .execute())
                f["analyses"] = a.data or []
            except Exception as ae:
                log.error(f"fetch analyses for field {f.get('id')}: {ae}")
                f["analyses"] = []
        return fields
    except Exception as e:
        log.error(f"db_get_farmer_fields: {e}")
        return []


def db_save_analysis(field_id, farmer_id, analysis_data):
    """
    Save satellite analysis result linked to a field.
    Returns: analysis dict with id
    """
    if not sb_ok:
        return None
    try:
        res = sb.table("analyses").insert({
            "field_id":   field_id,
            "farmer_id":  farmer_id,
            "ndvi":       analysis_data.get("ndvi"),
            "evi":        analysis_data.get("evi"),
            "savi":       analysis_data.get("savi"),
            "mndwi":      analysis_data.get("mndwi"),
            "lswi":       analysis_data.get("lswi"),
            "rain":       analysis_data.get("rain"),
            "source":     analysis_data.get("source", "regional_db"),
            "province":   analysis_data.get("province"),
            "area_ha":    analysis_data.get("area_ha"),
            "full_data":  json.dumps(analysis_data),
            "analysed_at": datetime.utcnow().isoformat()
        }).execute()
        log.info(f"✓ Analysis saved for field {field_id}")
        saved = res.data[0] if res.data else None
        # Auto-ingest into RAG (background, non-blocking)
        if rag_ok and saved:
            province = analysis_data.get("province", "")
            chunk = (
                f"Field analysis — {province}: "
                f"NDVI={analysis_data.get('ndvi')}, water={analysis_data.get('mndwi')}, "
                f"rain={analysis_data.get('rain')}mm, area={analysis_data.get('area_jereb')}jereb, "
                f"soil={analysis_data.get('soil', {}).get('texture', '')}, "
                f"date={datetime.utcnow().strftime('%Y-%m-%d')}"
            )
            threading.Thread(
                target=rag_store,
                args=(chunk,),
                kwargs={"source": "analysis",
                        "metadata": {"field_id": str(field_id),
                                     "farmer_id": str(farmer_id),
                                     "province": province}},
                daemon=True
            ).start()
        return saved
    except Exception as e:
        log.error(f"db_save_analysis: {e}")
        return None


def db_get_cached_analysis(field_id, max_age_hours=24):
    """Return the most recent analysis for field_id if within max_age_hours, else None."""
    if not sb_ok or not field_id:
        return None
    try:
        from datetime import timezone
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()
        res = (sb.table("analyses")
                 .select("full_data, analysed_at, source")
                 .eq("field_id", field_id)
                 .gte("analysed_at", cutoff)
                 .order("analysed_at", desc=True)
                 .limit(1)
                 .execute())
        if res.data:
            row = res.data[0]
            cached = json.loads(row["full_data"]) if isinstance(row["full_data"], str) else row["full_data"]
            cached["_cached"]       = True
            cached["_cached_at"]    = row["analysed_at"]
            cached["_cache_source"] = row.get("source", "db")
            log.info(f"Cache hit for field {field_id} (analysed {row['analysed_at'][:16]})")
            return cached
    except Exception as e:
        log.warning(f"db_get_cached_analysis: {e}")
    return None


def db_save_chat(farmer_id, field_id, question, answer, language):
    """
    Save AI conversation to database.
    Builds a valuable Dari/Pashto farming dataset over time.
    """
    if not sb_ok:
        return None
    try:
        sb.table("conversations").insert({
            "farmer_id": farmer_id,
            "field_id":  field_id,
            "question":  question,
            "answer":    answer,
            "language":  language,
            "asked_at":  datetime.utcnow().isoformat()
        }).execute()
        log.info(f"✓ Chat saved — lang:{language}")
        # Auto-ingest Q&A pair into RAG (background, non-blocking)
        if rag_ok and question and answer:
            chunk = f"Q: {question}\nA: {answer}"
            threading.Thread(
                target=rag_store,
                args=(chunk,),
                kwargs={"source": "conversation",
                        "metadata": {"farmer_id": str(farmer_id or ""),
                                     "language": language}},
                daemon=True
            ).start()
    except Exception as e:
        log.error(f"db_save_chat: {e}")


# ── Alert helpers ─────────────────────────────────────────────────────────────
# Requires Supabase table: farmer_alerts
# SQL: CREATE TABLE farmer_alerts (
#   id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
#   farmer_id uuid NOT NULL, field_id uuid,
#   alert_type TEXT NOT NULL,   -- ndvi_low | water_stress | rain_deficit | harvest_window
#   threshold FLOAT, crop TEXT, province TEXT,
#   is_active BOOLEAN DEFAULT true,
#   created_at TIMESTAMP DEFAULT NOW(), last_triggered TIMESTAMP
# );

def db_save_alert(farmer_id, alert_type, threshold=None, crop="", province="Afghanistan", field_id=None):
    if not sb_ok or not farmer_id:
        return None
    try:
        res = sb.table("farmer_alerts").insert({
            "farmer_id":  farmer_id,
            "field_id":   field_id,
            "alert_type": alert_type,
            "threshold":  threshold,
            "crop":       crop,
            "province":   province,
            "is_active":  True,
            "created_at": datetime.utcnow().isoformat()
        }).execute()
        log.info(f"✓ Alert saved: {alert_type} farmer={farmer_id}")
        return res.data[0] if res.data else None
    except Exception as e:
        log.error(f"db_save_alert: {e}")
        return None


def db_get_alerts(farmer_id):
    if not sb_ok or not farmer_id:
        return []
    try:
        res = (sb.table("farmer_alerts").select("*")
               .eq("farmer_id", farmer_id).eq("is_active", True)
               .order("created_at", desc=True).execute())
        return res.data or []
    except Exception as e:
        log.error(f"db_get_alerts: {e}")
        return []


def db_delete_alert(alert_id, farmer_id):
    if not sb_ok:
        return False
    try:
        sb.table("farmer_alerts").update({"is_active": False}) \
          .eq("id", alert_id).eq("farmer_id", farmer_id).execute()
        return True
    except Exception as e:
        log.error(f"db_delete_alert: {e}")
        return False


def check_alerts_fire(alerts, ndvi, mndwi, rain, province, full_data=None):
    """Return alerts that fire given current satellite readings."""
    fired = []
    month = datetime.now().month
    fd    = full_data or {}

    # Seasonal rain helper: expected rain for current month from monthly_rain list
    monthly_rain = fd.get("monthly_rain") or []
    cur_month_rain = monthly_rain[month - 1] if len(monthly_rain) == 12 else None

    for a in alerts:
        atype = a.get("alert_type", "")
        thr   = a.get("threshold")
        try:
            if atype == "ndvi_low" and ndvi is not None and thr is not None and ndvi < thr:
                fired.append({**a, "value": ndvi,
                              "msg": f"NDVI {ndvi} below {thr} — crop stress detected"})

            elif atype == "water_stress" and mndwi is not None and thr is not None and mndwi < thr:
                fired.append({**a, "value": mndwi,
                              "msg": f"Water index {mndwi} below {thr} — irrigate soon"})

            elif atype == "rain_deficit":
                # Improved: compare current month's satellite rain against 60% of
                # expected monthly average for this province and crop type.
                if cur_month_rain is not None and rain is not None:
                    # Estimate this month's actual rain from annual CHIRPS + monthly fraction
                    ptype = get_province_type(province)
                    fracs = MONTHLY_RAIN_FRACTION.get(ptype, MONTHLY_RAIN_FRACTION["central"])
                    expected_month = rain * fracs[month - 1]
                    deficit_threshold = expected_month * 0.60
                    if cur_month_rain < deficit_threshold:
                        pct = round(cur_month_rain / expected_month * 100) if expected_month > 0 else 0
                        fired.append({**a, "value": cur_month_rain,
                                      "msg": (f"Rain deficit: {cur_month_rain:.0f}mm this month "
                                              f"({pct}% of expected {expected_month:.0f}mm) — "
                                              f"crop water stress likely")})
                elif rain is not None and thr is not None and rain < thr:
                    # Fallback: compare annual rain against threshold
                    fired.append({**a, "value": rain,
                                  "msg": f"Annual rainfall {rain}mm below {thr}mm threshold"})

            elif atype == "harvest_window":
                crop  = a.get("crop", "wheat")
                ptype = get_province_type(province)
                cal   = CROP_CALENDAR.get(crop, CROP_CALENDAR.get("wheat", {}))
                zone  = cal.get(ptype, list(cal.values())[0] if cal else {})
                if month in zone.get("harvest", []):
                    fired.append({**a, "value": month,
                                  "msg": f"Harvest window open for {crop} — act now"})

            elif atype == "disease_detected":
                # Fires if the last /diagnose for this field found a disease with
                # severity "severe" or "high" within the past 7 days.
                disease_sev  = fd.get("disease_severity", "")
                disease_name = fd.get("disease_name", "")
                disease_date = fd.get("disease_diagnosed_at", "")
                is_recent    = True
                if disease_date:
                    try:
                        from datetime import timezone
                        diag_dt = datetime.fromisoformat(disease_date.replace("Z", "+00:00"))
                        now_utc = datetime.now(timezone.utc)
                        is_recent = (now_utc - diag_dt).days <= 7
                    except Exception:
                        pass
                if disease_sev in ("severe", "high") and disease_name and is_recent:
                    fired.append({**a, "value": disease_sev,
                                  "msg": (f"Disease alert: {disease_name} detected "
                                          f"(severity: {disease_sev}) — treat immediately")})

        except Exception:
            pass
    return fired


# ════════════════════════════════════════════════════════════════════════════════
# REGIONAL DATABASE  (unchanged from v6.0)
# ════════════════════════════════════════════════════════════════════════════════
PROVINCES = [
    (36.4,37.2,68.2,69.2,"Kunduz",0.33,0.24,0.28,-0.14,-0.09,287,
     {2019:0.40,2020:0.38,2021:0.35,2022:0.22,2023:0.27,2024:0.33,2025:0.35}),
    (36.4,37.1,66.5,67.3,"Balkh",0.31,0.22,0.26,-0.18,-0.12,245,
     {2019:0.37,2020:0.35,2021:0.31,2022:0.19,2023:0.24,2024:0.31,2025:0.33}),
    (33.8,35.0,61.5,63.5,"Herat",0.28,0.19,0.23,-0.20,-0.14,195,
     {2019:0.33,2020:0.31,2021:0.27,2022:0.15,2023:0.21,2024:0.28,2025:0.29}),
    (33.8,34.6,70.0,71.5,"Nangarhar",0.38,0.28,0.32,-0.12,-0.07,320,
     {2019:0.44,2020:0.41,2021:0.37,2022:0.26,2023:0.31,2024:0.38,2025:0.40}),
    (34.2,34.9,68.7,69.5,"Kabul",0.27,0.18,0.22,-0.22,-0.16,305,
     {2019:0.32,2020:0.29,2021:0.25,2022:0.14,2023:0.20,2024:0.27,2025:0.28}),
    (31.3,32.1,65.2,66.2,"Kandahar",0.22,0.14,0.18,-0.28,-0.21,175,
     {2019:0.27,2020:0.24,2021:0.20,2022:0.11,2023:0.16,2024:0.22,2025:0.23}),
    (30.8,32.2,63.5,65.5,"Helmand",0.25,0.16,0.20,-0.25,-0.18,148,
     {2019:0.30,2020:0.27,2021:0.23,2022:0.13,2023:0.18,2024:0.25,2025:0.26}),
    (36.5,38.5,70.0,72.0,"Badakhshan",0.41,0.30,0.35,-0.10,-0.06,420,
     {2019:0.47,2020:0.44,2021:0.40,2022:0.29,2023:0.35,2024:0.41,2025:0.43}),
    (36.4,37.2,69.0,70.5,"Takhar",0.36,0.26,0.30,-0.15,-0.10,340,
     {2019:0.42,2020:0.39,2021:0.35,2022:0.24,2023:0.29,2024:0.36,2025:0.38}),
    (35.8,36.6,68.2,69.2,"Baghlan",0.34,0.25,0.29,-0.16,-0.11,295,
     {2019:0.40,2020:0.37,2021:0.33,2022:0.21,2023:0.27,2024:0.34,2025:0.36}),
    (35.0,36.0,64.0,66.0,"Faryab",0.29,0.20,0.24,-0.19,-0.13,220,
     {2019:0.35,2020:0.32,2021:0.27,2022:0.16,2023:0.22,2024:0.29,2025:0.31}),
    (35.5,36.5,65.5,67.0,"Jawzjan",0.30,0.21,0.25,-0.17,-0.12,240,
     {2019:0.36,2020:0.33,2021:0.28,2022:0.17,2023:0.23,2024:0.30,2025:0.32}),
    (32.0,33.5,67.0,68.5,"Ghazni",0.24,0.15,0.19,-0.21,-0.15,185,
     {2019:0.29,2020:0.26,2021:0.22,2022:0.12,2023:0.18,2024:0.24,2025:0.25}),
    (34.5,35.5,67.0,68.5,"Bamyan",0.27,0.18,0.22,-0.18,-0.13,270,
     {2019:0.32,2020:0.29,2021:0.25,2022:0.14,2023:0.20,2024:0.27,2025:0.28}),
    (33.0,34.0,69.0,70.5,"Logar",0.26,0.17,0.21,-0.20,-0.14,260,
     {2019:0.31,2020:0.28,2021:0.24,2022:0.13,2023:0.19,2024:0.26,2025:0.27}),
    (32.5,33.5,68.0,69.5,"Paktia",0.28,0.19,0.23,-0.18,-0.12,285,
     {2019:0.33,2020:0.30,2021:0.26,2022:0.15,2023:0.21,2024:0.28,2025:0.29}),
]

def get_regional_data(lat, lon):
    for (lat_min,lat_max,lon_min,lon_max,name,ndvi,evi,savi,mndwi,lswi,rain,trend) in PROVINCES:
        if lat_min<=lat<=lat_max and lon_min<=lon<=lon_max:
            return {"province":name,"ndvi":ndvi,"evi":evi,"savi":savi,
                    "mndwi":mndwi,"lswi":lswi,"rain":rain,"trend":trend,"source":"regional_db"}
    return get_climate_zone_fallback(lat, lon)


def get_climate_zone_fallback(lat, lon):
    """Climate-zone based fallback for any global location when GEE is unavailable."""
    alat = abs(lat)
    # Tropical / equatorial
    if alat < 10:
        ndvi,evi,savi,mndwi,rain = 0.62,0.45,0.52,0.12,1800
        zone = "tropical"
    # Sub-tropical savanna / monsoon
    elif alat < 20:
        ndvi,evi,savi,mndwi,rain = 0.42,0.30,0.36,-0.05,820
        zone = "subtropical"
    # Arid / semi-arid (Sahara, Arabian Peninsula, Central Asia, Atacama)
    elif alat < 32:
        ndvi,evi,savi,mndwi,rain = 0.16,0.10,0.14,-0.38,140
        zone = "arid"
    # Mediterranean / semi-arid steppe
    elif alat < 42:
        ndvi,evi,savi,mndwi,rain = 0.34,0.24,0.29,-0.12,420
        zone = "mediterranean"
    # Temperate oceanic / continental
    elif alat < 56:
        ndvi,evi,savi,mndwi,rain = 0.46,0.33,0.39,0.04,660
        zone = "temperate"
    # Boreal / subarctic
    else:
        ndvi,evi,savi,mndwi,rain = 0.28,0.18,0.24,-0.06,380
        zone = "boreal"
    lswi = round(mndwi + 0.05, 4)
    trend = {yr: round(ndvi + (0.02 if yr >= 2022 else 0.04) * (-1 if yr==2022 else 1), 4)
             for yr in range(2019, 2026)}
    return {"ndvi":ndvi,"evi":evi,"savi":savi,"mndwi":mndwi,"lswi":lswi,"rain":rain,
            "trend":trend,"source":f"climate_zone_{zone}"}

# ── Soil, Crop Calendar, Area calc — all unchanged from v6.0 ─────────────────
AFGHAN_SOILS = {
    "Kunduz":{"ph":7.4,"clay":22,"sand":38,"silt":40,"soc":0.9,"texture":"Silty loam"},
    "Balkh":{"ph":7.6,"clay":18,"sand":52,"silt":30,"soc":0.7,"texture":"Sandy loam"},
    "Herat":{"ph":7.8,"clay":15,"sand":58,"silt":27,"soc":0.5,"texture":"Sandy loam"},
    "Nangarhar":{"ph":7.2,"clay":28,"sand":32,"silt":40,"soc":1.2,"texture":"Loam"},
    "Kabul":{"ph":7.5,"clay":20,"sand":42,"silt":38,"soc":0.8,"texture":"Loam"},
    "Kandahar":{"ph":8.0,"clay":12,"sand":65,"silt":23,"soc":0.3,"texture":"Sandy"},
    "Helmand":{"ph":7.9,"clay":14,"sand":60,"silt":26,"soc":0.4,"texture":"Sandy loam"},
    "Badakhshan":{"ph":6.8,"clay":30,"sand":28,"silt":42,"soc":1.8,"texture":"Clay loam"},
    "Takhar":{"ph":7.3,"clay":24,"sand":35,"silt":41,"soc":1.1,"texture":"Silty loam"},
    "Baghlan":{"ph":7.4,"clay":22,"sand":38,"silt":40,"soc":1.0,"texture":"Silty loam"},
    "Faryab":{"ph":7.7,"clay":16,"sand":55,"silt":29,"soc":0.6,"texture":"Sandy loam"},
    "Jawzjan":{"ph":7.6,"clay":17,"sand":53,"silt":30,"soc":0.6,"texture":"Sandy loam"},
    "Ghazni":{"ph":7.5,"clay":20,"sand":44,"silt":36,"soc":0.7,"texture":"Loam"},
    "Bamyan":{"ph":7.1,"clay":26,"sand":32,"silt":42,"soc":1.4,"texture":"Clay loam"},
    "Logar":{"ph":7.4,"clay":23,"sand":36,"silt":41,"soc":1.0,"texture":"Silty loam"},
    "Paktia":{"ph":7.2,"clay":25,"sand":34,"silt":41,"soc":1.2,"texture":"Loam"},
}

def classify_soil_texture(clay,sand,silt):
    if sand>=70: return "Sandy"
    if sand>=50 and clay<20: return "Sandy loam"
    if clay>=40: return "Clay"
    if clay>=27 and clay<40: return "Clay loam"
    if silt>=50 and clay<27: return "Silty loam"
    if silt>=80: return "Silt"
    return "Loam"

def soil_recommendations(texture,ph,soc,province):
    recs=[]
    if ph<6.5: recs.append(f"Acidic soil (pH {ph}) — apply lime 200-300 kg/jereb")
    elif ph>8.0: recs.append(f"Alkaline soil (pH {ph}) — add organic matter")
    elif ph>7.5: recs.append(f"Slightly alkaline (pH {ph}) — use ammonium sulfate over urea")
    else: recs.append(f"Good pH {ph} — suitable for wheat, vegetables, most crops")
    if soc<0.5: recs.append("Very low organic carbon — add 3-4 tonnes compost/jereb")
    elif soc<1.0: recs.append("Low organic carbon — add 2 tonnes compost/jereb annually")
    else: recs.append(f"Organic carbon {soc}% — maintain with annual compost")
    if "Sandy" in texture: recs.append("Sandy soil — use drip irrigation, split fertilizer doses")
    elif "Clay" in texture: recs.append("Clay soil — avoid overwatering, good nutrient retention")
    elif "Loam" in texture: recs.append("Loam soil — best for most crops")
    elif "Silty" in texture: recs.append("Silty soil — fertile but prone to crusting, add compost")
    return recs

def get_soil_data(lat,lon,province="Afghanistan"):
    try:
        props=["phh2o","clay","sand","silt","soc","bdod"]
        prop_str="&".join(f"property={p}" for p in props)
        url=(f"https://rest.soilgrids.org/soilgrids/v2.0/properties/query"
             f"?lon={lon}&lat={lat}&{prop_str}&depth=0-30cm&value=mean")
        resp=requests.get(url,timeout=12,headers={"User-Agent":"ZaminAI/7.0"})
        if resp.status_code==200:
            layers=resp.json().get("properties",{}).get("layers",[])
            vals={}
            for layer in layers:
                name=layer.get("name","")
                v=layer.get("depths",[{}])[0].get("values",{}).get("mean")
                if v is not None: vals[name]=v
            if vals:
                ph=round(vals.get("phh2o",75)/10,1)
                clay=round(vals.get("clay",200)/10,1)
                sand=round(vals.get("sand",400)/10,1)
                silt=round(vals.get("silt",300)/10,1)
                soc=round(vals.get("soc",80)/100,2)
                bd=round(vals.get("bdod",130)/100,2)
                texture=classify_soil_texture(clay,sand,silt)
                return {"ph":ph,"clay":clay,"sand":sand,"silt":silt,"soc":soc,
                        "bulk_density":bd,"texture":texture,
                        "recommendations":soil_recommendations(texture,ph,soc,province),
                        "source":"soilgrids_api","resolution":"250m"}
    except Exception as e:
        log.warning(f"SoilGrids failed: {e}")
    soil=AFGHAN_SOILS.get(province,{"ph":7.5,"clay":20,"sand":45,"silt":35,"soc":0.8,"texture":"Loam"})
    return {"ph":soil["ph"],"clay":soil["clay"],"sand":soil["sand"],"silt":soil["silt"],
            "soc":soil["soc"],"bulk_density":1.35,"texture":soil["texture"],
            "recommendations":soil_recommendations(soil["texture"],soil["ph"],soil["soc"],province),
            "source":"provincial_db","resolution":"province-level"}

CROP_CALENDAR={
    "wheat":{"north":{"plant":[10,11],"harvest":[6,7],"peak_ndvi_month":5},
             "central":{"plant":[10,11],"harvest":[7,8],"peak_ndvi_month":6},
             "south":{"plant":[11,12],"harvest":[4,5],"peak_ndvi_month":3},
             "west":{"plant":[11,12],"harvest":[5,6],"peak_ndvi_month":4},
             "east":{"plant":[10,11],"harvest":[5,6],"peak_ndvi_month":4}},
    "saffron":{"all":{"plant":[9,10],"harvest":[10,11],"peak_ndvi_month":10}},
    "vegetables":{"north":{"plant":[3,4],"harvest":[7,9],"peak_ndvi_month":6},
                  "south":{"plant":[2,3],"harvest":[5,7],"peak_ndvi_month":4},
                  "all":{"plant":[3,4],"harvest":[7,9],"peak_ndvi_month":6}},
}

MONTHLY_RAIN_FRACTION={
    "north":[0.04,0.07,0.14,0.16,0.14,0.08,0.03,0.02,0.03,0.05,0.10,0.14],
    "central":[0.05,0.08,0.15,0.15,0.12,0.06,0.02,0.01,0.02,0.05,0.12,0.17],
    "south":[0.07,0.10,0.16,0.13,0.09,0.04,0.02,0.02,0.03,0.06,0.13,0.15],
    "west":[0.08,0.11,0.16,0.12,0.08,0.04,0.02,0.01,0.02,0.06,0.14,0.16],
    "east":[0.06,0.09,0.14,0.14,0.11,0.07,0.08,0.07,0.04,0.05,0.09,0.06],
}

def get_province_type(province):
    north=["Kunduz","Balkh","Takhar","Baghlan","Faryab","Jawzjan","Badakhshan","Samangan"]
    south=["Kandahar","Helmand","Zabul","Uruzgan","Nimroz","Farah"]
    west=["Herat","Ghor","Badghis"]
    east=["Nangarhar","Kunar","Laghman","Nuristan","Khost","Paktia","Paktika"]
    if province in north: return "north"
    if province in south: return "south"
    if province in west:  return "west"
    if province in east:  return "east"
    return "central"

def get_monthly_rain(annual_rain,province):
    ptype=get_province_type(province)
    factors=MONTHLY_RAIN_FRACTION.get(ptype,MONTHLY_RAIN_FRACTION["central"])
    return [round(annual_rain*f,1) for f in factors]

def get_current_season_advice(province,ndvi,mndwi):
    month=datetime.now().month
    ptype=get_province_type(province)
    advice=[]
    wc=CROP_CALENDAR["wheat"].get(ptype,CROP_CALENDAR["wheat"]["central"])
    if month in wc["plant"]:
        advice.append({"type":"now","crop":"wheat","action":"Plant wheat now — optimal sowing window"})
    elif month in wc["harvest"]:
        advice.append({"type":"now","crop":"wheat","action":"Harvest wheat now — peak maturity window"})
    sc=CROP_CALENDAR["saffron"]["all"]
    if month in sc["plant"]:
        advice.append({"type":"now","crop":"saffron","action":"Plant saffron corms now — only window"})
    elif month in sc["harvest"]:
        advice.append({"type":"now","crop":"saffron","action":"Harvest saffron flowers now — 2-3 week window"})
    if mndwi<-0.15:
        days=2 if mndwi<-0.25 else 4
        advice.append({"type":"urgent","crop":"all","action":f"Irrigate within {days} days — water index low"})
    return advice


# Crop emoji lookup
_CROP_EMOJI = {
    "wheat": "🌾", "saffron": "🌸", "vegetables": "🥦",
    "orchard": "🍎", "bare_fallow": "🏜️",
}

# Stage definitions per crop: (months_list, stage_key, stage_label, color)
_SEASON_STAGES = {
    "wheat": [
        ([10, 11],           "planting",     "Planting Window",    "green"),
        ([12, 1, 2],         "vegetative",   "Vegetative Growth",  "green"),
        ([3, 4],             "heading",      "Jointing & Heading", "green"),
        ([5],                "grain_filling","Grain Filling",      "amber"),
        ([6, 7],             "harvest",      "Harvest Now",        "red"),
        ([8, 9],             "fallow",       "Post-Harvest Fallow","gray"),
    ],
    "saffron": [
        ([9, 10],            "planting",     "Saffron Planting",   "green"),
        ([10, 11],           "harvest",      "Saffron Harvest",    "amber"),
        ([12, 1, 2, 3, 4, 5, 6, 7, 8], "dormant", "Dormant Season", "gray"),
    ],
    "vegetables": [
        ([3, 4],             "planting",     "Planting Season",    "green"),
        ([5, 6],             "growing",      "Active Growing",     "green"),
        ([7, 8, 9],          "harvest",      "Harvest Season",     "amber"),
        ([10, 11, 12, 1, 2], "fallow",       "Off Season",         "gray"),
    ],
    "orchard": [
        ([2, 3],             "flowering",    "Flowering",          "green"),
        ([4, 5, 6, 7],       "fruit_set",    "Fruit Development",  "green"),
        ([8, 9, 10],         "harvest",      "Harvest Season",     "amber"),
        ([11, 12, 1],        "dormant",      "Winter Dormancy",    "gray"),
    ],
}

def get_season_stage(province, crops, month):
    """Returns current crop calendar stage for the sidebar crop band."""
    primary_crop = "wheat"
    if isinstance(crops, list) and crops:
        c0 = crops[0].get("crop", "wheat") if crops else "wheat"
        if c0 in _SEASON_STAGES:
            primary_crop = c0

    stages = _SEASON_STAGES.get(primary_crop, _SEASON_STAGES["wheat"])
    for months_list, stage, stage_label, color in stages:
        if month in months_list:
            harvest_months = sorted(set(m for ml, st, _, _ in stages if st == "harvest" for m in ml))
            days_to_harvest = None
            if stage not in ("harvest", "fallow", "dormant") and harvest_months:
                hm = next((m for m in harvest_months if m > month), harvest_months[0])
                delta = (hm - month) % 12
                days_to_harvest = delta * 30 if delta > 0 else None
            crop_label = primary_crop.replace("_", " ").title()
            days_str = f" · Harvest in ~{days_to_harvest} days" if days_to_harvest else ""
            return {
                "crop": primary_crop,
                "crop_label": crop_label,
                "emoji": _CROP_EMOJI.get(primary_crop, "🌿"),
                "stage": stage,
                "stage_label": stage_label,
                "color": color,
                "days_to_harvest": days_to_harvest,
                "message": f"{crop_label} — {stage_label}{days_str}",
                "month": month,
            }
    crop_label = primary_crop.replace("_", " ").title()
    return {"crop": primary_crop, "crop_label": crop_label, "emoji": "🌿",
            "stage": "growing", "stage_label": "Growing Season", "color": "green",
            "days_to_harvest": None, "message": f"{crop_label} — Growing Season", "month": month}

def detect_crop(ndvi,evi,savi,mndwi,lswi,month,province):
    candidates=[]
    ptype=get_province_type(province)
    if ndvi<0.12:
        candidates.append({"crop":"bare_fallow","label_en":"Bare / Fallow land",
            "label_fa":"زمین خالی / بایر","label_ps":"خالي / بایره ځمکه",
            "confidence":0.90,"reason":f"NDVI {ndvi} < 0.12"})
        return candidates
    if month in range(3,8) and 0.25<=ndvi<=0.60 and evi<0.38:
        conf=0.80 if 0.32<=ndvi<=0.52 else 0.72
        candidates.append({"crop":"wheat","label_en":"Wheat (گندم)",
            "label_fa":"گندم","label_ps":"غنم",
            "confidence":round(min(conf,0.92),2),"reason":f"NDVI {ndvi} wheat signature"})
    if ndvi>=0.38 and evi>=0.28 and lswi>=-0.10:
        candidates.append({"crop":"vegetables","label_en":"Vegetables",
            "label_fa":"سبزیجات","label_ps":"سبزیجات",
            "confidence":0.78,"reason":f"High NDVI+LSWI — vegetables"})
    if ndvi>=0.42 and evi>=0.30:
        candidates.append({"crop":"orchard","label_en":"Orchard / Trees (باغ)",
            "label_fa":"باغ","label_ps":"باغ","confidence":0.72,"reason":"High NDVI — orchard"})
    if not candidates:
        candidates.append({"crop":"mixed_unknown","label_en":"Mixed / Unknown",
            "label_fa":"مختلط","label_ps":"مخلوط","confidence":0.40,"reason":"Unclear signature"})
    return sorted(candidates,key=lambda x:x["confidence"],reverse=True)

def coords_or_bbox(coords, geojson_geometry=None):
    """Return coords if valid (≥3 points), otherwise derive a bounding-box
    rectangle from the raw GeoJSON geometry so analysis never hard-errors
    on GAUL GeometryCollection features."""
    if coords and len(coords) >= 3:
        return coords
    if not geojson_geometry:
        return coords
    try:
        lats, lons = [], []
        def _collect(g):
            t = g.get("type","")
            c = g.get("coordinates",[])
            if t in ("Point",):
                lons.append(c[0]); lats.append(c[1])
            elif t in ("LineString","MultiPoint"):
                for pt in c: lons.append(pt[0]); lats.append(pt[1])
            elif t in ("Polygon","MultiLineString"):
                for ring in c:
                    for pt in ring: lons.append(pt[0]); lats.append(pt[1])
            elif t == "MultiPolygon":
                for poly in c:
                    for ring in poly:
                        for pt in ring: lons.append(pt[0]); lats.append(pt[1])
            elif t == "GeometryCollection":
                for gm in g.get("geometries",[]): _collect(gm)
        _collect(geojson_geometry)
        if lats and lons:
            s,n,w,e = min(lats),max(lats),min(lons),max(lons)
            return [[s,w],[n,w],[n,e],[s,e],[s,w]]
    except Exception:
        pass
    return coords


def calc_area_ha(coords):
    n=len(coords)
    if n<3: return 0.0
    area=0.0; R=6371000
    for i in range(n):
        j=(i+1)%n
        lat1,lon1=math.radians(coords[i][0]),math.radians(coords[i][1])
        lat2,lon2=math.radians(coords[j][0]),math.radians(coords[j][1])
        area+=(lon2-lon1)*(2+math.sin(lat1)+math.sin(lat2))
    return round(abs(area)*R*R/2/10000,2)

def call_gemini(prompt):
    if not GEMINI_KEY: return None
    # gemini-1.5-flash: non-thinking, reliable, good for farming advice
    # gemini-2.0-flash: faster, non-thinking
    # gemini-2.5-flash: thinking model — needs higher token budget
    models = [
        ("gemini-1.5-flash",  {"temperature": 0.6, "maxOutputTokens": 500}),
        ("gemini-2.0-flash",  {"temperature": 0.6, "maxOutputTokens": 500}),
        ("gemini-2.5-flash",  {"temperature": 0.6, "maxOutputTokens": 2000}),
    ]
    for model, gen_cfg in models:
        try:
            url=(f"https://generativelanguage.googleapis.com/v1beta"
                 f"/models/{model}:generateContent?key={GEMINI_KEY}")
            resp=requests.post(url,json={
                "contents":[{"parts":[{"text":prompt}]}],
                "safetySettings":[{"category":c,"threshold":"BLOCK_NONE"} for c in [
                    "HARM_CATEGORY_HARASSMENT","HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT","HARM_CATEGORY_DANGEROUS_CONTENT"]],
                "generationConfig": gen_cfg
            },timeout=20)
            if resp.status_code==200:
                cands=resp.json().get("candidates",[])
                if cands:
                    # Collect all non-thinking text parts (gemini-2.5 splits thinking + answer)
                    parts = cands[0].get("content",{}).get("parts",[])
                    txt = " ".join(
                        p.get("text","") for p in parts
                        if not p.get("thought", False) and p.get("text","")
                    ).strip()
                    if txt and len(txt) > 8:
                        return txt
        except Exception as e:
            log.error(f"Gemini {model}: {e}")
    return None

def smart_fallback(question,ndvi,water,rain,area_j,lang,province="Afghanistan"):
    q=question.lower()
    days=2 if water<-0.20 else 4 if water<-0.10 else 9
    fert=round(area_j*35); cost=round(area_j*400)
    is_irr=any(w in q for w in ["irrigat","water","آبیاری","اوبه","آب"])
    is_crop=any(w in q for w in ["crop","plant","grow","کشت","محصول","وکارم"])
    is_fert=any(w in q for w in ["fertil","urea","dap","کود","سره"])
    if lang=="fa":
        if is_irr:
            return (f"🚨 آبیاری فوری — شاخص آب {water}. زمین را در {days}–{days+2} روز آبیاری کنید. هزینه: ~{cost} افغانی." if water<-0.05
                    else f"آب متوسط است. در ۷–۱۰ روز آبیاری کنید. باران سالانه {rain}mm.")
        if is_crop:
            return (f"با {rain}mm آب کم: ۱) زعفران — ۵۰ برابر گندم سود. ۲) کتان — مقاوم. ۳) نخود — کم‌آب." if rain<200
                    else f"با {rain}mm: ۱) گندم — پایه. ۲) سبزیجات — ۳ برابر سود. ۳) کتان.")
        if is_fert: return f"NDVI {ndvi} — کود نیاز دارید. یوریا: {fert} کیلوگرام. DAP: {round(area_j*20)} کیلوگرام."
        return f"زمین {area_j} جریب — NDVI {ndvi}, آب {water}, باران {rain}mm. سوال خاص؟"
    elif lang=="ps":
        if is_irr:
            return (f"🚨 بیړي اوبه — {water}. {days}–{days+2} ورځو کې اوبه ورکړئ. ~{cost} افغاني." if water<-0.05
                    else f"اوبه متوسط دي. ۷–۱۰ ورځو کې اوبه ورکړئ.")
        if is_crop:
            return (f"د {rain}mm لږو اوبو: ۱) زعفران — ۵۰ ځله ډیره ګټه. ۲) کتان. ۳) نخود." if rain<200
                    else f"د {rain}mm: ۱) گندم. ۲) سبزیجات — ۳ ځله ګټه. ۳) کتان.")
        if is_fert: return f"NDVI {ndvi} — سرې ته اړتیا ده. یوریا: {fert} کیلوګرام."
        return f"ستاسو {area_j} جریب — NDVI {ndvi}, اوبه {water}, باران {rain}mm. پوښتنه؟"
    elif lang=="ar":
        if is_irr:
            return (f"🚨 ري عاجل — مؤشر المياه {water}. اسقِ الأرض خلال {days}–{days+2} أيام. التكلفة: ~{cost} أفغاني." if water<-0.05
                    else f"المياه معتدلة. اسقِ خلال 7–10 أيام. هطول الأمطار: {rain}mm/سنة.")
        if is_crop:
            return (f"مع {rain}mm أمطار قليلة: ١) الزعفران — ٥٠ ضعف أرباح القمح. ٢) الكتان. ٣) الحمص." if rain<200
                    else f"مع {rain}mm: ١) القمح. ٢) الخضروات — ٣ أضعاف الدخل. ٣) الكتان.")
        if is_fert: return f"NDVI {ndvi} — أضف اليوريا: {fert}كجم + DAP: {round(area_j*20)}كجم/جريب."
        return f"أرضك {area_j} جريب — NDVI {ndvi}، مياه {water}، أمطار {rain}mm. سؤال؟"
    elif lang=="ur":
        if is_irr:
            return (f"🚨 فوری آبپاشی — پانی کا اشاریہ {water}۔ {days}–{days+2} دنوں میں آبپاشی کریں۔ لاگت: ~{cost} افغانی۔" if water<-0.05
                    else f"پانی اوسط ہے۔ 7–10 دنوں میں آبپاشی کریں۔ سالانہ بارش: {rain}mm۔")
        if is_crop:
            return (f"{rain}mm کم بارش: 1) زعفران — گندم سے 50 گنا منافع۔ 2) السی۔ 3) چنا۔" if rain<200
                    else f"{rain}mm کے ساتھ: 1) گندم۔ 2) سبزیاں — 3 گنا آمدنی۔ 3) السی۔")
        if is_fert: return f"NDVI {ndvi} — یوریا: {fert}کلو + DAP: {round(area_j*20)}کلو ڈالیں۔"
        return f"آپ کا {area_j} جریب — NDVI {ndvi}، پانی {water}، بارش {rain}mm۔ سوال؟"
    elif lang=="hi":
        if is_irr:
            return (f"🚨 तुरंत सिंचाई — जल सूचकांक {water}। {days}–{days+2} दिनों में सिंचाई करें। लागत: ~{cost} AFN।" if water<-0.05
                    else f"पानी मध्यम है। 7–10 दिनों में सिंचाई करें। वार्षिक वर्षा: {rain}mm।")
        if is_crop:
            return (f"कम वर्षा ({rain}mm): 1) केसर — गेहूं से 50 गुना लाभ। 2) अलसी। 3) चना।" if rain<200
                    else f"{rain}mm वर्षा: 1) गेहूं। 2) सब्जियां — 3 गुना आय। 3) अलसी।")
        if is_fert: return f"NDVI {ndvi} — यूरिया: {fert}किग्रा + DAP: {round(area_j*20)}किग्रा डालें।"
        return f"आपका {area_j} जरीब — NDVI {ndvi}, जल {water}, वर्षा {rain}mm। प्रश्न?"
    elif lang=="bn":
        if is_irr:
            return (f"🚨 জরুরি সেচ — জলের সূচক {water}। {days}–{days+2} দিনের মধ্যে সেচ দিন।" if water<-0.05
                    else f"জল মাঝারি। ৭–১০ দিনের মধ্যে সেচ দিন। বার্ষিক বৃষ্টিপাত: {rain}mm।")
        if is_crop:
            return (f"কম বৃষ্টি ({rain}mm): ১) জাফরান — গমের ৫০ গুণ লাভ। ২) তিসি। ৩) ছোলা।" if rain<200
                    else f"{rain}mm বৃষ্টি: ১) গম। ২) সবজি — ৩ গুণ আয়। ৩) তিসি।")
        if is_fert: return f"NDVI {ndvi} — ইউরিয়া: {fert}কেজি + DAP: {round(area_j*20)}কেজি দিন।"
        return f"আপনার {area_j} জেরিব — NDVI {ndvi}, জল {water}, বৃষ্টি {rain}mm। প্রশ্ন?"
    elif lang=="sw":
        if is_irr:
            return (f"🚨 Umwagiliaji wa haraka — kiwango cha maji {water}. Mwagilia shamba ndani ya siku {days}–{days+2}." if water<-0.05
                    else f"Maji ni wastani. Mwagilia ndani ya siku 7–10. Mvua ya mwaka: {rain}mm.")
        if is_crop:
            return (f"Mvua kidogo ({rain}mm): 1) Zafarani — faida mara 50 ya ngano. 2) Kitani. 3) Dengu." if rain<200
                    else f"Mvua {rain}mm: 1) Ngano. 2) Mboga — faida mara 3. 3) Kitani.")
        if is_fert: return f"NDVI {ndvi} — weka Urea: {fert}kg + DAP: {round(area_j*20)}kg/jerib."
        return f"Shamba lako {area_j} jerib — NDVI {ndvi}, maji {water}, mvua {rain}mm. Swali?"
    elif lang=="es":
        if is_irr:
            return (f"🚨 Riego urgente — índice hídrico {water}. Riegue en {days}–{days+2} días. Costo: ~{cost} AFN." if water<-0.05
                    else f"Agua moderada. Riegue en 7–10 días. Precipitación anual: {rain}mm.")
        if is_crop:
            return (f"Poca agua ({rain}mm): 1) Azafrán — 50× ganancia del trigo. 2) Lino. 3) Garbanzos." if rain<200
                    else f"Con {rain}mm: 1) Trigo. 2) Verduras — 3× ingresos. 3) Lino.")
        if is_fert: return f"NDVI {ndvi} — aplique Urea: {fert}kg + DAP: {round(area_j*20)}kg/jerib."
        return f"Su campo {area_j} jerib — NDVI {ndvi}, agua {water}, lluvia {rain}mm. ¿Pregunta?"
    elif lang=="fr":
        if is_irr:
            return (f"🚨 Irrigation urgente — indice eau {water}. Irriguez dans {days}–{days+2} jours. Coût: ~{cost} AFN." if water<-0.05
                    else f"Eau modérée. Irriguez dans 7–10 jours. Précipitations: {rain}mm/an.")
        if is_crop:
            return (f"Peu d'eau ({rain}mm): 1) Safran — 50× profit du blé. 2) Lin. 3) Pois chiche." if rain<200
                    else f"Avec {rain}mm: 1) Blé. 2) Légumes — 3× revenus. 3) Lin.")
        if is_fert: return f"NDVI {ndvi} — appliquer Urée: {fert}kg + DAP: {round(area_j*20)}kg/jerib."
        return f"Votre champ {area_j} jerib — NDVI {ndvi}, eau {water}, pluie {rain}mm. Question?"
    elif lang=="pt":
        if is_irr:
            return (f"🚨 Irrigação urgente — índice hídrico {water}. Irrigue em {days}–{days+2} dias. Custo: ~{cost} AFN." if water<-0.05
                    else f"Água moderada. Irrigue em 7–10 dias. Precipitação: {rain}mm/ano.")
        if is_crop:
            return (f"Pouca água ({rain}mm): 1) Açafrão — 50× lucro do trigo. 2) Linho. 3) Grão-de-bico." if rain<200
                    else f"Com {rain}mm: 1) Trigo. 2) Vegetais — 3× renda. 3) Linho.")
        if is_fert: return f"NDVI {ndvi} — aplique Ureia: {fert}kg + DAP: {round(area_j*20)}kg/jerib."
        return f"Seu campo {area_j} jerib — NDVI {ndvi}, água {water}, chuva {rain}mm. Pergunta?"
    elif lang=="am":
        if is_irr:
            return (f"🚨 አስቸኳይ መስኖ — የውሃ ጠቋሚ {water}። በ{days}–{days+2} ቀናት ውስጥ ያጠጡ።" if water<-0.05
                    else f"ውሃ መካከለኛ ነው። በ7–10 ቀናት ያጠጡ። የዓመታዊ ዝናብ: {rain}mm።")
        if is_crop:
            return (f"{rain}mm ትንሽ ዝናብ: 1) ኩርኩም — ከስንዴ 50 እጥፍ ትርፍ። 2) ተልባ። 3) ሽምብራ።" if rain<200
                    else f"{rain}mm ዝናብ: 1) ስንዴ። 2) አትክልቶች — 3 እጥፍ ገቢ። 3) ተልባ።")
        if is_fert: return f"NDVI {ndvi} — ዩሪያ: {fert}ኪሎ + DAP: {round(area_j*20)}ኪሎ ይጠቀሙ።"
        return f"እርስዎ {area_j} ጀሪብ — NDVI {ndvi}, ውሃ {water}, ዝናብ {rain}mm። ጥያቄ?"
    else:
        if is_irr:
            return (f"🚨 Irrigate within {days}–{days+2} days — water index {water} is low. Cost: ~{cost:,} AFN." if water<-0.05
                    else f"Water moderate (MNDWI={water}). Irrigate in 7–10 days. Rainfall: {rain}mm/yr.")
        if is_crop:
            return (f"With {rain}mm low water: 1) Saffron — 50× wheat profit. 2) Flax. 3) Chickpeas." if rain<200
                    else f"With {rain}mm: 1) Wheat — reliable. 2) Vegetables — 3× income. 3) Flax.")
        if is_fert: return f"NDVI {ndvi} — apply Urea: {fert}kg + DAP: {round(area_j*20)}kg/jereb."
        return f"Your {area_j} jereb — NDVI {ndvi}, water {water}, rain {rain}mm. What question?"

def gee_analyse(coords, year, clat, clon):
    import ee
    poly     = ee.Geometry.Polygon([[[c[1], c[0]] for c in coords]])
    end_date = min(f"{year}-07-31", datetime.now().strftime("%Y-%m-%d"))

    # ── Sentinel-2 (10 m, 2015+) ─────────────────────────────────────────────
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(poly).filterDate(f"{year}-04-01", end_date)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
          .sort("CLOUDY_PIXEL_PERCENTAGE").limit(5).median().clip(poly))

    def s2m(img, band):
        v = img.reduceRegion(ee.Reducer.mean(), poly, 10, maxPixels=1e8).get(band).getInfo()
        return round(float(v), 4) if v is not None else None

    ndvi  = s2m(s2.normalizedDifference(["B8","B4"]).rename("nd"), "nd")
    evi   = s2m(s2.expression("2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))",
                {"NIR":s2.select("B8"),"RED":s2.select("B4"),"BLUE":s2.select("B2")}).rename("evi"), "evi")
    savi  = s2m(s2.expression("((NIR-RED)/(NIR+RED+0.5))*1.5",
                {"NIR":s2.select("B8"),"RED":s2.select("B4")}).rename("savi"), "savi")
    mndwi = s2m(s2.normalizedDifference(["B3","B11"]).rename("nd"), "nd")
    lswi  = s2m(s2.normalizedDifference(["B8","B11"]).rename("nd"), "nd")
    ndre  = s2m(s2.normalizedDifference(["B8A","B5"]).rename("nd"), "nd")
    bsi   = s2m(s2.expression("((SWIR1+RED)-(NIR+BLUE))/((SWIR1+RED)+(NIR+BLUE))",
                {"SWIR1":s2.select("B11"),"RED":s2.select("B4"),
                 "NIR":s2.select("B8"),"BLUE":s2.select("B2")}).rename("bsi"), "bsi")
    rain  = s2m(ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(poly)
                .filterDate(f"{year}-01-01", f"{year}-12-31")
                .select("precipitation").sum().clip(poly), "precipitation")

    # ── Landsat 8 + 9 (30 m, 2013+) ─────────────────────────────────────────
    # Merged L8/L9 collection → median composite → scale factors applied
    landsat_data = None
    try:
        def lsm(img, band):
            v = img.reduceRegion(ee.Reducer.mean(), poly, 30, maxPixels=1e8).get(band).getInfo()
            return round(float(v), 4) if v is not None else None

        ls_col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                  .filterBounds(poly).filterDate(f"{year}-04-01", end_date)
                  .filter(ee.Filter.lt("CLOUD_COVER", 20))
                  .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
                         .filterBounds(poly).filterDate(f"{year}-04-01", end_date)
                         .filter(ee.Filter.lt("CLOUD_COVER", 20)))
                  .sort("CLOUD_COVER").limit(8))
        ls    = ls_col.median().clip(poly)
        ls_sc = ls.select("SR_B.*").multiply(0.0000275).add(-0.2)

        ls_ndvi  = lsm(ls_sc.normalizedDifference(["SR_B5","SR_B4"]).rename("nd"), "nd")
        ls_evi   = lsm(ls_sc.expression("2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))",
                        {"NIR":ls_sc.select("SR_B5"),"RED":ls_sc.select("SR_B4"),
                         "BLUE":ls_sc.select("SR_B2")}).rename("evi"), "evi")
        ls_mndwi = lsm(ls_sc.normalizedDifference(["SR_B3","SR_B6"]).rename("nd"), "nd")
        ls_lswi  = lsm(ls_sc.normalizedDifference(["SR_B5","SR_B6"]).rename("nd"), "nd")
        ls_savi  = lsm(ls_sc.expression("((NIR-RED)/(NIR+RED+0.5))*1.5",
                        {"NIR":ls_sc.select("SR_B5"),"RED":ls_sc.select("SR_B4")}).rename("savi"), "savi")
        landsat_data = {
            "ndvi": ls_ndvi, "evi": ls_evi, "mndwi": ls_mndwi,
            "lswi": ls_lswi, "savi": ls_savi,
            "source": "landsat_8_9", "resolution_m": 30
        }
        log.info(f"✓ Landsat L8/L9 NDVI={ls_ndvi}")
    except Exception as e:
        log.warning(f"Landsat composite failed: {e}")

    # ── Sentinel-2 NDVI trend (2019 – present) ───────────────────────────────
    s2_trend = {}
    for yr in range(2019, datetime.now().year + 1):
        try:
            yr_end = min(f"{yr}-07-31", datetime.now().strftime("%Y-%m-%d"))
            c2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(poly)
                  .filterDate(f"{yr}-05-01", yr_end)
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 25)).median().clip(poly))
            v  = (c2.normalizedDifference(["B8","B4"])
                  .reduceRegion(ee.Reducer.mean(), poly, 10, maxPixels=1e8).get("nd").getInfo())
            s2_trend[yr] = round(float(v), 4) if v else None
        except:
            s2_trend[yr] = None

    # ── Landsat NDVI trend (2013 – 2018, pre-Sentinel era) ───────────────────
    ls_trend = {}
    for yr in range(2013, 2019):
        try:
            yr_end = min(f"{yr}-07-31", datetime.now().strftime("%Y-%m-%d"))
            lc    = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                     .filterBounds(poly).filterDate(f"{yr}-04-01", yr_end)
                     .filter(ee.Filter.lt("CLOUD_COVER", 25)).median().clip(poly))
            lc_sc = lc.select("SR_B.*").multiply(0.0000275).add(-0.2)
            v     = (lc_sc.normalizedDifference(["SR_B5","SR_B4"])
                     .reduceRegion(ee.Reducer.mean(), poly, 30, maxPixels=1e8).get("nd").getInfo())
            ls_trend[yr] = round(float(v), 4) if v else None
        except:
            ls_trend[yr] = None

    # Combined 2013-present: Landsat fills pre-S2 years, Sentinel-2 from 2019
    combined_trend = {**ls_trend, **s2_trend}

    # ── Sentinel-1 SAR (10 m, cloud-free radar) ──────────────────────────────
    # Works through clouds — critical for Afghanistan monsoon/winter seasons.
    # VV = soil moisture / flood. VH = crop canopy. VH-VV ratio = structure.
    sar_data = None
    try:
        def sarm(img, band):
            v = img.reduceRegion(ee.Reducer.mean(), poly, 10, maxPixels=1e8).get(band).getInfo()
            return round(float(v), 3) if v is not None else None

        s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
              .filterBounds(poly)
              .filterDate(f"{year}-04-01", end_date)
              .filter(ee.Filter.eq("instrumentMode", "IW"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
              .select(["VV", "VH"])
              .median().clip(poly))
        vv = sarm(s1, "VV")   # dB — wetter soil → less negative
        vh = sarm(s1, "VH")   # dB — denser vegetation → less negative
        vh_vv = round(vh - vv, 3) if (vv and vh) else None  # structure ratio
        sar_data = {
            "vv_db":    vv,      # soil moisture proxy (typical: -20 to -5 dB)
            "vh_db":    vh,      # vegetation density proxy
            "vh_vv_db": vh_vv,   # crop structure (more negative = sparser)
            "source":   "sentinel1_SAR_IW",
            "resolution_m": 10,
            "cloud_free": True
        }
        log.info(f"✓ SAR VV={vv} VH={vh}")
    except Exception as e:
        log.warning(f"Sentinel-1 SAR failed: {e}")

    # ── MODIS Land Surface Temperature (1 km, 8-day composite) ───────────────
    # Frost risk detection, heat stress. Scale: pixel × 0.02 − 273.15 → °C
    modis_data = None
    try:
        def modt(col):
            img = col.mean()
            def mv(band):
                v = img.reduceRegion(ee.Reducer.mean(), poly, 1000, maxPixels=1e8).get(band).getInfo()
                return round(float(v) * 0.02 - 273.15, 1) if v is not None else None
            return mv("LST_Day_1km"), mv("LST_Night_1km")

        lst = ee.ImageCollection("MODIS/061/MOD11A2").filterBounds(poly)
        t_sum_d, t_sum_n = modt(lst.filterDate(f"{year}-06-01", f"{year}-08-31"))
        t_win_d, t_win_n = modt(lst.filterDate(f"{year}-01-01", f"{year}-03-31"))
        modis_data = {
            "summer_day_c":  t_sum_d,   # peak heat stress
            "summer_night_c": t_sum_n,
            "winter_day_c":  t_win_d,
            "winter_night_c": t_win_n,  # frost risk if < 0
            "frost_risk":    (t_win_n < 0) if t_win_n is not None else None,
            "source": "modis_MOD11A2"
        }
        log.info(f"✓ MODIS LST summer={t_sum_d}°C winter_night={t_win_n}°C")
    except Exception as e:
        log.warning(f"MODIS LST failed: {e}")

    return {
        "ndvi": ndvi, "evi": evi, "savi": savi, "mndwi": mndwi, "water": mndwi,
        "lswi": lswi, "ndre": ndre, "bsi": bsi, "rain": rain,
        "trend": s2_trend, "ndvi_trend": s2_trend,
        "landsat":       landsat_data,
        "landsat_trend": ls_trend,
        "combined_trend": combined_trend,
        "sar":   sar_data,
        "modis": modis_data,
        "lat": round(clat, 5), "lon": round(clon, 5),
        "source": "gee_live", "image_date": end_date
    }


# ════════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ════════════════════════════════════════════════════════════════════════════════

_HERE = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
@app.route("/index.html")
def serve_index():
    resp = send_from_directory(_HERE, "index.html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resp

@app.route("/officer.html")
def serve_officer():
    resp = send_from_directory(_HERE, "officer.html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resp


@app.route("/health")
def health():
    rag_chunks = 0
    if rag_ok and sb_ok:
        try:
            rag_chunks = sb.table("knowledge_chunks").select("id", count="exact").execute().count or 0
        except Exception:
            pass
    return jsonify({
        "status": "ok", "version": "8.0", "gee": gee_ok,
        "database": sb_ok, "rag": rag_ok, "rag_chunks": rag_chunks,
        "ai": "gemini" if GEMINI_KEY else "smart_only",
        "claude_vision": bool(ANTHROPIC_KEY),
        "satellites": ["sentinel2_10m", "landsat8_9_30m", "sentinel1_SAR_10m", "modis_LST_1km"],
        "indices": ["ndvi","evi","savi","mndwi","lswi","ndre","bsi"],
        "trend_years": "2013–present (Landsat 2013-2018 + Sentinel-2 2019+)",
        "telegram": bool(TELEGRAM_TOKEN),
        "endpoints": [
            "/health", "/analyse", "/ask", "/ndvi_tile",
            "/crop_detect", "/monthly_rain", "/soil", "/diagnose",
            "/db/farmer", "/db/field/save", "/db/fields/<id>",
            "/db/analysis/save", "/db/chat/save",
            "/alerts/save", "/alerts/<farmer_id>", "/alerts/delete",
            "/alerts/check", "/alerts/daily",
            "/telegram/webhook", "/telegram/setup",
            "/rag/setup", "/rag/seed", "/rag/ingest", "/rag/search", "/rag/stats"
        ]
    })


# ── DATABASE ROUTES ───────────────────────────────────────────────────────────

@app.route("/db/farmer", methods=["POST","OPTIONS"])
def db_farmer():
    """Register or retrieve a farmer by phone number."""
    if request.method=="OPTIONS": return jsonify({}),200
    try:
        d        = request.get_json(force=True)
        phone    = d.get("phone","").strip()
        language = d.get("language","en")
        province = d.get("province","Afghanistan")
        if not phone:
            return jsonify({"error":"Phone number required"}),400
        farmer = db_get_or_create_farmer(phone,language,province)
        if not farmer:
            return jsonify({"error":"Database unavailable","db_ok":False}),503
        fields = db_get_farmer_fields(farmer["id"])
        return jsonify({
            "status":  "ok",
            "farmer":  farmer,
            "fields":  fields,
            "field_count": len(fields),
            "is_new":  farmer.get("joined_at","")[:16] == datetime.utcnow().isoformat()[:16]
        })
    except Exception as e:
        log.error(f"/db/farmer: {e}")
        return jsonify({"error":str(e)}),500

@app.route("/db/field/delete", methods=["POST","OPTIONS"])
def db_field_delete():
    """Delete a field and its analyses for a farmer."""
    if request.method=="OPTIONS": return jsonify({}),200
    try:
        d         = request.get_json(force=True)
        field_id  = d.get("field_id")
        farmer_id = d.get("farmer_id")
        if not field_id or not farmer_id:
            return jsonify({"error":"field_id and farmer_id required"}),400
        if not sb_ok:
            return jsonify({"error":"Database unavailable","ok":False}),503
        try:
            sb.table("analyses").delete().eq("field_id", field_id).execute()
        except Exception as ae:
            log.error(f"delete analyses: {ae}")
        sb.table("fields").delete().eq("id", field_id).eq("farmer_id", farmer_id).execute()
        log.info(f"✓ Field {field_id} deleted")
        return jsonify({"ok":True,"status":"deleted"})
    except Exception as e:
        log.error(f"/db/field/delete: {e}")
        return jsonify({"error":str(e),"ok":False}),500
@app.route("/db/field/save", methods=["POST","OPTIONS"])
def db_field_save():
    """Save a drawn field polygon for a farmer."""
    if request.method=="OPTIONS": return jsonify({}),200
    try:
        d          = request.get_json(force=True)
        farmer_id  = d.get("farmer_id")
        coords     = d.get("coords",[])
        label      = d.get("label","My Field")
        province   = d.get("province","Afghanistan")
        area_ha    = d.get("area_ha",0)
        area_jereb = d.get("area_jereb",0)
        if not farmer_id or len(coords)<3:
            return jsonify({"error":"farmer_id and coords required"}),400
        field = db_save_field(farmer_id,coords,label,province,area_ha,area_jereb)
        if not field:
            return jsonify({"error":"Could not save field","db_ok":sb_ok}),503
        return jsonify({"status":"ok","field":field})
    except Exception as e:
        log.error(f"/db/field/save: {e}")
        return jsonify({"error":str(e)}),500


@app.route("/db/fields/<farmer_id>", methods=["GET","OPTIONS"])
def db_fields_get(farmer_id):
    """Get all saved fields for a farmer."""
    if request.method=="OPTIONS": return jsonify({}),200
    try:
        fields = db_get_farmer_fields(farmer_id)
        return jsonify({"status":"ok","fields":fields,"count":len(fields)})
    except Exception as e:
        return jsonify({"error":str(e)}),500


@app.route("/db/analysis/save", methods=["POST","OPTIONS"])
def db_analysis_save():
    """Save a satellite analysis result."""
    if request.method=="OPTIONS": return jsonify({}),200
    try:
        d         = request.get_json(force=True)
        field_id  = d.get("field_id")
        farmer_id = d.get("farmer_id")
        data      = d.get("analysis_data",{})
        result    = db_save_analysis(field_id,farmer_id,data)
        return jsonify({"status":"ok","saved":result is not None})
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route("/db/field/share", methods=["POST","OPTIONS"])
def db_field_share():
    """
    Generate a shareable 8-char token for a field.
    Requires schema: ALTER TABLE fields ADD COLUMN IF NOT EXISTS share_token VARCHAR(8);
    POST body: {field_id, farmer_id}
    Returns: {token, share_url}
    """
    if request.method == "OPTIONS": return jsonify({}), 200
    try:
        d         = request.get_json(force=True)
        field_id  = d.get("field_id")
        farmer_id = d.get("farmer_id")
        if not field_id:
            return jsonify({"error": "field_id required"}), 400
        if not sb_ok:
            return jsonify({"error": "Database not available"}), 503

        # Reuse existing token if present
        existing = (sb.table("fields").select("share_token")
                     .eq("id", field_id).limit(1).execute())
        if existing.data and existing.data[0].get("share_token"):
            token = existing.data[0]["share_token"]
        else:
            # Generate unique 8-char alphanumeric token
            import string, secrets
            alphabet = string.ascii_lowercase + string.digits
            for _ in range(10):
                token = ''.join(secrets.choice(alphabet) for _ in range(8))
                check = (sb.table("fields").select("id")
                           .eq("share_token", token).limit(1).execute())
                if not check.data:
                    break
            sb.table("fields").update({"share_token": token}).eq("id", field_id).execute()

        share_url = f"{request.host_url}field/{token}"
        return jsonify({"ok": True, "token": token, "share_url": share_url})
    except Exception as e:
        log.error(f"/db/field/share: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/field/<token>", methods=["GET"])
def field_share_view(token):
    """
    Read-only public view of a shared field analysis.
    Fetches latest analysis for the field and returns a minimal HTML page.
    """
    if not sb_ok:
        return "<h2>Database unavailable</h2>", 503
    try:
        # Look up field by token
        field_res = (sb.table("fields").select("id, label, province, area_ha, area_jereb, farmer_id")
                       .eq("share_token", token.lower()).limit(1).execute())
        if not field_res.data:
            return "<h2>Field not found or link expired.</h2>", 404

        field     = field_res.data[0]
        field_id  = field["id"]
        label     = field.get("label", "Field")
        province  = field.get("province", "")
        area_ha   = field.get("area_ha", 0)
        area_j    = field.get("area_jereb", 0)

        # Latest analysis
        an_res = (sb.table("analyses").select("full_data, analysed_at, source")
                    .eq("field_id", field_id).order("analysed_at", desc=True).limit(1).execute())
        if not an_res.data:
            return f"<h2>{label}</h2><p>No analysis available yet.</p>", 200

        row      = an_res.data[0]
        an_date  = row["analysed_at"][:10] if row.get("analysed_at") else "unknown"
        src      = row.get("source", "satellite")
        data     = json.loads(row["full_data"]) if isinstance(row.get("full_data"), str) else (row.get("full_data") or {})
        ndvi     = data.get("ndvi", 0)
        mndwi    = data.get("mndwi", 0)
        rain     = data.get("rain", 0)
        advice   = data.get("season", [])
        advice_html = ''.join(
            f'<div style="padding:8px 0;border-bottom:1px solid #e5e7eb"><span style="font-size:16px">'
            f'{"🚨" if a.get("type")=="urgent" else "📅"}</span> {a.get("action","")}</div>'
            for a in (advice or [])
        )
        ndvi_col  = '#16a34a' if ndvi >= 0.30 else '#d97706' if ndvi >= 0.18 else '#dc2626'
        ndvi_lbl  = 'Healthy' if ndvi >= 0.30 else 'Moderate stress' if ndvi >= 0.18 else 'Stressed'

        html_page = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ZaminAI — {label}</title>
<style>
  body{{font-family:-apple-system,sans-serif;margin:0;background:#f9fafb;color:#111}}
  .hd{{background:#0d1117;color:#fff;padding:14px 20px;display:flex;align-items:center;gap:10px}}
  .hd-logo{{font-size:20px;color:#4ade80;font-weight:700;letter-spacing:.04em}}
  .hd-sub{{font-size:11px;color:rgba(255,255,255,.5);margin-top:2px}}
  .card{{background:#fff;border-radius:12px;padding:20px;margin:16px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
  .metric{{display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid #f3f4f6}}
  .metric:last-child{{border:none}}
  .m-label{{color:#6b7280;font-size:13px}}
  .m-val{{font-weight:700;font-size:14px}}
  .badge{{display:inline-block;padding:4px 10px;border-radius:20px;font-size:11px;font-weight:600}}
  .ft{{text-align:center;color:#9ca3af;font-size:11px;padding:20px}}
</style>
</head>
<body>
<div class="hd">
  <div>
    <div class="hd-logo">🌱 ZaminAI</div>
    <div class="hd-sub">Satellite field intelligence · Shared view</div>
  </div>
</div>

<div class="card">
  <h2 style="margin:0 0 4px">{label}</h2>
  <div style="color:#6b7280;font-size:12px">{province} · {area_j} jerib ({area_ha} ha) · Analysed {an_date}</div>
  <div style="margin-top:14px">
    <span style="font-size:42px;font-weight:700;color:{ndvi_col}">{ndvi}</span>
    <span style="font-size:14px;color:{ndvi_col};margin-left:6px">NDVI · {ndvi_lbl}</span>
  </div>
  <div style="background:#f3f4f6;border-radius:6px;height:10px;margin-top:10px">
    <div style="width:{min(100,int(ndvi*100))}%;height:100%;background:{ndvi_col};border-radius:6px"></div>
  </div>
</div>

<div class="card">
  <div style="font-weight:700;margin-bottom:10px">Field Metrics</div>
  <div class="metric"><span class="m-label">Water index (MNDWI)</span><span class="m-val" style="color:{'#0284c7' if mndwi>-0.1 else '#d97706'}">{mndwi}</span></div>
  <div class="metric"><span class="m-label">Annual rainfall</span><span class="m-val">{rain} mm</span></div>
  <div class="metric"><span class="m-label">Data source</span><span class="m-val">{src.replace('_',' ').title()}</span></div>
</div>

{"<div class='card'><div style='font-weight:700;margin-bottom:10px'>Seasonal Advice</div>" + advice_html + "</div>" if advice_html else ""}

<div class="ft">
  Powered by <strong>ZaminAI</strong> · satellite.zaminai.org<br>
  This is a read-only shared view · data as of {an_date}
</div>
</body></html>"""
        return html_page, 200, {"Content-Type": "text/html; charset=utf-8"}
    except Exception as e:
        log.error(f"/field/<token>: {e}")
        return "<h2>Error loading field data.</h2>", 500




@app.route("/db/chat/save", methods=["POST","OPTIONS"])
def db_chat_save():
    """Save an AI conversation."""
    if request.method=="OPTIONS": return jsonify({}),200
    try:
        d         = request.get_json(force=True)
        farmer_id = d.get("farmer_id")
        field_id  = d.get("field_id")
        question  = d.get("question","")
        answer    = d.get("answer","")
        language  = d.get("language","en")
        db_save_chat(farmer_id,field_id,question,answer,language)
        return jsonify({"status":"ok"})
    except Exception as e:
        return jsonify({"error":str(e)}),500


# ── Telegram helpers ─────────────────────────────────────────────────────────

def send_telegram(chat_id, text):
    """Send a message to a Telegram user. Returns True on success."""
    if not TELEGRAM_TOKEN or not chat_id:
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10
        )
        return resp.status_code == 200
    except Exception as e:
        log.error(f"Telegram send failed: {e}")
        return False


def build_alert_message(fired, ndvi, mndwi, rain, province, lang="en"):
    """Build a multilingual alert message for Telegram."""
    now = datetime.now().strftime("%d %b %Y")

    # Per-type translated messages
    _TYPE_FA = {
        "ndvi_low":          "🌿 فشار بر محصول: شاخص NDVI پایین است — آبیاری یا کود دهی لازم است",
        "water_stress":      "💧 کمبود آب: شاخص رطوبت پایین — فوری آبیاری کنید",
        "rain_deficit":      "🌦 کمبود باران: آب کافی برای محصول نیست — آبیاری اضافه کنید",
        "harvest_window":    "🌾 وقت برداشت فرا رسیده — هرچه زودتر اقدام کنید",
        "disease_detected":  "🦠 بیماری شناسایی شد",
    }
    _TYPE_PS = {
        "ndvi_low":          "🌿 د فصل فشار: NDVI ټیټ دی — اوبه ورکول یا سره لپاره لازمه ده",
        "water_stress":      "💧 د اوبو کمښت: سمدستي اوبه ورکړئ",
        "rain_deficit":      "🌦 د باران کمښت: کافي اوبه نشته — اضافي اوبه ورکړئ",
        "harvest_window":    "🌾 د لیو کولو موسم دی — چټک اقدام وکړئ",
        "disease_detected":  "🦠 ناروغي وموندل شوه",
    }

    def _render_alert(a, translations):
        atype = a.get("alert_type", "")
        base  = translations.get(atype, "")
        if atype == "disease_detected":
            val   = a.get("value", "")
            msg_e = a.get("msg", "")
            disease_part = msg_e.split("Disease alert:")[-1].split("—")[0].strip() if "Disease alert:" in msg_e else msg_e
            return f"🚨 {base}: {disease_part}" if base else f"🚨 {msg_e}"
        if base:
            return f"🚨 {base}"
        return f"🚨 {a.get('msg', '')}"

    if lang == "fa":
        lines = [f"🌾 <b>ZaminAI هشدار</b>", f"📍 {province}  ·  {now}"]
        for a in fired:
            lines.append(_render_alert(a, _TYPE_FA))
        lines.append(f"\n📊 NDVI: {ndvi}  |  آب: {mndwi}  |  باران: {rain}mm")
        lines.append("🌐 zaminai.org")
    elif lang == "ps":
        lines = [f"🌾 <b>ZaminAI خبرداری</b>", f"📍 {province}  ·  {now}"]
        for a in fired:
            lines.append(_render_alert(a, _TYPE_PS))
        lines.append(f"\n📊 NDVI: {ndvi}  |  اوبه: {mndwi}  |  باران: {rain}mm")
        lines.append("🌐 zaminai.org")
    else:
        lines = [f"🌾 <b>ZaminAI Alert</b>", f"📍 {province}  ·  {now}"]
        for a in fired:
            lines.append(f"🚨 {a.get('msg', '')}")
        lines.append(f"\n📊 NDVI: {ndvi}  |  Water: {mndwi}  |  Rain: {rain}mm")
        lines.append("🌐 zaminai.org")
    return "\n".join(lines)


def db_link_telegram(phone, chat_id):
    """Link a farmer's phone number to their Telegram chat_id."""
    if not sb_ok:
        return None
    try:
        res = sb.table("farmers").select("*").eq("phone", phone).execute()
        if not res.data:
            return None
        sb.table("farmers").update({
            "telegram_chat_id": str(chat_id),
            "last_seen": datetime.utcnow().isoformat()
        }).eq("phone", phone).execute()
        log.info(f"✓ Telegram linked: {phone} → {chat_id}")
        return res.data[0]
    except Exception as e:
        log.error(f"db_link_telegram: {e}")
        return None


def run_daily_alerts():
    """
    Check every farmer's saved alerts against their latest analysis.
    Sends Telegram notifications for any that fire.
    Called by POST /alerts/daily.
    """
    if not sb_ok or not TELEGRAM_TOKEN:
        return {"sent": 0, "reason": "db or telegram not configured"}
    try:
        farmers_res = (sb.table("farmers").select("*")
                       .not_.is_("telegram_chat_id", "null").execute())
        farmers = farmers_res.data or []
        sent = skipped = 0

        for farmer in farmers:
            chat_id   = farmer.get("telegram_chat_id")
            farmer_id = farmer.get("id")
            lang      = farmer.get("language", "en")
            if not chat_id or not farmer_id:
                continue

            # Latest analysis for this farmer
            a_res = (sb.table("analyses").select("*")
                     .eq("farmer_id", farmer_id)
                     .order("analysed_at", desc=True).limit(1).execute())
            if not a_res.data:
                skipped += 1
                continue

            latest   = a_res.data[0]
            ndvi     = latest.get("ndvi")
            mndwi    = latest.get("mndwi")
            rain     = latest.get("rain")
            province = latest.get("province", "Afghanistan")
            try:
                full_data = json.loads(latest["full_data"]) if isinstance(latest.get("full_data"), str) else (latest.get("full_data") or {})
            except Exception:
                full_data = {}

            alerts = db_get_alerts(farmer_id)
            fired  = check_alerts_fire(alerts, ndvi, mndwi, rain, province, full_data=full_data)

            if fired:
                msg = build_alert_message(fired, ndvi, mndwi, rain, province, lang)
                if send_telegram(chat_id, msg):
                    sent += 1
                    for a in fired:
                        try:
                            sb.table("farmer_alerts").update({
                                "last_triggered": datetime.utcnow().isoformat()
                            }).eq("id", a["id"]).execute()
                        except:
                            pass
            else:
                skipped += 1

        log.info(f"Daily alerts: {sent} sent, {skipped} skipped")
        return {"sent": sent, "skipped": skipped, "total_farmers": len(farmers)}
    except Exception as e:
        log.error(f"run_daily_alerts: {e}")
        return {"error": str(e)}


# ── ALERT ROUTES ─────────────────────────────────────────────────────────────

@app.route("/alerts/save", methods=["POST","OPTIONS"])
def alerts_save():
    if request.method == "OPTIONS": return jsonify({}), 200
    try:
        d          = request.get_json(force=True)
        farmer_id  = d.get("farmer_id")
        alert_type = d.get("alert_type")
        if not farmer_id or not alert_type:
            return jsonify({"error": "farmer_id and alert_type required"}), 400
        result = db_save_alert(
            farmer_id, alert_type,
            threshold = d.get("threshold"),
            crop      = d.get("crop", ""),
            province  = d.get("province", "Afghanistan"),
            field_id  = d.get("field_id")
        )
        if not result:
            return jsonify({"error": "Could not save alert — run SQL migration first", "ok": False}), 503
        return jsonify({"status": "ok", "alert": result})
    except Exception as e:
        log.error(f"/alerts/save: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/alerts/<farmer_id>", methods=["GET","OPTIONS"])
def alerts_get(farmer_id):
    if request.method == "OPTIONS": return jsonify({}), 200
    try:
        alerts = db_get_alerts(farmer_id)
        return jsonify({"status": "ok", "alerts": alerts, "count": len(alerts)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/alerts/delete", methods=["POST","OPTIONS"])
def alerts_delete():
    if request.method == "OPTIONS": return jsonify({}), 200
    try:
        d         = request.get_json(force=True)
        alert_id  = d.get("alert_id")
        farmer_id = d.get("farmer_id")
        if not alert_id or not farmer_id:
            return jsonify({"error": "alert_id and farmer_id required"}), 400
        ok = db_delete_alert(alert_id, farmer_id)
        return jsonify({"status": "ok" if ok else "error", "ok": ok})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/alerts/check", methods=["POST","OPTIONS"])
def alerts_check():
    """Check which of a farmer's alerts fire against current satellite readings."""
    if request.method == "OPTIONS": return jsonify({}), 200
    try:
        d         = request.get_json(force=True)
        farmer_id = d.get("farmer_id")
        ndvi      = d.get("ndvi")
        mndwi     = d.get("mndwi")
        rain      = d.get("rain")
        province  = d.get("province", "Afghanistan")
        alerts    = db_get_alerts(farmer_id)
        fired     = check_alerts_fire(alerts, ndvi, mndwi, rain, province)
        return jsonify({"status": "ok", "fired": fired, "total": len(alerts), "fired_count": len(fired)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── TELEGRAM ROUTES ──────────────────────────────────────────────────────────

@app.route("/telegram/webhook", methods=["POST"])
def telegram_webhook():
    """
    Receives messages from Telegram users.
    Farmer registration flow:
      1. Farmer opens t.me/ZaminAIBot → sends /start
      2. Bot asks for phone number
      3. Farmer sends phone → bot links account → farmer receives alerts
    """
    try:
        data    = request.get_json(force=True) or {}
        msg     = data.get("message", {})
        chat_id = str(msg.get("chat", {}).get("id", ""))
        text    = msg.get("text", "").strip()
        if not chat_id or not text:
            return jsonify({}), 200

        if text == "/start":
            send_telegram(chat_id,
                "🌾 <b>ZaminAI</b> — Satellite Farming Intelligence\n\n"
                "Send your phone number to link your account and receive field alerts.\n\n"
                "📱 Example: +93701234567\n\n"
                "🌐 zaminai.org")

        elif text == "/stop":
            # Unlink telegram from farmer account
            if sb_ok:
                try:
                    sb.table("farmers").update({"telegram_chat_id": None}) \
                      .eq("telegram_chat_id", chat_id).execute()
                except:
                    pass
            send_telegram(chat_id, "✅ Alerts stopped. Send /start to re-enable.")

        elif text == "/status":
            # Show farmer's current field status
            if sb_ok:
                try:
                    f_res = sb.table("farmers").select("*").eq("telegram_chat_id", chat_id).execute()
                    if f_res.data:
                        farmer = f_res.data[0]
                        a_res  = (sb.table("analyses").select("*")
                                  .eq("farmer_id", farmer["id"])
                                  .order("analysed_at", desc=True).limit(1).execute())
                        if a_res.data:
                            a = a_res.data[0]
                            send_telegram(chat_id,
                                f"🌾 <b>Your latest field analysis</b>\n"
                                f"📍 {a.get('province','Afghanistan')}\n"
                                f"📊 NDVI: {a.get('ndvi','—')}  |  Water: {a.get('mndwi','—')}\n"
                                f"🌧️ Rain: {a.get('rain','—')}mm\n"
                                f"📅 {str(a.get('analysed_at',''))[:10]}\n\n"
                                f"🌐 zaminai.org")
                        else:
                            send_telegram(chat_id, "No analysis yet. Open zaminai.org to analyse your field.")
                    else:
                        send_telegram(chat_id, "Account not linked. Send your phone number first.")
                except Exception as e:
                    send_telegram(chat_id, "Could not fetch status. Try again later.")

        else:
            # Try to link phone number
            phone = text.replace(" ","").replace("-","")
            if phone.startswith("+") or phone.isdigit():
                farmer = db_link_telegram(phone, chat_id)
                if farmer:
                    lang = farmer.get("language","en")
                    if lang == "fa":
                        send_telegram(chat_id,
                            f"✅ <b>حساب متصل شد!</b>\n\n"
                            f"📱 شماره: {phone}\n"
                            f"📍 {farmer.get('province','Afghanistan')}\n\n"
                            f"از این پس هشدارهای زمین خود را اینجا دریافت می‌کنید.\n"
                            f"برای دیدن وضعیت زمین: /status\n"
                            f"🌐 zaminai.org")
                    elif lang == "ps":
                        send_telegram(chat_id,
                            f"✅ <b>حساب وصل شو!</b>\n\n"
                            f"📱 شمیره: {phone}\n"
                            f"📍 {farmer.get('province','Afghanistan')}\n\n"
                            f"له دې وروسته به دلته د ځمکې خبرداریونه ترلاسه کوئ.\n"
                            f"د ځمکې وضعیت: /status\n"
                            f"🌐 zaminai.org")
                    else:
                        send_telegram(chat_id,
                            f"✅ <b>Account linked!</b>\n\n"
                            f"📱 Phone: {phone}\n"
                            f"📍 {farmer.get('province','Afghanistan')}\n\n"
                            f"You will now receive ZaminAI field alerts here.\n"
                            f"Check field status: /status\n"
                            f"🌐 zaminai.org")
                else:
                    send_telegram(chat_id,
                        "⚠️ Phone number not found.\n\n"
                        "Please register first at zaminai.org, then send your number here.")
            else:
                send_telegram(chat_id,
                    "🌾 Send your phone number to link your account.\n"
                    "Example: +93701234567\n\n"
                    "Commands:\n/status — latest field data\n/stop — stop alerts")

    except Exception as e:
        log.error(f"/telegram/webhook: {e}")
    return jsonify({}), 200


@app.route("/telegram/setup", methods=["GET"])
def telegram_setup():
    """Register the webhook URL with Telegram. Call once after deployment."""
    if not TELEGRAM_TOKEN:
        return jsonify({"error": "TELEGRAM_BOT_TOKEN not set"}), 400
    webhook_url = f"https://zaminai.onrender.com/telegram/webhook"
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook",
            json={"url": webhook_url, "allowed_updates": ["message"]},
            timeout=10
        )
        result = resp.json()
        log.info(f"Telegram webhook set: {result}")
        return jsonify({"status": "ok", "webhook": webhook_url, "telegram": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/alerts/daily", methods=["POST", "GET"])
def alerts_daily():
    """
    Run daily alert checks for all farmers with Telegram linked.
    Call this once per day from a cron job or scheduler.
    """
    result = run_daily_alerts()
    return jsonify({"status": "ok", **result})


# ── ANALYSIS ROUTE (unchanged logic, added db save) ───────────────────────────

# Async task store for farmer field analysis (same pattern as officer/analyse)
_farmer_analyse_tasks = {}

def get_weather_forecast(lat, lon):
    """
    Fetch 7-day daily forecast from Open-Meteo (free, no API key).
    Returns list of {date, rain_mm, temp_max, temp_min, condition}.
    Falls back to empty list on any error.
    """
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&daily=precipitation_sum,temperature_2m_max,temperature_2m_min,weathercode"
            "&forecast_days=7&timezone=auto"
        )
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return []
        d = r.json().get("daily", {})
        WMO = {
            0:"Clear", 1:"Mainly clear", 2:"Partly cloudy", 3:"Overcast",
            45:"Fog", 48:"Icy fog",
            51:"Light drizzle", 53:"Drizzle", 55:"Heavy drizzle",
            61:"Light rain", 63:"Rain", 65:"Heavy rain",
            71:"Light snow", 73:"Snow", 75:"Heavy snow",
            80:"Rain showers", 81:"Rain showers", 82:"Violent showers",
            85:"Snow showers", 86:"Heavy snow showers",
            95:"Thunderstorm", 96:"Thunderstorm+hail", 99:"Thunderstorm+hail",
        }
        forecast = []
        dates     = d.get("time", [])
        rain      = d.get("precipitation_sum", [])
        tmax      = d.get("temperature_2m_max", [])
        tmin      = d.get("temperature_2m_min", [])
        wcode     = d.get("weathercode", [])
        for i in range(len(dates)):
            forecast.append({
                "date":      dates[i],
                "rain_mm":   rain[i]  if i < len(rain)  else 0,
                "temp_max":  tmax[i]  if i < len(tmax)  else None,
                "temp_min":  tmin[i]  if i < len(tmin)  else None,
                "condition": WMO.get(wcode[i] if i < len(wcode) else 0, "Unknown"),
            })
        return forecast
    except Exception as e:
        log.warning(f"weather forecast failed: {e}")
        return []


@app.route("/analyse", methods=["POST","OPTIONS"])
def analyse():
    """Async farmer field analysis. Returns {task_id} immediately;
    poll GET /analyse-result/<task_id> for the satellite result.
    Regional fallback is instant — live GEE runs in background.
    """
    if request.method=="OPTIONS": return jsonify({}),200
    try:
        data      = request.get_json(force=True)
        coords    = data.get("coords",[])
        year      = int(data.get("year",datetime.now().year))
        label     = data.get("label","Field")
        farmer_id = data.get("farmer_id")
        field_id  = data.get("field_id")
        if len(coords)<3:
            return jsonify({"error":"Need ≥3 coordinate points"}),400

        lats=[c[0] for c in coords]; lons=[c[1] for c in coords]
        clat=sum(lats)/len(lats); clon=sum(lons)/len(lons)
        area_ha=calc_area_ha(coords); area_jereb=round(area_ha*5,1)
        month=datetime.now().month
        task_id = str(uuid.uuid4())
        _farmer_analyse_tasks[task_id] = {"status":"pending"}

        def _worker():
            try:
                result = {}
                # Check 24-hour cache first — skip GEE if fresh result exists
                if field_id:
                    cached = db_get_cached_analysis(field_id, max_age_hours=24)
                    if cached:
                        cached.update({"label": label, "area_ha": area_ha, "area_jereb": area_jereb})
                        _farmer_analyse_tasks[task_id] = {"status": "done", "data": cached}
                        return
                if gee_ok:
                    gee_exc = None
                    for _attempt in range(3):
                        try:
                            result = gee_analyse(coords, year, clat, clon)
                            reg = get_regional_data(clat, clon)
                            result.update({"label":label,"area_ha":area_ha,"area_jereb":area_jereb,
                                           "status":"success","province":reg["province"],
                                           "data_source": result.get("source","gee_satellite")})
                            result["crops"] = detect_crop(result["ndvi"],result["evi"],result["savi"],
                                result["mndwi"],result["lswi"],month,reg["province"])
                            result["season"] = get_current_season_advice(reg["province"],result["ndvi"],result["mndwi"])
                            result["season_stage"] = get_season_stage(reg["province"],result.get("crops",[]),month)
                            result["monthly_rain"] = get_monthly_rain(result["rain"] or reg["rain"],reg["province"])
                            result["soil"] = get_soil_data(clat,clon,reg["province"])
                            result["weather_forecast"] = get_weather_forecast(clat, clon)
                            if result.get("trend"):
                                tv=[v for v in result["trend"].values() if v]
                                if tv:
                                    h_min=min(tv); h_max=max(tv); cur=result["ndvi"] or 0
                                    result["vci"] = round((cur-h_min)/(h_max-h_min+0.001)*100,1) if h_max>h_min else None
                            log.info(f"GEE ok (attempt {_attempt+1}) — source: {result.get('data_source')}")
                            if farmer_id and field_id:
                                db_save_analysis(field_id, farmer_id, result)
                            _farmer_analyse_tasks[task_id] = {"status":"done","data":result}
                            return
                        except Exception as e:
                            gee_exc = e
                            if _attempt < 2:
                                wait = 2 ** _attempt
                                log.warning(f"GEE attempt {_attempt+1}/3 failed: {e} — retrying in {wait}s")
                                time.sleep(wait)
                            else:
                                log.error(f"GEE failed after 3 attempts: {e}")

                # Regional fallback
                reg = get_regional_data(clat, clon)
                result = {
                    "label":label,"status":"success","source":reg["source"],
                    "data_source":reg["source"],
                    "province":reg["province"],"ndvi":reg["ndvi"],"evi":reg["evi"],
                    "savi":reg["savi"],"mndwi":reg["mndwi"],"water":reg["mndwi"],
                    "lswi":reg["lswi"],"rain":reg["rain"],"area_ha":area_ha,
                    "area_jereb":area_jereb,"trend":reg["trend"],"ndvi_trend":reg["trend"],
                    "year":year,"lat":round(clat,5),"lon":round(clon,5),
                    "latest_date":f"{year}-05-15",
                    "crops":detect_crop(reg["ndvi"],reg["evi"],reg["savi"],reg["mndwi"],reg["lswi"],month,reg["province"]),
                    "season":get_current_season_advice(reg["province"],reg["ndvi"],reg["mndwi"]),
                    "season_stage":get_season_stage(reg["province"],[],month),
                    "monthly_rain":get_monthly_rain(reg["rain"],reg["province"]),
                    "soil":get_soil_data(clat,clon,reg["province"]),
                    "weather_forecast":get_weather_forecast(clat,clon),
                    "ndre":round(reg["ndvi"]*0.75,4),"gndvi":round(reg["ndvi"]*0.88,4),
                    "ndmi":round(reg["mndwi"]+0.08,4),"ndwi":round(reg["mndwi"]+0.05,4),
                    "vci":None,"drought_index":round(reg["mndwi"]-reg["ndvi"],4),
                }
                if farmer_id and field_id:
                    db_save_analysis(field_id, farmer_id, result)
                _farmer_analyse_tasks[task_id] = {"status":"done","data":result}
            except Exception as ex:
                log.error(f"/analyse worker: {ex}")
                _farmer_analyse_tasks[task_id] = {"status":"error","error":str(ex)}

        threading.Thread(target=_worker, daemon=True).start()
        return jsonify({"task_id": task_id, "status": "pending"})
    except Exception as e:
        log.error(f"/analyse: {e}"); return jsonify({"error":str(e)}),500


@app.route("/analyse-result/<task_id>", methods=["GET","OPTIONS"])
def analyse_result(task_id):
    """Poll for farmer field analysis result."""
    if request.method == "OPTIONS": return jsonify({}), 200
    task = _farmer_analyse_tasks.get(task_id)
    if not task: return jsonify({"status":"error","error":"Task not found"}), 404
    if task["status"] == "pending": return jsonify({"status":"pending"})
    if task["status"] == "error":
        _farmer_analyse_tasks.pop(task_id, None)
        return jsonify({"status":"error","error":task["error"]}), 500
    data = task.get("data", {})
    _farmer_analyse_tasks.pop(task_id, None)
    return jsonify({"status":"done","data":data})


@app.route("/ask", methods=["POST","OPTIONS"])
def ask():
    if request.method=="OPTIONS": return jsonify({}),200
    try:
        data     =request.get_json(force=True)
        question =data.get("question","")
        language =data.get("language","en")
        context  =data.get("context","")
        fd       =data.get("field_data",{})
        farmer_id=data.get("farmer_id")
        field_id =data.get("field_id")
        if not question: return jsonify({"error":"No question"}),400
        ndvi=0.28; water=-0.19; rain=240; area_j=5.0; province="Afghanistan"
        if isinstance(fd,dict) and fd:
            ndvi=float(fd.get("ndvi",0.28)); water=float(fd.get("mndwi",fd.get("water",-0.19)))
            rain=float(fd.get("rain",240)); area_j=float(fd.get("area_jereb",fd.get("area_ha",1)*5))
            province=fd.get("province","Afghanistan")
            context=(f"Field: NDVI={ndvi}, Water={water}, Rain={rain}mm, Area={area_j}jereb, Province={province}")
        _LANG_INST_ASK = {
            "fa": "Afghan Dari (دری). Use دهقان for farmer, جریب for land. Eastern Arabic numerals ۱۲۳.",
            "ps": "Pashto (پښتو). Proper Pashto farming terms. Eastern Arabic numerals.",
            "ar": "Arabic (العربية). Use right-to-left text. Clear farming terms.",
            "ur": "Urdu (اردو). Right-to-left. Simple farming language.",
            "hi": "Hindi (हिंदी). Simple farming terms a village farmer understands.",
            "bn": "Bengali (বাংলা). Simple farming language.",
            "sw": "Swahili (Kiswahili). Practical farming advice.",
            "es": "Spanish (Español). Clear, practical farming language.",
            "fr": "French (Français). Simple farming terms.",
            "pt": "Portuguese (Português). Practical farming advice.",
            "am": "Amharic (አማርኛ). Simple farming language.",
            "en": "English. Concise and specific.",
        }
        lang_inst = _LANG_INST_ASK.get(language, "English. Concise and specific.")
        # Retrieve relevant knowledge chunks from vector DB
        rag_chunks = rag_retrieve(question, top_k=4, threshold=0.50)
        rag_section = (
            "\n\nVerified agronomic knowledge (WUR/FAO/ICARDA/FEWS NET):\n" +
            "\n\n".join(f"• {c}" for c in rag_chunks)
        ) if rag_chunks else ""
        word_limit = "under 120 words" if rag_chunks else "under 90 words"
        _location_label = province if province and province != "Afghanistan" else "the farmer's region"
        prompt=(f"You are ZaminAI, expert agricultural advisor for smallholder farmers in {_location_label}.\n"
                f"Satellite data: {context}{rag_section}\n\nRespond ONLY in {lang_inst}\n"
                f"Rules: use the verified knowledge above, give exact amounts in local units (hectares or local equivalent), {word_limit}, "
                f"speak as trusted local expert.\n\nQuestion: {question}")
        reply=call_gemini(prompt)
        if not reply or len(reply)<8:
            reply=smart_fallback(question,ndvi,water,rain,area_j,language,province)
            model="smart"
        else:
            model="gemini"
        # Save conversation to database
        if farmer_id:
            db_save_chat(farmer_id,field_id,question,reply,language)
        return jsonify({"reply":reply,"answer":reply,"model":model})
    except Exception as e:
        log.error(f"/ask: {e}"); return jsonify({"error":str(e)}),500


@app.route("/crop_detect",methods=["POST","OPTIONS"])
def crop_detect():
    if request.method=="OPTIONS": return jsonify({}),200
    try:
        d=request.get_json(force=True)
        crops=detect_crop(float(d.get("ndvi",0)),float(d.get("evi",0)),float(d.get("savi",0)),
            float(d.get("mndwi",0)),float(d.get("lswi",0)),
            int(d.get("month",datetime.now().month)),d.get("province","Afghanistan"))
        return jsonify({"status":"ok","crops":crops})
    except Exception as e: return jsonify({"error":str(e)}),500


@app.route("/monthly_rain",methods=["POST","OPTIONS"])
def monthly_rain():
    if request.method=="OPTIONS": return jsonify({}),200
    try:
        d=request.get_json(force=True)
        monthly=get_monthly_rain(float(d.get("annual_rain",250)),d.get("province","Afghanistan"))
        return jsonify({"status":"ok","monthly":monthly,
            "labels":["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
            "province":d.get("province","Afghanistan"),"annual":d.get("annual_rain",250)})
    except Exception as e: return jsonify({"error":str(e)}),500


@app.route("/ndvi_tile",methods=["POST","OPTIONS"])
def ndvi_tile():
    if request.method=="OPTIONS": return jsonify({}),200
    if not gee_ok: return jsonify({"status":"error","error":"GEE not available"}),503
    try:
        import ee
        d=request.get_json(force=True)
        coords=d.get("coords",[]); year=int(d.get("year",datetime.now().year))
        poly=ee.Geometry.Polygon([[[c[1],c[0]] for c in coords]])
        ed=f"{year}-07-31" if year<2025 else "2025-05-31"
        col=(ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(poly)
             .filterDate(f"{year}-04-01",ed).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",20))
             .median().clip(poly))
        url=col.normalizedDifference(["B8","B4"]).getThumbURL({
            "min":0,"max":0.7,"palette":["#d73027","#fc8d59","#fee08b","#d9ef8b","#91cf60","#1a9850"],
            "dimensions":512,"format":"png"})
        return jsonify({"status":"success","tile_url":url})
    except Exception as e: return jsonify({"status":"error","error":str(e)}),500


@app.route("/soil",methods=["POST","OPTIONS"])
def soil():
    if request.method=="OPTIONS": return jsonify({}),200
    try:
        d=request.get_json(force=True)
        s=get_soil_data(float(d.get("lat",34.5)),float(d.get("lon",67.7)),d.get("province","Afghanistan"))
        s["status"]="ok"; return jsonify(s)
    except Exception as e: return jsonify({"error":str(e)}),500

# ============================================================
# ZaminAI WEATHER FORECAST ENDPOINT
# Add this to your app.py
# Free Open-Meteo API — no key needed
# ============================================================

import requests
from functools import lru_cache
from datetime import datetime, timedelta

# Simple in-memory cache (1 hour) to reduce API calls
_weather_cache = {}

WEATHER_CODE_MAP = {
    0:  {"icon": "☀️", "en": "Clear sky",          "dr": "آسمان صاف",     "ps": "روښانه آسمان"},
    1:  {"icon": "🌤️", "en": "Mainly clear",        "dr": "اکثرا صاف",     "ps": "اکثرا روښانه"},
    2:  {"icon": "⛅",  "en": "Partly cloudy",      "dr": "نیمه ابری",     "ps": "نیمه وریځو"},
    3:  {"icon": "☁️", "en": "Cloudy",              "dr": "ابری",         "ps": "وریځو"},
    45: {"icon": "🌫️", "en": "Foggy",               "dr": "مه",           "ps": "لړه"},
    48: {"icon": "🌫️", "en": "Foggy",               "dr": "مه",           "ps": "لړه"},
    51: {"icon": "🌦️", "en": "Light drizzle",       "dr": "باران سبک",    "ps": "سپک باران"},
    53: {"icon": "🌦️", "en": "Drizzle",             "dr": "باران",        "ps": "باران"},
    55: {"icon": "🌧️", "en": "Heavy drizzle",       "dr": "باران شدید",   "ps": "سخت باران"},
    61: {"icon": "🌧️", "en": "Light rain",          "dr": "باران سبک",    "ps": "سپک باران"},
    63: {"icon": "🌧️", "en": "Rain",                "dr": "باران",        "ps": "باران"},
    65: {"icon": "⛈️", "en": "Heavy rain",          "dr": "باران شدید",   "ps": "سخت باران"},
    71: {"icon": "🌨️", "en": "Light snow",          "dr": "برف سبک",     "ps": "سپک واوره"},
    73: {"icon": "🌨️", "en": "Snow",                "dr": "برف",         "ps": "واوره"},
    75: {"icon": "❄️", "en": "Heavy snow",          "dr": "برف شدید",    "ps": "سخت واوره"},
    80: {"icon": "🌧️", "en": "Rain showers",         "dr": "رگبار",        "ps": "بارانونه"},
    81: {"icon": "🌧️", "en": "Heavy showers",        "dr": "رگبار شدید",   "ps": "سخت بارانونه"},
    82: {"icon": "⛈️", "en": "Violent showers",      "dr": "رگبار شدید",   "ps": "ډیر سخت بارانونه"},
    85: {"icon": "🌨️", "en": "Snow showers",         "dr": "برف رگبار",    "ps": "د واورې بارانونه"},
    86: {"icon": "❄️", "en": "Heavy snow showers",   "dr": "برف شدید",     "ps": "سخت د واورې بارانونه"},
    95: {"icon": "⛈️", "en": "Thunderstorm",         "dr": "طوفان رعد",    "ps": "د تندر طوفان"},
    96: {"icon": "⛈️", "en": "Thunderstorm + hail", "dr": "طوفان + ژاله", "ps": "طوفان + ږلۍ"},
}

DAY_NAMES = {
    "en": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    "dr": ["دوشنبه", "سه‌شنبه", "چهارشنبه", "پنجشنبه", "جمعه", "شنبه", "یکشنبه"],
    "ps": ["دوشنبه", "درېشنبه", "چهارشنبه", "پنجشنبه", "جمعه", "شنبه", "یکشنبه"]
}


@app.route("/weather", methods=["POST"])
def weather_forecast():
    """7-day weather forecast for any field location.
    Uses free Open-Meteo API — no key, no rate limit issues.
    """
    try:
        data = request.get_json()
        lat = float(data.get("lat"))
        lon = float(data.get("lon"))
        lang = data.get("lang", "en")  # en, dr, ps

        # Cache key — round to 0.1° (~11km) so nearby fields share cache
        cache_key = f"{round(lat, 1)},{round(lon, 1)}"

        # Check cache (valid for 1 hour)
        if cache_key in _weather_cache:
            cached_time, cached_data = _weather_cache[cache_key]
            if datetime.now() - cached_time < timedelta(hours=1):
                return jsonify({**cached_data, "cached": True})

        # Call Open-Meteo (free, no API key needed)
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode,windspeed_10m_max",
            "current_weather": "true",
            "timezone": "auto",
            "forecast_days": 7
        }

        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        raw = r.json()

        # Build clean response
        daily = raw.get("daily", {})
        current = raw.get("current_weather", {})

        forecast = []
        for i in range(len(daily.get("time", []))):
            date_str = daily["time"][i]
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            day_idx = dt.weekday()  # 0=Mon

            code = daily["weathercode"][i]
            code_info = WEATHER_CODE_MAP.get(code, WEATHER_CODE_MAP[3])

            forecast.append({
                "date": date_str,
                "day_name": DAY_NAMES[lang][day_idx] if lang in DAY_NAMES else DAY_NAMES["en"][day_idx],
                "temp_max": round(daily["temperature_2m_max"][i]),
                "temp_min": round(daily["temperature_2m_min"][i]),
                "rain_mm": round(daily["precipitation_sum"][i], 1),
                "wind_kmh": round(daily["windspeed_10m_max"][i]),
                "weather_code": code,
                "icon": code_info["icon"],
                "description": code_info[lang] if lang in code_info else code_info["en"]
            })

        # Detect alerts for the agent system later
        alerts = []
        for day in forecast:
            if day["rain_mm"] >= 20:
                alerts.append({"day": day["day_name"], "type": "heavy_rain", "value": day["rain_mm"]})
            if day["temp_min"] <= 0:
                alerts.append({"day": day["day_name"], "type": "frost", "value": day["temp_min"]})
            if day["temp_max"] >= 40:
                alerts.append({"day": day["day_name"], "type": "extreme_heat", "value": day["temp_max"]})
            if day["wind_kmh"] >= 50:
                alerts.append({"day": day["day_name"], "type": "high_wind", "value": day["wind_kmh"]})

        result = {
            "ok": True,
            "current": {
                "temp": round(current.get("temperature", 0)),
                "wind": round(current.get("windspeed", 0))
            },
            "forecast": forecast,
            "alerts": alerts,
            "location": {"lat": lat, "lon": lon},
            "cached": False
        }

        # Save to cache
        _weather_cache[cache_key] = (datetime.now(), result)

        return jsonify(result)

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════════════
# OFFICER DASHBOARD — worldwide regional analysis
# ════════════════════════════════════════════════════════════════════════════════

def gee_analyse_officer(coords, year, clat, clon, scale=500):
    """Regional GEE analysis at coarser resolution for large admin polygons."""
    import ee
    poly  = ee.Geometry.Polygon([[[c[1], c[0]] for c in coords]])
    today = datetime.now().strftime("%Y-%m-%d")

    # Hemisphere-aware growing season (same logic as detect-fields)
    if clat >= 10:
        s_start = f"{year}-04-01";  s_end = min(f"{year}-09-30", today)
    elif clat <= -10:
        s_start = f"{year-1}-10-01"; s_end = min(f"{year}-04-30", today)
    else:
        s_start = f"{year}-01-01";  s_end = min(f"{year}-12-31", today)
    if s_start > today:
        s_start = f"{year-1}{s_start[4:]}"; s_end = f"{year-1}{s_end[4:]}"

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(poly).filterDate(s_start, s_end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 35))
          .sort("CLOUDY_PIXEL_PERCENTAGE").limit(8).median().clip(poly))

    # ── All S2 indices in one reduceRegion call (was 4 separate getInfo calls) ──
    indices = (s2.normalizedDifference(["B8","B4"]).rename("ndvi")
               .addBands(s2.expression(
                   "2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))",
                   {"NIR":s2.select("B8"),"RED":s2.select("B4"),"BLUE":s2.select("B2")}
               ).rename("evi"))
               .addBands(s2.expression(
                   "((NIR-RED)/(NIR+RED+0.5))*1.5",
                   {"NIR":s2.select("B8"),"RED":s2.select("B4")}
               ).rename("savi"))
               .addBands(s2.normalizedDifference(["B3","B11"]).rename("mndwi")))

    idx_stats = indices.reduceRegion(
        ee.Reducer.mean(), poly, scale, maxPixels=1e9
    ).getInfo()
    def _f(k): v = idx_stats.get(k); return round(float(v), 4) if v is not None else None
    ndvi  = _f("ndvi")
    evi   = _f("evi")
    savi  = _f("savi")
    mndwi = _f("mndwi")

    # ── CHIRPS rain — separate collection, one call ──────────────────────────
    rain = None
    try:
        rv = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(poly)
              .filterDate(f"{year}-01-01", f"{year}-12-31")
              .select("precipitation").sum().clip(poly)
              .reduceRegion(ee.Reducer.mean(), poly, scale, maxPixels=1e9)
              .get("precipitation").getInfo())
        rain = round(float(rv), 1) if rv is not None else None
    except Exception as e:
        log.warning(f"Officer CHIRPS failed: {e}")

    # For very large polygons sample every other year to stay within timeout budget
    trend_step = 2 if scale >= 2000 else 1
    cur_year = datetime.now().year

    # NDVI trend S2 (2019-present)
    s2_trend = {}
    for yr in range(2019, cur_year + 1, trend_step):
        try:
            yr_end = min(f"{yr}-07-31", datetime.now().strftime("%Y-%m-%d"))
            c2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(poly)
                  .filterDate(f"{yr}-05-01", yr_end)
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 25)).median().clip(poly))
            v  = c2.normalizedDifference(["B8","B4"]).reduceRegion(
                     ee.Reducer.mean(), poly, scale, maxPixels=1e9).get("nd").getInfo()
            s2_trend[yr] = round(float(v), 4) if v else None
        except:
            s2_trend[yr] = None

    # NDVI trend Landsat (2013-2018 pre-Sentinel era)
    ls_trend = {}
    for yr in range(2013, 2019, trend_step):
        try:
            yr_end = min(f"{yr}-07-31", datetime.now().strftime("%Y-%m-%d"))
            lc    = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                     .filterBounds(poly).filterDate(f"{yr}-04-01", yr_end)
                     .filter(ee.Filter.lt("CLOUD_COVER", 25)).median().clip(poly))
            lc_sc = lc.select("SR_B.*").multiply(0.0000275).add(-0.2)
            v     = lc_sc.normalizedDifference(["SR_B5","SR_B4"]).reduceRegion(
                        ee.Reducer.mean(), poly, max(scale, 100), maxPixels=1e9).get("nd").getInfo()
            ls_trend[yr] = round(float(v), 4) if v else None
        except:
            ls_trend[yr] = None

    combined_trend = {**ls_trend, **s2_trend}

    # Sentinel-1 SAR
    sar_data = None
    try:
        s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
              .filterBounds(poly).filterDate(s_start, s_end)
              .filter(ee.Filter.eq("instrumentMode", "IW"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
              .select(["VV","VH"]).median().clip(poly))
        def sarm(img, band):
            v = img.reduceRegion(ee.Reducer.mean(), poly, max(scale,100), maxPixels=1e9).get(band).getInfo()
            return round(float(v), 3) if v is not None else None
        vv = sarm(s1, "VV"); vh = sarm(s1, "VH")
        sar_data = {"vv_db":vv, "vh_db":vh, "vh_vv_db":round(vh-vv,3) if (vv and vh) else None,
                    "source":"sentinel1_SAR_IW", "cloud_free":True}
    except Exception as e:
        log.warning(f"Officer SAR failed: {e}")

    # MODIS LST
    modis_data = None
    try:
        lst = ee.ImageCollection("MODIS/061/MOD11A2").filterBounds(poly)
        def modt(col):
            img = col.mean()
            def mv(b):
                v = img.reduceRegion(ee.Reducer.mean(), poly, 1000, maxPixels=1e9).get(b).getInfo()
                return round(float(v)*0.02-273.15,1) if v is not None else None
            return mv("LST_Day_1km"), mv("LST_Night_1km")
        td, tn = modt(lst.filterDate(f"{year}-06-01", f"{year}-08-31"))
        wd, wn = modt(lst.filterDate(f"{year}-01-01", f"{year}-03-31"))
        modis_data = {"summer_day_c":td,"summer_night_c":tn,"winter_day_c":wd,"winter_night_c":wn,
                      "frost_risk":(wn<0) if wn is not None else None,"source":"modis_MOD11A2"}
    except Exception as e:
        log.warning(f"Officer MODIS failed: {e}")

    area_ha  = calc_area_ha(coords)
    area_km2 = area_ha / 100.0

    # ── WorldPop population (global 100 m, 2000-2020) ──────────────────────────
    pop_data = None
    try:
        pop_yr = min(year, 2020)
        pop_img = (ee.ImageCollection("WorldPop/GP/100m/pop")
                   .filterBounds(poly)
                   .filterDate(f"{pop_yr}-01-01", f"{pop_yr}-12-31")
                   .first())
        total_pop = (pop_img.clip(poly)
                     .reduceRegion(ee.Reducer.sum(), poly, 100, maxPixels=1e10, bestEffort=True)
                     .get("population").getInfo())
        if total_pop is not None:
            density = round(float(total_pop) / max(area_km2, 0.01), 1)
            pop_data = {"total": int(round(float(total_pop))), "per_km2": density,
                        "year": pop_yr, "source": "WorldPop"}
    except Exception as e:
        log.warning(f"WorldPop failed: {e}")

    # ── Dynamic World land cover (global 10 m, 2015+) ─────────────────────────
    landcover_data = None
    try:
        lc_scale = 10 if area_km2 < 100 else 30 if area_km2 < 1000 else 100 if area_km2 < 10000 else 300
        dw_mode = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                   .filterBounds(poly)
                   .filterDate(s_start, s_end)
                   .select("label")
                   .mode()
                   .clip(poly))
        CLASS_NAMES = ["water","trees","grass","flooded_veg","crops",
                       "shrub_scrub","built_up","bare_ground","snow_ice"]
        hist = (dw_mode.reduceRegion(
            ee.Reducer.frequencyHistogram(), poly, lc_scale,
            maxPixels=1e10, bestEffort=True
        ).get("label").getInfo()) or {}
        total_px = sum(hist.values()) or 1
        lc_pct = {}
        for k, v in hist.items():
            idx  = int(float(k))
            name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
            lc_pct[name] = round(v / total_px * 100, 1)
        crop_pct = lc_pct.get("crops", 0)
        landcover_data = {
            "classes":   lc_pct,
            "crop_pct":  crop_pct,
            "crop_ha":   round(area_ha * crop_pct / 100, 1),
            "tree_pct":  lc_pct.get("trees", 0),
            "grass_pct": lc_pct.get("grass", 0),
            "built_pct": lc_pct.get("built_up", 0),
            "water_pct": lc_pct.get("water", 0),
            "bare_pct":  lc_pct.get("bare_ground", 0),
            "source":    f"DynamicWorld_v1_{lc_scale}m"
        }
    except Exception as e:
        log.warning(f"DynamicWorld failed: {e}")

    # ── Terrain — SRTM 30 m (single getInfo call) ────────────────────────────
    terrain_data = None
    try:
        ter_scale = max(90, scale)
        dem   = ee.Image("USGS/SRTMGL1_003").clip(poly)
        slope = ee.Terrain.slope(dem)
        # Stack elevation + slope into one image → one getInfo() round-trip
        ter_img = dem.rename("elevation").addBands(slope.rename("slope"))
        combined_reducer = ee.Reducer.mean().combine(
            reducer2=ee.Reducer.minMax(), sharedInputs=True)
        ter_stats = ter_img.reduceRegion(
            combined_reducer, poly, ter_scale, maxPixels=1e9, bestEffort=True
        ).getInfo()
        def _tf(key):
            v = ter_stats.get(key)
            return round(float(v), 1) if v is not None else None
        terrain_data = {
            "elev_mean_m": _tf("elevation_mean"),
            "elev_min_m":  _tf("elevation_min"),
            "elev_max_m":  _tf("elevation_max"),
            "slope_deg":   _tf("slope_mean"),
            "source": "SRTM_90m"
        }
    except Exception as e:
        log.warning(f"SRTM terrain failed: {e}")

    # ── JRC Global Surface Water — single getInfo() via mean of binary mask ────
    water_bodies = None
    try:
        jrc_scale = max(30, min(scale, 300))
        gsw  = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").clip(poly)
        perm = gsw.gte(50)   # binary: 1 = water ≥50% of observed years
        w_mean = (perm.reduceRegion(
            ee.Reducer.mean(), poly, jrc_scale, maxPixels=1e10, bestEffort=True
        ).get("occurrence").getInfo())
        w_pct = round(float(w_mean) * 100, 2) if w_mean is not None else 0
        water_bodies = {
            "pct": w_pct,
            "ha":  round(area_ha * w_pct / 100, 1),
            "source": "JRC_GSW_1.4"
        }
    except Exception as e:
        log.warning(f"JRC surface water failed: {e}")

    # ── Monthly NDVI profile — all 12 months for villages, bimonthly for large ──
    ndvi_monthly = {}
    if area_km2 < 20000:
        try:
            # Villages get fine monthly data; provinces get every-2-months
            months = list(range(1,13)) if area_km2 < 200 else [1,3,5,7,9,11]
            cal_scale = max(scale, 10 if area_km2 < 5 else 30 if area_km2 < 50 else 100)
            for mo in months:
                mo_end = f"{year}-{(mo%12)+1:02d}-01" if mo < 12 else f"{year+1}-01-01"
                mc = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                      .filterBounds(poly).filterDate(f"{year}-{mo:02d}-01", mo_end)
                      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
                      .sort("CLOUDY_PIXEL_PERCENTAGE").limit(3).median().clip(poly))
                v = mc.normalizedDifference(["B8","B4"]).reduceRegion(
                        ee.Reducer.mean(), poly, cal_scale, maxPixels=1e9
                    ).get("nd").getInfo()
                ndvi_monthly[mo] = round(float(v), 3) if v else None
        except Exception as e:
            log.warning(f"Monthly NDVI profile failed: {e}")

    return {
        "ndvi":ndvi,"evi":evi,"savi":savi,"mndwi":mndwi,"water":mndwi,"rain":rain,
        "trend":s2_trend,"ndvi_trend":s2_trend,"landsat_trend":ls_trend,"combined_trend":combined_trend,
        "sar":sar_data,"modis":modis_data,
        "population":   pop_data,
        "landcover":    landcover_data,
        "terrain":      terrain_data,
        "water_bodies": water_bodies,
        "ndvi_monthly": ndvi_monthly,
        "lat":round(clat,5),"lon":round(clon,5),
        "source":"gee_live","image_date":s_end,"analysis_scale_m":scale
    }


@app.route("/officer/farmers", methods=["GET"])
def officer_farmers():
    """List registered farmers for a province with masked phone and field count."""
    province = request.args.get("province", "")
    if not sb_ok:
        return jsonify({"farmers": [], "count": 0})
    if not province:
        return jsonify({"error": "province required"}), 400
    try:
        res = (sb.table("farmers")
                 .select("id,phone,language,province,created_at")
                 .eq("province", province)
                 .order("created_at", desc=True)
                 .execute())
        farmers = res.data or []
        # Batch field counts — one query instead of one per farmer
        farmer_ids = [f["id"] for f in farmers]
        field_counts = {}
        if farmer_ids:
            try:
                fc_res = (sb.table("fields").select("farmer_id")
                            .in_("farmer_id", farmer_ids).execute())
                for row in (fc_res.data or []):
                    fid = row["farmer_id"]
                    field_counts[fid] = field_counts.get(fid, 0) + 1
            except Exception as e:
                log.warning(f"Officer batch field count failed: {e}")
        result = []
        for f in farmers:
            phone = f.get("phone", "")
            masked = (phone[:3] + "****" + phone[-3:]) if len(phone) > 6 else "****"
            result.append({
                "phone": masked, "language": f.get("language","en"),
                "province": f.get("province",""),
                "field_count": field_counts.get(f["id"], 0),
                "joined": (f.get("created_at","") or "")[:10]
            })
        return jsonify({"farmers": result, "count": len(result)})
    except Exception as e:
        log.error(f"/officer/farmers: {e}")
        return jsonify({"error": str(e)}), 500


# In-memory task store for async field detection
# Render free tier has a hard 30-second request timeout — GEE takes 30-120 s.
# Solution: POST returns a task_id immediately; client polls GET until done.
_detect_tasks = {}

@app.route("/officer/detect-fields", methods=["POST","OPTIONS"])
def officer_detect_fields():
    """Start async field detection. Returns {task_id} immediately.
    Client polls GET /officer/detect-fields/<task_id> for the result.
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200
    if not gee_ok:
        return jsonify({"error": "Satellite analysis not available"}), 503
    try:
        import ee
        data    = request.get_json(force=True)
        raw_coords = data.get("coords", [])
        geometry   = data.get("geometry")      # raw GeoJSON geometry (optional, for bbox fallback)
        coords  = coords_or_bbox(raw_coords, geometry)
        year    = int(data.get("year", datetime.now().year))
        if len(coords) < 3:
            return jsonify({"error": "Need ≥3 coordinate points — check boundary data"}), 400

        poly     = ee.Geometry.Polygon([[[c[1], c[0]] for c in coords]])
        area_km2 = calc_area_ha(coords) / 100.0
        today    = datetime.now().strftime("%Y-%m-%d")

        # Hemisphere-aware growing season
        clat = sum(c[0] for c in coords) / len(coords)
        if clat >= 10:
            s_start = f"{year}-04-01";  s_end = min(f"{year}-09-30", today)
        elif clat <= -10:
            s_start = f"{year-1}-10-01"; s_end = min(f"{year}-04-30", today)
        else:
            s_start = f"{year}-01-01";  s_end = min(f"{year}-12-31", today)
        if s_start > today:
            s_start = f"{year-1}{s_start[4:]}"; s_end = f"{year-1}{s_end[4:]}"

        # Scale: coarser for larger areas so full coverage fits in one response.
        # Target: ~500-2000 polygons for the whole district/province.
        # DW has 9 classes → naturally few large polygons at any scale.
        # Satellite index layers have 15+ classes → need coarser scale.
        if area_km2 < 50:      seg_scale = 20
        elif area_km2 < 300:   seg_scale = 30
        elif area_km2 < 1000:  seg_scale = 50
        else:                   seg_scale = 100

        # Per-pixel QA60 cloud masking — keeps valid pixels even from cloudy images.
        # Critical for Netherlands, UK, tropical monsoon regions where no image
        # passes a strict CLOUDY_PIXEL_PERCENTAGE < 35 filter.
        def _mask_s2(img):
            qa = img.select("QA60")
            return img.updateMask(
                qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
            )
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(poly)
              .filterDate(s_start, s_end)
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
              .sort("CLOUDY_PIXEL_PERCENTAGE").limit(10)
              .map(_mask_s2)
              .median().clip(poly))

        ndvi = s2.normalizedDifference(["B8","B4"]).rename("ndvi")

        # ── Full land cover map via Dynamic World ─────────────────────────────
        # DW V1 classifies every S2 pixel into 9 classes globally at 10 m:
        #   0=water  1=trees  2=grass  3=flooded_veg  4=crops
        #   5=shrub  6=built_up  7=bare_ground  8=snow_ice
        # Vectorize all classes so the officer sees crops, buildings, bare soil,
        # forest, water — everything — not just agricultural fields.
        # Falls back to NDVI quantisation for years before DW coverage (< 2016).
        if year >= 2016:
            dw_label = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                        .filterBounds(poly)
                        .filterDate(s_start, s_end)
                        .select("label").limit(200).mode().clip(poly))
            # unmask NDVI with 0 — if S2 composite is masked (cloud) the DW labels
            # must still produce polygons. addBands(masked_image) would mask the
            # entire pixel so DW features would disappear. unmask(0) keeps them.
            ndvi_safe = ndvi.unmask(ee.Image(0).rename("ndvi"))
            fc = (dw_label.toInt().addBands(ndvi_safe).reduceToVectors(
                geometry=poly, scale=seg_scale,
                geometryType="polygon", eightConnected=True,
                reducer=ee.Reducer.mean(),
                labelProperty="lc",
                maxPixels=1e10, bestEffort=True, tileScale=4)
                .limit(4000))   # GEE hard limit is 5000; stay under it
        else:
            ndvi_q = (ndvi.multiply(20).floor().int()
                      .updateMask(ndvi.gt(0.10).And(ndvi.lt(0.95))))
            fc = (ndvi_q.reduceToVectors(
                geometry=poly, scale=seg_scale,
                geometryType="polygon", eightConnected=True,
                reducer=ee.Reducer.mean(),
                labelProperty="field",
                maxPixels=1e10, bestEffort=True, tileScale=4)
                .filter(ee.Filter.gte("mean", 0.10))
                .limit(4000))

        task_id = str(uuid.uuid4())
        _detect_tasks[task_id] = {"status": "pending"}

        def gee_worker():
            try:
                result = fc.getInfo()
                count  = len(result.get("features", []))
                log.info(f"detect-fields task {task_id[:8]}: {count} fields, scale={seg_scale}m")
                _detect_tasks[task_id] = {"status": "done", "data": result}
            except Exception as ex:
                log.error(f"detect-fields task {task_id[:8]} error: {ex}")
                _detect_tasks[task_id] = {"status": "error", "error": str(ex)}

        threading.Thread(target=gee_worker, daemon=True).start()
        return jsonify({"task_id": task_id, "status": "pending"})

    except Exception as e:
        log.error(f"/officer/detect-fields: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/officer/detect-fields/<task_id>", methods=["GET","OPTIONS"])
def officer_detect_fields_poll(task_id):
    """Poll for async field detection result."""
    if request.method == "OPTIONS":
        return jsonify({}), 200
    task = _detect_tasks.get(task_id)
    if not task:
        return jsonify({"status": "error", "error": "Task not found or expired"}), 404
    if task["status"] == "pending":
        return jsonify({"status": "pending"})
    if task["status"] == "error":
        _detect_tasks.pop(task_id, None)
        return jsonify({"status": "error", "error": task["error"]}), 500
    # done
    data = task.get("data", {"type":"FeatureCollection","features":[]})
    _detect_tasks.pop(task_id, None)
    return jsonify({"status": "done", "data": data})


@app.route("/officer/fields", methods=["GET"])
def officer_fields():
    """Return all farmer fields for a province with their latest satellite analysis."""
    province = request.args.get("province", "")
    if not sb_ok:
        return jsonify({"fields": [], "count": 0})
    if not province:
        return jsonify({"error": "province required"}), 400
    try:
        res = (sb.table("fields")
                 .select("id,label,coords,province,area_ha,area_jereb,created_at")
                 .eq("province", province)
                 .execute())
        fields = res.data or []
        # Batch analyses — one query instead of one per field (N+1 fix)
        field_ids = [f["id"] for f in fields]
        analyses_map = {}
        if field_ids:
            try:
                a_res = (sb.table("analyses")
                           .select("field_id,ndvi,evi,mndwi,rain,savi")
                           .in_("field_id", field_ids)
                           .order("analysed_at", desc=True)
                           .execute())
                for a in (a_res.data or []):
                    fid = a["field_id"]
                    if fid not in analyses_map:   # first = most recent (ordered desc)
                        analyses_map[fid] = a
            except Exception as ae:
                log.warning(f"Officer batch analyses failed: {ae}")
        for f in fields:
            if isinstance(f.get("coords"), str):
                try: f["coords"] = json.loads(f["coords"])
                except: f["coords"] = []
            f["analysis"] = analyses_map.get(f["id"])
        return jsonify({"fields": fields, "count": len(fields)})
    except Exception as e:
        log.error(f"/officer/fields: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/officer/parcel-thumbnail", methods=["POST","OPTIONS"])
def officer_parcel_thumbnail():
    """Generate a GEE satellite/index image of a polygon for download.
    layer: natural | ndvi | landcover | water | baresoil | croptype
    Returns {url} — a public PNG thumbnail URL from GEE.
    """
    if request.method == "OPTIONS": return jsonify({}), 200
    if not gee_ok: return jsonify({"error": "Satellite imagery unavailable"}), 503
    try:
        import ee
        data   = request.get_json(force=True)
        coords = data.get("coords", [])
        year   = int(data.get("year", datetime.now().year))
        layer  = data.get("layer", "natural")
        if len(coords) < 3:
            return jsonify({"error": "Need ≥3 coordinate points"}), 400

        poly  = ee.Geometry.Polygon([[[c[1], c[0]] for c in coords]])
        today = datetime.now().strftime("%Y-%m-%d")

        # S2 composite for most layers
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(poly)
              .filterDate(f"{year}-01-01", f"{year}-12-31")
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 35))
              .sort("CLOUDY_PIXEL_PERCENTAGE").limit(8).median().clip(poly))

        if layer == "natural":
            img = s2.select(["B4","B3","B2"])
            vis = {"min":0,"max":3000,"bands":["B4","B3","B2"]}
        elif layer == "ndvi":
            img = s2.normalizedDifference(["B8","B4"])
            vis = {"min":0,"max":0.8,
                   "palette":["d32f2f","ef6c00","f9a825","558b2f","2e7d32","1b5e20"]}
        elif layer == "landcover":
            dw  = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                   .filterBounds(poly).filterDate(f"{year}-01-01",f"{year}-12-31")
                   .select("label").limit(300).mode().clip(poly))
            img = dw
            vis = {"min":0,"max":8,
                   "palette":["2980b9","1a7a40","8bc34a","1abc9c",
                               "f39c12","95a5a6","e74c3c","a04000","ecf0f1"]}
        elif layer == "water":
            img = s2.normalizedDifference(["B3","B11"])
            vis = {"min":-0.5,"max":0.5,
                   "palette":["6d4c41","a1887f","80cbc4","29b6f6","0288d1","01579b"]}
        elif layer == "baresoil":
            img = s2.expression(
                "((SWIR+RED)-(NIR+BLUE))/((SWIR+RED)+(NIR+BLUE))",
                {"SWIR":s2.select("B11"),"RED":s2.select("B4"),
                 "NIR":s2.select("B8"),"BLUE":s2.select("B2")})
            vis = {"min":-0.3,"max":0.3,
                   "palette":["1b5e20","ffe0b2","ffa726","f57c00","e64a19","bf360c"]}
        elif layer == "false_color":
            # NIR false colour — crops appear red, urban grey, bare soil brown
            img = s2.select(["B8","B4","B3"])
            vis = {"min":0,"max":3500,"bands":["B8","B4","B3"]}
        else:
            img = s2.select(["B4","B3","B2"])
            vis = {"min":0,"max":3000,"bands":["B4","B3","B2"]}

        url = img.getThumbURL({**vis, "region":poly,
                               "dimensions":768, "format":"png"})
        log.info(f"parcel-thumbnail: layer={layer} year={year}")
        return jsonify({"url": url, "layer": layer, "year": year})
    except Exception as e:
        log.error(f"/officer/parcel-thumbnail: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/officer/village-crops", methods=["POST","OPTIONS"])
def officer_village_crops():
    """High-resolution crop type map for small areas (villages, fields).
    Returns GeoJSON polygons with crop class + NDVI at 10-20m resolution.
    Only suitable for area < 50 km² — uses native S2 10m resolution.
    """
    if request.method == "OPTIONS": return jsonify({}), 200
    if not gee_ok: return jsonify({"error": "GEE not available"}), 503
    try:
        import ee
        data   = request.get_json(force=True)
        coords = data.get("coords", [])
        year   = int(data.get("year", datetime.now().year))
        if len(coords) < 3: return jsonify({"error": "Need ≥3 points"}), 400

        area_km2 = calc_area_ha(coords) / 100.0
        if area_km2 > 50:
            return jsonify({"error": "Area too large for village crop map. Use district/province level analysis."}), 400

        task_id = str(uuid.uuid4())
        _analyse_farmer_tasks[task_id] = {"status": "pending"}

        def _worker():
            try:
                poly = ee.Geometry.Polygon([[[c[1],c[0]] for c in coords]])
                today = datetime.now().strftime("%Y-%m-%d")
                clat  = sum(c[0] for c in coords) / len(coords)

                if clat >= 10:
                    s_start, s_end = f"{year}-04-01", min(f"{year}-09-30", today)
                elif clat <= -10:
                    s_start, s_end = f"{year-1}-10-01", min(f"{year}-04-30", today)
                else:
                    s_start, s_end = f"{year}-01-01", min(f"{year}-12-31", today)

                def _qamask(img):
                    qa = img.select("QA60")
                    return img.updateMask(qa.bitwiseAnd(1<<10).eq(0).And(qa.bitwiseAnd(1<<11).eq(0)))

                s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                      .filterBounds(poly).filterDate(s_start, s_end)
                      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 35))
                      .sort("CLOUDY_PIXEL_PERCENTAGE").limit(6)
                      .map(_qamask).median().clip(poly))

                ndvi = s2.normalizedDifference(["B8","B4"]).rename("ndvi")
                evi  = s2.expression("2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))",
                                     {"NIR":s2.select("B8"),"RED":s2.select("B4"),"BLUE":s2.select("B2")}).rename("evi")
                lswi = s2.normalizedDifference(["B8","B11"]).rename("lswi")

                month = datetime.now().month
                # Pixel-level crop classification (same rules as detect_crop scalar logic)
                crop_map = (ee.Image(4)  # default: bare/fallow
                    .where(ndvi.gte(0.42).And(evi.gte(0.30)), 3)           # orchard
                    .where(ndvi.gte(0.38).And(evi.gte(0.28)).And(lswi.gte(-0.10)), 2)  # vegetables
                    .where(ndvi.gte(0.25).And(ndvi.lte(0.60)).And(evi.lt(0.38))
                           .And(ee.Image(1 if month in range(3,8) else 0).eq(1)), 1)   # wheat
                    .where(ndvi.lt(0.12), 4)                                # bare
                ).rename("crop_class")

                # Only show vegetated pixels
                crop_masked = crop_map.updateMask(ndvi.gt(0.08))

                # Vectorize at 10m (fine detail for villages)
                fc = (crop_masked.addBands(ndvi)
                      .reduceToVectors(
                          geometry=poly, scale=10,
                          geometryType="polygon", eightConnected=False,
                          labelProperty="crop_class", reducer=ee.Reducer.mean(),
                          maxPixels=1e10, bestEffort=True, tileScale=4)
                      .filter(ee.Filter.gte("count", 10))   # min ~0.1 ha
                      .limit(500))

                result = fc.getInfo()
                count  = len(result.get("features", []))
                log.info(f"village-crops: {count} crop polygons, scale=10m, area={area_km2:.2f}km²")
                _analyse_farmer_tasks[task_id] = {"status":"done","data":result}
            except Exception as ex:
                log.error(f"village-crops worker: {ex}")
                _analyse_farmer_tasks[task_id] = {"status":"error","error":str(ex)}

        threading.Thread(target=_worker, daemon=True).start()
        return jsonify({"task_id": task_id, "status": "pending", "type": "village_crops"})
    except Exception as e:
        log.error(f"/officer/village-crops: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/officer/proxy-image", methods=["GET"])
def officer_proxy_image():
    """Proxy GEE thumbnail URLs so the browser can draw them on canvas
    without CORS errors. Only allows earthengine.googleapis.com URLs."""
    url = request.args.get("url", "")
    if not url.startswith("https://earthengine.googleapis.com"):
        return "Only GEE URLs allowed", 403
    try:
        r = requests.get(url, timeout=40, headers={"User-Agent": "ZaminAI/1.0"})
        return r.content, 200, {
            "Content-Type": "image/png",
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "public,max-age=3600"
        }
    except Exception as e:
        return str(e), 500


# In-memory cache for boundary data (key: "ISO_level", value: geojson dict)
_gadm_cache  = {}
_gadm_tasks  = {}   # task_id -> {status, data/error}

# ISO3 → country name in FAO/GAUL dataset
_ISO3_GAUL = {
    "AFG":"Afghanistan","GHA":"Ghana","KEN":"Kenya","NGA":"Nigeria",
    "ETH":"Ethiopia","TZA":"Tanzania","UGA":"Uganda","RWA":"Rwanda",
    "MOZ":"Mozambique","ZMB":"Zambia","ZWE":"Zimbabwe","MWI":"Malawi",
    "ZAF":"South Africa","NAM":"Namibia","BWA":"Botswana","LSO":"Lesotho",
    "SWZ":"Swaziland","EGY":"Egypt","MAR":"Morocco","TUN":"Tunisia",
    "DZA":"Algeria","LBY":"Libya","SDN":"Sudan","SSD":"South Sudan",
    "SOM":"Somalia","DJI":"Djibouti","ERI":"Eritrea","CMR":"Cameroon",
    "COD":"Democratic Republic of the Congo","COG":"Republic of Congo",
    "GAB":"Gabon","CAF":"Central African Republic","SEN":"Senegal",
    "MLI":"Mali","BFA":"Burkina Faso","NER":"Niger","CIV":"Cote d Ivoire",
    "LBR":"Liberia","SLE":"Sierra Leone","GIN":"Guinea","GNB":"Guinea-Bissau",
    "GMB":"Gambia","MRT":"Mauritania","BEN":"Benin","TGO":"Togo",
    "TCD":"Chad","AGO":"Angola","IND":"India","PAK":"Pakistan",
    "BGD":"Bangladesh","LKA":"Sri Lanka","NPL":"Nepal","MMR":"Myanmar",
    "THA":"Thailand","VNM":"Viet Nam","KHM":"Cambodia","LAO":"Lao PDR",
    "MYS":"Malaysia","IDN":"Indonesia","PHL":"Philippines","PNG":"Papua New Guinea",
    "CHN":"China","JPN":"Japan","KOR":"Republic of Korea","MNG":"Mongolia",
    "IRN":"Iran  (Islamic Republic of)","IRQ":"Iraq","SYR":"Syrian Arab Republic",
    "JOR":"Jordan","SAU":"Saudi Arabia","YEM":"Yemen","OMN":"Oman",
    "ARE":"United Arab Emirates","QAT":"Qatar","KWT":"Kuwait","TUR":"Turkey",
    "AZE":"Azerbaijan","ARM":"Armenia","GEO":"Georgia","KAZ":"Kazakhstan",
    "UZB":"Uzbekistan","TKM":"Turkmenistan","KGZ":"Kyrgyzstan","TJK":"Tajikistan",
    "RUS":"Russian Federation","UKR":"Ukraine","BLR":"Belarus","MDA":"Republic of Moldova",
    "POL":"Poland","CZE":"Czech Republic","SVK":"Slovakia","HUN":"Hungary",
    "ROU":"Romania","BGR":"Bulgaria","SRB":"Serbia","HRV":"Croatia",
    "BIH":"Bosnia and Herzegovina","ALB":"Albania","MKD":"The former Yugoslav Republic of Macedonia",
    "SVN":"Slovenia","MNE":"Montenegro","GRC":"Greece","CYP":"Cyprus",
    "DEU":"Germany","FRA":"France","ESP":"Spain","PRT":"Portugal",
    "ITA":"Italy","GBR":"United Kingdom of Great Britain and Northern Ireland",
    "NLD":"Netherlands","BEL":"Belgium","LUX":"Luxembourg","CHE":"Switzerland",
    "AUT":"Austria","DNK":"Denmark","SWE":"Sweden","NOR":"Norway",
    "FIN":"Finland","IRL":"Ireland","LVA":"Latvia","LTU":"Lithuania",
    "EST":"Estonia","USA":"United States of America","CAN":"Canada",
    "MEX":"Mexico","BRA":"Brazil","ARG":"Argentina","CHL":"Chile",
    "COL":"Colombia","VEN":"Venezuela","PER":"Peru","BOL":"Bolivia",
    "ECU":"Ecuador","PRY":"Paraguay","URY":"Uruguay","AUS":"Australia",
    "NZL":"New Zealand",
}


def _remap_gaul_props(feats, iso, level):
    """Remap FAO/GAUL property names to GADM-compatible names the frontend expects."""
    for feat in feats:
        p = feat["properties"]
        adm1 = p.get("ADM1_NAME", "")
        adm2 = p.get("ADM2_NAME", "")
        p["NAME_1"] = adm1
        p["GID_1"]  = f"{iso}.{adm1.replace(' ','_')}_1"
        if level >= 2:
            p["NAME_2"] = adm2
            p["GID_2"]  = f"{iso}.{adm1.replace(' ','_')}.{adm2.replace(' ','_')}_1"
        if level >= 3:
            adm3 = p.get("ADM3_NAME", "")
            p["NAME_3"] = adm3
            p["GID_3"]  = (f"{iso}.{adm1.replace(' ','_')}"
                           f".{adm2.replace(' ','_')}.{adm3.replace(' ','_')}_1")


def _fetch_gadm_boundaries(iso, level, province_filter=None):
    """Background worker: fetch FAO/GAUL/2015 boundaries from GEE.
    province_filter: ADM1_NAME string to restrict level-2/3 to one province only
                     (avoids loading thousands of districts for large countries).
    """
    cache_key = f"{iso}_{level}_{province_filter or ''}"
    if cache_key in _gadm_cache:
        return _gadm_cache[cache_key]

    try:
        import ee
        country_name = _ISO3_GAUL.get(iso)
        if gee_ok and country_name:
            # FAO/GAUL/2015 — confirmed GEE dataset, full global coverage
            gaul_col = f"FAO/GAUL/2015/level{min(level, 2)}"

            fc = (ee.FeatureCollection(gaul_col)
                  .filter(ee.Filter.eq("ADM0_NAME", country_name)))

            # For level 2/3 scope to one province to keep response small and fast
            if province_filter and level >= 2:
                fc = fc.filter(ee.Filter.eq("ADM1_NAME", province_filter))

            # Simplify geometry properly — setGeometry keeps Polygon/MultiPolygon type
            error_m = 2000 if level == 1 else 1000
            fc = (fc.map(
                lambda f: f.setGeometry(f.geometry().simplify(maxError=error_m)))
                .limit(4000))   # GEE caps FeatureCollection.getInfo() at 5000

            result = fc.getInfo()
            feats  = result.get("features", [])
            if feats:
                _remap_gaul_props(feats, iso, level)
                log.info(f"GAUL {iso} L{level} prov={province_filter}: {len(feats)} features")
                _gadm_cache[cache_key] = result
                return result
            log.warning(f"GAUL returned 0 features for {iso} L{level} country='{country_name}'")

        # Fallback: ucdavis direct download (only works for small countries fast enough)
        url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{iso}_{level}.json"
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            data = r.json()
            _gadm_cache[cache_key] = data
            return data
        return None

    except Exception as e:
        log.error(f"_fetch_gadm_boundaries {iso} L{level}: {e}")
        return None


@app.route("/gadm/<iso>/<int:level>", methods=["GET"])
def gadm_proxy(iso, level):
    """Return admin boundaries async. Render's 30s limit is never hit.
    Optional ?province=<ADM1_NAME> narrows level-2/3 to one province only.
    """
    if level not in (1, 2, 3):
        return jsonify({"error": "Level must be 1, 2, or 3"}), 400
    iso      = iso.upper()[:3]
    province = request.args.get("province", "").strip() or None
    cache_key = f"{iso}_{level}_{province or ''}"

    # Serve instantly from cache
    if cache_key in _gadm_cache:
        return jsonify(_gadm_cache[cache_key]), 200, {
            "Content-Type": "application/json", "Cache-Control": "public,max-age=86400"
        }

    # Start async fetch
    task_id = str(uuid.uuid4())
    _gadm_tasks[task_id] = {"status": "pending"}

    def _worker():
        result = _fetch_gadm_boundaries(iso, level, province)
        if result:
            _gadm_tasks[task_id] = {"status": "done", "data": result}
        else:
            _gadm_tasks[task_id] = {
                "status": "error",
                "error": f"No boundary data for {iso} L{level}"
                         + (f" province={province}" if province else "")
            }

    threading.Thread(target=_worker, daemon=True).start()
    return jsonify({"status": "pending", "task_id": task_id}), 202


@app.route("/gadm-result/<task_id>", methods=["GET"])
def gadm_result(task_id):
    """Poll for async GADM boundary result."""
    task = _gadm_tasks.get(task_id)
    if not task:
        return jsonify({"status": "error", "error": "Task not found"}), 404
    if task["status"] == "pending":
        return jsonify({"status": "pending"})
    if task["status"] == "error":
        _gadm_tasks.pop(task_id, None)
        return jsonify({"status": "error", "error": task["error"]}), 500
    data = task["data"]
    _gadm_tasks.pop(task_id, None)
    return jsonify(data), 200, {
        "Content-Type": "application/json", "Cache-Control": "public,max-age=86400"
    }


# ── Async satellite layer tasks (Step 2) ─────────────────────────────────────
# Same async-task pattern as detect-fields: POST starts the GEE job,
# GET /<task_id> polls for the result.  Each layer is computed on demand.
_layer_tasks = {}

@app.route("/officer/layer/<layer_name>", methods=["POST","OPTIONS"])
def officer_layer(layer_name):
    """Start async computation of a named satellite layer.
    Supported: ndvi, water, baresoil, croptype
    Returns {task_id} immediately; poll GET /officer/layer-result/<task_id>.
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200
    if not gee_ok:
        return jsonify({"error": "Satellite analysis not available"}), 503
    if layer_name not in ("ndvi", "water", "baresoil", "croptype", "forest"):
        return jsonify({"error": f"Unknown layer: {layer_name}"}), 400
    try:
        import ee
        data   = request.get_json(force=True)
        coords = data.get("coords", [])
        year   = int(data.get("year", datetime.now().year))
        if len(coords) < 3:
            return jsonify({"error": "Need ≥3 coordinate points"}), 400

        poly     = ee.Geometry.Polygon([[[c[1], c[0]] for c in coords]])
        area_km2 = calc_area_ha(coords) / 100.0
        today    = datetime.now().strftime("%Y-%m-%d")
        clat     = sum(c[0] for c in coords) / len(coords)

        # Hemisphere-aware season
        if clat >= 10:
            s_start = f"{year}-04-01"; s_end = min(f"{year}-09-30", today)
        elif clat <= -10:
            s_start = f"{year-1}-10-01"; s_end = min(f"{year}-04-30", today)
        else:
            s_start = f"{year}-01-01"; s_end = min(f"{year}-12-31", today)
        if s_start > today:
            s_start = f"{year-1}{s_start[4:]}"; s_end = f"{year-1}{s_end[4:]}"

        if area_km2 < 50:     seg_scale = 20
        elif area_km2 < 300:  seg_scale = 30
        elif area_km2 < 1000: seg_scale = 50
        else:                  seg_scale = 100

        def _mask_s2(img):
            qa = img.select("QA60")
            return img.updateMask(
                qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
            )
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(poly).filterDate(s_start, s_end)
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
              .sort("CLOUDY_PIXEL_PERCENTAGE").limit(10)
              .map(_mask_s2).median().clip(poly))

        if layer_name == "ndvi":
            # NDVI: (B8−B4)/(B8+B4), quantised to 0.05 classes, cropped to 0.10–0.95
            index = s2.normalizedDifference(["B8","B4"]).rename("ndvi")
            label_img = (index.multiply(20).floor().int()
                         .updateMask(index.gt(0.10).And(index.lt(0.95))))
            label_prop = "ndvi_class"
            def feat_props(f):
                return f  # mean already in properties

        elif layer_name == "water":
            # MNDWI: (B3−B11)/(B3+B11) — positive = water/moisture
            mndwi = s2.normalizedDifference(["B3","B11"]).rename("mndwi")
            # Also compute NDWI (B3−B8)/(B3+B8) as secondary confirmation
            ndwi  = s2.normalizedDifference(["B3","B8"]).rename("ndwi")
            # Keep pixels where MNDWI > −0.1 (water, flooded, moist soil)
            water_mask = mndwi.gt(-0.10)
            index       = mndwi
            label_img  = (index.multiply(20).floor().int()
                          .updateMask(water_mask))
            label_prop = "mndwi_class"

        elif layer_name == "baresoil":
            # BSI: ((B11+B4)−(B8+B2))/((B11+B4)+(B8+B2))
            bsi = s2.expression(
                "((SWIR+RED)-(NIR+BLUE))/((SWIR+RED)+(NIR+BLUE))",
                {"SWIR":s2.select("B11"),"RED":s2.select("B4"),
                 "NIR":s2.select("B8"),"BLUE":s2.select("B2")}
            ).rename("bsi")
            # Bare soil: BSI > 0  (positive = exposed soil/sand)
            index     = bsi
            label_img = (index.multiply(20).floor().int()
                         .updateMask(index.gt(0.0)))
            label_prop = "bsi_class"

        elif layer_name == "forest":
            # Forest / dense vegetation: NDVI > 0.45 year-round.
            # Use the full date window (not just growing season) to catch
            # evergreen forest that stays green all year.
            s2_full = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                       .filterBounds(poly)
                       .filterDate(f"{year}-01-01", f"{year}-12-31")
                       .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
                       .sort("CLOUDY_PIXEL_PERCENTAGE").limit(12)
                       .map(_mask_s2).median().clip(poly))
            ndvi_f = s2_full.normalizedDifference(["B8","B4"])
            # Forest threshold: high NDVI (0.45+) — distinguishes forest from crops
            index     = ndvi_f.rename("ndvi")
            label_img = (index.multiply(20).floor().int()
                         .updateMask(index.gte(0.45)))
            label_prop = "forest_class"

        elif layer_name == "croptype":
            # Pixel-level crop classification using the same rules as detect_crop().
            # Encoded as integer: 1=wheat 2=vegetables 3=orchard 4=bare_fallow
            ndvi  = s2.normalizedDifference(["B8","B4"])
            evi   = s2.expression(
                "2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))",
                {"NIR":s2.select("B8"),"RED":s2.select("B4"),"BLUE":s2.select("B2")})
            lswi  = s2.normalizedDifference(["B8","B11"])
            month = datetime.now().month

            # Rule-based classification (mirrors detect_crop scalar logic as raster)
            is_bare  = ndvi.lt(0.12)
            is_wheat = (ndvi.gte(0.25).And(ndvi.lte(0.60))
                        .And(evi.lt(0.38))
                        .And(ee.Image(1).multiply(1 if month in range(3,8) else 0).eq(1)))
            is_veg   = ndvi.gte(0.38).And(evi.gte(0.28)).And(lswi.gte(-0.10))
            is_orch  = ndvi.gte(0.42).And(evi.gte(0.30))

            crop_map = (ee.Image(4).where(is_orch, 3)
                                   .where(is_veg,  2)
                                   .where(is_wheat,1)
                                   .where(is_bare, 4))
            # Only show pixels with some vegetation signal
            label_img = crop_map.updateMask(ndvi.gt(0.05))
            index     = ndvi  # use NDVI for the mean reducer
            label_prop = "crop_class"

        task_id = str(uuid.uuid4())
        _layer_tasks[task_id] = {"status": "pending", "layer": layer_name}

        ndvi_safe = s2.normalizedDifference(["B8","B4"]).rename("ndvi").unmask(ee.Image(0).rename("ndvi"))
        label_with_ndvi = label_img.toInt().addBands(ndvi_safe)

        def _worker():
            try:
                fc = (label_with_ndvi.reduceToVectors(
                    geometry=poly, scale=seg_scale,
                    geometryType="polygon", eightConnected=True,
                    reducer=ee.Reducer.mean(),
                    labelProperty=label_prop,
                    maxPixels=1e10, bestEffort=True, tileScale=4)
                    .limit(4000))
                result = fc.getInfo()
                count  = len(result.get("features", []))
                log.info(f"layer/{layer_name} task {task_id[:8]}: {count} polygons")
                _layer_tasks[task_id] = {"status": "done", "data": result, "layer": layer_name}
            except Exception as ex:
                log.error(f"layer/{layer_name} task {task_id[:8]} error: {ex}")
                _layer_tasks[task_id] = {"status": "error", "error": str(ex), "layer": layer_name}

        threading.Thread(target=_worker, daemon=True).start()
        return jsonify({"task_id": task_id, "status": "pending", "layer": layer_name})

    except Exception as e:
        log.error(f"/officer/layer/{layer_name}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/officer/layer-result/<task_id>", methods=["GET","OPTIONS"])
def officer_layer_result(task_id):
    """Poll for async satellite layer result."""
    if request.method == "OPTIONS":
        return jsonify({}), 200
    task = _layer_tasks.get(task_id)
    if not task:
        return jsonify({"status": "error", "error": "Task not found"}), 404
    if task["status"] == "pending":
        return jsonify({"status": "pending", "layer": task["layer"]})
    if task["status"] == "error":
        _layer_tasks.pop(task_id, None)
        return jsonify({"status": "error", "error": task["error"]}), 500
    data = task.get("data", {"type":"FeatureCollection","features":[]})
    _layer_tasks.pop(task_id, None)
    return jsonify({"status": "done", "layer": task["layer"], "data": data})


# In-memory task store for async officer/analyse (same pattern as detect-fields)
_analyse_tasks = {}

@app.route("/officer/analyse", methods=["POST","OPTIONS"])
def officer_analyse():
    """Start async regional analysis. Returns {task_id} immediately.
    Client polls GET /officer/analyse-result/<task_id> for the result.
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:
        data     = request.get_json(force=True)
        coords   = data.get("coords", [])
        year     = int(data.get("year", datetime.now().year))
        country  = data.get("country", "")
        province = data.get("province", "")
        district = data.get("district", "")
        village  = data.get("village", "")

        if len(coords) < 3:
            return jsonify({"error": "Need ≥3 coordinate points"}), 400

        lats = [c[0] for c in coords]; lons = [c[1] for c in coords]
        clat = sum(lats)/len(lats);     clon  = sum(lons)/len(lons)
        area_ha  = calc_area_ha(coords)
        area_km2 = area_ha / 100.0

        # Fine scale for small areas (villages, fields) — coarser for large regions
        if area_km2 < 1:       scale = 10    # tiny village / single field
        elif area_km2 < 10:    scale = 20    # village
        elif area_km2 < 50:    scale = 30    # small district
        elif area_km2 < 200:   scale = 50    # district
        elif area_km2 < 1000:  scale = 100   # province
        elif area_km2 < 5000:  scale = 300
        elif area_km2 < 20000: scale = 500
        else:                   scale = 1000

        task_id = str(uuid.uuid4())
        _analyse_tasks[task_id] = {"status": "pending"}

        def _worker():
            try:
                result = {}
                if gee_ok:
                    try:
                        result = gee_analyse_officer(coords, year, clat, clon, scale=scale)
                        log.info(f"Officer GEE ok — {country}/{province}/{district} {area_km2:.0f}km² scale={scale}m")
                    except Exception as e:
                        log.error(f"Officer GEE failed: {e}")

                if not result:
                    reg = get_regional_data(clat, clon)
                    result = {
                        "ndvi":reg["ndvi"],"evi":reg["evi"],"savi":reg["savi"],
                        "mndwi":reg["mndwi"],"water":reg["mndwi"],"rain":reg["rain"],
                        "trend":reg.get("trend",{}),"ndvi_trend":reg.get("trend",{}),
                        "landsat_trend":{},"combined_trend":reg.get("trend",{}),
                        "lat":round(clat,5),"lon":round(clon,5),
                        "source":reg.get("source","climate_zone_fallback"),
                        "image_date":f"{year}-05-15","analysis_scale_m":None,
                        "sar":None,"modis":None,
                        "population":None,"landcover":None,"terrain":None,
                        "water_bodies":None,"ndvi_monthly":{}
                    }

                result.update({
                    "area_km2": round(area_km2, 1),
                    "area_ha":  round(area_ha, 1),
                    "country":  country, "province": province,
                    "district": district, "village":  village,
                    "admin_level": "village" if village else "district" if district else "province" if province else "country",
                    "year": year,
                })

                farmer_count = None
                if sb_ok and province:
                    try:
                        r = sb.table("farmers").select("id", count="exact").eq("province", province).execute()
                        farmer_count = r.count
                    except Exception as e:
                        log.warning(f"Officer farmer count failed: {e}")
                result["farmer_count"] = farmer_count

                _analyse_tasks[task_id] = {"status": "done", "data": result}
            except Exception as ex:
                log.error(f"officer_analyse task {task_id[:8]} error: {ex}")
                _analyse_tasks[task_id] = {"status": "error", "error": str(ex)}

        threading.Thread(target=_worker, daemon=True).start()
        return jsonify({"task_id": task_id, "status": "pending"})
    except Exception as e:
        log.error(f"/officer/analyse: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/officer/analyse-result/<task_id>", methods=["GET","OPTIONS"])
def officer_analyse_result(task_id):
    """Poll for async officer/analyse result."""
    if request.method == "OPTIONS":
        return jsonify({}), 200
    task = _analyse_tasks.get(task_id)
    if not task:
        return jsonify({"status": "error", "error": "Task not found"}), 404
    if task["status"] == "pending":
        return jsonify({"status": "pending"})
    if task["status"] == "error":
        _analyse_tasks.pop(task_id, None)
        return jsonify({"status": "error", "error": task["error"]}), 500
    data = task.get("data", {})
    _analyse_tasks.pop(task_id, None)
    return jsonify({"status": "done", "data": data})


# ════════════════════════════════════════════════════════════════════════════════
# MULTI-AGENT SYSTEM
# ════════════════════════════════════════════════════════════════════════════════

# Agent task store — same async pattern as GEE analysis tasks
_agent_tasks = {}

# Lazy-load Anthropic client (avoids import error if key not set)
_anthropic_client = None
def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None and ANTHROPIC_KEY:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    return _anthropic_client


def _build_tool_context():
    """Build the context dict passed to every tool execution."""
    def _monthly_rain_fn(lat, lon, year):
        try:
            import ee
            pt  = ee.Geometry.Point([lon, lat])
            out = {}
            for mo in range(1, 13):
                mo_end = f"{year}-{(mo%12)+1:02d}-01" if mo < 12 else f"{year+1}-01-01"
                rain = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                        .filterDate(f"{year}-{mo:02d}-01", mo_end)
                        .sum().sample(pt, 5000).first()
                        .get("precipitation").getInfo())
                out[mo] = round(float(rain), 1) if rain else None
            return {"monthly_mm": out, "annual_mm": round(sum(v for v in out.values() if v),1)}
        except Exception as e:
            return {"error": str(e)}

    def _soil_fn(lat, lon):
        try:
            url = (f"https://rest.isric.org/soilgrids/v2.0/properties/query"
                   f"?lon={lon}&lat={lat}&property=phh2o&property=soc&property=clay&depth=0-30cm&value=mean")
            r = requests.get(url, timeout=10)
            return r.json() if r.ok else {"error": f"SoilGrids {r.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def _get_all_fields_fn():
        if not sb_ok: return []
        try:
            r = sb.table("fields").select("id,farmer_id,label,coords,province").limit(200).execute()
            return r.data or []
        except: return []

    def _get_farmer_fields_fn(farmer_id):
        if not sb_ok: return []
        try:
            r = sb.table("fields").select("*").eq("farmer_id", farmer_id).execute()
            return r.data or []
        except: return []

    def _save_alert_fn(farmer_id, field_id=None, message="", severity="info", language="en"):
        if not sb_ok: return
        try:
            sb.table("farmer_alerts").insert({
                "farmer_id": farmer_id, "field_id": field_id,
                "message": message, "severity": severity,
                "created_at": datetime.now().isoformat(),
            }).execute()
        except Exception as e:
            log.warning(f"save_alert failed: {e}")

    return {
        "gee_analyse_officer": gee_analyse_officer if gee_ok else None,
        "monthly_rain_fn":     _monthly_rain_fn,
        "soil_fn":             _soil_fn,
        "get_all_fields_fn":   _get_all_fields_fn,
        "get_farmer_fields_fn":_get_farmer_fields_fn,
        "save_alert_fn":       _save_alert_fn,
        # Wrap detect_crop — it needs 7 args; tools.py calls it with 5
        "detect_crop_fn": lambda nd,ev,ls,sv,mo: detect_crop(nd,ev,sv,0,ls,mo,""),
    }


# In-memory conversation history (keyed by session_id)
_agent_sessions = {}


@app.route("/agent/chat", methods=["POST","OPTIONS"])
def agent_chat():
    """Main agentic chat endpoint — async task pattern.
    Returns {task_id} immediately; client polls /agent/result/<task_id>.
    This avoids Render's reverse-proxy 60s timeout on long Claude+GEE calls."""
    if request.method == "OPTIONS": return jsonify({}), 200

    data       = request.get_json(force=True)
    question   = data.get("question", "").strip()
    session_id = data.get("session_id", str(uuid.uuid4()))
    role       = data.get("role", "farmer")
    language   = data.get("language", "en")
    stream     = data.get("stream", False)
    coords     = data.get("coords")
    farmer_id  = data.get("farmer_id")

    if not question:
        return jsonify({"error": "Question required"}), 400

    anthropic_client = _get_anthropic()
    if not anthropic_client and not GEMINI_KEY:
        return jsonify({
            "answer":     "No AI key configured. Set GEMINI_API_KEY (free) or ANTHROPIC_API_KEY.",
            "tool_calls": [], "fallback": True, "session_id": session_id
        })

    # Start background task — returns task_id immediately (avoids 60s proxy timeout)
    task_id = str(uuid.uuid4())
    _agent_tasks[task_id] = {"status": "pending"}

    def _worker():
        try:
            from agents.tools        import TOOLS
            from agents.prompts      import ORCHESTRATOR_PROMPT, OFFICER_AGENT_PROMPT

            q = question
            if coords:
                q = f"[Field coordinates: {coords[:3]}…] {q}"

            system   = OFFICER_AGENT_PROMPT if role == "officer" else ORCHESTRATOR_PROMPT
            # Inject RAG knowledge into the system prompt
            rag_ctx = rag_retrieve(question, top_k=3, threshold=0.50)
            if rag_ctx:
                system = system + "\n\n## Relevant domain knowledge\n" + "\n\n".join(rag_ctx)
            history  = _agent_sessions.get(session_id, [])
            tool_ctx = _build_tool_context()

            if anthropic_client:
                from agents.orchestrator import run_agent
                out     = run_agent(q, system, TOOLS, tool_ctx, anthropic_client,
                                    history=history, language=language)
                backend = "anthropic"
            else:
                from agents.orchestrator import run_gemini_agent
                out     = run_gemini_agent(q, system, TOOLS, tool_ctx,
                                           language=language, history=history)
                backend = "gemini"

            log.info(f"[Agent] backend={backend} tools={len(out.get('tool_calls',[]))} iter={out.get('iterations')}")

            history.append({"role":"user",      "content":q})
            history.append({"role":"assistant", "content":out.get("answer","")})
            _agent_sessions[session_id] = history[-20:]

            if sb_ok and farmer_id:
                try:
                    sb.table("conversations").insert({
                        "farmer_id":  farmer_id, "question": q,
                        "answer":     out.get("answer",""),
                        "tool_calls": json.dumps(out.get("tool_calls",[])),
                        "created_at": datetime.now().isoformat(),
                    }).execute()
                except Exception as e:
                    log.warning(f"Save conversation: {e}")

            # Extract location data from satellite tool calls for map display
            map_data = None
            for tc in out.get("tool_calls", []):
                if tc.get("tool") == "query_satellite_data" and tc.get("output"):
                    o = tc["output"]
                    if o.get("ndvi") is not None:
                        map_data = {
                            "ndvi":       o.get("ndvi"),
                            "rain":       o.get("rainfall_mm"),
                            "lat":        o.get("lat"),
                            "lon":        o.get("lon"),
                            "area_km2":   o.get("area_km2"),
                            "land_cover": o.get("land_cover"),
                            "population": o.get("population"),
                        }
                        break

            _agent_tasks[task_id] = {
                "status":     "done",
                "answer":     out.get("answer",""),
                "tool_calls": out.get("tool_calls",[]),
                "iterations": out.get("iterations"),
                "session_id": session_id,
                "backend":    backend,
                "usage":      out.get("usage"),
                "map_data":   map_data,
            }
        except Exception as e:
            log.error(f"agent worker: {e}")
            _agent_tasks[task_id] = {"status": "error", "error": str(e),
                                      "session_id": session_id}

    threading.Thread(target=_worker, daemon=True).start()
    return jsonify({"status": "pending", "task_id": task_id,
                    "session_id": session_id}), 202


@app.route("/agent/result/<task_id>", methods=["GET"])
def agent_result(task_id):
    """Poll for async agent response."""
    task = _agent_tasks.get(task_id)
    if not task:
        return jsonify({"status": "error", "error": "Task not found"}), 404
    if task["status"] == "pending":
        return jsonify({"status": "pending"})
    _agent_tasks.pop(task_id, None)
    if task["status"] == "error":
        return jsonify({"status": "error", "error": task.get("error","Unknown")}), 500
    return jsonify({"status": "done", **task})


@app.route("/agent/stream", methods=["POST","OPTIONS"])
def agent_stream():
    """SSE streaming agent chat — events arrive word-by-word."""
    if request.method == "OPTIONS": return jsonify({}), 200
    data = request.get_json(force=True)
    data["stream"] = True
    # Re-use agent_chat with stream=True via internal redirect
    with app.test_request_context(
        "/agent/chat", method="POST",
        data=json.dumps(data), content_type="application/json"
    ):
        return agent_chat()


@app.route("/agent/monitor", methods=["POST","OPTIONS"])
def agent_monitor():
    """Trigger autonomous field monitoring loop (can be called by cron).
    Checks all registered fields, detects NDVI drops, issues alerts."""
    if request.method == "OPTIONS": return jsonify({}), 200
    client = _get_anthropic()
    if not client:
        return jsonify({"error": "Anthropic API key required for monitoring"}), 503

    task_id = str(uuid.uuid4())
    _analyse_farmer_tasks[task_id] = {"status": "pending"}

    def _worker():
        try:
            from agents.orchestrator import run_field_monitor
            results = run_field_monitor(_build_tool_context(), client)
            _analyse_farmer_tasks[task_id] = {"status":"done","data":results}
        except Exception as e:
            _analyse_farmer_tasks[task_id] = {"status":"error","error":str(e)}

    threading.Thread(target=_worker, daemon=True).start()
    return jsonify({"task_id": task_id, "status": "pending",
                    "message": "Monitoring loop started — poll /analyse-result/<task_id>"})


@app.route("/agent/weekly-report", methods=["POST","OPTIONS"])
def agent_weekly_report():
    """Generate a weekly satellite intelligence report for a province."""
    if request.method == "OPTIONS": return jsonify({}), 200
    data     = request.get_json(force=True)
    province = data.get("province", "Kunduz")
    country  = data.get("country",  "Afghanistan")
    client   = _get_anthropic()
    if not client:
        return jsonify({"error": "Anthropic API key required"}), 503

    task_id = str(uuid.uuid4())
    _analyse_farmer_tasks[task_id] = {"status":"pending"}

    def _worker():
        try:
            from agents.orchestrator import run_weekly_officer_report
            result = run_weekly_officer_report(
                province, country, _build_tool_context(), client
            )
            _analyse_farmer_tasks[task_id] = {"status":"done","data":result}
        except Exception as e:
            _analyse_farmer_tasks[task_id] = {"status":"error","error":str(e)}

    threading.Thread(target=_worker, daemon=True).start()
    return jsonify({"task_id": task_id, "status": "pending",
                    "province": province, "country": country})


@app.route("/agent/history/<session_id>", methods=["GET"])
def agent_history(session_id):
    """Return conversation history for a session."""
    history = _agent_sessions.get(session_id, [])
    return jsonify({"session_id": session_id, "turns": len(history)//2,
                    "history": history[-10:]})


@app.route("/agent/status", methods=["GET"])
def agent_status():
    """Return agent system status."""
    has_anthropic = bool(_get_anthropic())
    has_gemini    = bool(GEMINI_KEY)
    active_backend = ("anthropic" if has_anthropic else
                      "gemini"    if has_gemini    else "none")
    return jsonify({
        "anthropic":       has_anthropic,
        "gemini":          has_gemini,
        "active_backend":  active_backend,
        "gee":             gee_ok,
        "database":        sb_ok,
        "active_sessions": len(_agent_sessions),
        "tools_available": 12,
        "models": {
            "anthropic_main":  "claude-sonnet-4-6",
            "anthropic_fast":  "claude-haiku-4-5-20251001",
            "gemini_default":  "gemini-1.5-flash",
        }
    })


@app.route("/agent")
def agent_ui():
    """Serve the agentic chat UI."""
    return send_from_directory(".", "agent.html")


# ── CROP PHOTO DIAGNOSIS ──────────────────────────────────────────────────────

def _parse_diagnosis_confidence(text):
    """
    Infer confidence level from Claude's language in the diagnosis.
    Returns "high" | "medium" | "low"
    """
    if not text:
        return "low"
    t = text.lower()
    # Explicit uncertainty markers → low
    if any(w in t for w in ["cannot determine", "unclear", "unable to", "not enough",
                              "hard to tell", "difficult to identify", "no clear"]):
        return "low"
    # Hedging language → medium
    if any(w in t for w in ["possibly", "possibly", "may be", "might be", "could be",
                              "appears to", "seems to", "likely", "probable", "suspect"]):
        return "medium"
    # Healthy / no problem → high (confident assessment)
    if any(w in t for w in ["healthy", "no disease", "no sign", "no pest", "looks good"]):
        return "high"
    # Disease named directly without hedging → high
    if any(w in t for w in ["confirmed", "clearly", "definite", "identified", "detected",
                              "infected with", "caused by", "rust", "blight", "mildew",
                              "aphid", "locust", "wilt", "rot"]):
        return "high"
    return "medium"


@app.route("/diagnose", methods=["POST", "OPTIONS"])
def diagnose():
    """
    Photo-based crop disease / pest detection.
    Stage 1 — YOLO v8m (fast bounding-box detection, 38 disease classes).
    Stage 2 — Claude Haiku (detailed multilingual diagnosis + treatment advice).
    Body: {image: <base64 data-URL or raw base64>, language: "en|fa|ps", crop: "wheat|..."}
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:
        d         = request.get_json(force=True)
        image_b64 = d.get("image", "")
        language  = d.get("language", "en")
        crop_hint = d.get("crop", "").strip()
        mode      = d.get("mode", "disease").strip().lower()  # disease | pest | yield | soil

        if not image_b64:
            return jsonify({"error": "image required (base64)"}), 400

        # Strip data-URL prefix if present (e.g. "data:image/jpeg;base64,...")
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception as e:
            return jsonify({"error": f"Invalid base64 image: {e}"}), 400

        # ── Stage 1: YOLO fast detection ──────────────────────────────────────
        # Skip YOLO if env flag set (useful on memory-constrained hosts like Render 512MB)
        _yolo_disabled = os.environ.get("DISABLE_YOLO", "").lower() in ("1", "true", "yes")
        yolo_result = {"ok": False, "yolo_available": False, "detections": []}
        if not _yolo_disabled:
            try:
                from crop_vision import run_inference
                yolo_result = run_inference(image_bytes)
            except MemoryError:
                log.warning("YOLO OOM — falling back to Vision AI only")
            except Exception as e:
                log.warning(f"YOLO import/run failed: {e}")

        # ── Build shared prompt ───────────────────────────────────────────────
        _LANG_INST_DIAG = {
            "fa": "Respond ONLY in Dari (Afghan Persian). Use simple farming language a village farmer understands.",
            "ps": "Respond ONLY in Pashto. Use simple farming language a village farmer understands.",
            "ar": "Respond ONLY in Arabic (العربية). Use right-to-left text. Simple farming language.",
            "ur": "Respond ONLY in Urdu (اردو). Right-to-left. Simple farming language.",
            "hi": "Respond ONLY in Hindi (हिंदी). Simple language a village farmer understands.",
            "bn": "Respond ONLY in Bengali (বাংলা). Simple farming language.",
            "sw": "Respond ONLY in Swahili (Kiswahili). Practical farming language.",
            "es": "Respond ONLY in Spanish (Español). Clear, practical farming language.",
            "fr": "Respond ONLY in French (Français). Simple farming terms.",
            "pt": "Respond ONLY in Portuguese (Português). Practical farming language.",
            "am": "Respond ONLY in Amharic (አማርኛ). Simple farming language.",
        }
        lang_inst = _LANG_INST_DIAG.get(language, "Respond in English. Use simple, practical language.")
        country    = d.get("country", "").strip() or "Afghanistan"
        land_unit  = "jerib" if country.lower() in ("afghanistan",) else "hectare"

        yolo_ctx = ""
        if yolo_result.get("ok") and yolo_result.get("detections"):
            top_det = yolo_result["detections"][0]
            yolo_ctx = f"YOLO model detected: {top_det['label_en']} ({top_det['confidence']*100:.0f}% confidence). "

        crop_ctx = f"The farmer says this is a {crop_hint} crop. " if crop_hint else ""
        location_ctx = f"Location: {country}. " if country and country != "Afghanistan" else ""

        # Augment with RAG knowledge
        rag_diag_ctx = ""
        if rag_ok:
            rag_kw = yolo_result.get("detections", [{}])[0].get("label_en", mode) if yolo_result.get("detections") else mode
            rag_query = f"{crop_hint or 'crop'} {rag_kw} Afghanistan"
            rag_chunks = rag_retrieve(rag_query, top_k=2, threshold=0.45)
            if rag_chunks:
                rag_diag_ctx = "\n\nRelevant agronomic knowledge:\n" + "\n\n".join(rag_chunks)

        _MODE_PROMPTS = {
            "disease": (
                "Examine this crop/plant photo carefully and answer:\n"
                "1. What disease or infection do you see? (name it; if none say plant looks healthy)\n"
                "2. Severity: mild / moderate / severe\n"
                "3. What must the farmer do RIGHT NOW? (3-4 numbered steps)\n"
                "4. Which fungicide/product to apply, exact dose, and when?\n"
                "5. One sentence: how to prevent this disease next season."
            ),
            "pest": (
                "Examine this crop/plant photo carefully and answer:\n"
                "1. What pest(s) do you see? Name them; estimate count or infestation % if visible.\n"
                "2. Infestation level: light / moderate / heavy\n"
                "3. What must the farmer do RIGHT NOW? (3-4 numbered steps)\n"
                "4. Which pesticide to apply, exact dose per litre and per "+land_unit+", and timing?\n"
                "5. One sentence: how to prevent this pest next season."
            ),
            "yield": (
                "Examine this crop/plant photo carefully and answer:\n"
                "1. What crop and growth stage do you see? Is it ready to harvest?\n"
                "2. Estimated days until optimal harvest window (give a range)\n"
                "3. What should the farmer check or do before harvest? (3-4 numbered steps)\n"
                "4. Any quality issues visible — grain filling, pest damage, moisture problems?\n"
                "5. One tip to maximise yield or grain quality before harvest."
            ),
            "soil": (
                "Examine this soil photo carefully and answer:\n"
                "1. What soil type and colour do you see? Sandy / loam / clay / silty?\n"
                "2. Moisture level: dry / moist / wet / waterlogged\n"
                "3. Any visible signs of compaction, erosion, salinity, or nutrient deficiency? (3-4 steps to improve)\n"
                "4. What organic or chemical amendments should this farmer add, and how much per "+land_unit+"?\n"
                "5. Which crops grow best in this soil type in Afghanistan?"
            ),
        }
        mode_prompt = _MODE_PROMPTS.get(mode, _MODE_PROMPTS["disease"])
        diagnosis_prompt = (
            f"{location_ctx}{crop_ctx}{yolo_ctx}"
            f"{mode_prompt}\n\n"
            f"{lang_inst}\n"
            f"Be concise. Farmers in {country} will act directly on this advice. Use {land_unit}s as the land measurement unit."
            f"{rag_diag_ctx}"
        )

        # ── Stage 2a: Claude Vision (preferred) ──────────────────────────────
        ai_diagnosis = None
        ai_model_used = None
        if ANTHROPIC_KEY:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=ANTHROPIC_KEY, timeout=55.0)
                msg = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=800,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {
                                "type": "base64", "media_type": "image/jpeg", "data": image_b64,
                            }},
                            {"type": "text", "text": diagnosis_prompt},
                        ],
                    }],
                )
                ai_diagnosis  = msg.content[0].text
                ai_model_used = "claude-haiku-4-5"
            except Exception as e:
                log.error(f"Claude Vision error: {e}")

        # ── Stage 2b: Gemini Vision fallback ─────────────────────────────────
        if ai_diagnosis is None and GEMINI_KEY:
            try:
                for gmodel in ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-flash-latest"]:
                    url = (f"https://generativelanguage.googleapis.com/v1beta"
                           f"/models/{gmodel}:generateContent?key={GEMINI_KEY}")
                    resp = requests.post(url, json={
                        "contents": [{"parts": [
                            {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                            {"text": diagnosis_prompt},
                        ]}],
                        "safetySettings": [{"category": c, "threshold": "BLOCK_NONE"} for c in [
                            "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                            "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]],
                        "generationConfig": {"temperature": 0.5, "maxOutputTokens": 600},
                    }, timeout=30)
                    if resp.status_code == 200:
                        cands = resp.json().get("candidates", [])
                        if cands:
                            txt = cands[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                            if txt and len(txt) > 10:
                                ai_diagnosis  = txt.strip()
                                ai_model_used = gmodel
                                break
            except Exception as e:
                log.error(f"Gemini Vision error: {e}")

        top        = yolo_result["detections"][0] if yolo_result.get("detections") else None
        confidence = _parse_diagnosis_confidence(ai_diagnosis)
        # Boost to high if YOLO has a strong detection
        if top and not top.get("is_healthy") and top.get("confidence", 0) > 0.75:
            confidence = "high"

        # Derive disease name and severity for alert storage
        disease_name = ""
        disease_severity = ""
        if top and not top.get("is_healthy"):
            disease_name = top.get("label_en", top.get("label", ""))
            disease_severity = "severe" if confidence == "high" else "moderate" if confidence == "medium" else "mild"

        # Optionally persist disease result to field's latest analysis for alerts
        field_id_diag = d.get("field_id")
        if sb_ok and field_id_diag and (disease_name or ai_diagnosis):
            try:
                an_res = (sb.table("analyses").select("id, full_data")
                            .eq("field_id", field_id_diag)
                            .order("analysed_at", desc=True).limit(1).execute())
                if an_res.data:
                    an_row = an_res.data[0]
                    fd = json.loads(an_row["full_data"]) if isinstance(an_row.get("full_data"), str) else (an_row.get("full_data") or {})
                    fd["disease_name"]         = disease_name or ""
                    fd["disease_severity"]     = disease_severity or ""
                    fd["disease_diagnosed_at"] = datetime.utcnow().isoformat()
                    fd["disease_diagnosis"]    = ai_diagnosis[:800] if ai_diagnosis else ""
                    sb.table("analyses").update({"full_data": json.dumps(fd)}).eq("id", an_row["id"]).execute()
                    log.info(f"Disease result persisted to analyses for field {field_id_diag}")
            except Exception as de:
                log.warning(f"Could not persist disease to analyses: {de}")

        return jsonify({
            "ok":             True,
            "detections":     yolo_result.get("detections", []),
            "top_detection":  top,
            "is_healthy":     bool(top and top.get("is_healthy")),
            "yolo_ok":        yolo_result.get("ok", False),
            "yolo_available": yolo_result.get("yolo_available", False),
            "diagnosis":      ai_diagnosis,
            "confidence":     confidence,
            "disease_name":   disease_name,
            "disease_severity": disease_severity,
            "model":          f"yolov8m-plant-disease + {ai_model_used or 'no-vision-key'}",
            "language":       language,
        })
    except Exception as e:
        log.error(f"/diagnose: {e}")
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════════════
# RAG ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════════

_RAG_SETUP_SQL = """
-- Run once in Supabase SQL Editor (Database → SQL Editor)

create extension if not exists vector;

create table if not exists knowledge_chunks (
    id         uuid    primary key default gen_random_uuid(),
    content    text    not null,
    embedding  vector(768),
    source     text    default 'manual',
    metadata   jsonb   default '{}',
    created_at timestamp default now()
);

create or replace function match_knowledge_chunks (
    query_embedding  vector(768),
    match_count      int     default 4,
    match_threshold  float   default 0.70
)
returns table (id uuid, content text, source text, metadata jsonb, similarity float)
language sql stable as $$
    select id, content, source, metadata,
           1 - (embedding <=> query_embedding) as similarity
    from knowledge_chunks
    where 1 - (embedding <=> query_embedding) > match_threshold
    order by embedding <=> query_embedding
    limit match_count;
$$;

create index if not exists knowledge_chunks_embedding_idx
    on knowledge_chunks
    using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);
""".strip()

_RAG_SEED_DOCS = [
    # ── Satellite indices ─────────────────────────────────────────────────────

    ("NDVI (Normalized Difference Vegetation Index) crop health thresholds "
     "[Source: Tucker 1979, Remote Sensing of Environment; NASA MODIS Land Team; "
     "WUR Laboratory of Geo-information Science and Remote Sensing]: "
     "NDVI < 0.10: bare soil or rock — no active vegetation cover. "
     "NDVI 0.10-0.20: very sparse vegetation — severe drought stress or early seedling stage. "
     "NDVI 0.20-0.40: moderate vegetation — typical for dryland cereals in semi-arid regions. "
     "NDVI 0.40-0.60: healthy dense vegetation — well-irrigated crops or established orchards. "
     "NDVI > 0.60: very dense canopy — tropical or heavily irrigated crops. "
     "For winter wheat in Afghanistan, peak NDVI at grain fill (April-May north, March-April south) "
     "is most predictive of final yield: correlation r²=0.65-0.80 (ICARDA remote sensing studies)."),

    ("VCI (Vegetation Condition Index) for drought monitoring and early warning "
     "[Source: Kogan 1990 NOAA/NESDIS; FAO Global Information and Early Warning System; "
     "FEWS NET Vegetation Index product]: "
     "VCI = (NDVI_current − NDVI_min) / (NDVI_max − NDVI_min) × 100 "
     "using the multi-year minimum and maximum for the same pixel and calendar week. "
     "VCI < 10: extreme drought — crop failure likely without irrigation. "
     "VCI 10-35: severe drought — significant yield loss expected (40-70% reduction). "
     "VCI 35-50: moderate drought — yield reduction 20-40%. "
     "VCI 50-75: near-normal vegetation condition. "
     "VCI > 75: above-average vegetation condition, favourable growing season."),

    ("MNDWI (Modified Normalized Difference Water Index) for irrigation scheduling "
     "[Source: Xu 2006, International Journal of Remote Sensing 27(14); "
     "WUR Remote Sensing and GIS chair; ESA Sentinel-2 applications]: "
     "MNDWI = (Green − SWIR) / (Green + SWIR) using Sentinel-2 B3 and B11. "
     "MNDWI < -0.25: severe soil water deficit — irrigate within 2-3 days. "
     "MNDWI -0.25 to -0.10: moderate water stress — irrigate within 4-7 days. "
     "MNDWI -0.10 to 0.00: mild stress — irrigate within 7-10 days. "
     "MNDWI > 0.00: adequate soil moisture or recent rainfall — hold irrigation."),

    ("Sentinel-1 SAR (C-band radar) for cloud-free soil moisture monitoring "
     "[Source: ESA Sentinel-1 Mission; Ulaby et al. 1978 microwave remote sensing; "
     "Wagner et al. 1999 TU Wien soil moisture retrieval]: "
     "VV polarisation backscatter is the primary soil moisture proxy. "
     "VV < -15 dB: dry soil surface, high water deficit. "
     "VV -15 to -8 dB: moderate soil moisture. "
     "VV > -8 dB: wet soil or near-surface water presence. "
     "SAR penetrates cloud and smoke cover — critical for Afghanistan monsoon-season and winter monitoring "
     "when optical satellites (Sentinel-2, Landsat) are frequently cloud-obscured."),

    ("MODIS MOD11A2 Land Surface Temperature for frost and heat stress monitoring "
     "[Source: NASA MODIS Science Team; WMO agrometeorology guidelines; "
     "Porter & Gawith 1999 critical temperature thresholds for wheat]: "
     "MODIS provides 8-day composite LST at 1 km resolution globally. "
     "Frost damage thresholds for winter wheat by growth stage: "
     "Tillering: < -5°C for 2+ hours causes significant leaf damage. "
     "Stem extension: < -2°C causes significant tiller death. "
     "Anthesis (flowering): < 0°C for 2 hours can cause sterility and poor grain set — most critical stage. "
     "Heat stress at grain fill: > 34°C day temperature accelerates senescence and reduces grain weight. "
     "High-elevation provinces (Badakhshan, Bamyan, Ghor) have >90 frost days per year at altitude."),

    # ── Afghan wheat production ───────────────────────────────────────────────

    ("Winter wheat production systems in Afghanistan "
     "[Source: CIMMYT/ICARDA Afghanistan wheat improvement programme; "
     "FAO Afghanistan country cereal assessment 2022; WFP ADAM food security database]: "
     "Afghanistan produces approximately 4.5-5.5 million tonnes wheat/year on ~2.5 million ha. "
     "Wheat provides over 60% of national caloric intake — staple for 40+ million people. "
     "Irrigated wheat (~60% of area): stable yields 2.5-3.5 t/ha under good management. "
     "Rainfed wheat (~40% of area): highly variable yields 0.5-2.0 t/ha depending on rainfall. "
     "Actual average farm yield 1.5-2.5 t/ha is 40-60% below demonstrated potential of 4-5 t/ha. "
     "Main yield gap causes (ICARDA/CIMMYT field surveys): "
     "1. Suboptimal fertilizer application. "
     "2. Water stress at critical growth stages (tillering, anthesis, grain fill). "
     "3. Late sowing beyond the optimum window. "
     "4. Low-quality or susceptible seed varieties."),

    ("Winter wheat sowing calendar for Afghanistan by agro-ecological zone "
     "[Source: FEWS NET Afghanistan seasonal calendar; FAO GIEWS; ICARDA agronomic recommendations]: "
     "Northern lowlands (Kunduz, Balkh, Baghlan, Takhar, Jawzjan, Faryab — below 900m): "
     "Sow October 15 – November 15. Harvest June 15 – July 15. "
     "Southern lowlands (Kandahar, Helmand, Zabul, Nimroz — below 1000m): "
     "Sow November 1 – December 15. Harvest April 15 – May 31. "
     "Eastern lowlands (Nangarhar, Laghman, Kunar — below 800m): "
     "Sow October 15 – November 30. Harvest May 1 – June 15. "
     "Western lowlands (Herat, Farah, Badghis — below 1200m): "
     "Sow November 1 – December 15. Harvest May 15 – June 30. "
     "Central highlands (Kabul, Logar, Wardak, Ghazni — 1500-2200m): "
     "Sow October 10 – November 10. Harvest July 1 – July 31. "
     "High mountain zones (Bamyan, Badakhshan — above 2200m): "
     "Spring wheat only: Sow March-April. Harvest August-September."),

    ("Wheat fertilizer recommendations for Afghanistan "
     "[Source: ICARDA Central Asia and Caucasus wheat program; "
     "CIMMYT fertilizer response trials in South Asia; "
     "FAO plant nutrition guidelines for semi-arid cereals]: "
     "Baseline nitrogen requirement: 60-90 kg N/ha rainfed wheat; 100-120 kg N/ha irrigated wheat. "
     "Phosphorus is frequently the most limiting nutrient in Afghan calcareous alkaline soils (pH > 7.5) "
     "because P fixation by calcium reduces plant availability. "
     "DAP (18-46-0): 100-150 kg/ha at sowing as basal dose (provides P + starter N). "
     "Urea (46-0-0): 100-150 kg/ha total, split — half at tillering (Zadoks 20-25), half at stem extension (Zadoks 30-32). "
     "Split urea application improves nitrogen use efficiency (NUE) from 30-35% to 45-55% "
     "(WUR NUE research; CIMMYT agronomic best practices). "
     "Zinc deficiency (ZnSO4 at 25 kg/ha) is widespread in calcareous soils — symptoms: white striping on young leaves. "
     "Potassium deficiency is rare in Afghan soils which are naturally high in K."),

    ("Wheat rust diseases in Afghanistan "
     "[Source: CIMMYT global wheat rust surveillance network; "
     "ICARDA Afghanistan rust monitoring 2010-2023; "
     "Bockus et al. Compendium of Wheat Diseases, APS Press]: "
     "Yellow (stripe) rust — Puccinia striiformis f.sp. tritici: "
     "Most damaging wheat disease in Afghanistan; present in all wheat-growing provinces. "
     "Symptoms: yellow-orange pustules in parallel stripes along leaf veins. "
     "Optimal temperature 8-15°C — severe in cool moist springs; northern provinces most at risk. "
     "Management: Tebuconazole 250 EC or Propiconazole 250 EC at first pustule detection; "
     "resistant varieties Mazar-99, Roshan, Zurmat, Jahan. "
     "Stem rust — Puccinia graminis f.sp. tritici: "
     "Orange-brown pustules on stems and leaf sheaths; favoured by warm temperatures 18-25°C. "
     "Ug99 race and derivatives are present in the region — a serious threat. "
     "Leaf rust — Puccinia triticina: "
     "Circular brown pustules on upper leaf surface; less severe but widespread."),

    # ── Soils ─────────────────────────────────────────────────────────────────

    ("Afghan agricultural soil characteristics "
     "[Source: ISRIC SoilGrids v2.0 — Wageningen University & Research; "
     "FAO-UNESCO World Soil Map; WUR Soil Geography and Landscape group]: "
     "Dominant soil types: Calcisols and Cambisols — calcareous, alkaline (pH 7.0-8.5). "
     "Soil organic carbon (SOC) is critically low across most of Afghanistan: 0.3-1.8% "
     "vs. the FAO-recommended minimum of 2% for productive agricultural soils. "
     "Low SOC severely limits water-holding capacity, nutrient retention, and microbial activity. "
     "Northern provinces (Kunduz, Baghlan, Takhar): silty loam, SOC ~0.9-1.1%, best fertility nationally. "
     "Southern provinces (Kandahar, Helmand, Nimroz): sandy loam to sandy, SOC 0.3-0.5%, very low fertility. "
     "Western provinces (Herat, Farah): sandy loam, pH ~7.8, SOC 0.5%. "
     "Eastern provinces (Nangarhar, Laghman): loam, pH ~7.2, SOC ~1.2%, highest in lowlands. "
     "Central highlands (Kabul, Ghazni, Bamyan): loam to clay loam, pH 7.1-7.5. "
     "Badakhshan (mountain): clay loam, pH 6.8, SOC 1.8% — highest SOC in country."),

    ("Soil organic carbon improvement in calcareous dryland soils "
     "[Source: ISRIC / WUR Soil Geography and Landscape; "
     "FAO Voluntary Guidelines for Sustainable Soil Management 2017; "
     "Minasny et al. 2017 Soil Carbon 4 per mille initiative, Geoderma]: "
     "Building SOC from 0.5% to 1.5% requires sustained 5-10 years of organic matter addition. "
     "Approximate SOC increase per tonne of compost applied: 0.04-0.08% per year. "
     "Most cost-effective SOC-building practices ranked: "
     "1. Crop residue retention (not burning): avoids 50-80% of residue carbon loss. "
     "2. Legume cover crops (lentil, chickpea) fix 50-150 kg N/ha and add root carbon. "
     "3. Manure application: 5-10 t/ha builds SOC and supplies N+P+K. "
     "4. Compost: more stable carbon than fresh manure, slower release. "
     "SOC increase of 0.1% per year across Afghan cropland would sequester ~1.5 Mt CO2/year."),

    ("Improving alkaline calcareous soils for crop production "
     "[Source: FAO plant nutrition for sustainable food production; "
     "ICARDA soil fertility guidelines for West Asia and North Africa; "
     "WUR Plant Nutrition group research]: "
     "Soils with pH > 7.8 fix phosphorus as calcium phosphate — apply P in bands near roots not broadcast. "
     "Sulfur (elemental S): 300-500 kg/ha acidifies soil by ~0.5 pH unit over one season via soil bacteria. "
     "Ammonium sulfate fertilizer acidifies the root zone compared to urea — preferred for alkaline soils. "
     "Gypsum (calcium sulfate): 1-2 t/ha improves structure of sodic and heavy clay soils; no pH effect. "
     "Organic matter (compost/manure) buffers pH extremes and improves phosphorus availability. "
     "Iron and manganese deficiency can occur at pH > 8.0 — foliar spray more effective than soil application."),

    # ── Water and irrigation ──────────────────────────────────────────────────

    ("Rainfall climatology for Afghanistan "
     "[Source: FAO AQUASTAT Afghanistan irrigation and water resources; "
     "CHIRPS v2.0 satellite rainfall dataset (UCSB); "
     "FEWS NET climate monitoring; World Bank climate data portal]: "
     "Afghanistan is predominantly semi-arid to arid — national average ~250 mm/year. "
     "Northern provinces: 245-420 mm/year (Kunduz ~287mm, Badakhshan ~420mm). "
     "Western provinces: 140-220 mm/year (Herat ~195mm, Farah ~140mm). "
     "Southern provinces: 90-180 mm/year (Helmand ~148mm, Kandahar ~175mm). "
     "Eastern provinces: 280-400 mm/year (Nangarhar ~320mm, Khost ~350mm). "
     "Seasonal pattern: 75-85% of annual rainfall falls October-April (Mediterranean winter regime). "
     "Summer (June-September) is nearly completely dry in all provinces — drought for rainfed crops. "
     "Snow melt from Hindu Kush and Pamir mountains feeds rivers through June — critical for irrigation."),

    ("Irrigation systems and water management in Afghanistan "
     "[Source: FAO AQUASTAT Afghanistan water report 2008; "
     "World Bank Afghanistan National Solidarity Programme rural water review; "
     "Asian Development Bank Afghanistan water sector strategy]: "
     "Afghanistan has ~3.3 million ha of potentially irrigable land; ~2.5 million ha currently irrigated. "
     "Traditional karez (qanat) underground channels: gravity-fed, minimal maintenance, fixed flow rate; "
     "prevalent in piedmont zones of Kandahar, Herat, Ghazni. "
     "River diversion (jui) canal systems: dominant in Helmand, Kunduz, Baghlan river valleys. "
     "Tube wells: common in Nangarhar and Kabul where groundwater is accessible. "
     "Irrigation water use efficiency in Afghanistan: typically 30-50% due to unlined earthen canals. "
     "Drip and sprinkler irrigation: <2% adoption nationally but proven in horticulture — saves 40-50% water. "
     "Water allocation unit varies by region — 'ab' (water turn) system governs irrigation scheduling."),

    # ── Crop calendar and systems ─────────────────────────────────────────────

    ("Afghan agricultural crop calendar — all major crops "
     "[Source: FEWS NET Afghanistan seasonal calendar (updated 2023); "
     "FAO GIEWS country brief; MAIL Afghanistan crop reporting system]: "
     "Winter wheat (main staple): sown Oct-Nov (north), Nov-Dec (south/west); harvested May-Jun (south), Jun-Jul (north). "
     "Spring barley: sown Mar-Apr at altitude >1500m; harvested Aug-Sep. "
     "Maize: sown Apr-May in eastern lowlands (Nangarhar, Laghman, Kunar); harvested Aug-Sep. "
     "Rice: sown May-Jun in Kunduz, Baghlan, Nangarhar irrigated areas; harvested Sep-Oct. "
     "Cotton: sown Apr-May in Kunduz, Baghlan, Balkh; harvested Oct-Nov. "
     "Saffron: corms planted Aug-Sep; flowers harvested Oct-Nov (3-week window only). "
     "Melon/watermelon: sown Apr-May in southern lowlands; harvested Jul-Aug. "
     "Potato: sown Apr-May (lowlands); May-Jun (highlands >2000m); harvested Aug-Oct. "
     "Chickpea (lentil): sown Feb-Mar; harvested May-Jun in warmer zones."),

    # ── Saffron ───────────────────────────────────────────────────────────────

    ("Saffron (Crocus sativus) cultivation — Afghanistan "
     "[Source: FAO Afghanistan saffron value chain analysis 2016; "
     "USDA Foreign Agricultural Service GAIN report Afghanistan 2019; "
     "Gresta et al. 2008 review, Agronomy for Sustainable Development]: "
     "Afghanistan is among the world's top saffron producers alongside Iran and Spain. "
     "Primary production areas: Herat province (70% of national output), Ghor, Farah, Badghis. "
     "Planting: corms planted August-September at 15-20 cm depth, 10 cm inter-corm spacing. "
     "Corm density: 60-80 corms/m² for commercial production. "
     "Flowering window: October-November — each corm produces 1-3 flowers. "
     "Harvest: red stigmas must be hand-picked daily at dawn within 3 weeks; delays reduce quality. "
     "Yield: 3-8 kg dried saffron per hectare depending on corm age, density, and soil fertility. "
     "Corms multiply underground — fields productive 8-15 years without replanting. "
     "Water requirement: 150-250 mm/year — highly drought-tolerant once established. "
     "Soil requirement: well-drained, pH 6.5-8.0; waterlogging causes corm rot (Fusarium)."),

    # ── Horticulture ─────────────────────────────────────────────────────────

    ("Pomegranate and grape production in Afghanistan "
     "[Source: FAO Afghanistan horticulture sector study 2013; "
     "USAID Afghanistan Vouchers for Increased Production in Agriculture (VIPA) programme; "
     "Sarkhosh et al. 2021 pomegranate review, Scientia Horticulturae]: "
     "Afghanistan is historically renowned for pomegranate (Punica granatum — انار) quality. "
     "Kandahar Anar variety is internationally recognised for sweetness and large fruit size. "
     "Pomegranate production area: ~20,000-30,000 ha concentrated in Kandahar, Zabul, Logar, Kapisa, Laghman. "
     "Pomegranate is drought-tolerant: established trees require only 350-600 mm water/year. "
     "Trees begin bearing at 3-4 years, reach peak production at 8-12 years, productive to 50+ years. "
     "Grapes (Vitis vinifera — انگور): Kandahar, Herat, Parwan, Kapisa. "
     "Raisins (kishmish) from Kandahar and Kabul are a historic Afghan export product. "
     "Income comparison (FAO data): pomegranate and grapes generate 3-5× higher gross income per ha than wheat."),

    # ── Food security and WUR research ───────────────────────────────────────

    ("WUR research on smallholder food systems and yield gaps in Afghanistan "
     "[Source: WUR Food Systems group; WUR-FAO Afghanistan food security collaboration; "
     "van Ittersum et al. 2013 yield gap methodology, Field Crops Research; "
     "GYGA (Global Yield Gap Atlas) Afghanistan data — WUR/University of Nebraska]: "
     "Afghan smallholders operate on average 1.5-2.5 ha (7-12 jeribs) of fragmented land holdings. "
     "Water-limited yield potential for irrigated wheat in Afghanistan: 4.0-5.5 t/ha. "
     "Actual average farm yield: 1.5-2.5 t/ha — a yield gap of 40-60%. "
     "Nitrogen Use Efficiency (NUE) in Afghan wheat systems: typically 30-45% "
     "vs. the achievable 50-70% with split fertilizer application and timing optimisation. "
     "Remote sensing (NDVI) can explain 60-70% of district-level yield variation in Afghanistan "
     "when calibrated against ground truth data (ICARDA/WUR joint studies). "
     "Key WUR recommendation: satellite-based NDVI monitoring combined with farmer advisory "
     "services can reduce the yield gap by 20-35% at low cost."),

    ("Food security context for Afghanistan "
     "[Source: WFP Afghanistan Food Security Monitoring System (FSMS); "
     "FAO/WFP Crop and Food Supply Assessment Mission (CFSAM) 2022-2023; "
     "IPC Acute Food Insecurity Classification Afghanistan 2023]: "
     "Afghanistan is among the world's most acute food insecurity situations — "
     "17-22 million people (IPC Phase 3+) in 2022-2024 assessments. "
     "Wheat provides ~75% of daily caloric intake in rural households. "
     "Key food security shocks documented: "
     "2018 drought: NDVI anomaly -25 to -35% below 20-year average; wheat production fell 26%. "
     "2021-2022 La Niña drought: worst in 27 years; NDVI anomaly -35 to -45%; production fell 33%. "
     "Dryland rainfed wheat (40% of wheat area) shows near-zero yield in severe drought years. "
     "CHIRPS satellite rainfall data and MODIS NDVI anomaly maps are the primary early-warning tools "
     "used by WFP, FAO, and FEWS NET for food security monitoring in Afghanistan."),

    # ── Crop rotation ─────────────────────────────────────────────────────────

    ("Crop rotation systems for Afghan smallholder farms "
     "[Source: ICARDA dryland farming systems for West Asia and North Africa; "
     "FAO Crop Rotation and Soil Fertility 2018; "
     "WUR Farming Systems Ecology group]: "
     "Wheat–fallow (the most common Afghan rotation): one wheat crop followed by bare fallow. "
     "Limitation: fallow loses 30-50% of stored soil moisture to evaporation and adds no organic carbon. "
     "Wheat–legume rotation (recommended): replace fallow with lentil, chickpea, or mung bean. "
     "Benefits: legumes fix 50-150 kg N/ha, reducing next wheat fertilizer need by 30-40 kg N/ha. "
     "Wheat NDVI and yield typically 10-18% higher after legume than after bare fallow (ICARDA trials). "
     "Wheat–cotton rotation (Kunduz, Baghlan, Balkh): cotton is high N-demanding; follow with wheat + extra urea. "
     "Wheat–maize–rice in irrigated Nangarhar: intensive triple crop possible but exhausts soil; "
     "include a legume rest year every 3-4 cycles. "
     "Sainfoin (Onobrychis) and alfalfa (yonjo) 2-year stands improve soil structure before wheat — "
     "traditional practice that also provides fodder for livestock."),

    ("Legume crops for nitrogen fixation and soil health in Afghanistan "
     "[Source: ICARDA legume improvement programme for Central and West Asia; "
     "FAO Legumes: Seeds of Change report 2016; "
     "WUR Plant Sciences group legume-cereal intercropping research]: "
     "Chickpea (Cicer arietinum — نخود): most widely grown legume in Afghanistan. "
     "Sown Feb-Mar, harvested May-Jun. Fixes 60-120 kg N/ha with Mesorhizobium ciceri inoculant. "
     "Lentil (Lens culinaris — عدس): sown Feb-Mar, harvested May-Jun. Fixes 40-90 kg N/ha. "
     "Drought-tolerant: survives on 200-350 mm rainfall. "
     "Mung bean (Vigna radiata): sown Apr-May, harvested Jul-Aug. Short-season — fits as gap crop. "
     "Fixes 40-80 kg N/ha. High market value. "
     "Fenugreek (Trigonella): sown Mar-Apr, harvested Jun. Dual-purpose: grain and fodder. "
     "Key practice: rhizobium seed inoculant (available through FAO/MAIL extension) "
     "increases N fixation 25-40% vs. uninoculated seed in calcareous Afghan soils. "
     "Intercropping lentil + wheat at 70:30 ratio reduces rust severity and improves total protein yield."),

    # ── Regenerative agriculture ──────────────────────────────────────────────

    ("Regenerative agriculture principles for Afghan dryland smallholders "
     "[Source: FAO Voluntary Guidelines for Sustainable Soil Management 2017; "
     "WUR Farming Systems Ecology — agroecology principles; "
     "Rodale Institute regenerative organic agriculture research (applied globally); "
     "ICARDA Conservation Agriculture for Central Asia programme]: "
     "Regenerative agriculture rebuilds soil health, increases water retention, and restores biodiversity. "
     "Five core practices applicable to Afghan smallholders: "
     "1. Minimum tillage / conservation tillage: reduces soil disturbance, preserves soil structure and water. "
     "   In semi-arid Afghanistan, zero-till can improve wheat yield by 5-12% vs. deep ploughing in drought years. "
     "2. Permanent soil cover: keep crop residues on soil surface — reduces evaporation 15-30%, "
     "   prevents erosion, feeds soil microbes. Avoid burning residues. "
     "3. Crop diversity and rotation: alternate cereals with legumes every year (see rotation guide). "
     "4. Living roots in soil as long as possible: cover crops in fallow period maintain soil biology. "
     "5. Integrate livestock: managed grazing and composting manure returns nutrients to soil. "
     "Documented benefits in similar dryland systems (Iran, Pakistan, Central Asia): "
     "SOC +0.05-0.15% per year; water infiltration +20-40%; input cost -15-25% over 5 years."),

    ("Conservation tillage and no-till for Afghan semi-arid soils "
     "[Source: ICARDA Conservation Agriculture for West Asia North Africa (WANA); "
     "FAO Conservation Agriculture: A Manual for Farmers and Extension Workers 2014; "
     "WUR Soil Physics and Land Management group]: "
     "Conventional deep ploughing (30-40 cm) in semi-arid soils: "
     "breaks soil aggregates, accelerates organic matter oxidation, increases erosion risk. "
     "Reduced-till (15-20 cm rip, then surface seeding): saves fuel 35-50%, maintains surface mulch layer. "
     "Zero-till (direct seeding into untilled soil): "
     "  - Requires a seed drill with disc coulters (available through MAIL/NGO programmes). "
     "  - Wheat in zero-till plots: soil moisture at anthesis +10-18% vs. ploughed (key in dry springs). "
     "  - Yields equal or better than conventional after a 2-3 year transition period. "
     "Subsoil pan formation from repeated shallow ploughing at the same depth: "
     "  - Breaks up with single deep rip every 5-7 years then revert to zero-till. "
     "Stubble retention (not burning): surface residue reduces soil temperature 3-6°C in summer, "
     "critical for soil microbial survival in hot Afghan summers."),

    ("Agroforestry and windbreaks for Afghan farming systems "
     "[Source: FAO Agroforestry for sustainable land management 2017; "
     "WUR Forest Ecology and Forest Management group; "
     "ICARDA agroforestry practices for dryland systems in Central Asia]: "
     "Agroforestry integrates trees with crops and/or livestock for multiple ecosystem services. "
     "Windbreaks (shelterbelts): "
     "  - Poplars (Populus nigra) or willows along field edges reduce wind erosion — "
     "    major problem in Kunduz, Balkh, Herat, Faryab. "
     "  - Windbreak 5-10× tree height protects soil. Trees at 10m height protect a 50-100m field strip. "
     "  - Wind speed reduction 30-50% in sheltered zone; transpiration losses from wheat reduced 10-20%. "
     "Fruit trees on field borders: "
     "  - Mulberry (toot), almond, walnut: no root competition with field crops at >3m from tree trunk. "
     "  - Mulberry provides silk production, fruit, and firewood. Leaves used as livestock fodder. "
     "  - Walnut (Juglans regia) in Kunar, Nuristan, Badakhshan highlands: 20-40 year investment, "
     "    high income once bearing; highly drought-tolerant. "
     "Nitrogen-fixing trees: Elaeagnus (Russian olive / ziziphus) fixes atmospheric N; "
     "leaf litter adds 20-60 kg N/ha per year to adjacent soil."),

    ("Integrated Pest Management (IPM) for Afghan smallholders "
     "[Source: FAO IPM Farmer Field Schools methodology (global); "
     "ICARDA IPM for dryland cereals in West Asia; "
     "WUR Entomology and Biological Control group research]: "
     "IPM minimises pesticide use by combining biological, cultural, and chemical controls. "
     "Sequence of actions: Monitor first, then act. "
     "Step 1 — Field scouting (observe weekly): count pest density and natural enemy populations. "
     "Step 2 — Economic threshold: spray only when pest density exceeds the damage threshold. "
     "For aphids on wheat: economic threshold is >500 aphids per tiller at booting stage. "
     "Step 3 — Cultural controls first: "
     "  - Planting date adjustment: early sowing (Oct 15 north) reduces Hessian fly risk. "
     "  - Crop rotation: breaks pest cycles — reduces Sunn pest (Eurygaster) and wireworm populations. "
     "  - Resistant varieties: Mazar-99, Roshan are both rust-resistant and Hessian-fly tolerant. "
     "Step 4 — Biological control: "
     "  - Parasitic wasps naturally control aphids and armyworm; pesticides kill beneficial insects. "
     "  - Bt (Bacillus thuringiensis) products for caterpillar pests are safe and effective. "
     "Step 5 — Chemical control only if threshold exceeded, use targeted low-toxicity products. "
     "Sunn pest (Eurygaster integriceps) — the most damaging wheat pest in Afghanistan: "
     "  - Causes grain shrivelling and gluten degradation. "
     "  - Control: spray nymphs in May with pyrethroid or neonicotinoid when >1 nymph/m²."),
# ── East Africa — Kenya, Ethiopia, Tanzania ───────────────────────────────

    ("Maize (corn) production systems in East Africa — Kenya, Tanzania, Uganda "
     "[Source: CIMMYT Eastern Africa maize programme; "
     "FAO GIEWS East Africa crop calendar; "
     "IITA Maize Agronomy Research, Ibadan; "
     "WUR Plant Production Systems group — sub-Saharan Africa]: "
     "Maize is the primary staple crop across East Africa, providing 50-70% of caloric intake in rural areas. "
     "Kenya: long rains (March-July) — main season; short rains (Oct-Dec) — secondary season. "
     "Tanzania: unimodal (south/central) — single season Nov-May; bimodal (north) — two maize seasons. "
     "Uganda: two seasons — Mar-Jun (first) and Aug-Nov (second). "
     "NDVI signature for healthy East African maize: 0.45-0.70 at 6-8 weeks after emergence. "
     "Planting density: 44,000-55,000 plants/ha (75 cm row × 25-30 cm within row). "
     "Fertilizer: 60-90 kg N/ha + 40-60 kg P2O5/ha; DAP at planting + CAN (calcium ammonium nitrate) top-dress. "
     "Key pests: Fall Armyworm (Spodoptera frugiperda, since 2016) causes 20-50% yield loss if uncontrolled. "
     "Scouting: examine 20 plants/field; spray if >20% show leaf damage and egg masses present. "
     "Improved varieties: H614D, H6213, WH403 (Kenya); Seedco SC403, SC627 (Tanzania/Uganda). "
     "Yield gap: actual 1.2-2.5 t/ha vs. potential 5-8 t/ha — mainly N-deficiency and water stress."),

    ("East African highland coffee production — Kenya, Ethiopia, Tanzania "
     "[Source: ICO (International Coffee Organisation) crop reports; "
     "Coffee Research Institute Kenya (CRI); "
     "Ethiopian Coffee and Tea Authority; CABI Crop Protection Compendium]: "
     "Arabica coffee (Coffea arabica) dominates East African production — prized for cup quality. "
     "Kenya: SL28 and SL34 varieties; grown 1400-1800 m altitude; bimodal rainfall — long rains flowering trigger. "
     "Ethiopia: Heirloom (landrace) varieties from birthplace of Arabica diversity; Sidamo, Yirgacheffe, Harrar zones. "
     "Tanzania: Kilimanjaro and Mbeya regions; altitude 1200-1900 m; volcanic Andosol soils. "
     "Coffee Berry Disease (CBD — Colletotrichum kahawae): "
     "  - Most destructive coffee disease in Africa; unique to African Arabica. "
     "  - Symptoms: brown/black lesions on green berries; premature drop of infected berries. "
     "  - Copper-based fungicides (copper hydroxide 77%) applied every 2-3 weeks during fruiting. "
     "  - Resistant varieties: Ruiru 11, Batian (Kenya) provide best protection. "
     "Coffee Leaf Rust (CLR — Hemileia vastatrix): "
     "  - Orange-yellow powder on underside of leaves; defoliation reduces yield 40-70% in severe cases. "
     "  - Conditions: high humidity >80%, temperatures 18-28°C optimal for spore germination. "
     "  - Control: copper oxychloride or triazole fungicides; prune for airflow. "
     "Optimal NDVI for healthy coffee canopy: 0.50-0.70; below 0.40 indicates stress or heavy disease."),

    ("Teff (Eragrostis tef) — Ethiopia's unique staple crop "
     "[Source: EIAR (Ethiopian Institute of Agricultural Research); "
     "FAO Ethiopia crop production statistics; "
     "WUR Plant Breeding group — orphan crops research]: "
     "Teff is grown exclusively in Ethiopia and Eritrea — the most important cereal in Ethiopia by area. "
     "Planted on approximately 3 million ha; contributes 15-20% of total caloric supply. "
     "Crop characteristics: C4 photosynthesis; extremely small grain (1 mg — 150× smaller than wheat). "
     "Nutritional advantage: high iron (7.6 mg/100g), calcium (180 mg/100g); gluten-free. "
     "Altitude range: 1000-2800 m — extremely broad adaptation. "
     "Seasonal calendar (highland Ethiopia): "
     "  Kiremt (main) season: sow June-July; harvest October-November. "
     "  Belg (short) season: sow February-March; harvest June-July (limited to favourable areas). "
     "Rainfall requirement: 300-450 mm/season — moderate drought tolerance. "
     "NDVI at peak vegetative stage: 0.35-0.55 (lower than wheat/maize due to fine-leafed canopy). "
     "Fertilizer: DAP 100 kg/ha + Urea 100 kg/ha. NUE lower than wheat — risk of lodging at high N. "
     "Key disease: teff head smut (Ustilago teff); rust (Uromyces eragrostidis). "
     "Water stress at panicle emergence (tillering) causes greatest yield loss."),

    ("East African seasonal rainfall and crop calendar "
     "[Source: FEWS NET East Africa Seasonal Calendar; "
     "ICPAC Greater Horn of Africa Climate Outlook; "
     "FAO GIEWS East Africa bulletin; "
     "NASA SERVIR East Africa regional remote sensing hub]: "
     "East Africa rainfall patterns are driven by the Intertropical Convergence Zone (ITCZ). "
     "Bimodal rainfall zones (Kenya, Uganda, northern Tanzania, southern Ethiopia): "
     "  Long Rains (Masika): March-May, peak April. Primary cropping season. "
     "  Short Rains (Vuli): October-December, peak November. Secondary season. "
     "Unimodal rainfall zones (southern Tanzania, southern Ethiopia, Rwanda, Burundi): "
     "  Single season October/November-April/May. "
     "El Niño (ENSO warm phase): enhances Short Rains in East Africa — flood risk October-December. "
     "La Niña (ENSO cool phase): suppresses rainfall — drought risk in Long Rains. "
     "NDVI anomaly monitoring: MODIS/Sentinel-2 NDVI vs. long-term average detects crop stress. "
     "NDVI anomaly < -0.10 for 3+ weeks during growing season → likely crop failure without intervention. "
     "Satellite rainfall products used for East Africa: CHIRPS v2.0 (UCSB), TAMSAT (University of Reading). "
     "Altitude profoundly modifies rainfall: highlands (>1500m) receive 1000-2000 mm; "
     "semi-arid lowlands (<500m) receive 200-500 mm (northern Kenya, eastern Ethiopia)."),

    ("East African soil types — Nitisols, Vertisols, and Ferralsols "
     "[Source: ISRIC SoilGrids v2.0 — WUR; FAO-UNESCO World Soil Map; "
     "IITA soil characterisation for sub-Saharan Africa]: "
     "Nitisols (Niti soils): deep, well-structured, clay-rich, high fertility; "
     "  dominant in Ethiopian and Kenyan highlands — naturally high SOC (3-6%), good water retention. "
     "  Most productive agricultural soils in East Africa; high suitability for coffee, tea, maize. "
     "  pH 5.5-6.5 — slightly acidic; may need lime for pH-sensitive crops. "
     "Vertisols (black cracking clays): "
     "  Dominant in Ethiopian Rift Valley, parts of Tanzania, eastern Kenya lowlands. "
     "  High clay content (>30% montmorillonite) — shrink-crack in dry season, waterlog when wet. "
     "  Difficult to work: extremely hard when dry, sticky when wet. "
     "  Highly fertile when managed correctly: 40-60 cm deep ripping once per 3 years improves water infiltration. "
     "  Tied ridges (Fanya juu / broad beds) dramatically improve drainage and yield. "
     "Ferralsols (Oxisols): found in humid lowland East Africa; highly weathered; "
     "  very low nutrient reserves; low pH (4.5-5.5) — require lime + all macronutrients. "
     "Red Andosols (volcanic soils): Tanzania/Kilimanjaro, Kenya/Mt. Elgon; "
     "  excellent fertility and structure — ideal for coffee and tea."),

    ("Irrigation and water harvesting in East Africa — smallholder methods "
     "[Source: FAO AQUASTAT East Africa water resources reports; "
     "IWMI (International Water Management Institute) East Africa programmes; "
     "WUR Land and Water Management group; IFAD smallholder water harvesting project data]: "
     "Less than 5% of arable land in East Africa is irrigated — vast untapped potential. "
     "Smallholder irrigation systems in practice: "
     "  1. Motor pumps (petrol/diesel): lifting water from rivers/boreholes to furrow irrigation. "
     "     Cost: 0.05-0.15 USD per cubic meter water. "
     "  2. Treadle pumps: foot-powered, low capital cost ($50-100), lifts 5-8 m head — widely used in Tanzania. "
     "  3. Drip kits (drum/bucket drip): 200-500 USD investment; saves 40-60% water vs. furrow; "
     "     ideal for high-value vegetables (tomato, onion). "
     "Water harvesting for rainfed agriculture: "
     "  Zai pits (planting basins): 20-30 cm diameter holes concentrate water and organic matter — "
     "  yield increase 30-100% in Burkina Faso, Ethiopia, Mali (WFP / ICRISAT studies). "
     "  Half-moon catchments (demi-lunes): capture runoff for trees and crops in arid zones. "
     "  Runoff farming (Ethiopia Tigray): stone bunds (soil and water conservation SWC) on slopes — "
     "  reduce runoff 50-70%, increase NDVI 0.05-0.10 vs. unbunded fields. "
     "NDVI signal from irrigation: water-stressed NDVI 0.20-0.30; irrigated NDVI 0.45-0.65 for same crop stage. "
     "SAR (Sentinel-1 VV) can detect flooded fields: VV > -5 dB indicates surface water presence."),

    ("Fall Armyworm (Spodoptera frugiperda) management in East and West Africa "
     "[Source: CABI Fall Armyworm monitoring network; "
     "FAO Integrated Management of Fall Armyworm on Maize (2018); "
     "IITA biocontrol programme for Fall Armyworm in Africa]: "
     "Fall Armyworm (FAW) invaded Africa in 2016; now present in 44+ sub-Saharan countries. "
     "Damage: larvae feed on maize leaves, silk, and ears — window of 6 weeks from emergence to silking most critical. "
     "Identification: characteristic Y-shaped mark on head capsule; 4 black dots on 8th abdominal segment. "
     "Scouting: check 20-30 plants randomly; damage threshold >20% plants with fresh leaf damage + egg masses. "
     "Cultural controls: "
     "  - Early planting avoids peak FAW moth flight (synchronise with rains). "
     "  - Push-pull intercropping (Desmodium understorey + Napier grass border): "
     "    reduces FAW damage 80% (ICIPE, Rothamsted Research). "
     "  - Maize varieties with moderate FAW resistance: Bt varieties (Wema project East Africa). "
     "Biological control: "
     "  - Bacillus thuringiensis (Bt) products: apply to whorl as spray, 2-3 applications. "
     "  - Metarhizium (entomopathogenic fungus): effective under humid conditions >70% RH. "
     "  - Parasitic wasps (Telenomus remus): mass-released in Nigeria, Ghana — promising. "
     "Chemical: lambda-cyhalothrin, emamectin benzoate; apply to whorl at night when larvae feed. "
     "NDVI + Sentinel-2 green band anomalies can detect FAW-damaged fields (< 0.40 NDVI vs normal 0.55+)."),

    # ── South Asia — Bangladesh, Pakistan ─────────────────────────────────────

    ("Rice production systems in Bangladesh — Boro, Aman, Aus seasons "
     "[Source: Bangladesh Rice Research Institute (BRRI); "
     "FAO Bangladesh country crop production report; "
     "IRRI (International Rice Research Institute) South Asia programme; "
     "WUR Plant Production Systems — rice systems]: "
     "Bangladesh is one of the world's most rice-intensive countries — rice grown on ~11 million ha. "
     "Three rice seasons: "
     "  Boro (dry season, irrigated): Dec/Jan transplanting → May/June harvest. "
     "    Accounts for ~55% of national rice output; requires full irrigation (groundwater/river). "
     "    Hybrid and high-yielding varieties: BRRI dhan28, dhan29 average 5.5-7.0 t/ha. "
     "  Aman (wet season, rainfed + transplanted): July-Aug transplanting → Nov-Dec harvest. "
     "    Dominates rice area; flash-flood tolerant varieties critical (BRRI dhan51, dhan52 — Sub1 gene). "
     "  Aus (pre-kharif, rainfed): March-April sowing → June-July harvest. Declining due to water stress. "
     "Fertilizer (Boro): 150-180 kg N/ha (Urea split 3×), 60-80 kg P2O5 (TSP), 70-90 kg K2O (MoP). "
     "NDVI for healthy Boro rice at heading (booting stage): 0.60-0.75. "
     "Post-harvest Boro: NDVI drops sharply to 0.10-0.20 (harvest + stubble). "
     "Aman transplanting visible by NDVI increase June-July after slow flooded-field establishment."),

    ("Flooding and waterlogging management for South Asian smallholders "
     "[Source: IRRI Flood-tolerant rice research programme; "
     "BRRI Bangladesh flood research; "
     "FAO Coping with water scarcity and floods in South Asia; "
     "WUR Water Resources Management group]: "
     "Bangladesh experiences annual monsoon floods (June-October) affecting 20-30% of the country. "
     "Flash floods: sudden river overbank flooding, duration 3-20 days, depth 0.5-3.0 m. "
     "SUB1 (submergence tolerance) gene in rice: "
     "  Varieties with SUB1 (BRRI dhan51, dhan52, Swarna-Sub1) survive 14-17 days complete submergence. "
     "  Standard varieties die after 3-7 days underwater. "
     "  Sub1 varieties show full yield recovery if water recedes within 14 days. "
     "Satellite monitoring of floods: "
     "  Sentinel-1 SAR (C-band VV/VH): water surface backscatter drops to < -15 dB. "
     "  MODIS Near-Daily surface reflectance: MNDWI > 0.0 indicates open water or saturated soil. "
     "  Flood mapping within 24-48 hours using Sentinel-1 overpasses (6-day repeat, free). "
     "Waterlogging management: "
     "  Raised bed cultivation: 30-45 cm beds with 30 cm furrows — eliminates waterlogging for vegetables. "
     "  Surface drainage channels at 30-50 m intervals drain fields in 24-48 hours. "
     "  Aerobic rice (direct seeding on non-flooded fields): saves 30-50% irrigation water vs. flooded rice. "
     "Post-flood recovery: apply 20 kg Urea/ha as foliar spray within 3 days of waterlogging; "
     "restores N-loss from anaerobic leaching."),

    ("Pakistan wheat production — Punjab and Sindh "
     "[Source: Pakistan Agricultural Research Council (PARC); "
     "CIMMYT Pakistan wheat improvement programme; "
     "FAO Pakistan country food and agriculture profile; "
     "ICARDA West Asia North Africa wheat research]: "
     "Pakistan is the 8th largest wheat producer globally; wheat area ~8.5-9.0 million ha. "
     "Punjab province produces ~75% of national wheat on irrigated Indus Plains. "
     "Sowing calendar: "
     "  Punjab (irrigated): Oct 25 – Nov 25 optimal; Nov 30+ = late sowing with 1-2% yield loss per day. "
     "  Sindh (irrigated): Nov 1 – Nov 30. "
     "  Rainfed (Barani) Potohar Plateau: Oct 15 – Nov 15. "
     "NDVI satellite calibration for Pakistan Punjab: "
     "  Peak NDVI (grain fill, March-April): 0.55-0.75 irrigated; 0.30-0.50 rainfed. "
     "  NDVI < 0.40 at flag leaf stage → severe stress likely reducing yield 30-50%. "
     "Fertilizer (PARC recommendations for irrigated wheat): "
     "  N: 120-150 kg/ha total — half DAP at sowing, half Urea at crown root initiation (Zadoks 13). "
     "  P: 60-75 kg P2O5/ha as DAP at sowing. "
     "  K: only if soil K < 80 ppm (test required); SOP (sulfate of potash) 50 kg/ha. "
     "Key varieties: Galaxy-13, Faisalabad-08, NARC-11, Borlaug-16 (high yield, stripe rust tolerant). "
     "Yellow (stripe) rust (Puccinia striiformis) is the #1 wheat disease in Pakistan — "
     "  spreads from Kyrgyzstan/Afghanistan via wind; triazole fungicides (tebuconazole, propiconazole) effective."),

    ("Waterlogging and salinity in Pakistan's Indus Plains "
     "[Source: Pakistan WAPDA (Water and Power Development Authority) drainage reports; "
     "FAO Pakistan land and water resources; "
     "IWMI Pakistan irrigation and drainage research; "
     "WUR Soil Geography and Landscape group]: "
     "Pakistan's canal-irrigated Indus Plains suffer from twin problems of waterlogging and soil salinity. "
     "Extent: ~6 million ha waterlogged, ~4 million ha salt-affected — limiting yields on 30-40% of Punjab irrigated land. "
     "Cause: over-irrigation + poor drainage raises water table to < 2 m depth → waterlogging → salt accumulation. "
     "Salinity effects on wheat: "
     "  EC < 6 dS/m: no yield effect. "
     "  EC 6-10 dS/m: 10-25% yield reduction. "
     "  EC > 10 dS/m: severe reduction; salt-tolerant varieties required. "
     "Satellite detection: "
     "  Saline patches show low NDVI (< 0.20 even when neighbours are healthy 0.55+). "
     "  White efflorescence on surface visible in Sentinel-2 true-colour (B4-B3-B2). "
     "  High BSI (Bare Soil Index) > 0.1 on Sentinel-2 in growing season = likely salt-affected. "
     "Management options: "
     "  Vertical drainage (tubewells): pump groundwater below root zone — used successfully in Punjab. "
     "  Horizontal drainage tiles at 1.5-2 m depth: expensive but permanent. "
     "  Gypsum (calcium sulfate) application: 5-10 t/ha reclaims sodic (Na-rich) soils. "
     "  Salt-tolerant varieties: Kharchia-65, LU-26 (Pakistan wheat); BRRI dhan47 (rice Bangladesh)."),

    ("Cotton production in Pakistan — Bt cotton, irrigation, and pest management "
     "[Source: Pakistan Central Cotton Research Institute (CCRI), Multan; "
     "FAO Pakistan cotton sector report; "
     "CABI Cotton Crop Protection Compendium]: "
     "Pakistan is the 4th largest cotton producer globally; grown on ~2.2 million ha (Punjab, Sindh). "
     "Cotton contributes ~5% GDP and 55% of export earnings — economically critical crop. "
     "Bt cotton (Bacillus thuringiensis gene): 95%+ adoption in Pakistan; controls bollworm larvae. "
     "Sowing: April 15 – May 30 in Punjab; earlier (Mar-Apr) in Sindh. "
     "Irrigation: 8-12 irrigations per season; water-sensitive stages: flowering and boll formation. "
     "Satellite monitoring (Sentinel-2): "
     "  Healthy cotton at boll-opening stage: NDVI 0.40-0.65; EVI 0.30-0.50. "
     "  MNDWI at picking (Sep-Oct) rises as boll opens — canopy dries out. "
     "  Cotton Leaf Curl Virus (CLCuV): plant shows curling, yellowing; NDVI drops to 0.20-0.35. "
     "Major pests (post-Bt emergence): "
     "  Whitefly (Bemisia tabaci): primary vector of CLCuV; thrives in hot dry conditions. "
     "  Mealybug (Phenacoccus solenopsis): invasive since 2005; sprayed with chlorpyrifos, profenofos. "
     "  Pink bollworm (Pectinophora gossypiella): Bt controls this effectively. "
     "Harvesting: 3-4 machine pickings Sep-Nov; hand-picking still common in smallholder fields."),

    ("South Asian Rabi (winter) crop calendar — wheat, chickpea, mustard "
     "[Source: ICAR (Indian Council of Agricultural Research); "
     "FAO South Asia food security brief; "
     "CIMMYT South Asia wheat programme; "
     "WUR Plant Production Systems group]: "
     "Rabi (winter) season in South Asia: crops sown October-December, harvested March-May. "
     "Wheat: "
     "  India (major belt): Punjab, Haryana, UP — sow Nov 10-30; harvest April-May. "
     "  Pakistan Punjab: sow Oct 25 – Nov 25; harvest April-May. "
     "  Bangladesh (very limited): Dec-Jan sowing (only possible after Aman rice harvest). "
     "  NDVI at grain fill (March): 0.55-0.75 irrigated Punjab/Haryana. "
     "Chickpea (Cicer arietinum — major Rabi pulse): "
     "  Sown October-November after Kharif; harvested February-April. "
     "  India is the world's largest producer (65% of global supply). "
     "  Drought-tolerant — grown rainfed on residual soil moisture. "
     "  NDVI at pod-fill: 0.35-0.55 (lower than cereals due to compact canopy). "
     "Mustard/rapeseed (Brassica juncea): "
     "  Sown October-November; harvested February-March. "
     "  Punjab (Pakistan/India): 2-3 million ha under mustard/rapeseed. "
     "  NDVI peak at flowering (yellow flowers): relatively low (0.35-0.50) vs. wheat. "
     "Kharif (summer monsoon) season: June-September — rice, cotton, maize, sugarcane."),

    # ── Central America — Guatemala, Honduras, coffee, maize ──────────────────

    ("Coffee Leaf Rust (La Roya) — Hemileia vastatrix — Central America "
     "[Source: CABI Crop Protection Compendium — Hemileia vastatrix; "
     "PROMECAFE (Central American Coffee Research Programme); "
     "World Coffee Research (WCR) disease management guides; "
     "Vandermeer et al. 2010 — Ecology of Coffee Agroecosystems]: "
     "Coffee Leaf Rust (La Roya) is the most destructive coffee disease globally, including Central America. "
     "2012-2013 epidemic across Central America and Mexico destroyed ~50% of coffee crop in affected areas. "
     "Economic losses exceeded $1 billion; 400,000 rural workers lost income (IICA data). "
     "Causal agent: Hemileia vastatrix — obligate fungal pathogen. "
     "Symptoms: orange-yellow powdery pustules on lower leaf surface; defoliation weakens the plant. "
     "Favourable conditions: temperature 21-25°C, humidity >80%, frequent light rain or heavy dew. "
     "At altitude >1400m: cooler temperatures slow rust development — less severe historically, but "
     "climate change is shifting the disease upslope (new threat to traditionally safe high-altitude farms). "
     "Fungicide management: "
     "  Preventive: copper hydroxide or copper oxychloride sprays every 3-4 weeks during rainy season. "
     "  Curative: triazole fungicides (tebuconazole, myclobutanil, hexaconazole) after first sign. "
     "  Apply in 3-spray programme at budbreak, after fruit set, and 60 days later. "
     "Resistant varieties: Catimor derivatives (Lempira, IHCAFE 90 — Honduras); "
     "  Marsellesa, Centroamericano (WCR); Geisha (tolerant but low resistance). "
     "NDVI for coffee under La Roya attack: 0.35-0.45 (vs. healthy 0.55-0.70 for same age)."),

    ("Maize (milpa) systems in Guatemala and Honduras "
     "[Source: FAO Regional Office for Latin America — maize programme; "
     "CIMMYT Latin America maize improvement (LAMAIZE); "
     "WFP Guatemala food security assessment; "
     "WUR Farming Systems Ecology — Mesoamerican food systems]: "
     "The traditional Maya milpa system is the dominant smallholder farming system in Guatemala and Honduras. "
     "Milpa: intercrop of maize (Zea mays) + beans (Phaseolus vulgaris) + squash (Cucurbita pepo). "
     "Ecological benefits: beans fix 40-100 kg N/ha (reduces fertilizer need); "
     "  squash ground cover reduces weeds 40-60% and retains soil moisture. "
     "Maize seasonal calendar (Central America): "
     "  Primera season (1st rains): April-May sowing → August-September harvest (major). "
     "  Postrera season (2nd rains): July-August sowing → October-December harvest (secondary). "
     "  Apante (Honduras Pacific slope): September → January-February (limited area). "
     "NDVI signature of milpa vs. monoculture maize: 0.05-0.10 higher NDVI in milpa due to bean/squash fill. "
     "Soils: Guatemala highlands dominated by volcanic Andosols — naturally fertile but vulnerable to erosion. "
     "  Slopes >15%: terraces (curvas de nivel) or agroforestry essential to prevent topsoil loss. "
     "Fertilizer use: relatively low by regional standards — 60-90 kg N/ha for maize (DAP + Urea). "
     "Food security: maize (tortilla) provides 60-70% of caloric intake in indigenous communities. "
     "Grey leaf spot (Cercospora zeae-maydis) and common rust (Puccinia sorghi) are primary maize diseases."),

    ("Coffee production systems in Central America — agroforestry and altitude "
     "[Source: WCR (World Coffee Research) Annual Report 2022-23; "
     "PROMECAFE technical bulletins; "
     "Somarriba et al. 2013 — agroforestry coffee systems, Agroforestry Systems journal; "
     "CABI Coffee Crop Protection Compendium]: "
     "Central America (Guatemala, Honduras, Costa Rica, El Salvador, Nicaragua) produces "
     "premium Arabica coffee grown under shade agroforestry and in open sun systems. "
     "Honduras: largest coffee producer in Central America (~8 million 60kg bags/year). "
     "  Altitude 1000-2200 m; best quality above 1500 m (SHG — Strictly Hard Bean grade). "
     "Guatemala: Antigua, Huehuetenango, Atitlán volcanic zones; altitude 1200-1900 m. "
     "Shade-grown (agroforestry) coffee: "
     "  Inga (Inga edulis) — primary shade tree; fixes 60-100 kg N/ha via leaf litter decomposition. "
     "  Benefits: biodiversity, microclimate regulation (reduces heat stress), lower fertilizer cost. "
     "  NDVI of shade coffee: 0.50-0.70 (higher than sun coffee due to shade tree canopy). "
     "Sun coffee (tecnificado): NDVI 0.40-0.60; higher yield per plant but greater input requirements. "
     "Flowering: triggered by first rain after dry season — synchronised across a farm. "
     "Harvest (veraison to ripe cherry): 8-9 months from flower to harvest. "
     "Altitude effect: every 100 m increase above 1000 m slows cherry development ~5 days — "
     "higher altitude = slower maturation = more complex cup profile."),

    ("Maize gray leaf spot and corn diseases — Central America and global "
     "[Source: CABI Gray Leaf Spot Crop Compendium; "
     "CIMMYT maize disease management guide; "
     "Lipps 1998 — gray leaf spot of corn, Ohio State University Extension; "
     "WUR Laboratory of Plant Pathology]: "
     "Gray Leaf Spot (Cercospora zeae-maydis): "
     "  Most economically important foliar disease of maize in humid tropics and subtropics. "
     "  Symptoms: rectangular grey-brown lesions bounded by leaf veins; lesions 1-5 cm long. "
     "  Optimal conditions: minimum temperature 22°C, extended leaf wetness >12 hours per night. "
     "  Management: resistant hybrids (most important — check local CIMMYT variety trials); "
     "    azoxystrobin or mancozeb fungicide at V10-VT growth stage if >5% leaf area affected. "
     "Common rust (Puccinia sorghi): "
     "  Brick-red pustules on both leaf surfaces; spores wind-dispersed. "
     "  Less severe in lowlands (hot) but damaging in cool highlands (above 1500m). "
     "  Management: resistant varieties; foliar triazole fungicide if applied early. "
     "Corn smut (Ustilago maydis): "
     "  Silver-grey galls on ears, leaves, tassels — rupture to release black spore masses. "
     "  Higher in dryland stressed fields; cultural control (remove galls before rupture). "
     "  No effective chemical control — resistant hybrid selection best management. "
     "Northern Corn Leaf Blight (Exserohilum turcicum): "
     "  Large cigar-shaped grey-green lesions; severe in cool humid conditions. "
     "  Propiconazole or azoxystrobin fungicide + resistant hybrids. "
     "NDVI monitoring: gray leaf spot causes NDVI decline 0.10-0.20 units vs. healthy adjacent fields."),

    ("Volcanic soil (Andosol) management in Central America and East Africa "
     "[Source: ISRIC SoilGrids v2.0 — WUR; "
     "FAO Soils of Volcanic Regions 2011; "
     "Dahlgren et al. 2004 Andosols review, Advances in Agronomy]: "
     "Andosols (Andisols) form on volcanic ash deposits — present in Guatemala, Honduras, Costa Rica, "
     "Colombia, Japan, New Zealand, and East African Rift Valley (Ethiopia, Tanzania, Uganda). "
     "Natural advantages: "
     "  High SOC: 5-15% (vs. 0.5-2% in most tropical soils) — dark colour from allophane minerals. "
     "  High water holding capacity: 50-200% of soil weight — drought buffer. "
     "  Light, porous structure — excellent drainage and root penetration. "
     "  Generally high pH (5.5-7.0) and medium to high CEC (cation exchange capacity). "
     "Key challenge — phosphorus fixation: "
     "  Allophane and imogolite in Andosols strongly fix phosphorus — largest nutrient constraint. "
     "  P fixation capacity 1000-3000 mg P/kg soil (vs. 50-200 in other soils). "
     "  Solution: band P fertilizer close to seed/root; apply organic matter to reduce P fixation sites. "
     "  High P application rates required: 100-200 kg P2O5/ha initial; 60-90 kg/ha maintenance. "
     "  Residual P builds up over 3-5 years — reduce rate after establishment. "
     "Potassium and zinc status: variable; test annually — volcanic soils can be deficient in K and Zn. "
     "NDVI of crops on Andosols: typically 0.05-0.15 higher than adjacent non-volcanic soils "
     "  due to superior water and nutrient supply."),

    ("Banana and plantain production — Honduras and Central America "
     "[Source: FAO Banana Market Review 2022; "
     "CORBANA (Costa Rica Banana Research Corporation); "
     "Bioversity International Musa research; "
     "WUR Plant Sciences group — banana production systems]: "
     "Honduras is one of the world's top banana exporters — Cavendish variety on north coast. "
     "Central American banana belt: Honduras (Sula Valley), Guatemala (Pacific), Costa Rica (Caribbean). "
     "Production system: intensive monoculture with drip irrigation, fertigation, and full pest management. "
     "NDVI of healthy banana plantation: 0.60-0.80 (high biomass, large leaf area). "
     "Seasonal production: bananas bear fruit 9-11 months after planting; year-round harvest. "
     "Major diseases: "
     "  Fusarium wilt (Panama disease Race 4 — TR4): "
     "    Destroys vascular system; no chemical cure; spreads via infected soil, water, tools. "
     "    Cavendish (currently grown worldwide) is susceptible to Tropical Race 4 — existential threat. "
     "    Management: strict quarantine, soil testing, biocontrol agents (Trichoderma). "
     "    NDVI drops sharply 0.20-0.30 units below healthy before visual wilting visible. "
     "  Black Sigatoka (Mycosphaerella fijiensis): "
     "    Fungal leaf disease; reduces photosynthesis, causes premature fruit ripening. "
     "    Control: 40-60 aerial or knapsack fungicide applications per year (oil-based formulations). "
     "    Biological control research ongoing (WUR/Bioversity International). "
     "  Moko (Ralstonia — bacterial wilt): affects Gros Michel plantains; spreads via infected tools. "
     "Plantain (cooking banana): smallholder staple across Central America and West/Central Africa. "
     "  NDVI 0.50-0.70; intercropped with maize, beans, or coffee as shade."),

    ("Climate-smart agriculture for smallholders — global principles "
     "[Source: FAO Climate-Smart Agriculture Sourcebook 2013 (updated 2023); "
     "CGIAR Research Program on Climate Change, Agriculture and Food Security (CCAFS); "
     "WUR Climate Change and Biosphere Group; "
     "IPCC AR6 Working Group II — Food, Fibre and Forest Products chapter]: "
     "Climate-Smart Agriculture (CSA) achieves three goals simultaneously: "
     "  1. Sustainably increase productivity and incomes. "
     "  2. Adapt and build resilience to climate change. "
     "  3. Reduce or remove greenhouse gas emissions where possible. "
     "Satellite-based NDVI and LST monitoring is a core CSA tool for: "
     "  - Detecting drought onset 2-4 weeks before visible crop wilting. "
     "  - Identifying heat stress events using MODIS Land Surface Temperature (LST > 35°C at anthesis). "
     "  - Tracking long-term NDVI trend decline as indicator of land degradation. "
     "Adaptation practices applicable across smallholder contexts globally: "
     "  Drought: drought-tolerant varieties; mulching; soil organic carbon building; water harvesting. "
     "  Flooding: raised beds; SUB1 gene rice; flood-tolerant maize; improved drainage. "
     "  Heat stress: shade trees (agroforestry); irrigation timing (night/early morning); "
     "    early-maturing varieties that escape terminal heat at grain fill. "
     "  Late onset of rains: direct seeding instead of transplanting to save days; "
     "    short-duration varieties; soil moisture conservation (mulch, minimum till). "
     "Most cost-effective interventions by satellite signal: "
     "  - Fields with VCI < 35: drought stress — prioritise water; yield loss 20-40% likely. "
     "  - Fields with NDVI anomaly < -0.15 for 3+ weeks: structural problem — fertilizer or disease. "
     "  - MNDWI > 0.10 in non-riparian fields: waterlogging — open surface drainage channels."),
]


@app.route("/rag/setup", methods=["GET"])
def rag_setup():
    """Return the one-time SQL migration needed to enable pgvector in Supabase."""
    return jsonify({
        "sql":          _RAG_SETUP_SQL,
        "instructions": (
            "1. Open Supabase dashboard → Database → SQL Editor. "
            "2. Paste the SQL above and click Run. "
            "3. Restart the ZaminAI API. "
            "4. Call POST /rag/seed to load built-in Afghan farming knowledge."
        ),
        "rag_ok":       rag_ok
    })


@app.route("/rag/seed", methods=["POST", "OPTIONS"])
def rag_seed():
    """Seed the vector DB with built-in Afghan farming knowledge. Call once after setup."""
    if request.method == "OPTIONS": return jsonify({}), 200
    if not rag_ok:
        return jsonify({"error": "RAG not ready — call GET /rag/setup first", "rag_ok": False}), 503
    if not GEMINI_KEY:
        return jsonify({"error": "GEMINI_API_KEY required for embeddings"}), 503
    stored = 0
    for doc in _RAG_SEED_DOCS:
        if rag_store(doc, source="seed_knowledge", metadata={"type": "domain_knowledge"}):
            stored += 1
    return jsonify({
        "ok":        stored > 0,
        "stored":    stored,
        "total":     len(_RAG_SEED_DOCS),
        "message":   f"Seeded {stored}/{len(_RAG_SEED_DOCS)} knowledge chunks"
    })


@app.route("/rag/ingest", methods=["POST", "OPTIONS"])
def rag_ingest():
    """Add knowledge chunks to the vector DB.
    Body: {text: "..."} for one chunk  OR  {chunks: ["...", "..."]} for bulk.
    Optional: source (str), metadata (dict).
    """
    if request.method == "OPTIONS": return jsonify({}), 200
    if not rag_ok:
        return jsonify({"error": "RAG not ready — call GET /rag/setup first", "rag_ok": False}), 503
    if not GEMINI_KEY:
        return jsonify({"error": "GEMINI_API_KEY required for embeddings"}), 503
    try:
        d        = request.get_json(force=True)
        source   = d.get("source", "manual")
        metadata = d.get("metadata", {})

        if "text" in d:
            ok = rag_store(d["text"], source=source, metadata=metadata)
            return jsonify({"ok": ok, "stored": 1 if ok else 0})

        chunks = d.get("chunks", [])
        if not chunks:
            return jsonify({"error": "Provide 'text' or 'chunks'"}), 400
        stored = sum(
            1 for c in chunks
            if isinstance(c, str) and c.strip() and rag_store(c, source=source, metadata=metadata)
        )
        return jsonify({"ok": stored > 0, "stored": stored, "total": len(chunks)})
    except Exception as e:
        log.error(f"/rag/ingest: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/rag/search", methods=["POST", "OPTIONS"])
def rag_search():
    """Test RAG retrieval for a question.
    Body: {question: "...", top_k: 4, threshold: 0.65}
    """
    if request.method == "OPTIONS": return jsonify({}), 200
    if not rag_ok:
        return jsonify({"error": "RAG not ready", "rag_ok": False}), 503
    try:
        d         = request.get_json(force=True)
        question  = d.get("question", "").strip()
        top_k     = int(d.get("top_k", 4))
        threshold = float(d.get("threshold", 0.50))
        if not question:
            return jsonify({"error": "question required"}), 400
        chunks = rag_retrieve(question, top_k=top_k, threshold=threshold)
        return jsonify({
            "ok": True, "question": question,
            "chunks": chunks, "count": len(chunks), "threshold": threshold
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/rag/stats", methods=["GET"])
def rag_stats():
    """Return RAG database statistics."""
    if not sb_ok:
        return jsonify({"ok": False, "rag_ok": False, "error": "Database not available"}), 503
    try:
        res    = sb.table("knowledge_chunks").select("source", count="exact").execute()
        total  = res.count or 0
        by_src = {}
        for row in (res.data or []):
            s = row.get("source", "unknown")
            by_src[s] = by_src.get(s, 0) + 1
        return jsonify({
            "ok":           True,
            "rag_ok":       rag_ok,
            "total_chunks": total,
            "by_source":    by_src,
            "embed_model":  EMBED_MODEL,
            "embed_dim":    EMBED_DIM,
            "vector_db":    "supabase_pgvector"
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════════════
# REGENERATIVE AGRICULTURE — Cultivation Story + Crop Rotation
# Endpoints: POST /regen/log  · GET /regen/history/<farmer_id>
#            POST /regen/recommend  · GET /regen/setup
# ════════════════════════════════════════════════════════════════════════════════

CROP_ROTATION_RULES = {
    "wheat":       {"next_best":["chickpea","mung_bean","lentil"],"next_good":["vegetables","sunflower"],"avoid":["wheat"],"regen_reason":"Legumes fix nitrogen after wheat exhausts soil N — reduces fertilizer need 30–40%.","regen_reason_fa":"بقولات بعد از گندم نیتروژن خاک را تثبیت می‌کند و نیاز به کود کمتر می‌شود.","regen_reason_ps":"بقولات د غنمو وروسته د خاورې نایتروجن ثابتوي. کود کمیږي."},
    "chickpea":    {"next_best":["wheat","vegetables"],"next_good":["saffron","sunflower"],"avoid":["chickpea","lentil","mung_bean"],"regen_reason":"After chickpea, soil is N-rich — plant wheat or vegetables to use that fixed nitrogen.","regen_reason_fa":"بعد از نخود، خاک پر از نیتروژن است — گندم یا سبزیجات بکارید.","regen_reason_ps":"د نخود وروسته خاوره د نایتروجن ډکه ده — غنم یا سبزیجات وکرئ."},
    "lentil":      {"next_best":["wheat","vegetables"],"next_good":["cotton","sunflower"],"avoid":["lentil","chickpea"],"regen_reason":"Lentil enriches soil nitrogen. Follow with wheat or high-demand vegetables to use it.","regen_reason_fa":"عدس خاک را غنی می‌کند. بعد از آن گندم یا سبزیجات پرتقاضا بکارید.","regen_reason_ps":"مسور خاوره شتمنه کوي. د هغه وروسته غنم یا سبزیجات وکرئ."},
    "mung_bean":   {"next_best":["wheat","vegetables"],"next_good":["cotton"],"avoid":["mung_bean","chickpea"],"regen_reason":"Mung bean fixes N and breaks wheat disease cycles — excellent summer bridge crop.","regen_reason_fa":"ماش نیتروژن ثابت می‌کند و چرخه بیماری گندم را می‌شکند.","regen_reason_ps":"ماش نایتروجن ثابتوي او د غنمو د ناروغۍ دوره ماتوي."},
    "vegetables":  {"next_best":["wheat","chickpea","lentil"],"next_good":["sunflower"],"avoid":["vegetables"],"regen_reason":"Vegetables deplete soil fast. Rest with wheat or restore N with legumes next season.","regen_reason_fa":"سبزیجات خاک را سریع خسته می‌کند. با گندم استراحت یا با بقولات نیتروژن تجدید کنید.","regen_reason_ps":"سبزیجات خاوره ژر ستړې کوي. د غنم سره آرام یا د بقولاتو سره نایتروجن بیارغاوئ."},
    "saffron":     {"next_best":["chickpea","wheat"],"next_good":["vegetables"],"avoid":["saffron"],"regen_reason":"Saffron is perennial (7–10 yr). Intercrop rows with legumes to maintain soil health between bulbs.","regen_reason_fa":"زعفران چندین ساله است. ردیف‌های بین آن را با بقولات بکارید تا خاک سالم بماند.","regen_reason_ps":"زعفران ددې کلیزې دی. د هغه تر منځ قطارونه د بقولاتو سره وکرئ."},
    "cotton":      {"next_best":["wheat","chickpea"],"next_good":["vegetables"],"avoid":["cotton"],"regen_reason":"Cotton depletes K and micronutrients. Rotate with wheat then legumes to rebuild soil.","regen_reason_fa":"پنبه پتاسیم خاک را تخلیه می‌کند. با گندم و بقولات تناوب کنید.","regen_reason_ps":"کپاس د خاورې پوتاشیم کموي. د غنم او بقولاتو سره تناوب وکرئ."},
    "rice":        {"next_best":["wheat","lentil"],"next_good":["vegetables"],"avoid":["rice"],"regen_reason":"After rice, drain and plant dry-season wheat or legumes to prevent waterlogged soil degradation.","regen_reason_fa":"بعد از برنج، زمین را خشک کنید و گندم یا بقولات فصل خشک بکارید.","regen_reason_ps":"د وريجو وروسته، ځمکه وچه کړئ او وچ موسم کې غنم یا بقولات وکرئ."},
    "sunflower":   {"next_best":["wheat","chickpea"],"next_good":["vegetables"],"avoid":["sunflower"],"regen_reason":"Sunflower deep roots break hardpan. Follow with shallow-rooted wheat to take advantage.","regen_reason_fa":"ریشه عمیق آفتابگردان لایه سخت خاک را می‌شکند. بعد از آن گندم بکارید.","regen_reason_ps":"د لمروال ژورې ريښې د سختې پوستکۍ ماتوي. د هغه وروسته غنم وکرئ."},
    "pomegranate": {"next_best":["chickpea","lentil"],"next_good":["wheat"],"avoid":[],"regen_reason":"Intercrop pomegranate rows with legumes to fix N and reduce fertilizer dependency.","regen_reason_fa":"ردیف‌های انار را با بقولات بکارید تا نیتروژن ثابت شود.","regen_reason_ps":"د انار قطارونه د بقولاتو سره وکرئ ترڅو نایتروجن ثابت شي."},
    "orchard":     {"next_best":["chickpea","lentil"],"next_good":["wheat"],"avoid":[],"regen_reason":"Intercrop orchard rows with legumes to fix N and protect topsoil from erosion.","regen_reason_fa":"ردیف‌های باغ را با بقولات بکارید تا نیتروژن ثابت شود.","regen_reason_ps":"د باغ قطارونه د بقولاتو سره وکرئ ترڅو نایتروجن ثابت شي."},
    "bare_fallow": {"next_best":["chickpea","wheat","vegetables"],"next_good":["lentil","mung_bean"],"avoid":[],"regen_reason":"Fallow field — now is the time to plant. Start with chickpea to build soil N, then wheat next year.","regen_reason_fa":"زمین بایر است — وقت کاشت است. نخود بکارید تا خاک را بسازید، سپس سال آینده گندم.","regen_reason_ps":"ځمکه خالي ده — د کرلو وخت دی. لومړی نخود وکرئ، راتلونکي کال غنم."},
    "mixed_unknown":{"next_best":["wheat","chickpea"],"next_good":["vegetables"],"avoid":[],"regen_reason":"Start a cultivation record this season — we can give better rotation advice as history builds.","regen_reason_fa":"سابقه کشت را این فصل ثبت کنید — سال بعد توصیه بهتری خواهیم داد.","regen_reason_ps":"دا موسم د کرلو ریکارډ ثبت کړئ — راتلونکي کال به ښه مشوره درکو."},
}

CROP_VALUE_TABLE = {
    "saffron":    {"label_en":"Saffron","label_fa":"زعفران","label_ps":"زعفران","value_usd_ha":"3,000–8,000","yield_kg_ha":"5–8 kg","water":"Low","soil_benefit":"Medium","market":"Export (premium)","regen_score":4,"notes":"Best in Herat, Kandahar. Perennial 7–10 yr."},
    "vegetables": {"label_en":"Vegetables","label_fa":"سبزیجات","label_ps":"سبزیجات","value_usd_ha":"800–2,000","yield_kg_ha":"8,000–20,000 kg","water":"High","soil_benefit":"Low","market":"Local / urban","regen_score":2,"notes":"High value but needs irrigation. Depletes soil fast."},
    "pomegranate":{"label_en":"Pomegranate","label_fa":"انار","label_ps":"انار","value_usd_ha":"500–1,500","yield_kg_ha":"5,000–15,000 kg","water":"Medium","soil_benefit":"Medium","market":"Export (Kandahar)","regen_score":3,"notes":"Kandahar specialty. 3-yr establishment before first yield."},
    "wheat":      {"label_en":"Wheat","label_fa":"گندم","label_ps":"غنم","value_usd_ha":"300–600","yield_kg_ha":"2,000–4,000 kg","water":"Medium","soil_benefit":"Low","market":"Staple / food security","regen_score":2,"notes":"Food security crop. Low cash return but consistent demand."},
    "chickpea":   {"label_en":"Chickpea","label_fa":"نخود","label_ps":"نخود","value_usd_ha":"250–500","yield_kg_ha":"800–1,500 kg","water":"Low","soil_benefit":"High (N-fix)","market":"Local + export","regen_score":5,"notes":"Best regenerative crop. Fixes N, saves up to 40% fertilizer."},
    "lentil":     {"label_en":"Lentil","label_fa":"عدس","label_ps":"مسور","value_usd_ha":"200–400","yield_kg_ha":"600–1,200 kg","water":"Low","soil_benefit":"High (N-fix)","market":"Local + export","regen_score":5,"notes":"Excellent N-fixer. Low water need. Strong local demand."},
    "mung_bean":  {"label_en":"Mung Bean","label_fa":"ماش","label_ps":"ماش","value_usd_ha":"300–500","yield_kg_ha":"600–1,200 kg","water":"Low","soil_benefit":"High (N-fix)","market":"Local","regen_score":5,"notes":"Summer legume. Breaks wheat disease cycles."},
    "cotton":     {"label_en":"Cotton","label_fa":"پنبه","label_ps":"کپاس","value_usd_ha":"400–700","yield_kg_ha":"1,500–2,500 kg","water":"High","soil_benefit":"Depletes","market":"Export (south)","regen_score":1,"notes":"South only (Helmand). Heavy on water and soil nutrients."},
    "sunflower":  {"label_en":"Sunflower","label_fa":"آفتابگردان","label_ps":"لمروال","value_usd_ha":"250–500","yield_kg_ha":"1,500–2,500 kg","water":"Medium","soil_benefit":"Medium","market":"Oil / local","regen_score":3,"notes":"Deep taproot breaks compaction. Good transition crop."},
    "rice":       {"label_en":"Rice","label_fa":"برنج","label_ps":"وريجي","value_usd_ha":"400–900","yield_kg_ha":"2,000–4,000 kg","water":"Very High","soil_benefit":"Low","market":"Local / premium","regen_score":1,"notes":"North (Kunduz/Takhar) only. Very high water need."},
    "orchard":    {"label_en":"Orchard / Trees","label_fa":"باغ","label_ps":"باغ","value_usd_ha":"400–1,200","yield_kg_ha":"3,000–10,000 kg","water":"Medium","soil_benefit":"Medium","market":"Local + export","regen_score":3,"notes":"Perennial. Intercrop with legumes for soil health."},
}


def db_save_cultivation(farmer_id, field_id, crop, year, season, notes, province,
                        source="farmer_reported", ndvi=None):
    if not sb_ok:
        return None
    try:
        res = sb.table("cultivation_history").insert({
            "farmer_id":     farmer_id,
            "field_id":      field_id,
            "crop":          crop,
            "year":          year,
            "season":        season,
            "notes":         notes,
            "province":      province,
            "source":        source,
            "ndvi_at_peak":  ndvi,
            "created_at":    datetime.utcnow().isoformat()
        }).execute()
        log.info(f"✓ Cultivation logged: {crop} {year} farmer={farmer_id}")
        return res.data[0] if res.data else None
    except Exception as e:
        log.error(f"db_save_cultivation: {e}")
        return None


def db_get_cultivation_history(farmer_id, field_id=None):
    if not sb_ok:
        return []
    try:
        q = sb.table("cultivation_history").select("*").eq("farmer_id", farmer_id)
        if field_id:
            q = q.eq("field_id", field_id)
        res = (q.order("year", desc=True)
                .order("created_at", desc=True)
                .limit(20).execute())
        return res.data or []
    except Exception as e:
        log.error(f"db_get_cultivation_history: {e}")
        return []


def regen_build_recommendation(history, province, current_crop=None, rain=None):
    last_crop = current_crop or (history[0]["crop"] if history else "mixed_unknown")
    rule  = CROP_ROTATION_RULES.get(last_crop, CROP_ROTATION_RULES["mixed_unknown"])
    ptype = get_province_type(province)

    def province_ok(c):
        if c == "saffron"     and ptype not in ("west","south"): return False
        if c == "rice"        and ptype != "north":              return False
        if c == "cotton"      and ptype != "south":              return False
        if c == "pomegranate" and ptype not in ("south","east"): return False
        return True

    def water_rank(c):
        w = CROP_VALUE_TABLE.get(c, {}).get("water","Medium")
        return {"Low":0,"Medium":1,"High":2,"Very High":3}.get(w, 1)

    best = [c for c in rule["next_best"] if province_ok(c)]
    good = [c for c in rule["next_good"] if province_ok(c)]
    if rain and rain < 150:
        best = sorted(best, key=water_rank)
        good = sorted(good, key=water_rank)

    recommended = []
    for c in best[:3]:
        recommended.append({"crop":c,"priority":"best",**CROP_VALUE_TABLE.get(c,{})})
    for c in good[:2]:
        recommended.append({"crop":c,"priority":"good",**CROP_VALUE_TABLE.get(c,{})})

    return {
        "last_crop":   last_crop,
        "recommended": recommended,
        "explanation": {
            "why":          rule["regen_reason"],
            "why_fa":       rule["regen_reason_fa"],
            "why_ps":       rule["regen_reason_ps"],
            "last_crop":    last_crop,
            "avoid":        rule.get("avoid",[]),
            "years_of_data":len(history),
            "regen_tips": [
                "Add 2–3 tons compost per jereb before planting — improves water retention 20%",
                "Leave crop stubble on field after harvest to feed soil microbes",
                "Minimum tillage — fewer passes protects soil structure and reduces erosion",
                "Mulch between crop rows to reduce water evaporation by 30%"
            ]
        },
        "crop_comparison": [{"crop":c,**v} for c,v in CROP_VALUE_TABLE.items()]
    }


@app.route("/regen/setup", methods=["GET"])
def regen_setup_sql():
    sql = (
        "-- Run once in Supabase SQL Editor\n"
        "CREATE TABLE IF NOT EXISTS cultivation_history (\n"
        "  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),\n"
        "  farmer_id    uuid,\n"
        "  field_id     uuid,\n"
        "  crop         TEXT NOT NULL,\n"
        "  season       TEXT,\n"
        "  year         INTEGER,\n"
        "  notes        TEXT,\n"
        "  source       TEXT DEFAULT 'farmer_reported',\n"
        "  ndvi_at_peak FLOAT,\n"
        "  province     TEXT,\n"
        "  created_at   TIMESTAMP DEFAULT NOW()\n"
        ");\n"
        "CREATE INDEX IF NOT EXISTS idx_cult_farmer ON cultivation_history(farmer_id);\n"
        "CREATE INDEX IF NOT EXISTS idx_cult_field  ON cultivation_history(field_id);\n"
    )
    return jsonify({"ok": True, "sql": sql,
                    "steps": ["1. Open Supabase → SQL Editor",
                              "2. Paste the sql above and click Run",
                              "3. POST /regen/log to start logging"]})


@app.route("/regen/log", methods=["POST","OPTIONS"])
def regen_log():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    d    = request.get_json(silent=True) or {}
    crop = (d.get("crop") or "").strip()
    if not crop:
        return jsonify({"ok": False, "error": "crop is required"}), 400
    year = int(d.get("year") or datetime.utcnow().year)
    saved = db_save_cultivation(
        farmer_id = d.get("farmer_id"),
        field_id  = d.get("field_id"),
        crop      = crop,
        year      = year,
        season    = d.get("season",""),
        notes     = d.get("notes",""),
        province  = d.get("province",""),
        source    = "farmer_reported",
        ndvi      = d.get("ndvi")
    )
    return jsonify({"ok": True, "saved": saved})


@app.route("/regen/history/<farmer_id>", methods=["GET","OPTIONS"])
def regen_history(farmer_id):
    if request.method == "OPTIONS":
        return jsonify({}), 200
    field_id = request.args.get("field_id")
    history  = db_get_cultivation_history(farmer_id, field_id)
    return jsonify({"ok": True, "history": history, "count": len(history)})


@app.route("/regen/recommend", methods=["POST","OPTIONS"])
def regen_recommend():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    d            = request.get_json(silent=True) or {}
    farmer_id    = d.get("farmer_id")
    field_id     = d.get("field_id")
    province     = d.get("province") or "Kabul"
    current_crop = d.get("current_crop")
    rain         = d.get("rain")
    history      = db_get_cultivation_history(farmer_id, field_id) if (farmer_id and sb_ok) else []
    if not current_crop and history:
        current_crop = history[0]["crop"]
    rec = regen_build_recommendation(history, province, current_crop, rain)
    return jsonify({"ok": True, "history": history, "recommendation": rec,
                    "crop_values": CROP_VALUE_TABLE})


# Auto-seed RAG on startup if knowledge base is empty (handles fresh Render deploys)
if rag_ok and GEMINI_KEY:
    threading.Thread(target=_auto_seed_rag, daemon=True).start()



@app.route("/country", methods=["POST","OPTIONS"])
def country_lookup():
    """Reverse geocode lat/lng → country name via OpenStreetMap Nominatim."""
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:
        d   = request.get_json(force=True)
        lat = d.get("lat")
        lng = d.get("lng")
        if lat is None or lng is None:
            return jsonify({"country": "Unknown", "error": "lat/lng required"}), 400
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"format": "json", "lat": lat, "lon": lng, "zoom": 3},
            headers={"User-Agent": "ZaminAI/1.0 (zaminai.onrender.com)"},
            timeout=8,
        )
        if resp.status_code == 200:
            data    = resp.json()
            country = data.get("address", {}).get("country", "Unknown")
            code    = data.get("address", {}).get("country_code", "").upper()
            return jsonify({"country": country, "country_code": code})
        return jsonify({"country": "Unknown"})
    except Exception as e:
        log.warning(f"/country lookup failed: {e}")
        return jsonify({"country": "Unknown"})



@app.route("/report/<field_id>", methods=["GET"])
def field_report(field_id):
    """
    Render a print-ready HTML field report from stored Supabase data.
    GET /report/<field_id>?lang=en
    Returns text/html — open in browser, print to PDF.
    """
    lang = request.args.get("lang", "en")
    try:
        if not sb_ok:
            return "<h2>Database not connected</h2>", 503

        # Fetch latest analysis for this field
        a_res = (sb.table("analyses")
                   .select("full_data, analysed_at, ndvi, rain, province, source, area_ha")
                   .eq("field_id", field_id)
                   .order("analysed_at", desc=True)
                   .limit(1)
                   .execute())
        if not a_res.data:
            return "<h2>No analysis found for this field.</h2>", 404

        row  = a_res.data[0]
        data = json.loads(row["full_data"]) if isinstance(row["full_data"], str) else (row["full_data"] or {})

        # Fetch field metadata
        f_res = (sb.table("fields")
                   .select("label, province, area_ha, area_jereb, coords")
                   .eq("id", field_id)
                   .limit(1)
                   .execute())
        field_meta = f_res.data[0] if f_res.data else {}

        label     = field_meta.get("label") or data.get("label", "Field")
        province  = field_meta.get("province") or data.get("province", "")
        area_ha   = field_meta.get("area_ha")  or data.get("area_ha", 0)
        area_j    = field_meta.get("area_jereb") or data.get("area_jereb", round(float(area_ha or 0)*5,1))
        ndvi      = data.get("ndvi") or row.get("ndvi") or 0
        mndwi     = data.get("mndwi") or data.get("water") or 0
        rain      = data.get("rain")  or row.get("rain")  or 0
        source    = data.get("source") or row.get("source") or "satellite"
        sat_date  = data.get("latest_date") or row.get("analysed_at","")[:10]
        analysed  = row.get("analysed_at","")[:16].replace("T"," ")

        ndvi  = float(ndvi  or 0)
        mndwi = float(mndwi or 0)
        rain  = float(rain  or 0)

        # Health classification
        if ndvi >= 0.35:   health, hcol = "Healthy vegetation", "#16a34a"
        elif ndvi >= 0.20: health, hcol = "Moderate stress", "#d97706"
        else:              health, hcol = "Severe stress / bare soil", "#dc2626"

        # Crops detected
        crops_html = ""
        for c in (data.get("crops") or []):
            conf = int(float(c.get("confidence",0))*100)
            crops_html += f'<tr><td style="padding:8px 12px;border-bottom:1px solid #f3f4f6">{c.get("label_en","")}</td><td style="padding:8px 12px;border-bottom:1px solid #f3f4f6;color:#16a34a;font-weight:700">{conf}%</td></tr>'

        # Season advice
        season_html = ""
        for s in (data.get("season") or []):
            icon = s.get("icon","•")
            season_html += f'<li style="margin-bottom:6px">{icon} {s.get("text","")}</li>'

        # Soil
        soil = data.get("soil") or {}
        soil_rows = ""
        if soil:
            for k,v in [("Type", soil.get("texture","")), ("pH", soil.get("ph","")),
                        ("Organic matter", soil.get("organic_carbon","")),
                        ("N content", soil.get("nitrogen",""))]:
                if v: soil_rows += f'<tr><td style="padding:8px 12px;border-bottom:1px solid #f3f4f6;color:#6b7280">{k}</td><td style="padding:8px 12px;border-bottom:1px solid #f3f4f6;font-weight:600">{v}</td></tr>'

        # Weather forecast strip
        wx_html = ""
        WMO_EMOJI = {"Clear":"☀️","Mainly clear":"🌤️","Partly cloudy":"⛅","Overcast":"☁️",
                     "Light rain":"🌧️","Rain":"🌧️","Heavy rain":"⛈️","Thunderstorm":"⛈️",
                     "Light drizzle":"🌦️","Drizzle":"🌧️","Snow":"❄️","Fog":"🌫️"}
        for day in (data.get("weather_forecast") or [])[:7]:
            ico  = WMO_EMOJI.get(day.get("condition",""), "🌡️")
            tmax = f'{round(day["temp_max"])}°' if day.get("temp_max") is not None else "--"
            rmm  = f'{day.get("rain_mm",0):.1f}mm'
            dt   = day.get("date","")[-5:]  # MM-DD
            wx_html += (f'<div style="flex:1;text-align:center;padding:8px 4px;background:#f8fafc;'
                        f'border-radius:8px;border:1px solid #e5e7eb;min-width:60px">'
                        f'<div style="font-size:9px;color:#6b7280">{dt}</div>'
                        f'<div style="font-size:20px">{ico}</div>'
                        f'<div style="font-size:11px;font-weight:700">{tmax}</div>'
                        f'<div style="font-size:10px;color:#3b82f6">{rmm}</div></div>')
        wx_section = (f'<div style="margin-bottom:22px"><div style="font-size:14px;font-weight:700;'
                      f'margin-bottom:10px;padding-bottom:8px;border-bottom:2px solid #f3f4f6">7-Day Forecast</div>'
                      f'<div style="display:flex;gap:6px;flex-wrap:wrap">{wx_html}</div></div>') if wx_html else ""

        # Lat/lon for map link
        lat = data.get("lat") or data.get("clat") or ""
        lon = data.get("lon") or data.get("clon") or data.get("lng") or ""
        map_link = (f'<a href="https://www.google.com/maps?q={lat},{lon}" target="_blank" '
                    f'style="color:#16a34a;font-size:11px">View on Google Maps →</a>') if lat and lon else ""

        html_out = f"""<!DOCTYPE html>
<html lang="{lang}"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ZaminAI Field Report — {label}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#1f2937;background:#fff;font-size:14px;line-height:1.6}}
.page{{max-width:780px;margin:0 auto;padding:28px 28px 48px}}
table{{border-collapse:collapse;width:100%}}
.no-print{{margin-bottom:20px}}
@media print{{.no-print{{display:none!important}}.page{{padding:12mm 15mm;max-width:none}}body{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}}}
@page{{margin:15mm}}
</style></head><body><div class="page">

<div class="no-print">
  <button onclick="window.print()" style="padding:10px 22px;background:#16a34a;color:#fff;border:none;border-radius:8px;font-size:14px;cursor:pointer;font-weight:600">🖨️ Print / Save as PDF</button>
</div>

<div style="display:flex;justify-content:space-between;align-items:flex-start;border-bottom:3px solid #16a34a;padding-bottom:16px;margin-bottom:22px">
  <div><div style="font-size:26px;font-weight:800;color:#16a34a">🌱 ZaminAI</div>
  <div style="font-size:12px;color:#6b7280;margin-top:2px">Satellite Farming Intelligence</div></div>
  <div style="text-align:right"><div style="font-size:10px;color:#9ca3af;text-transform:uppercase">Report generated</div>
  <div style="font-weight:700;font-size:13px">{analysed} UTC</div></div>
</div>

<div style="background:#f0fdf4;border-radius:12px;padding:18px;margin-bottom:22px;display:grid;grid-template-columns:1fr 1fr;gap:14px">
  <div><div style="font-size:10px;color:#6b7280;text-transform:uppercase;margin-bottom:3px">Field</div>
  <div style="font-size:18px;font-weight:700">{label}</div>{map_link}</div>
  <div><div style="font-size:10px;color:#6b7280;text-transform:uppercase;margin-bottom:3px">Area</div>
  <div style="font-size:18px;font-weight:700">{area_j} jereb · {area_ha:.2f} ha</div></div>
  <div><div style="font-size:10px;color:#6b7280;text-transform:uppercase;margin-bottom:3px">Province / Region</div>
  <div style="font-size:15px;font-weight:600">{province}</div></div>
  <div><div style="font-size:10px;color:#6b7280;text-transform:uppercase;margin-bottom:3px">Satellite date</div>
  <div style="font-size:13px;font-weight:600">{sat_date} · {source}</div></div>
</div>

<div style="background:#fff;border:2px solid {hcol};border-radius:12px;padding:18px;margin-bottom:22px;display:flex;align-items:center;gap:16px">
  <div style="width:56px;height:56px;background:{hcol};border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:24px;color:#fff;flex-shrink:0">{"✓" if ndvi>=0.30 else "!" if ndvi>=0.18 else "⚠"}</div>
  <div style="flex:1"><div style="font-size:10px;text-transform:uppercase;color:{hcol};letter-spacing:.06em">Crop Health</div>
  <div style="font-size:22px;font-weight:800;color:{hcol}">{health}</div></div>
  <div style="background:#f9fafb;border-radius:10px;padding:10px 18px;text-align:center;flex-shrink:0">
  <div style="font-family:monospace;font-size:32px;font-weight:800;color:{hcol}">{ndvi:.3f}</div>
  <div style="font-size:10px;color:#6b7280;text-transform:uppercase">NDVI</div></div>
</div>

<div style="margin-bottom:22px"><div style="font-size:15px;font-weight:700;margin-bottom:10px;padding-bottom:8px;border-bottom:2px solid #f3f4f6">Satellite Measurements</div>
<table><tbody>
<tr><td style="padding:9px 12px;border-bottom:1px solid #f3f4f6;color:#6b7280">NDVI (vegetation health)</td><td style="padding:9px 12px;border-bottom:1px solid #f3f4f6;font-family:monospace;font-weight:700">{ndvi:.4f}</td></tr>
<tr><td style="padding:9px 12px;border-bottom:1px solid #f3f4f6;color:#6b7280">MNDWI (water / moisture)</td><td style="padding:9px 12px;border-bottom:1px solid #f3f4f6;font-family:monospace;font-weight:700">{mndwi:.4f}</td></tr>
<tr><td style="padding:9px 12px;border-bottom:1px solid #f3f4f6;color:#6b7280">Annual rainfall</td><td style="padding:9px 12px;border-bottom:1px solid #f3f4f6;font-family:monospace;font-weight:700">{round(rain)} mm</td></tr>
<tr><td style="padding:9px 12px;border-bottom:1px solid #f3f4f6;color:#6b7280">EVI</td><td style="padding:9px 12px;border-bottom:1px solid #f3f4f6;font-family:monospace">{data.get("evi") or "—"}</td></tr>
<tr><td style="padding:9px 12px;border-bottom:1px solid #f3f4f6;color:#6b7280">SAVI</td><td style="padding:9px 12px;border-bottom:1px solid #f3f4f6;font-family:monospace">{data.get("savi") or "—"}</td></tr>
<tr><td style="padding:9px 12px;color:#6b7280">VCI (vegetation condition)</td><td style="padding:9px 12px;font-family:monospace">{f'{data.get("vci")}%' if data.get("vci") is not None else "—"}</td></tr>
</tbody></table></div>

{f'<div style="margin-bottom:22px"><div style="font-size:15px;font-weight:700;margin-bottom:10px;padding-bottom:8px;border-bottom:2px solid #f3f4f6">Detected Crops</div><table><tbody>{crops_html}</tbody></table></div>' if crops_html else ""}

{f'<div style="margin-bottom:22px"><div style="font-size:15px;font-weight:700;margin-bottom:10px;padding-bottom:8px;border-bottom:2px solid #f3f4f6">Season Advice</div><ul style="padding-left:18px;color:#374151">{season_html}</ul></div>' if season_html else ""}

{f'<div style="margin-bottom:22px"><div style="font-size:15px;font-weight:700;margin-bottom:10px;padding-bottom:8px;border-bottom:2px solid #f3f4f6">Soil</div><table><tbody>{soil_rows}</tbody></table></div>' if soil_rows else ""}

{wx_section}

<div style="margin-top:36px;padding-top:14px;border-top:1px solid #e5e7eb;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
  <div style="font-size:11px;color:#9ca3af">Generated by ZaminAI · zaminai.onrender.com</div>
  <div style="font-size:11px;color:#9ca3af">For agricultural advisory purposes only.</div>
</div>
</div></body></html>"""

        return html_out, 200, {"Content-Type": "text/html; charset=utf-8"}

    except Exception as e:
        log.error(f"/report/{field_id}: {e}")
        return f"<h2>Report error: {e}</h2>", 500

if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    log.info(f"ZaminAI API v7.0 starting on port {port}")
    app.run(host="0.0.0.0",port=port,debug=False)
