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

import os, json, math, logging, requests, threading, uuid, base64
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
EMBED_DIM   = 3072

rag_ok = False
try:
    if sb_ok:
        sb.table("knowledge_chunks").select("id").limit(1).execute()
        rag_ok = True
        log.info("✓ RAG / pgvector ready")
except Exception:
    log.warning("RAG not ready — call GET /rag/setup for the SQL migration")


def embed_text(text: str) -> list | None:
    """Embed text using Google text-embedding-004 (768 dimensions)."""
    if not GEMINI_KEY or not text.strip():
        return None
    try:
        url = (f"https://generativelanguage.googleapis.com/v1beta"
               f"/models/{EMBED_MODEL}:embedContent?key={GEMINI_KEY}")
        resp = requests.post(url, json={
            "model":   f"models/{EMBED_MODEL}",
            "content": {"parts": [{"text": text[:8000]}]}
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


def rag_retrieve(question: str, top_k: int = 4, threshold: float = 0.70) -> list[str]:
    """Return top-k most similar knowledge chunks for a question."""
    if not sb_ok or not rag_ok or not GEMINI_KEY:
        return []
    embedding = embed_text(question)
    if not embedding:
        return []
    try:
        res = sb.rpc("match_knowledge_chunks", {
            "query_embedding": embedding,
            "match_count":     top_k,
            "match_threshold": threshold
        }).execute()
        return [r["content"] for r in (res.data or [])]
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


def check_alerts_fire(alerts, ndvi, mndwi, rain, province):
    """Return alerts that fire given current satellite readings."""
    fired = []
    month = datetime.now().month
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
            elif atype == "rain_deficit" and rain is not None and thr is not None and rain < thr:
                fired.append({**a, "value": rain,
                              "msg": f"Rainfall {rain}mm below {thr}mm threshold"})
            elif atype == "harvest_window":
                crop  = a.get("crop", "wheat")
                ptype = get_province_type(province)
                cal   = CROP_CALENDAR.get(crop, CROP_CALENDAR.get("wheat", {}))
                zone  = cal.get(ptype, list(cal.values())[0] if cal else {})
                if month in zone.get("harvest", []):
                    fired.append({**a, "value": month,
                                  "msg": f"Harvest window open for {crop} — act now"})
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
    models=["gemini-2.0-flash","gemini-2.5-flash","gemini-flash-latest"]
    for model in models:
        try:
            url=(f"https://generativelanguage.googleapis.com/v1beta"
                 f"/models/{model}:generateContent?key={GEMINI_KEY}")
            resp=requests.post(url,json={
                "contents":[{"parts":[{"text":prompt}]}],
                "safetySettings":[{"category":c,"threshold":"BLOCK_NONE"} for c in [
                    "HARM_CATEGORY_HARASSMENT","HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT","HARM_CATEGORY_DANGEROUS_CONTENT"]],
                "generationConfig":{"temperature":0.6,"maxOutputTokens":280}
            },timeout=14)
            if resp.status_code==200:
                cands=resp.json().get("candidates",[])
                if cands:
                    txt=cands[0].get("content",{}).get("parts",[{}])[0].get("text","")
                    if txt and len(txt)>8: return txt.strip()
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
    return jsonify({
        "status": "ok", "version": "8.0", "gee": gee_ok,
        "database": sb_ok, "rag": rag_ok,
        "ai": "gemini" if GEMINI_KEY else "smart_only",
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
            "/telegram/webhook", "/telegram/setup"
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
    if lang == "fa":
        lines = [f"🌾 <b>ZaminAI هشدار</b>", f"📍 {province}  ·  {now}"]
        for a in fired:
            lines.append(f"\n🚨 {a.get('msg','')}")
        lines.append(f"\n📊 NDVI: {ndvi}  |  آب: {mndwi}  |  باران: {rain}mm")
        lines.append("🌐 zaminai.org")
    elif lang == "ps":
        lines = [f"🌾 <b>ZaminAI خبرداری</b>", f"📍 {province}  ·  {now}"]
        for a in fired:
            lines.append(f"\n🚨 {a.get('msg','')}")
        lines.append(f"\n📊 NDVI: {ndvi}  |  اوبه: {mndwi}  |  باران: {rain}mm")
        lines.append("🌐 zaminai.org")
    else:
        lines = [f"🌾 <b>ZaminAI Alert</b>", f"📍 {province}  ·  {now}"]
        for a in fired:
            lines.append(f"\n🚨 {a.get('msg','')}")
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

            alerts = db_get_alerts(farmer_id)
            fired  = check_alerts_fire(alerts, ndvi, mndwi, rain, province)

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
                if gee_ok:
                    try:
                        result = gee_analyse(coords, year, clat, clon)
                        reg = get_regional_data(clat, clon)
                        result.update({"label":label,"area_ha":area_ha,"area_jereb":area_jereb,
                                       "status":"success","province":reg["province"]})
                        result["crops"] = detect_crop(result["ndvi"],result["evi"],result["savi"],
                            result["mndwi"],result["lswi"],month,reg["province"])
                        result["season"] = get_current_season_advice(reg["province"],result["ndvi"],result["mndwi"])
                        result["monthly_rain"] = get_monthly_rain(result["rain"] or reg["rain"],reg["province"])
                        result["soil"] = get_soil_data(clat,clon,reg["province"])
                        if result.get("trend"):
                            tv=[v for v in result["trend"].values() if v]
                            if tv:
                                h_min=min(tv); h_max=max(tv); cur=result["ndvi"] or 0
                                result["vci"] = round((cur-h_min)/(h_max-h_min+0.001)*100,1) if h_max>h_min else None
                        if farmer_id and field_id:
                            db_save_analysis(field_id, farmer_id, result)
                        _farmer_analyse_tasks[task_id] = {"status":"done","data":result}
                        return
                    except Exception as e:
                        log.error(f"GEE failed in /analyse worker: {e}")

                # Regional fallback
                reg = get_regional_data(clat, clon)
                result = {
                    "label":label,"status":"success","source":reg["source"],
                    "province":reg["province"],"ndvi":reg["ndvi"],"evi":reg["evi"],
                    "savi":reg["savi"],"mndwi":reg["mndwi"],"water":reg["mndwi"],
                    "lswi":reg["lswi"],"rain":reg["rain"],"area_ha":area_ha,
                    "area_jereb":area_jereb,"trend":reg["trend"],"ndvi_trend":reg["trend"],
                    "year":year,"lat":round(clat,5),"lon":round(clon,5),
                    "latest_date":f"{year}-05-15",
                    "crops":detect_crop(reg["ndvi"],reg["evi"],reg["savi"],reg["mndwi"],reg["lswi"],month,reg["province"]),
                    "season":get_current_season_advice(reg["province"],reg["ndvi"],reg["mndwi"]),
                    "monthly_rain":get_monthly_rain(reg["rain"],reg["province"]),
                    "soil":get_soil_data(clat,clon,reg["province"]),
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
        lang_inst={"fa":"Afghan Dari (دری). Use دهقان for farmer, جریب for land. Eastern Arabic numerals ۱۲۳.",
                   "ps":"Pashto (پښتو). Proper Pashto farming terms. Eastern Arabic numerals.",
                   "en":"English. Concise and specific."}.get(language,"English.")
        # Retrieve relevant knowledge chunks from vector DB
        rag_chunks = rag_retrieve(question, top_k=4, threshold=0.68)
        rag_section = ("\n\nRelevant local knowledge:\n" + "\n\n".join(rag_chunks)) if rag_chunks else ""
        prompt=(f"You are ZaminAI, expert agricultural advisor for Afghan smallholder farmers.\n"
                f"Satellite data: {context}{rag_section}\n\nRespond ONLY in {lang_inst}\n"
                f"Rules: exact amounts, under 90 words, speak as trusted local expert.\n\nQuestion: {question}")
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
        try:
            from crop_vision import run_inference
            yolo_result = run_inference(image_bytes)
        except Exception as e:
            log.warning(f"YOLO import/run failed: {e}")
            yolo_result = {"ok": False, "yolo_available": False, "detections": []}

        # ── Build shared prompt ───────────────────────────────────────────────
        lang_inst = {
            "fa": "Respond ONLY in Dari (Afghan Persian). Use simple farming language a village farmer understands.",
            "ps": "Respond ONLY in Pashto. Use simple farming language a village farmer understands.",
        }.get(language, "Respond in English. Use simple, practical language.")

        yolo_ctx = ""
        if yolo_result.get("ok") and yolo_result.get("detections"):
            top_det = yolo_result["detections"][0]
            yolo_ctx = f"YOLO model detected: {top_det['label_en']} ({top_det['confidence']*100:.0f}% confidence). "

        crop_ctx = f"The farmer says this is a {crop_hint} crop. " if crop_hint else ""

        diagnosis_prompt = (
            f"{crop_ctx}{yolo_ctx}"
            "Examine this crop/plant photo carefully and answer:\n"
            "1. What disease, pest, or problem do you see? (if none, say the plant looks healthy)\n"
            "2. Severity: mild / moderate / severe\n"
            "3. What must the farmer do RIGHT NOW? (specific, numbered steps)\n"
            "4. Which product to apply, what dose, when?\n"
            "5. One sentence: how to prevent this next season.\n\n"
            f"{lang_inst}\n"
            "Be concise. Afghan smallholder farmers will act directly on this advice."
        )

        # ── Stage 2a: Claude Vision (preferred) ──────────────────────────────
        ai_diagnosis = None
        ai_model_used = None
        if ANTHROPIC_KEY:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
                msg = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=600,
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

        top = yolo_result["detections"][0] if yolo_result.get("detections") else None
        return jsonify({
            "ok":             True,
            "detections":     yolo_result.get("detections", []),
            "top_detection":  top,
            "is_healthy":     bool(top and top.get("is_healthy")),
            "yolo_ok":        yolo_result.get("ok", False),
            "yolo_available": yolo_result.get("yolo_available", False),
            "diagnosis":      ai_diagnosis,
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
    embedding  vector(3072),
    source     text    default 'manual',
    metadata   jsonb   default '{}',
    created_at timestamp default now()
);

create or replace function match_knowledge_chunks (
    query_embedding  vector(3072),
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
        threshold = float(d.get("threshold", 0.65))
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


if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    log.info(f"ZaminAI API v7.0 starting on port {port}")
    app.run(host="0.0.0.0",port=port,debug=False)
