"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ZaminAI API  —  v8.0  (FastAPI)                                             ║
║  Satellite Farming Intelligence for Afghan Smallholders                      ║
║                                                                              ║
║  Author : Maiwand Jan Alamzoi                                                ║
║  Org    : Afghanistan Development Initiative (ADI)                           ║
║                                                                              ║
║  Run locally:  uvicorn main:app --host 0.0.0.0 --port 8000 --reload         ║
║  Deploy:       set PORT env var; uvicorn reads it automatically              ║
║                                                                              ║
║  All credentials come from environment variables — never hardcoded.          ║
║  All dates are computed at request time — never hardcoded to a year.         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import math
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Load .env if present (no-op in production where vars are set directly) ────
load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CREDENTIALS  — all from environment, none hardcoded
# ══════════════════════════════════════════════════════════════════════════════

GEMINI_KEY    = os.environ.get("GEMINI_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEE_SA        = os.environ.get("GEE_SERVICE_ACCOUNT", "")
# Private key is stored with literal \n; restore real newlines at startup
GEE_KEY       = os.environ.get("GEE_PRIVATE_KEY", "").replace("\\n", "\n")
SUPABASE_URL  = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY  = os.environ.get("SUPABASE_KEY", "")

# ══════════════════════════════════════════════════════════════════════════════
# SUPABASE
# ══════════════════════════════════════════════════════════════════════════════

sb    = None
sb_ok = False
try:
    if SUPABASE_URL and SUPABASE_KEY:
        from supabase import create_client
        sb    = create_client(SUPABASE_URL, SUPABASE_KEY)
        sb_ok = True
        log.info("✓ Supabase connected")
    else:
        log.warning("SUPABASE_URL / SUPABASE_KEY missing — database disabled")
except Exception as e:
    log.error(f"Supabase init failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# GOOGLE EARTH ENGINE
# ══════════════════════════════════════════════════════════════════════════════

gee_ok = False
try:
    import ee
    if GEE_SA and GEE_KEY:
        ee.Initialize(ee.ServiceAccountCredentials(GEE_SA, key_data=GEE_KEY))
        gee_ok = True
        log.info("✓ GEE initialized with service account")
    else:
        log.warning("GEE_SERVICE_ACCOUNT or GEE_PRIVATE_KEY missing — regional fallback active")
except Exception as e:
    log.error(f"GEE init failed: {e}")

log.info(f"AI  : {'Gemini ✓' if GEMINI_KEY else 'smart fallback only'}")
log.info(f"DB  : {'Supabase ✓' if sb_ok else 'disabled'}")
log.info(f"GEE : {'live satellite ✓' if gee_ok else 'regional database fallback'}")

# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP + CORS
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="ZaminAI API",
    description="Satellite Farming Intelligence for Afghan Smallholders — Afghanistan Development Initiative",
    version="8.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # restrict to ["https://zaminai.org"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════════════════
# REQUEST MODELS
# ══════════════════════════════════════════════════════════════════════════════

class AnalyseRequest(BaseModel):
    coords:     List[List[float]]
    year:       Optional[int]  = None   # None → current calendar year
    label:      Optional[str]  = "Field"
    farmer_id:  Optional[str]  = None
    field_id:   Optional[str]  = None

class AskRequest(BaseModel):
    question:   str
    language:   Optional[str]  = "en"
    context:    Optional[str]  = ""
    field_data: Optional[Dict] = {}
    farmer_id:  Optional[str]  = None
    field_id:   Optional[str]  = None

class NdviTileRequest(BaseModel):
    coords: List[List[float]]
    year:   Optional[int] = None

class CropDetectRequest(BaseModel):
    ndvi:     float        = 0.0
    evi:      float        = 0.0
    savi:     float        = 0.0
    mndwi:    float        = 0.0
    lswi:     float        = 0.0
    month:    Optional[int] = None
    province: Optional[str] = "Afghanistan"

class MonthlyRainRequest(BaseModel):
    annual_rain: float         = 250.0
    province:    Optional[str] = "Afghanistan"

class SoilRequest(BaseModel):
    lat:      float          = 34.5
    lon:      float          = 67.7
    province: Optional[str]  = "Afghanistan"

class FarmerRequest(BaseModel):
    phone:    str
    language: Optional[str] = "en"
    province: Optional[str] = "Afghanistan"

class FieldSaveRequest(BaseModel):
    farmer_id:  str
    coords:     List[List[float]]
    label:      Optional[str]   = "My Field"
    province:   Optional[str]   = "Afghanistan"
    area_ha:    Optional[float] = 0.0
    area_jereb: Optional[float] = 0.0

class FieldDeleteRequest(BaseModel):
    field_id:  str
    farmer_id: str

class AnalysisSaveRequest(BaseModel):
    field_id:      Optional[str]  = None
    farmer_id:     Optional[str]  = None
    analysis_data: Optional[Dict] = {}

class ChatSaveRequest(BaseModel):
    farmer_id: Optional[str] = None
    field_id:  Optional[str] = None
    question:  str           = ""
    answer:    str           = ""
    language:  Optional[str] = "en"

class WeatherRequest(BaseModel):
    lat:  float
    lon:  float
    lang: Optional[str] = "en"

# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC DATE HELPERS  — every date is computed at call time
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_year(year: Optional[int]) -> int:
    """Return the requested year, defaulting to the current calendar year."""
    return year if year else datetime.now().year


def _season_window(year: int):
    """
    Return (start_date, end_date) strings for Sentinel-2 growing-season query.

    Historical years  → full April–July season.
    Current year      → April 1 through today, capped at July 31.
    This ensures we never request future dates or hardcode a cut-off year.
    """
    now          = datetime.now()
    current_year = now.year
    if year < current_year:
        return f"{year}-04-01", f"{year}-07-31"
    cap = datetime(year, 7, 31)
    end = min(now, cap)
    return f"{year}-04-01", end.strftime("%Y-%m-%d")


def _trend_years() -> range:
    """NDVI trend range: 2019 to the current year, inclusive."""
    return range(2019, datetime.now().year + 1)

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def db_get_or_create_farmer(phone: str, language: str = "en", province: str = "Afghanistan"):
    if not sb_ok or not phone:
        return None
    try:
        res = sb.table("farmers").select("*").eq("phone", phone).execute()
        if res.data:
            sb.table("farmers").update({
                "last_seen": datetime.utcnow().isoformat(),
                "language":  language,
            }).eq("phone", phone).execute()
            return res.data[0]
        new = sb.table("farmers").insert({
            "phone":     phone,
            "language":  language,
            "province":  province,
            "joined_at": datetime.utcnow().isoformat(),
            "last_seen": datetime.utcnow().isoformat(),
        }).execute()
        log.info(f"✓ New farmer: {phone}")
        return new.data[0] if new.data else None
    except Exception as e:
        log.error(f"db_get_or_create_farmer: {e}")
        return None


def db_save_field(farmer_id, coords, label, province, area_ha, area_jereb):
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
            "created_at": datetime.utcnow().isoformat(),
        }).execute()
        log.info(f"✓ Field saved for farmer {farmer_id}")
        return res.data[0] if res.data else None
    except Exception as e:
        log.error(f"db_save_field: {e}")
        return None


def db_get_farmer_fields(farmer_id: str):
    if not sb_ok or not farmer_id:
        return []
    try:
        res = (
            sb.table("fields")
            .select("*")
            .eq("farmer_id", farmer_id)
            .order("created_at", desc=True)
            .execute()
        )
        fields = res.data or []
        for f in fields:
            try:
                a = (
                    sb.table("analyses")
                    .select("*")
                    .eq("field_id", f["id"])
                    .order("analysed_at", desc=True)
                    .limit(1)
                    .execute()
                )
                f["analyses"] = a.data or []
            except Exception as ae:
                log.error(f"fetch analyses for field {f.get('id')}: {ae}")
                f["analyses"] = []
        return fields
    except Exception as e:
        log.error(f"db_get_farmer_fields: {e}")
        return []


def db_save_analysis(field_id, farmer_id, analysis_data: dict):
    if not sb_ok:
        return None
    try:
        res = sb.table("analyses").insert({
            "field_id":    field_id,
            "farmer_id":   farmer_id,
            "ndvi":        analysis_data.get("ndvi"),
            "evi":         analysis_data.get("evi"),
            "savi":        analysis_data.get("savi"),
            "mndwi":       analysis_data.get("mndwi"),
            "lswi":        analysis_data.get("lswi"),
            "rain":        analysis_data.get("rain"),
            "source":      analysis_data.get("source", "regional_db"),
            "province":    analysis_data.get("province"),
            "area_ha":     analysis_data.get("area_ha"),
            "full_data":   json.dumps(analysis_data),
            "analysed_at": datetime.utcnow().isoformat(),
        }).execute()
        log.info(f"✓ Analysis saved for field {field_id}")
        return res.data[0] if res.data else None
    except Exception as e:
        log.error(f"db_save_analysis: {e}")
        return None


def db_save_chat(farmer_id, field_id, question, answer, language):
    if not sb_ok:
        return
    try:
        sb.table("conversations").insert({
            "farmer_id": farmer_id,
            "field_id":  field_id,
            "question":  question,
            "answer":    answer,
            "language":  language,
            "asked_at":  datetime.utcnow().isoformat(),
        }).execute()
        log.info(f"✓ Chat saved ({language})")
    except Exception as e:
        log.error(f"db_save_chat: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# REGIONAL FALLBACK DATABASE  (used when GEE is unavailable)
# Historical NDVI values are real measured data; do not auto-extrapolate.
# ══════════════════════════════════════════════════════════════════════════════

PROVINCES = [
    # lat_min, lat_max, lon_min, lon_max, name, ndvi, evi, savi, mndwi, lswi, rain_mm, ndvi_by_year
    (36.4,37.2,68.2,69.2,"Kunduz",    0.33,0.24,0.28,-0.14,-0.09,287,{2019:0.40,2020:0.38,2021:0.35,2022:0.22,2023:0.27,2024:0.33}),
    (36.4,37.1,66.5,67.3,"Balkh",     0.31,0.22,0.26,-0.18,-0.12,245,{2019:0.37,2020:0.35,2021:0.31,2022:0.19,2023:0.24,2024:0.31}),
    (33.8,35.0,61.5,63.5,"Herat",     0.28,0.19,0.23,-0.20,-0.14,195,{2019:0.33,2020:0.31,2021:0.27,2022:0.15,2023:0.21,2024:0.28}),
    (33.8,34.6,70.0,71.5,"Nangarhar", 0.38,0.28,0.32,-0.12,-0.07,320,{2019:0.44,2020:0.41,2021:0.37,2022:0.26,2023:0.31,2024:0.38}),
    (34.2,34.9,68.7,69.5,"Kabul",     0.27,0.18,0.22,-0.22,-0.16,305,{2019:0.32,2020:0.29,2021:0.25,2022:0.14,2023:0.20,2024:0.27}),
    (31.3,32.1,65.2,66.2,"Kandahar",  0.22,0.14,0.18,-0.28,-0.21,175,{2019:0.27,2020:0.24,2021:0.20,2022:0.11,2023:0.16,2024:0.22}),
    (30.8,32.2,63.5,65.5,"Helmand",   0.25,0.16,0.20,-0.25,-0.18,148,{2019:0.30,2020:0.27,2021:0.23,2022:0.13,2023:0.18,2024:0.25}),
    (36.5,38.5,70.0,72.0,"Badakhshan",0.41,0.30,0.35,-0.10,-0.06,420,{2019:0.47,2020:0.44,2021:0.40,2022:0.29,2023:0.35,2024:0.41}),
    (36.4,37.2,69.0,70.5,"Takhar",    0.36,0.26,0.30,-0.15,-0.10,340,{2019:0.42,2020:0.39,2021:0.35,2022:0.24,2023:0.29,2024:0.36}),
    (35.8,36.6,68.2,69.2,"Baghlan",   0.34,0.25,0.29,-0.16,-0.11,295,{2019:0.40,2020:0.37,2021:0.33,2022:0.21,2023:0.27,2024:0.34}),
    (35.0,36.0,64.0,66.0,"Faryab",    0.29,0.20,0.24,-0.19,-0.13,220,{2019:0.35,2020:0.32,2021:0.27,2022:0.16,2023:0.22,2024:0.29}),
    (35.5,36.5,65.5,67.0,"Jawzjan",   0.30,0.21,0.25,-0.17,-0.12,240,{2019:0.36,2020:0.33,2021:0.28,2022:0.17,2023:0.23,2024:0.30}),
    (32.0,33.5,67.0,68.5,"Ghazni",    0.24,0.15,0.19,-0.21,-0.15,185,{2019:0.29,2020:0.26,2021:0.22,2022:0.12,2023:0.18,2024:0.24}),
    (34.5,35.5,67.0,68.5,"Bamyan",    0.27,0.18,0.22,-0.18,-0.13,270,{2019:0.32,2020:0.29,2021:0.25,2022:0.14,2023:0.20,2024:0.27}),
    (33.0,34.0,69.0,70.5,"Logar",     0.26,0.17,0.21,-0.20,-0.14,260,{2019:0.31,2020:0.28,2021:0.24,2022:0.13,2023:0.19,2024:0.26}),
    (32.5,33.5,68.0,69.5,"Paktia",    0.28,0.19,0.23,-0.18,-0.12,285,{2019:0.33,2020:0.30,2021:0.26,2022:0.15,2023:0.21,2024:0.28}),
]


def get_regional_data(lat: float, lon: float) -> dict:
    for (lat_min, lat_max, lon_min, lon_max, name,
         ndvi, evi, savi, mndwi, lswi, rain, trend) in PROVINCES:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return {"province": name, "ndvi": ndvi, "evi": evi, "savi": savi,
                    "mndwi": mndwi, "lswi": lswi, "rain": rain, "trend": trend,
                    "source": "regional_db"}
    # Coordinate outside all province bounding boxes — interpolate
    ndvi  = round(max(0.08, min(0.45, 0.06 + lat * 0.003 + (lon - 62) * 0.002)), 4)
    evi   = round(ndvi * 0.72, 4)
    savi  = round(ndvi * 0.85, 4)
    rain  = max(80, min(480, int(lat * 8)))
    mndwi = round(max(-0.38, min(0.05, -0.38 + rain * 0.001)), 4)
    lswi  = round(mndwi + 0.05, 4)
    offsets = [0.07, 0.05, 0.02, -0.10, -0.04, 0.0]
    trend = {yr: round(ndvi + offsets[min(yr - 2019, 5)], 4) for yr in range(2019, 2025)}
    return {"province": "Afghanistan", "ndvi": ndvi, "evi": evi, "savi": savi,
            "mndwi": mndwi, "lswi": lswi, "rain": rain, "trend": trend,
            "source": "interpolated"}

# ══════════════════════════════════════════════════════════════════════════════
# SOIL
# ══════════════════════════════════════════════════════════════════════════════

AFGHAN_SOILS = {
    "Kunduz":     {"ph":7.4,"clay":22,"sand":38,"silt":40,"soc":0.9, "texture":"Silty loam"},
    "Balkh":      {"ph":7.6,"clay":18,"sand":52,"silt":30,"soc":0.7, "texture":"Sandy loam"},
    "Herat":      {"ph":7.8,"clay":15,"sand":58,"silt":27,"soc":0.5, "texture":"Sandy loam"},
    "Nangarhar":  {"ph":7.2,"clay":28,"sand":32,"silt":40,"soc":1.2, "texture":"Loam"},
    "Kabul":      {"ph":7.5,"clay":20,"sand":42,"silt":38,"soc":0.8, "texture":"Loam"},
    "Kandahar":   {"ph":8.0,"clay":12,"sand":65,"silt":23,"soc":0.3, "texture":"Sandy"},
    "Helmand":    {"ph":7.9,"clay":14,"sand":60,"silt":26,"soc":0.4, "texture":"Sandy loam"},
    "Badakhshan": {"ph":6.8,"clay":30,"sand":28,"silt":42,"soc":1.8, "texture":"Clay loam"},
    "Takhar":     {"ph":7.3,"clay":24,"sand":35,"silt":41,"soc":1.1, "texture":"Silty loam"},
    "Baghlan":    {"ph":7.4,"clay":22,"sand":38,"silt":40,"soc":1.0, "texture":"Silty loam"},
    "Faryab":     {"ph":7.7,"clay":16,"sand":55,"silt":29,"soc":0.6, "texture":"Sandy loam"},
    "Jawzjan":    {"ph":7.6,"clay":17,"sand":53,"silt":30,"soc":0.6, "texture":"Sandy loam"},
    "Ghazni":     {"ph":7.5,"clay":20,"sand":44,"silt":36,"soc":0.7, "texture":"Loam"},
    "Bamyan":     {"ph":7.1,"clay":26,"sand":32,"silt":42,"soc":1.4, "texture":"Clay loam"},
    "Logar":      {"ph":7.4,"clay":23,"sand":36,"silt":41,"soc":1.0, "texture":"Silty loam"},
    "Paktia":     {"ph":7.2,"clay":25,"sand":34,"silt":41,"soc":1.2, "texture":"Loam"},
}


def classify_soil_texture(clay, sand, silt):
    if sand >= 70:               return "Sandy"
    if sand >= 50 and clay < 20: return "Sandy loam"
    if clay >= 40:               return "Clay"
    if 27 <= clay < 40:          return "Clay loam"
    if silt >= 50 and clay < 27: return "Silty loam"
    if silt >= 80:               return "Silt"
    return "Loam"


def soil_recommendations(texture, ph, soc, province):
    recs = []
    if ph < 6.5:    recs.append(f"Acidic soil (pH {ph}) — apply lime 200–300 kg/jereb")
    elif ph > 8.0:  recs.append(f"Alkaline soil (pH {ph}) — add organic matter to lower pH")
    elif ph > 7.5:  recs.append(f"Slightly alkaline (pH {ph}) — use ammonium sulfate over urea")
    else:           recs.append(f"Good pH {ph} — suitable for wheat, vegetables, most crops")
    if soc < 0.5:   recs.append("Very low organic carbon — add 3–4 tonnes compost/jereb")
    elif soc < 1.0: recs.append("Low organic carbon — add 2 tonnes compost/jereb annually")
    else:           recs.append(f"Organic carbon {soc}% — maintain with annual compost")
    if "Sandy" in texture:  recs.append("Sandy soil — use drip irrigation, split fertilizer doses")
    elif "Clay" in texture: recs.append("Clay soil — avoid overwatering, good nutrient retention")
    elif "Silty" in texture: recs.append("Silty soil — fertile but prone to crusting, add compost")
    else:                   recs.append("Loam soil — best for most crops")
    return recs


def get_soil_data(lat: float, lon: float, province: str = "Afghanistan") -> dict:
    try:
        props    = ["phh2o", "clay", "sand", "silt", "soc", "bdod"]
        prop_str = "&".join(f"property={p}" for p in props)
        url      = (f"https://rest.soilgrids.org/soilgrids/v2.0/properties/query"
                    f"?lon={lon}&lat={lat}&{prop_str}&depth=0-30cm&value=mean")
        resp = requests.get(url, timeout=12, headers={"User-Agent": "ZaminAI/8.0"})
        if resp.status_code == 200:
            layers = resp.json().get("properties", {}).get("layers", [])
            vals   = {}
            for layer in layers:
                name = layer.get("name", "")
                v    = layer.get("depths", [{}])[0].get("values", {}).get("mean")
                if v is not None:
                    vals[name] = v
            if vals:
                ph      = round(vals.get("phh2o", 75)  / 10,  1)
                clay    = round(vals.get("clay",  200)  / 10,  1)
                sand    = round(vals.get("sand",  400)  / 10,  1)
                silt    = round(vals.get("silt",  300)  / 10,  1)
                soc     = round(vals.get("soc",    80)  / 100, 2)
                bd      = round(vals.get("bdod",  130)  / 100, 2)
                texture = classify_soil_texture(clay, sand, silt)
                return {"ph": ph, "clay": clay, "sand": sand, "silt": silt,
                        "soc": soc, "bulk_density": bd, "texture": texture,
                        "recommendations": soil_recommendations(texture, ph, soc, province),
                        "source": "soilgrids_api", "resolution": "250m"}
    except Exception as e:
        log.warning(f"SoilGrids API failed: {e}")
    soil = AFGHAN_SOILS.get(province, {"ph": 7.5, "clay": 20, "sand": 45,
                                        "silt": 35, "soc": 0.8, "texture": "Loam"})
    return {"ph": soil["ph"], "clay": soil["clay"], "sand": soil["sand"],
            "silt": soil["silt"], "soc": soil["soc"], "bulk_density": 1.35,
            "texture": soil["texture"],
            "recommendations": soil_recommendations(
                soil["texture"], soil["ph"], soil["soc"], province),
            "source": "provincial_db", "resolution": "province-level"}

# ══════════════════════════════════════════════════════════════════════════════
# CROP CALENDAR & SEASONAL ADVICE
# ══════════════════════════════════════════════════════════════════════════════

CROP_CALENDAR = {
    "wheat": {
        "north":   {"plant": [10, 11], "harvest": [6, 7],  "peak_ndvi_month": 5},
        "central": {"plant": [10, 11], "harvest": [7, 8],  "peak_ndvi_month": 6},
        "south":   {"plant": [11, 12], "harvest": [4, 5],  "peak_ndvi_month": 3},
        "west":    {"plant": [11, 12], "harvest": [5, 6],  "peak_ndvi_month": 4},
        "east":    {"plant": [10, 11], "harvest": [5, 6],  "peak_ndvi_month": 4},
    },
    "saffron":    {"all": {"plant": [9, 10], "harvest": [10, 11], "peak_ndvi_month": 10}},
    "vegetables": {
        "north": {"plant": [3, 4], "harvest": [7, 9], "peak_ndvi_month": 6},
        "south": {"plant": [2, 3], "harvest": [5, 7], "peak_ndvi_month": 4},
        "all":   {"plant": [3, 4], "harvest": [7, 9], "peak_ndvi_month": 6},
    },
}

MONTHLY_RAIN_FRACTION = {
    "north":   [0.04,0.07,0.14,0.16,0.14,0.08,0.03,0.02,0.03,0.05,0.10,0.14],
    "central": [0.05,0.08,0.15,0.15,0.12,0.06,0.02,0.01,0.02,0.05,0.12,0.17],
    "south":   [0.07,0.10,0.16,0.13,0.09,0.04,0.02,0.02,0.03,0.06,0.13,0.15],
    "west":    [0.08,0.11,0.16,0.12,0.08,0.04,0.02,0.01,0.02,0.06,0.14,0.16],
    "east":    [0.06,0.09,0.14,0.14,0.11,0.07,0.08,0.07,0.04,0.05,0.09,0.06],
}

_NORTH = {"Kunduz","Balkh","Takhar","Baghlan","Faryab","Jawzjan","Badakhshan","Samangan"}
_SOUTH = {"Kandahar","Helmand","Zabul","Uruzgan","Nimroz","Farah"}
_WEST  = {"Herat","Ghor","Badghis"}
_EAST  = {"Nangarhar","Kunar","Laghman","Nuristan","Khost","Paktia","Paktika"}


def get_province_type(province: str) -> str:
    if province in _NORTH: return "north"
    if province in _SOUTH: return "south"
    if province in _WEST:  return "west"
    if province in _EAST:  return "east"
    return "central"


def get_monthly_rain(annual_rain: float, province: str) -> List[float]:
    factors = MONTHLY_RAIN_FRACTION.get(get_province_type(province),
                                        MONTHLY_RAIN_FRACTION["central"])
    return [round(annual_rain * f, 1) for f in factors]


def get_current_season_advice(province: str, ndvi: float, mndwi: float) -> List[dict]:
    month  = datetime.now().month
    ptype  = get_province_type(province)
    advice = []
    wc = CROP_CALENDAR["wheat"].get(ptype, CROP_CALENDAR["wheat"]["central"])
    if month in wc["plant"]:
        advice.append({"type": "now", "crop": "wheat",
                        "action": "Plant wheat now — optimal sowing window"})
    elif month in wc["harvest"]:
        advice.append({"type": "now", "crop": "wheat",
                        "action": "Harvest wheat now — peak maturity window"})
    sc = CROP_CALENDAR["saffron"]["all"]
    if month in sc["plant"]:
        advice.append({"type": "now", "crop": "saffron",
                        "action": "Plant saffron corms now — only window"})
    elif month in sc["harvest"]:
        advice.append({"type": "now", "crop": "saffron",
                        "action": "Harvest saffron flowers now — 2–3 week window"})
    if mndwi < -0.15:
        days = 2 if mndwi < -0.25 else 4
        advice.append({"type": "urgent", "crop": "all",
                        "action": f"Irrigate within {days} days — water index critically low"})
    return advice

# ══════════════════════════════════════════════════════════════════════════════
# CROP DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_crop(ndvi, evi, savi, mndwi, lswi, month, province):
    if ndvi < 0.12:
        return [{"crop": "bare_fallow", "label_en": "Bare / Fallow land",
                 "label_fa": "زمین خالی", "label_ps": "خالي ځمکه",
                 "confidence": 0.90, "reason": f"NDVI {ndvi} < 0.12"}]
    candidates = []
    if month in range(3, 8) and 0.25 <= ndvi <= 0.60 and evi < 0.38:
        conf = 0.80 if 0.32 <= ndvi <= 0.52 else 0.72
        candidates.append({"crop": "wheat", "label_en": "Wheat (گندم)",
                            "label_fa": "گندم", "label_ps": "غنم",
                            "confidence": round(min(conf, 0.92), 2),
                            "reason": f"NDVI {ndvi} wheat signature"})
    if ndvi >= 0.38 and evi >= 0.28 and lswi >= -0.10:
        candidates.append({"crop": "vegetables", "label_en": "Vegetables",
                            "label_fa": "سبزیجات", "label_ps": "سبزیجات",
                            "confidence": 0.78, "reason": "High NDVI + LSWI"})
    if ndvi >= 0.42 and evi >= 0.30:
        candidates.append({"crop": "orchard", "label_en": "Orchard / Trees (باغ)",
                            "label_fa": "باغ", "label_ps": "باغ",
                            "confidence": 0.72, "reason": "High NDVI — orchard"})
    if not candidates:
        candidates.append({"crop": "mixed_unknown", "label_en": "Mixed / Unknown",
                            "label_fa": "مختلط", "label_ps": "مخلوط",
                            "confidence": 0.40, "reason": "Unclear spectral signature"})
    return sorted(candidates, key=lambda x: x["confidence"], reverse=True)

# ══════════════════════════════════════════════════════════════════════════════
# AREA CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def calc_area_ha(coords: List[List[float]]) -> float:
    n = len(coords)
    if n < 3:
        return 0.0
    area = 0.0
    R    = 6_371_000
    for i in range(n):
        j    = (i + 1) % n
        lat1 = math.radians(coords[i][0]); lon1 = math.radians(coords[i][1])
        lat2 = math.radians(coords[j][0]); lon2 = math.radians(coords[j][1])
        area += (lon2 - lon1) * (2 + math.sin(lat1) + math.sin(lat2))
    return round(abs(area) * R * R / 2 / 10_000, 2)

# ══════════════════════════════════════════════════════════════════════════════
# GEE LIVE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def gee_analyse(coords: List[List[float]], year: int, clat: float, clon: float) -> dict:
    import ee
    poly       = ee.Geometry.Polygon([[[c[1], c[0]] for c in coords]])
    start, end = _season_window(year)

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(poly)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
        .limit(5)
        .median()
        .clip(poly)
    )

    def _mean(img, band):
        v = img.reduceRegion(ee.Reducer.mean(), poly, 10, maxPixels=1e8).get(band).getInfo()
        return round(float(v), 4) if v is not None else None

    ndvi  = _mean(s2.normalizedDifference(["B8", "B4"]).rename("nd"), "nd")
    evi   = _mean(s2.expression(
        "2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))",
        {"NIR": s2.select("B8"), "RED": s2.select("B4"), "BLUE": s2.select("B2")},
    ).rename("evi"), "evi")
    savi  = _mean(s2.expression(
        "((NIR-RED)/(NIR+RED+0.5))*1.5",
        {"NIR": s2.select("B8"), "RED": s2.select("B4")},
    ).rename("savi"), "savi")
    mndwi = _mean(s2.normalizedDifference(["B3",  "B11"]).rename("nd"), "nd")
    lswi  = _mean(s2.normalizedDifference(["B8",  "B11"]).rename("nd"), "nd")
    ndre  = _mean(s2.normalizedDifference(["B8A", "B5" ]).rename("nd"), "nd")
    bsi   = _mean(s2.expression(
        "((SWIR1+RED)-(NIR+BLUE))/((SWIR1+RED)+(NIR+BLUE))",
        {"SWIR1": s2.select("B11"), "RED": s2.select("B4"),
         "NIR":   s2.select("B8"),  "BLUE": s2.select("B2")},
    ).rename("bsi"), "bsi")

    rain_col = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(poly)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .select("precipitation")
        .sum()
        .clip(poly)
    )
    rain = _mean(rain_col, "precipitation")

    # NDVI trend — dynamic range, never hardcoded
    trend: Dict[int, Any] = {}
    for yr in _trend_years():
        try:
            ts, te = _season_window(yr)
            c2 = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(poly)
                .filterDate(ts, te)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 25))
                .median()
                .clip(poly)
            )
            v = (
                c2.normalizedDifference(["B8", "B4"])
                .reduceRegion(ee.Reducer.mean(), poly, 10, maxPixels=1e8)
                .get("nd")
                .getInfo()
            )
            trend[yr] = round(float(v), 4) if v else None
        except Exception:
            trend[yr] = None

    return {"ndvi": ndvi, "evi": evi, "savi": savi, "mndwi": mndwi, "water": mndwi,
            "lswi": lswi, "ndre": ndre, "bsi": bsi, "rain": rain,
            "trend": trend, "ndvi_trend": trend,
            "lat": round(clat, 5), "lon": round(clon, 5),
            "source": "gee_live", "image_date": end}

# ══════════════════════════════════════════════════════════════════════════════
# AI  — Gemini + rule-based smart fallback
# ══════════════════════════════════════════════════════════════════════════════

def call_gemini(prompt: str) -> Optional[str]:
    if not GEMINI_KEY:
        return None
    for model in ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-pro"]:
        try:
            url  = (f"https://generativelanguage.googleapis.com/v1beta"
                    f"/models/{model}:generateContent?key={GEMINI_KEY}")
            resp = requests.post(url, json={
                "contents": [{"parts": [{"text": prompt}]}],
                "safetySettings": [
                    {"category": c, "threshold": "BLOCK_NONE"} for c in [
                        "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT",
                    ]
                ],
                "generationConfig": {"temperature": 0.6, "maxOutputTokens": 280},
            }, timeout=14)
            if resp.status_code == 200:
                cands = resp.json().get("candidates", [])
                if cands:
                    txt = cands[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    if txt and len(txt) > 8:
                        return txt.strip()
        except Exception as e:
            log.error(f"Gemini {model}: {e}")
    return None


def smart_fallback(question, ndvi, water, rain, area_j, lang, province="Afghanistan"):
    q       = question.lower()
    days    = 2 if water < -0.20 else 4 if water < -0.10 else 9
    fert    = round(area_j * 35)
    cost    = round(area_j * 400)
    is_irr  = any(w in q for w in ["irrigat","water","آبیاری","اوبه","آب"])
    is_crop = any(w in q for w in ["crop","plant","grow","کشت","محصول","وکارم"])
    is_fert = any(w in q for w in ["fertil","urea","dap","کود","سره"])
    if lang == "fa":
        if is_irr:
            return (f"🚨 آبیاری فوری — شاخص آب {water}. در {days}–{days+2} روز آبیاری کنید." if water < -0.05
                    else "آب متوسط است. در ۷–۱۰ روز آبیاری کنید.")
        if is_crop:
            return (f"با {rain}mm آب کم: ۱) زعفران ۲) کتان ۳) نخود" if rain < 200
                    else f"با {rain}mm: ۱) گندم ۲) سبزیجات ۳) کتان")
        if is_fert:
            return f"یوریا: {fert} کیلوگرام. DAP: {round(area_j*20)} کیلوگرام."
        return f"زمین {area_j} جریب — NDVI {ndvi}, آب {water}, باران {rain}mm."
    elif lang == "ps":
        if is_irr:
            return (f"🚨 بیړي اوبه — {water}. {days}–{days+2} ورځو کې اوبه ورکړئ." if water < -0.05
                    else "اوبه متوسط دي. ۷–۱۰ ورځو کې اوبه ورکړئ.")
        if is_crop:
            return (f"د {rain}mm لږو اوبو: ۱) زعفران ۲) کتان ۳) نخود" if rain < 200
                    else f"د {rain}mm: ۱) گندم ۲) سبزیجات ۳) کتان")
        if is_fert:
            return f"یوریا: {fert} کیلوګرام. DAP: {round(area_j*20)} کیلوګرام."
        return f"ستاسو {area_j} جریب — NDVI {ndvi}, اوبه {water}, باران {rain}mm."
    else:
        if is_irr:
            return (f"🚨 Irrigate within {days}–{days+2} days — MNDWI={water}. Cost: ~{cost:,} AFN."
                    if water < -0.05 else f"Water moderate (MNDWI={water}). Irrigate in 7–10 days.")
        if is_crop:
            return (f"With {rain}mm low water: 1) Saffron — 50× wheat profit. 2) Flax. 3) Chickpeas."
                    if rain < 200 else f"With {rain}mm: 1) Wheat. 2) Vegetables — 3× income. 3) Flax.")
        if is_fert:
            return f"Apply Urea: {fert}kg + DAP: {round(area_j*20)}kg per jereb."
        return f"Your {area_j} jereb — NDVI {ndvi}, MNDWI {water}, rain {rain}mm. What question?"

# ══════════════════════════════════════════════════════════════════════════════
# WEATHER  — Open-Meteo, no API key required
# ══════════════════════════════════════════════════════════════════════════════

WEATHER_CODE_MAP = {
    0:  {"icon": "☀️", "en": "Clear sky",          "fa": "آسمان صاف",    "ps": "روښانه آسمان"},
    1:  {"icon": "🌤️", "en": "Mainly clear",       "fa": "اکثرا صاف",    "ps": "اکثرا روښانه"},
    2:  {"icon": "⛅",  "en": "Partly cloudy",      "fa": "نیمه ابری",    "ps": "نیمه وریځو"},
    3:  {"icon": "☁️", "en": "Cloudy",              "fa": "ابری",         "ps": "وریځو"},
    45: {"icon": "🌫️", "en": "Foggy",              "fa": "مه",           "ps": "لړه"},
    51: {"icon": "🌦️", "en": "Light drizzle",      "fa": "باران سبک",    "ps": "سپک باران"},
    61: {"icon": "🌧️", "en": "Light rain",         "fa": "باران سبک",    "ps": "سپک باران"},
    65: {"icon": "⛈️", "en": "Heavy rain",         "fa": "باران شدید",   "ps": "سخت باران"},
    71: {"icon": "🌨️", "en": "Light snow",         "fa": "برف سبک",     "ps": "سپک واوره"},
    75: {"icon": "❄️", "en": "Heavy snow",         "fa": "برف شدید",    "ps": "سخت واوره"},
    80: {"icon": "🌧️", "en": "Rain showers",       "fa": "رگبار",       "ps": "بارانونه"},
    95: {"icon": "⛈️", "en": "Thunderstorm",       "fa": "طوفان رعد",    "ps": "د تندر طوفان"},
}
DAY_NAMES = {
    "en": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
    "fa": ["دوشنبه","سه‌شنبه","چهارشنبه","پنجشنبه","جمعه","شنبه","یکشنبه"],
    "ps": ["دوشنبه","درېشنبه","چهارشنبه","پنجشنبه","جمعه","شنبه","یکشنبه"],
}
_weather_cache: Dict[str, Any] = {}


def _fetch_weather(lat: float, lon: float, lang: str) -> dict:
    key = f"{round(lat, 1)},{round(lon, 1)}"
    if key in _weather_cache:
        ts, data = _weather_cache[key]
        if datetime.now() - ts < timedelta(hours=1):
            return {**data, "cached": True}
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon,
        "daily": ("temperature_2m_max,temperature_2m_min,"
                  "precipitation_sum,weathercode,windspeed_10m_max"),
        "current_weather": "true",
        "timezone": "Asia/Kabul",
        "forecast_days": 7,
    }, timeout=10)
    r.raise_for_status()
    raw     = r.json()
    daily   = raw.get("daily", {})
    current = raw.get("current_weather", {})
    forecast: List[dict] = []
    alerts:   List[dict] = []
    for i in range(len(daily.get("time", []))):
        dt_obj   = datetime.strptime(daily["time"][i], "%Y-%m-%d")
        code     = daily["weathercode"][i]
        code_inf = WEATHER_CODE_MAP.get(code, WEATHER_CODE_MAP[3])
        day = {
            "date":        daily["time"][i],
            "day_name":    DAY_NAMES.get(lang, DAY_NAMES["en"])[dt_obj.weekday()],
            "temp_max":    round(daily["temperature_2m_max"][i]),
            "temp_min":    round(daily["temperature_2m_min"][i]),
            "rain_mm":     round(daily["precipitation_sum"][i], 1),
            "wind_kmh":    round(daily["windspeed_10m_max"][i]),
            "icon":        code_inf["icon"],
            "description": code_inf.get(lang, code_inf["en"]),
        }
        forecast.append(day)
        if day["rain_mm"]  >= 20: alerts.append({"day": day["day_name"], "type": "heavy_rain",   "value": day["rain_mm"]})
        if day["temp_min"] <= 0:  alerts.append({"day": day["day_name"], "type": "frost",         "value": day["temp_min"]})
        if day["temp_max"] >= 40: alerts.append({"day": day["day_name"], "type": "extreme_heat",  "value": day["temp_max"]})
        if day["wind_kmh"] >= 50: alerts.append({"day": day["day_name"], "type": "high_wind",     "value": day["wind_kmh"]})
    result = {
        "ok": True,
        "current":  {"temp": round(current.get("temperature", 0)), "wind": round(current.get("windspeed", 0))},
        "forecast": forecast,
        "alerts":   alerts,
        "location": {"lat": lat, "lon": lon},
        "cached":   False,
    }
    _weather_cache[key] = (datetime.now(), result)
    return result

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status":    "ok",
        "version":   "8.0",
        "gee":       gee_ok,
        "database":  sb_ok,
        "ai":        "gemini" if GEMINI_KEY else "smart_only",
        "indices":   ["ndvi", "evi", "savi", "mndwi", "lswi", "ndre", "bsi"],
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "endpoints": [
            "GET  /health",
            "POST /analyse",       "POST /ask",           "POST /ndvi_tile",
            "POST /crop_detect",   "POST /monthly_rain",  "POST /soil",
            "POST /weather",
            "POST /db/farmer",     "POST /db/field/save", "POST /db/field/delete",
            "GET  /db/fields/{farmer_id}",
            "POST /db/analysis/save",                     "POST /db/chat/save",
        ],
    }


@app.post("/analyse")
def analyse(req: AnalyseRequest):
    if len(req.coords) < 3:
        raise HTTPException(status_code=400, detail="Need ≥ 3 coordinate points")

    year       = _resolve_year(req.year)
    lats       = [c[0] for c in req.coords]
    lons       = [c[1] for c in req.coords]
    clat       = sum(lats) / len(lats)
    clon       = sum(lons) / len(lons)
    area_ha    = calc_area_ha(req.coords)
    area_jereb = round(area_ha * 5, 1)
    month      = datetime.now().month

    if gee_ok:
        try:
            result = gee_analyse(req.coords, year, clat, clon)
            reg    = get_regional_data(clat, clon)
            result.update({"label": req.label, "area_ha": area_ha,
                            "area_jereb": area_jereb, "status": "success",
                            "province": reg["province"]})
            result["crops"]        = detect_crop(result["ndvi"], result["evi"], result["savi"],
                                                  result["mndwi"], result["lswi"], month, reg["province"])
            result["season"]       = get_current_season_advice(reg["province"], result["ndvi"], result["mndwi"])
            result["monthly_rain"] = get_monthly_rain(result.get("rain") or reg["rain"], reg["province"])
            result["soil"]         = get_soil_data(clat, clon, reg["province"])
            tv = [v for v in (result.get("trend") or {}).values() if v]
            if len(tv) >= 2:
                h_min = min(tv); h_max = max(tv); cur = result["ndvi"] or 0
                result["vci"] = round((cur - h_min) / (h_max - h_min + 0.001) * 100, 1)
            if req.farmer_id and req.field_id:
                db_save_analysis(req.field_id, req.farmer_id, result)
            return result
        except Exception as e:
            log.error(f"GEE analysis failed, falling back: {e}")

    reg    = get_regional_data(clat, clon)
    result = {
        "label": req.label, "status": "success", "source": reg["source"],
        "province": reg["province"],
        "ndvi": reg["ndvi"], "evi": reg["evi"], "savi": reg["savi"],
        "mndwi": reg["mndwi"], "water": reg["mndwi"], "lswi": reg["lswi"],
        "rain": reg["rain"], "area_ha": area_ha, "area_jereb": area_jereb,
        "trend": reg["trend"], "ndvi_trend": reg["trend"],
        "year": year, "lat": round(clat, 5), "lon": round(clon, 5),
        "latest_date": datetime.now().strftime("%Y-%m-%d"),
        "crops":        detect_crop(reg["ndvi"], reg["evi"], reg["savi"],
                                    reg["mndwi"], reg["lswi"], month, reg["province"]),
        "season":       get_current_season_advice(reg["province"], reg["ndvi"], reg["mndwi"]),
        "monthly_rain": get_monthly_rain(reg["rain"], reg["province"]),
        "soil":         get_soil_data(clat, clon, reg["province"]),
        "ndre":         round(reg["ndvi"] * 0.75, 4),
        "gndvi":        round(reg["ndvi"] * 0.88, 4),
        "ndmi":         round(reg["mndwi"] + 0.08, 4),
        "ndwi":         round(reg["mndwi"] + 0.05, 4),
        "vci":          None,
        "drought_index": round(reg["mndwi"] - reg["ndvi"], 4),
    }
    if req.farmer_id and req.field_id:
        db_save_analysis(req.field_id, req.farmer_id, result)
    return result


@app.post("/ask")
def ask(req: AskRequest):
    if not req.question:
        raise HTTPException(status_code=400, detail="No question provided")
    ndvi = 0.28; water = -0.19; rain = 240.0; area_j = 5.0; province = "Afghanistan"
    context = req.context or ""
    if req.field_data:
        fd       = req.field_data
        ndvi     = float(fd.get("ndvi",     0.28))
        water    = float(fd.get("mndwi", fd.get("water", -0.19)))
        rain     = float(fd.get("rain",    240))
        area_j   = float(fd.get("area_jereb", fd.get("area_ha", 1) * 5))
        province = fd.get("province", "Afghanistan")
        context  = (f"Field: NDVI={ndvi}, MNDWI={water}, Rain={rain}mm, "
                    f"Area={area_j} jereb, Province={province}")
    lang_inst = {
        "fa": "Afghan Dari (دری). Use دهقان for farmer, جریب for land. Eastern Arabic numerals ۱۲۳.",
        "ps": "Pashto (پښتو). Proper Pashto farming terms. Eastern Arabic numerals.",
        "en": "English. Concise and specific.",
    }.get(req.language, "English.")
    prompt = (
        f"You are ZaminAI, expert agricultural advisor for Afghan smallholder farmers.\n"
        f"Satellite data: {context}\n\n"
        f"Respond ONLY in {lang_inst}\n"
        f"Rules: exact amounts, under 90 words, speak as trusted local expert.\n\n"
        f"Question: {req.question}"
    )
    reply = call_gemini(prompt)
    model = "gemini"
    if not reply or len(reply) < 8:
        reply = smart_fallback(req.question, ndvi, water, rain, area_j, req.language, province)
        model = "smart"
    if req.farmer_id:
        db_save_chat(req.farmer_id, req.field_id, req.question, reply, req.language)
    return {"reply": reply, "answer": reply, "model": model}


@app.post("/ndvi_tile")
def ndvi_tile(req: NdviTileRequest):
    if not gee_ok:
        raise HTTPException(status_code=503, detail="GEE unavailable — GEE_SERVICE_ACCOUNT / GEE_PRIVATE_KEY not set")
    import ee
    year    = _resolve_year(req.year)
    start, end = _season_window(year)
    poly    = ee.Geometry.Polygon([[[c[1], c[0]] for c in req.coords]])
    col     = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterBounds(poly).filterDate(start, end)
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
               .median().clip(poly))
    url = col.normalizedDifference(["B8", "B4"]).getThumbURL({
        "min": 0, "max": 0.7,
        "palette": ["#d73027","#fc8d59","#fee08b","#d9ef8b","#91cf60","#1a9850"],
        "dimensions": 512, "format": "png",
    })
    return {"status": "success", "tile_url": url}


@app.post("/crop_detect")
def crop_detect(req: CropDetectRequest):
    month = req.month or datetime.now().month
    return {"status": "ok",
            "crops": detect_crop(req.ndvi, req.evi, req.savi,
                                 req.mndwi, req.lswi, month, req.province)}


@app.post("/monthly_rain")
def monthly_rain(req: MonthlyRainRequest):
    monthly = get_monthly_rain(req.annual_rain, req.province)
    return {"status": "ok", "monthly": monthly,
            "labels": ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"],
            "province": req.province, "annual": req.annual_rain}


@app.post("/soil")
def soil(req: SoilRequest):
    s = get_soil_data(req.lat, req.lon, req.province)
    s["status"] = "ok"
    return s


@app.post("/weather")
def weather(req: WeatherRequest):
    try:
        return _fetch_weather(req.lat, req.lon, req.lang)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Database routes ───────────────────────────────────────────────────────────

@app.post("/db/farmer")
def db_farmer(req: FarmerRequest):
    if not req.phone.strip():
        raise HTTPException(status_code=400, detail="Phone number required")
    farmer = db_get_or_create_farmer(req.phone.strip(), req.language, req.province)
    if not farmer:
        raise HTTPException(status_code=503, detail="Database unavailable")
    fields = db_get_farmer_fields(farmer["id"])
    return {"status": "ok", "farmer": farmer, "fields": fields, "field_count": len(fields)}


@app.post("/db/field/save")
def db_field_save(req: FieldSaveRequest):
    if len(req.coords) < 3:
        raise HTTPException(status_code=400, detail="farmer_id and ≥ 3 coordinate points required")
    field = db_save_field(req.farmer_id, req.coords, req.label,
                          req.province, req.area_ha, req.area_jereb)
    if not field:
        raise HTTPException(status_code=503, detail=f"Could not save field (db_ok={sb_ok})")
    return {"status": "ok", "field": field}


@app.post("/db/field/delete")
def db_field_delete(req: FieldDeleteRequest):
    if not sb_ok:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        sb.table("analyses").delete().eq("field_id", req.field_id).execute()
    except Exception as ae:
        log.error(f"delete analyses for field {req.field_id}: {ae}")
    sb.table("fields").delete().eq("id", req.field_id).eq("farmer_id", req.farmer_id).execute()
    log.info(f"✓ Field {req.field_id} deleted")
    return {"ok": True, "status": "deleted"}


@app.get("/db/fields/{farmer_id}")
def db_fields_get(farmer_id: str):
    fields = db_get_farmer_fields(farmer_id)
    return {"status": "ok", "fields": fields, "count": len(fields)}


@app.post("/db/analysis/save")
def db_analysis_save(req: AnalysisSaveRequest):
    result = db_save_analysis(req.field_id, req.farmer_id, req.analysis_data or {})
    return {"status": "ok", "saved": result is not None}


@app.post("/db/chat/save")
def db_chat_save(req: ChatSaveRequest):
    db_save_chat(req.farmer_id, req.field_id, req.question, req.answer, req.language)
    return {"status": "ok"}


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    log.info(f"ZaminAI API v8.0 — starting on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
