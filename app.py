"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ZaminAI API  —  v6.0                                                        ║
║  Satellite Farming Intelligence for Afghan Smallholders                      ║
║                                                                              ║
║  Author : Maiwand Jan Alamzoi                                                ║
║  Org    : Afghanistan Development Initiative (ADI)                           ║
║  Collab : Wageningen University & Research (WUR) + FAO                       ║
║                                                                              ║
║  Endpoints:                                                                  ║
║    GET  /health      — service status                                        ║
║    POST /analyse     — full field analysis (GEE or regional fallback)        ║
║    POST /ask         — AI question answering (Gemini + smart fallback)       ║
║    POST /ndvi_tile   — NDVI thumbnail URL for mini-map                       ║
║    POST /crop_detect — crop type detection from satellite indices             ║
║    POST /monthly_rain— monthly rainfall breakdown for a field                ║
║                                                                              ║
║  Satellite indices computed:                                                 ║
║    NDVI  — Normalized Difference Vegetation Index  (B8-B4)/(B8+B4)          ║
║    EVI   — Enhanced Vegetation Index (reduces atmosphere/soil noise)         ║
║    SAVI  — Soil Adjusted Vegetation Index (L=0.5 for Afghan soil)            ║
║    MNDWI — Modified Normalized Difference Water Index (B3-B11)/(B3+B11)     ║
║    LSWI  — Land Surface Water Index  (B8-B11)/(B8+B11)                      ║
║    NDRE  — Red Edge index (B8A-B5)/(B8A+B5) — crop stress sensitive         ║
║    BSI   — Bare Soil Index  — detects fallow/bare land                      ║
║                                                                              ║
║  Data sources:                                                               ║
║    Sentinel-2 SR Harmonized  — ESA / Google Earth Engine                    ║
║    CHIRPS Daily Rainfall      — UCSB Climate Hazards Group                  ║
║    Regional DB                — Provincial averages from MSc research        ║
║    SoilGrids (planned)        — ISRIC World Soil Information                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, json, math, logging, requests
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins="*")

# ── Environment variables ─────────────────────────────────────────────────────
GEMINI_KEY    = os.environ.get("GEMINI_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEE_SA        = os.environ.get("GEE_SERVICE_ACCOUNT", "")
GEE_KEY       = os.environ.get("GEE_PRIVATE_KEY", "").replace("\\n", "\n")

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

log.info(f"AI: {'Gemini' if GEMINI_KEY else 'Anthropic' if ANTHROPIC_KEY else 'Smart fallback only'}")


# ════════════════════════════════════════════════════════════════════════════════
# REGIONAL DATABASE
# Real Sentinel-2 values measured per Afghan province (MSc research + GEE)
# Updated: 2025 · Source: Maiwand Jan Alamzoi, IIT Kharagpur
# Columns: lat_min, lat_max, lon_min, lon_max, province,
#          ndvi, evi, savi, mndwi, lswi, rain_mm,
#          ndvi_trend {year: value}
# ════════════════════════════════════════════════════════════════════════════════
PROVINCES = [
    # Kunduz — major cropland, Amu Darya basin
    (36.4,37.2,68.2,69.2,"Kunduz",
     0.33, 0.24, 0.28, -0.14, -0.09, 287,
     {2019:0.40,2020:0.38,2021:0.35,2022:0.22,2023:0.27,2024:0.33,2025:0.35}),
    # Balkh — wheat + cotton belt
    (36.4,37.1,66.5,67.3,"Balkh",
     0.31, 0.22, 0.26, -0.18, -0.12, 245,
     {2019:0.37,2020:0.35,2021:0.31,2022:0.19,2023:0.24,2024:0.31,2025:0.33}),
    # Herat — western province, saffron + wheat
    (33.8,35.0,61.5,63.5,"Herat",
     0.28, 0.19, 0.23, -0.20, -0.14, 195,
     {2019:0.33,2020:0.31,2021:0.27,2022:0.15,2023:0.21,2024:0.28,2025:0.29}),
    # Nangarhar — subtropical, vegetables + wheat
    (33.8,34.6,70.0,71.5,"Nangarhar",
     0.38, 0.28, 0.32, -0.12, -0.07, 320,
     {2019:0.44,2020:0.41,2021:0.37,2022:0.26,2023:0.31,2024:0.38,2025:0.40}),
    # Kabul — highland, mixed crops
    (34.2,34.9,68.7,69.5,"Kabul",
     0.27, 0.18, 0.22, -0.22, -0.16, 305,
     {2019:0.32,2020:0.29,2021:0.25,2022:0.14,2023:0.20,2024:0.27,2025:0.28}),
    # Kandahar — arid, pomegranates + wheat
    (31.3,32.1,65.2,66.2,"Kandahar",
     0.22, 0.14, 0.18, -0.28, -0.21, 175,
     {2019:0.27,2020:0.24,2021:0.20,2022:0.11,2023:0.16,2024:0.22,2025:0.23}),
    # Helmand — Helmand River basin, cotton + opium historically
    (30.8,32.2,63.5,65.5,"Helmand",
     0.25, 0.16, 0.20, -0.25, -0.18, 148,
     {2019:0.30,2020:0.27,2021:0.23,2022:0.13,2023:0.18,2024:0.25,2025:0.26}),
    # Badakhshan — mountainous, diverse crops + saffron
    (36.5,38.5,70.0,72.0,"Badakhshan",
     0.41, 0.30, 0.35, -0.10, -0.06, 420,
     {2019:0.47,2020:0.44,2021:0.40,2022:0.29,2023:0.35,2024:0.41,2025:0.43}),
    # Takhar — fertile plains, wheat + vegetables
    (36.4,37.2,69.0,70.5,"Takhar",
     0.36, 0.26, 0.30, -0.15, -0.10, 340,
     {2019:0.42,2020:0.39,2021:0.35,2022:0.24,2023:0.29,2024:0.36,2025:0.38}),
    # Baghlan — sugar beet + wheat
    (35.8,36.6,68.2,69.2,"Baghlan",
     0.34, 0.25, 0.29, -0.16, -0.11, 295,
     {2019:0.40,2020:0.37,2021:0.33,2022:0.21,2023:0.27,2024:0.34,2025:0.36}),
    # Faryab — wheat + cotton
    (35.0,36.0,64.0,66.0,"Faryab",
     0.29, 0.20, 0.24, -0.19, -0.13, 220,
     {2019:0.35,2020:0.32,2021:0.27,2022:0.16,2023:0.22,2024:0.29,2025:0.31}),
    # Jawzjan — oil seeds + wheat
    (35.5,36.5,65.5,67.0,"Jawzjan",
     0.30, 0.21, 0.25, -0.17, -0.12, 240,
     {2019:0.36,2020:0.33,2021:0.28,2022:0.17,2023:0.23,2024:0.30,2025:0.32}),
    # Ghazni — highland wheat + potato
    (32.0,33.5,67.0,68.5,"Ghazni",
     0.24, 0.15, 0.19, -0.21, -0.15, 185,
     {2019:0.29,2020:0.26,2021:0.22,2022:0.12,2023:0.18,2024:0.24,2025:0.25}),
    # Bamyan — Hazarajat, potato + wheat
    (34.5,35.5,67.0,68.5,"Bamyan",
     0.27, 0.18, 0.22, -0.18, -0.13, 270,
     {2019:0.32,2020:0.29,2021:0.25,2022:0.14,2023:0.20,2024:0.27,2025:0.28}),
    # Logar — vegetables + wheat near Kabul
    (33.0,34.0,69.0,70.5,"Logar",
     0.26, 0.17, 0.21, -0.20, -0.14, 260,
     {2019:0.31,2020:0.28,2021:0.24,2022:0.13,2023:0.19,2024:0.26,2025:0.27}),
    # Paktia — forested, mixed crops
    (32.5,33.5,68.0,69.5,"Paktia",
     0.28, 0.19, 0.23, -0.18, -0.12, 285,
     {2019:0.33,2020:0.30,2021:0.26,2022:0.15,2023:0.21,2024:0.28,2025:0.29}),
]

def get_regional_data(lat, lon):
    """
    Return regional satellite data for a given coordinate.
    Matches to nearest province bounding box.
    Falls back to interpolated values if outside all known provinces.
    """
    for (lat_min,lat_max,lon_min,lon_max,name,
         ndvi,evi,savi,mndwi,lswi,rain,trend) in PROVINCES:
        if lat_min<=lat<=lat_max and lon_min<=lon<=lon_max:
            return {
                "province": name, "ndvi": ndvi, "evi": evi,
                "savi": savi, "mndwi": mndwi, "lswi": lswi,
                "rain": rain, "trend": trend, "source": "regional_db"
            }
    # Interpolate for locations outside known provinces
    ndvi  = round(max(0.08, min(0.45, 0.06 + lat*0.003 + (lon-62)*0.002)), 4)
    evi   = round(ndvi * 0.72, 4)
    savi  = round(ndvi * 0.85, 4)
    rain  = max(80, min(480, int(lat * 8)))
    mndwi = round(max(-0.38, min(0.05, -0.38 + rain*0.001)), 4)
    lswi  = round(mndwi + 0.05, 4)
    return {
        "province": "Afghanistan", "ndvi": ndvi, "evi": evi,
        "savi": savi, "mndwi": mndwi, "lswi": lswi, "rain": rain,
        "trend": {
            2019: round(ndvi+0.07,4), 2020: round(ndvi+0.05,4),
            2021: round(ndvi+0.02,4), 2022: round(ndvi-0.10,4),
            2023: round(ndvi-0.04,4), 2024: ndvi, 2025: round(ndvi+0.02,4)
        },
        "source": "interpolated"
    }

# ════════════════════════════════════════════════════════════════════════════════
# SOILGRIDS API — ISRIC World Soil Information
# Free REST API — no key needed
# Resolution: 250m — good for field-level in Afghanistan
# Source: https://soilgrids.org / https://www.isric.org
#
# Properties returned:
#   phh2o — soil pH in water (returned as pH×10, divide by 10)
#   clay  — clay content g/kg (divide by 10 for %)
#   sand  — sand content g/kg (divide by 10 for %)
#   silt  — silt content g/kg (divide by 10 for %)
#   soc   — soil organic carbon dg/kg (divide by 100 for %)
#   bdod  — bulk density cg/cm³ (divide by 100 for g/cm³)
#
# Soil texture classification (USDA triangle):
#   Sandy:      sand >70%
#   Sandy loam: sand 50-70%, clay <20%
#   Loam:       sand 25-50%, clay 10-27%, silt 28-50%
#   Clay loam:  clay 27-40%
#   Clay:       clay >40%
#   Silty loam: silt >50%, clay <27%
# ════════════════════════════════════════════════════════════════════════════════

# Afghan provincial soil database (fallback when SoilGrids API unavailable)
# Values from: FAO soil map of Afghanistan + ISRIC regional estimates
AFGHAN_SOILS = {
    "Kunduz":     {"ph":7.4,"clay":22,"sand":38,"silt":40,"soc":0.9,"texture":"Silty loam","color":"dark_brown"},
    "Balkh":      {"ph":7.6,"clay":18,"sand":52,"silt":30,"soc":0.7,"texture":"Sandy loam","color":"light_brown"},
    "Herat":      {"ph":7.8,"clay":15,"sand":58,"silt":27,"soc":0.5,"texture":"Sandy loam","color":"light"},
    "Nangarhar":  {"ph":7.2,"clay":28,"sand":32,"silt":40,"soc":1.2,"texture":"Loam","color":"dark_brown"},
    "Kabul":      {"ph":7.5,"clay":20,"sand":42,"silt":38,"soc":0.8,"texture":"Loam","color":"brown"},
    "Kandahar":   {"ph":8.0,"clay":12,"sand":65,"silt":23,"soc":0.3,"texture":"Sandy","color":"light_red"},
    "Helmand":    {"ph":7.9,"clay":14,"sand":60,"silt":26,"soc":0.4,"texture":"Sandy loam","color":"light"},
    "Badakhshan": {"ph":6.8,"clay":30,"sand":28,"silt":42,"soc":1.8,"texture":"Clay loam","color":"dark"},
    "Takhar":     {"ph":7.3,"clay":24,"sand":35,"silt":41,"soc":1.1,"texture":"Silty loam","color":"brown"},
    "Baghlan":    {"ph":7.4,"clay":22,"sand":38,"silt":40,"soc":1.0,"texture":"Silty loam","color":"brown"},
    "Faryab":     {"ph":7.7,"clay":16,"sand":55,"silt":29,"soc":0.6,"texture":"Sandy loam","color":"light_brown"},
    "Jawzjan":    {"ph":7.6,"clay":17,"sand":53,"silt":30,"soc":0.6,"texture":"Sandy loam","color":"light_brown"},
    "Ghazni":     {"ph":7.5,"clay":20,"sand":44,"silt":36,"soc":0.7,"texture":"Loam","color":"brown"},
    "Bamyan":     {"ph":7.1,"clay":26,"sand":32,"silt":42,"soc":1.4,"texture":"Clay loam","color":"dark_brown"},
    "Logar":      {"ph":7.4,"clay":23,"sand":36,"silt":41,"soc":1.0,"texture":"Silty loam","color":"brown"},
    "Paktia":     {"ph":7.2,"clay":25,"sand":34,"silt":41,"soc":1.2,"texture":"Loam","color":"dark_brown"},
}

def classify_soil_texture(clay, sand, silt):
    """
    Classify soil texture using USDA soil texture triangle.
    Input: clay%, sand%, silt% (should sum to ~100)
    Returns: texture class name
    """
    if sand >= 70:                              return "Sandy"
    if sand >= 50 and clay < 20:               return "Sandy loam"
    if clay >= 40:                             return "Clay"
    if clay >= 27 and clay < 40:               return "Clay loam"
    if silt >= 50 and clay < 27:               return "Silty loam"
    if silt >= 80:                             return "Silt"
    return "Loam"

def soil_recommendations(texture, ph, soc, province):
    """
    Generate crop and management recommendations based on soil properties.
    Returns list of recommendation strings in English.
    """
    recs = []

    # pH recommendations
    if ph < 6.5:
        recs.append(f"Acidic soil (pH {ph}) — apply lime 200-300 kg/jereb to raise pH")
    elif ph > 8.0:
        recs.append(f"Alkaline soil (pH {ph}) — add organic matter, avoid urea fertilizer")
    elif ph > 7.5:
        recs.append(f"Slightly alkaline (pH {ph}) — normal for {province}. Use ammonium sulfate over urea")
    else:
        recs.append(f"Good pH {ph} — suitable for wheat, vegetables, most crops")

    # Organic carbon
    if soc < 0.5:
        recs.append("Very low organic carbon — add 3-4 tonnes compost/jereb before planting")
    elif soc < 1.0:
        recs.append("Low organic carbon — add 2 tonnes compost/jereb annually")
    else:
        recs.append(f"Organic carbon {soc}% — reasonable. Maintain with annual compost")

    # Texture recommendations
    if "Sandy" in texture:
        recs.append("Sandy soil — water drains fast. Use drip irrigation, split fertilizer doses")
        recs.append("Best crops: saffron, flax, chickpeas — drought tolerant crops preferred")
    elif "Clay" in texture:
        recs.append("Clay soil — holds water well but can waterlog. Avoid overwatering")
        recs.append("Best crops: wheat, rice, vegetables — clay retains nutrients well")
    elif "Loam" in texture:
        recs.append("Loam soil — best for most crops. Good water and nutrient retention")
        recs.append("All crops suitable: wheat, vegetables, cotton, saffron")
    elif "Silty" in texture:
        recs.append("Silty soil — fertile but prone to crusting. Add compost to improve structure")

    return recs

def get_soil_data(lat, lon, province="Afghanistan"):
    """
    Fetch soil properties from SoilGrids REST API.
    Falls back to provincial database if API unavailable.

    Returns dict with:
        ph, clay, sand, silt, soc, bulk_density,
        texture, recommendations, source
    """
    # Try SoilGrids API first
    try:
        props = ["phh2o","clay","sand","silt","soc","bdod"]
        prop_str = "&".join(f"property={p}" for p in props)
        url = (f"https://rest.soilgrids.org/soilgrids/v2.0/properties/query"
               f"?lon={lon}&lat={lat}&{prop_str}&depth=0-30cm&value=mean")
        resp = requests.get(url, timeout=12,
                            headers={"User-Agent":"ZaminAI/6.0 (zaminai.org)"})
        if resp.status_code == 200:
            layers = resp.json().get("properties",{}).get("layers",[])
            vals   = {}
            for layer in layers:
                name = layer.get("name","")
                v    = layer.get("depths",[{}])[0].get("values",{}).get("mean")
                if v is not None:
                    vals[name] = v
            if vals:
                # Apply unit conversions
                ph   = round(vals.get("phh2o", 75) / 10, 1)
                clay = round(vals.get("clay",  200) / 10, 1)
                sand = round(vals.get("sand",  400) / 10, 1)
                silt = round(vals.get("silt",  300) / 10, 1)
                soc  = round(vals.get("soc",   80)  / 100, 2)
                bd   = round(vals.get("bdod",  130) / 100, 2)
                texture = classify_soil_texture(clay, sand, silt)
                return {
                    "ph": ph, "clay": clay, "sand": sand,
                    "silt": silt, "soc": soc, "bulk_density": bd,
                    "texture": texture,
                    "recommendations": soil_recommendations(texture, ph, soc, province),
                    "source": "soilgrids_api",
                    "resolution": "250m"
                }
    except Exception as e:
        log.warning(f"SoilGrids API failed: {e} — using provincial database")

    # Provincial database fallback
    soil = AFGHAN_SOILS.get(province, {
        "ph":7.5,"clay":20,"sand":45,"silt":35,
        "soc":0.8,"texture":"Loam","color":"brown"
    })
    return {
        "ph":          soil["ph"],
        "clay":        soil["clay"],
        "sand":        soil["sand"],
        "silt":        soil["silt"],
        "soc":         soil["soc"],
        "bulk_density": 1.35,
        "texture":     soil["texture"],
        "recommendations": soil_recommendations(
            soil["texture"], soil["ph"], soil["soc"], province),
        "source":      "provincial_db",
        "resolution":  "province-level"
    }




# ════════════════════════════════════════════════════════════════════════════════
# CROP CALENDAR
# Planting and harvest windows per crop per province type
# Based on: Afghan agricultural calendars + FAO country profiles
# Province types: north (Kunduz/Balkh/Takhar), central (Kabul/Ghazni),
#                 south (Kandahar/Helmand), west (Herat), east (Nangarhar)
# ════════════════════════════════════════════════════════════════════════════════
CROP_CALENDAR = {
    "wheat": {
        "north":   {"plant":[10,11], "harvest":[6,7],  "peak_ndvi_month": 5},
        "central": {"plant":[10,11], "harvest":[7,8],  "peak_ndvi_month": 6},
        "south":   {"plant":[11,12], "harvest":[4,5],  "peak_ndvi_month": 3},
        "west":    {"plant":[11,12], "harvest":[5,6],  "peak_ndvi_month": 4},
        "east":    {"plant":[10,11], "harvest":[5,6],  "peak_ndvi_month": 4},
    },
    "saffron": {
        "all": {"plant":[9,10], "harvest":[10,11], "peak_ndvi_month": 10}
    },
    "vegetables": {
        "north":   {"plant":[3,4],   "harvest":[7,9],  "peak_ndvi_month": 6},
        "south":   {"plant":[2,3],   "harvest":[5,7],  "peak_ndvi_month": 4},
        "all":     {"plant":[3,4],   "harvest":[7,9],  "peak_ndvi_month": 6},
    },
    "cotton": {
        "north":   {"plant":[4,5],   "harvest":[9,10], "peak_ndvi_month": 7},
        "south":   {"plant":[3,4],   "harvest":[8,9],  "peak_ndvi_month": 6},
    },
    "chickpeas": {
        "all": {"plant":[3,4], "harvest":[7,8], "peak_ndvi_month": 6}
    },
    "flax": {
        "all": {"plant":[3,4], "harvest":[7,8], "peak_ndvi_month": 6}
    },
}

# Monthly rainfall distribution by province type (fraction of annual total)
# Based on CHIRPS climatology for Afghanistan
MONTHLY_RAIN_FRACTION = {
    "north":   [0.04,0.07,0.14,0.16,0.14,0.08,0.03,0.02,0.03,0.05,0.10,0.14],
    "central": [0.05,0.08,0.15,0.15,0.12,0.06,0.02,0.01,0.02,0.05,0.12,0.17],
    "south":   [0.07,0.10,0.16,0.13,0.09,0.04,0.02,0.02,0.03,0.06,0.13,0.15],
    "west":    [0.08,0.11,0.16,0.12,0.08,0.04,0.02,0.01,0.02,0.06,0.14,0.16],
    "east":    [0.06,0.09,0.14,0.14,0.11,0.07,0.08,0.07,0.04,0.05,0.09,0.06],
}

def get_province_type(province):
    """Classify province into climate zone for calendar and rainfall lookups."""
    north   = ["Kunduz","Balkh","Takhar","Baghlan","Faryab","Jawzjan","Badakhshan","Samangan"]
    south   = ["Kandahar","Helmand","Zabul","Uruzgan","Nimroz","Farah"]
    west    = ["Herat","Ghor","Badghis"]
    east    = ["Nangarhar","Kunar","Laghman","Nuristan","Khost","Paktia","Paktika"]
    if province in north:   return "north"
    if province in south:   return "south"
    if province in west:    return "west"
    if province in east:    return "east"
    return "central"

def get_monthly_rain(annual_rain, province):
    """Return list of 12 monthly rainfall values (mm) from annual total."""
    ptype   = get_province_type(province)
    factors = MONTHLY_RAIN_FRACTION.get(ptype, MONTHLY_RAIN_FRACTION["central"])
    return [round(annual_rain * f, 1) for f in factors]

def get_current_season_advice(province, ndvi, mndwi):
    """
    Return what a farmer should do RIGHT NOW based on current month,
    province, and satellite indices.
    """
    month = datetime.now().month
    ptype = get_province_type(province)
    advice = []

    # Wheat planting window
    wc = CROP_CALENDAR["wheat"].get(ptype, CROP_CALENDAR["wheat"]["central"])
    if month in wc["plant"]:
        advice.append({"type":"now","crop":"wheat","action":"Plant wheat now — optimal sowing window"})
    elif month in wc["harvest"]:
        advice.append({"type":"now","crop":"wheat","action":"Harvest wheat now — peak maturity window"})
    elif month == wc["peak_ndvi_month"]:
        advice.append({"type":"now","crop":"wheat","action":"Wheat at peak growth — check water and fertilizer"})

    # Saffron
    sc = CROP_CALENDAR["saffron"]["all"]
    if month in sc["plant"]:
        advice.append({"type":"now","crop":"saffron","action":"Plant saffron corms now — September/October is the only window"})
    elif month in sc["harvest"]:
        advice.append({"type":"now","crop":"saffron","action":"Harvest saffron flowers now — window is only 2-3 weeks"})

    # Water stress urgency
    if mndwi < -0.15:
        days = 2 if mndwi < -0.25 else 4
        advice.append({"type":"urgent","crop":"all","action":f"Irrigate within {days} days — water index critically low"})

    return advice


# ════════════════════════════════════════════════════════════════════════════════
# CROP RECOGNITION
# Detect likely crop type from multi-index satellite signature
# Based on: spectral signatures of Afghan crops from literature +
#           MSc research field validation
#
# Key logic:
#   Wheat (May peak):  NDVI 0.35-0.55, EVI 0.25-0.40, LSWI low
#   Saffron (Oct):     NDVI 0.15-0.30, low biomass but distinctive timing
#   Vegetables:        NDVI 0.40-0.65, high EVI, high LSWI
#   Cotton:            NDVI 0.30-0.50, high LSWI in summer
#   Bare/Fallow:       NDVI < 0.15, high BSI
#   Orchard/Trees:     NDVI 0.45-0.70, persistent through seasons
# ════════════════════════════════════════════════════════════════════════════════

def detect_crop(ndvi, evi, savi, mndwi, lswi, month, province):
    """
    Detect probable crop type from multi-index signature.
    Returns: list of candidates with confidence score and reasoning.

    Confidence scale: 0.0 - 1.0
    Note: satellite-based crop detection has ~70-85% accuracy for
    dominant crops. Mixed/small fields may return lower confidence.
    """
    candidates = []
    ptype = get_province_type(province)

    # ── Bare / Fallow ─────────────────────────────────────────────────────────
    if ndvi < 0.12:
        candidates.append({
            "crop": "bare_fallow",
            "label_en": "Bare / Fallow land",
            "label_fa": "زمین خالی / بایر",
            "label_ps": "خالي / بایره ځمکه",
            "confidence": 0.90,
            "reason": f"NDVI {ndvi} < 0.12 — very low vegetation, likely bare or post-harvest"
        })
        return candidates

    # ── Wheat ────────────────────────────────────────────────────────────────
    wheat_cal = CROP_CALENDAR["wheat"].get(ptype, CROP_CALENDAR["wheat"]["central"])
    wheat_peak = wheat_cal["peak_ndvi_month"]
    wheat_months_active = wheat_cal["plant"] + list(range(wheat_cal["plant"][-1]+1, wheat_peak+1)) + wheat_cal["harvest"]

    if month in range(3,8) and 0.25 <= ndvi <= 0.60 and evi < 0.38:
        conf = 0.75
        if wheat_peak - 1 <= month <= wheat_peak + 1: conf = 0.85
        if 0.32 <= ndvi <= 0.52: conf += 0.05
        candidates.append({
            "crop": "wheat",
            "label_en": "Wheat (گندم)",
            "label_fa": "گندم",
            "label_ps": "غنم",
            "confidence": round(min(conf, 0.92), 2),
            "reason": f"NDVI {ndvi}, EVI {evi} — wheat spectral signature in growing season"
        })

    # ── Vegetables ───────────────────────────────────────────────────────────
    if ndvi >= 0.38 and evi >= 0.28 and lswi >= -0.10:
        conf = 0.70
        if ndvi >= 0.45: conf = 0.80
        if month in [4,5,6,7,8]: conf += 0.05
        candidates.append({
            "crop": "vegetables",
            "label_en": "Vegetables (سبزیجات)",
            "label_fa": "سبزیجات",
            "label_ps": "سبزیجات",
            "confidence": round(min(conf, 0.88), 2),
            "reason": f"NDVI {ndvi}, LSWI {lswi} — high biomass + water content typical of vegetables"
        })

    # ── Orchard / Trees ──────────────────────────────────────────────────────
    if ndvi >= 0.42 and evi >= 0.30 and savi >= 0.36:
        conf = 0.72
        candidates.append({
            "crop": "orchard",
            "label_en": "Orchard / Trees (باغ)",
            "label_fa": "باغ / درختان",
            "label_ps": "باغ / ونې",
            "confidence": round(conf, 2),
            "reason": f"High NDVI {ndvi} + EVI {evi} consistent with orchard or dense trees"
        })

    # ── Cotton ───────────────────────────────────────────────────────────────
    if month in [6,7,8,9] and 0.28 <= ndvi <= 0.52 and lswi >= -0.08 and ptype in ["north","south"]:
        conf = 0.68
        candidates.append({
            "crop": "cotton",
            "label_en": "Cotton (پنبه)",
            "label_fa": "پنبه",
            "label_ps": "پنبه",
            "confidence": round(conf, 2),
            "reason": f"NDVI {ndvi}, LSWI {lswi} in summer — cotton signature in {ptype}ern Afghanistan"
        })

    # ── Saffron ──────────────────────────────────────────────────────────────
    if month in [10,11] and 0.12 <= ndvi <= 0.32 and ptype in ["west","central","north"]:
        conf = 0.65
        candidates.append({
            "crop": "saffron",
            "label_en": "Saffron (زعفران)",
            "label_fa": "زعفران",
            "label_ps": "زعفران",
            "confidence": round(conf, 2),
            "reason": f"Low NDVI {ndvi} in October/November — possible saffron (low-biomass high-value crop)"
        })

    # ── Fallback if nothing detected ─────────────────────────────────────────
    if not candidates:
        candidates.append({
            "crop": "mixed_unknown",
            "label_en": "Mixed / Unknown crops",
            "label_fa": "محصولات مختلط / ناشناخته",
            "label_ps": "مخلوط / نامعلوم محصولات",
            "confidence": 0.40,
            "reason": f"NDVI {ndvi} — vegetation present but crop type unclear from current indices"
        })

    # Sort by confidence descending
    return sorted(candidates, key=lambda x: x["confidence"], reverse=True)


# ════════════════════════════════════════════════════════════════════════════════
# AREA CALCULATION
# Geodesic area using the shoelace formula on a spherical earth
# Accuracy: ±2-3% for typical field sizes (0.1 - 20 ha)
# For very irregular shapes or fields crossing UTM zones,
# consider upgrading to Vincenty formula or turf.js on frontend
# ════════════════════════════════════════════════════════════════════════════════
def calc_area_ha(coords):
    """
    Calculate geodesic polygon area in hectares.
    Input: list of [lat, lon] pairs
    Output: float (hectares), rounded to 2 decimal places
    """
    n = len(coords)
    if n < 3:
        return 0.0
    area = 0.0
    R = 6371000  # Earth radius in metres
    for i in range(n):
        j = (i + 1) % n
        lat1, lon1 = math.radians(coords[i][0]), math.radians(coords[i][1])
        lat2, lon2 = math.radians(coords[j][0]), math.radians(coords[j][1])
        area += (lon2 - lon1) * (2 + math.sin(lat1) + math.sin(lat2))
    area = abs(area) * R * R / 2
    return round(area / 10000, 2)  # m² → hectares


# ════════════════════════════════════════════════════════════════════════════════
# AI — GEMINI REST
# Direct HTTP call — no google-generativeai package needed
# Tries 3 models in order of quality/speed
# ════════════════════════════════════════════════════════════════════════════════
def call_gemini(prompt):
    """
    Call Gemini via REST API.
    Returns: response text string, or None if all models fail.
    """
    if not GEMINI_KEY:
        return None
    models = ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-pro"]
    for model in models:
        try:
            url = (f"https://generativelanguage.googleapis.com/v1beta"
                   f"/models/{model}:generateContent?key={GEMINI_KEY}")
            resp = requests.post(url, json={
                "contents": [{"parts": [{"text": prompt}]}],
                "safetySettings": [
                    {"category": c, "threshold": "BLOCK_NONE"} for c in [
                        "HARM_CATEGORY_HARASSMENT",
                        "HARM_CATEGORY_HATE_SPEECH",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "HARM_CATEGORY_DANGEROUS_CONTENT"
                    ]
                ],
                "generationConfig": {"temperature": 0.6, "maxOutputTokens": 280}
            }, timeout=14)
            if resp.status_code == 200:
                cands = resp.json().get("candidates", [])
                if cands:
                    txt = cands[0].get("content",{}).get("parts",[{}])[0].get("text","")
                    if txt and len(txt) > 8:
                        log.info(f"✓ Gemini {model}: {len(txt)} chars")
                        return txt.strip()
                    log.warning(f"Gemini {model}: empty — {cands[0].get('finishReason','?')}")
            elif resp.status_code == 429:
                log.warning("Gemini: rate limited")
                break
            else:
                log.warning(f"Gemini {model}: HTTP {resp.status_code}")
        except Exception as e:
            log.error(f"Gemini {model}: {e}")
    return None


# ════════════════════════════════════════════════════════════════════════════════
# AI — ANTHROPIC FALLBACK
# ════════════════════════════════════════════════════════════════════════════════
def call_anthropic(prompt):
    """Call Anthropic Claude as secondary AI fallback."""
    if not ANTHROPIC_KEY:
        return None
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_KEY,
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": "claude-haiku-4-5-20251001", "max_tokens": 280,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=14
        )
        if resp.status_code == 200:
            return resp.json()["content"][0]["text"].strip()
        log.warning(f"Anthropic: HTTP {resp.status_code}")
    except Exception as e:
        log.error(f"Anthropic: {e}")
    return None


# ════════════════════════════════════════════════════════════════════════════════
# SMART FALLBACK AI
# Rule-based responses using real satellite data
# Always gives a specific, useful answer even when all AI APIs fail
# Based on: Afghan agricultural extension knowledge + MSc research findings
# ════════════════════════════════════════════════════════════════════════════════
def smart_fallback(question, ndvi, water, rain, area_j, lang, province="Afghanistan"):
    """
    Generate satellite-data-driven farming advice without AI APIs.
    Matches question intent and returns specific, actionable guidance.
    Language: 'en', 'fa' (Dari), or 'ps' (Pashto)
    """
    q     = question.lower()
    days  = 2 if water < -0.20 else 4 if water < -0.10 else 9
    fert  = round(area_j * 35)
    cost  = round(area_j * 400)
    month = datetime.now().month

    # Intent detection
    is_irr    = any(w in q for w in ["irrigat","water","آبیاری","اوبه","آب","water"])
    is_crop   = any(w in q for w in ["crop","plant","grow","کشت","محصول","وکارم","بکارم"])
    is_profit = any(w in q for w in ["profit","income","earn","money","فایده","عاید","ګټه"])
    is_fert   = any(w in q for w in ["fertil","urea","dap","کود","سره","fertilizer"])
    is_rec    = any(w in q for w in ["recov","ndvi","2022","بهبود","بهتر"])
    is_crop_q = any(w in q for w in ["what crop","which crop","best crop","کدام","کوم"])

    if lang == "fa":
        nums = {"0":"۰","1":"۱","2":"۲","3":"۳","4":"۴","5":"۵","6":"۶","7":"۷","8":"۸","9":"۹"}
        def fa_num(n):
            return "".join(nums.get(c,c) for c in str(n))
        if is_irr:
            if water < -0.05:
                return (f"🚨 آبیاری فوری — شاخص آب {water} است. زمین {fa_num(area_j)} جریب شما را "
                        f"در {fa_num(days)}–{fa_num(days+2)} روز آبیاری کنید. "
                        f"۵۰–۷۰ ملیمتر آب، صبح زود با آبیاری قطره‌ای یا جوی. "
                        f"هزینه تخمینی: ~{fa_num(cost)} افغانی.")
            return (f"آب در سطح متوسط است (MNDWI={water}). "
                    f"در ۷–۱۰ روز آبیاری کنید. باران سالانه {fa_num(rain)} ملیمتر.")
        if is_crop or is_crop_q:
            if rain < 200 or water < -0.20:
                return (f"با {fa_num(rain)}mm آب کم: "
                        f"۱) زعفران — ۳۰۰mm آب کافی، فایده ۵۰ برابر گندم. "
                        f"۲) کتان — مقاوم به خشکی. ۳) نخود — خاک را بهبود می‌دهد.")
            return (f"با {fa_num(rain)}mm آب: "
                    f"۱) گندم — محصول پایه قابل اعتماد. "
                    f"۲) سبزیجات — فایده ۳ برابر. ۳) کتان — کم‌آب‌تر از گندم.")
        if is_profit:
            base = "زعفران ≈ ۱۵ میلیون AFN/هکتار" if rain < 200 else "گندم ≈ ۴۲۰۰ AFN/جریب"
            return f"برای زمین {fa_num(area_j)} جریب: {base}. مصرف تخمینی: ~{fa_num(round(area_j*3200))} افغانی."
        if is_fert:
            return (f"NDVI {ndvi} نشان می‌دهد کود نیاز دارید. "
                    f"یوریا: {fa_num(fert)} کیلوگرام (۳۰–۴۰ kg/جریب). "
                    f"DAP: {fa_num(round(area_j*20))} کیلوگرام. "
                    f"در آب آبیاری حل کنید. تاثیر در ۲–۳ هفته.")
        return (f"زمین {fa_num(area_j)} جریب — "
                f"NDVI {ndvi} ({'خوب' if ndvi>=0.30 else 'تحت فشار'}), "
                f"آب {water} ({'آبیاری کنید' if water<-0.05 else 'متوسط'}), "
                f"باران {fa_num(rain)}mm. چه سوال خاصی دارید؟")

    elif lang == "ps":
        if is_irr:
            if water < -0.05:
                return (f"🚨 بیړي اوبه ورکول — د اوبو شاخص {water} دی. "
                        f"{area_j} جریب ځمکه {days}–{days+2} ورځو کې اوبه ورکړئ. "
                        f"۵۰–۷۰ ملیمتر اوبه، سهار وختي. تخمیني لګښت: ~{cost} افغاني.")
            return (f"اوبه متوسط دي (MNDWI={water}). "
                    f"د ۷–۱۰ ورځو کې اوبه ورکړئ. کلنی باران {rain}mm دی.")
        if is_crop or is_crop_q:
            if rain < 200 or water < -0.20:
                return (f"د {rain}mm لږو اوبو سره: "
                        f"۱) زعفران — ۳۰۰mm اوبه کافي، د گندم ۵۰ ځله ډیره ګټه. "
                        f"۲) کتان — د خشکسالۍ مقاوم. ۳) نخود — خاوره ښه کوي.")
            return (f"د {rain}mm اوبو سره: "
                    f"۱) گندم — باوري. ۲) سبزیجات — ۳ ځله ډیره ګټه. "
                    f"۳) کتان — له گندم نه لږ اوبه.")
        if is_profit:
            base = "زعفران ≈ ۱۵ میلیون AFN/هکتار" if rain < 200 else "گندم ≈ ۴۲۰۰ AFN/جریب"
            return f"ستاسو {area_j} جریب: {base}. تخمیني لګښت: ~{round(area_j*3200)} افغاني."
        if is_fert:
            return (f"NDVI {ndvi} ښیي چې سرې ته اړتیا ده. "
                    f"یوریا: {fert} کیلوګرام (۳۰–۴۰ kg/جریب). "
                    f"د ۲–۳ اونیو کې اغیز ښکاره کیږي.")
        return (f"ستاسو {area_j} جریب — NDVI {ndvi} "
                f"({'ښه' if ndvi>=0.30 else 'تحت فشار'}), "
                f"اوبه {water} ({'اوبه ورکړئ' if water<-0.05 else 'متوسط'}), "
                f"باران {rain}mm. کومه ځانګړې پوښتنه؟")

    else:  # English
        if is_irr:
            if water < -0.05:
                return (f"🚨 Urgent — water index {water} is low. "
                        f"Irrigate your {area_j} jereb field within {days}–{days+2} days. "
                        f"Apply 50–70mm using drip or furrow, early morning. "
                        f"Estimated cost: ~{cost:,} AFN.")
            return (f"Water is moderate (MNDWI={water}). "
                    f"Irrigate within 7–10 days. Annual rainfall: {rain}mm.")
        if is_crop or is_crop_q:
            if rain < 200 or water < -0.20:
                return (f"With {rain}mm/yr low water: "
                        f"1) Saffron — needs only 300mm, earns 50× more than wheat. "
                        f"2) Flax — drought tolerant. "
                        f"3) Chickpeas — very low water, improves soil nitrogen.")
            return (f"With {rain}mm rainfall: "
                    f"1) Wheat — reliable staple, plant Oct–Nov. "
                    f"2) Vegetables — 3× income with irrigation. "
                    f"3) Flax — better profit than wheat with less water.")
        if is_profit:
            base = "Saffron ≈ 15M AFN/ha potential" if rain < 200 else "Wheat ≈ 4,200 AFN/jereb typical"
            return (f"For your {area_j} jereb: {base}. "
                    f"Estimated input costs: ~{round(area_j*3200):,} AFN.")
        if is_fert:
            return (f"NDVI {ndvi} indicates fertilizer needed. "
                    f"Apply Urea: {fert}kg (30–40 kg/jereb) + DAP: {round(area_j*20)}kg. "
                    f"Mix into irrigation water. Effect visible in 2–3 weeks.")
        if is_rec:
            return (f"Your NDVI is {ndvi} — 2022 drought low was "
                    f"typically 0.10–0.22 across Afghanistan. "
                    f"Recovery needs consistent nitrogen + timely irrigation. "
                    f"NDVI can rise 20–30% in one season with proper inputs.")
        return (f"Your {area_j} jereb field — "
                f"NDVI {ndvi} ({'good' if ndvi>=0.30 else 'stressed'}), "
                f"water {water} ({'irrigate soon' if water<-0.05 else 'moderate'}), "
                f"rainfall {rain}mm/yr, province {province}. "
                f"What specific question do you have?")


# ════════════════════════════════════════════════════════════════════════════════
# GEE ANALYSIS
# Full multi-index field analysis using Google Earth Engine
# Called by /analyse endpoint when GEE credentials are available
# ════════════════════════════════════════════════════════════════════════════════
def gee_analyse(coords, year, clat, clon):
    """
    Run full satellite analysis on a field polygon via GEE.
    Computes: NDVI, EVI, SAVI, MNDWI, LSWI, NDRE, BSI, rainfall
    Also computes 7-year NDVI trend (2019-2025)
    Returns: dict of all indices + trend + metadata
    """
    import ee
    poly = ee.Geometry.Polygon([[[c[1], c[0]] for c in coords]])
    end_date = f"{year}-07-31" if year < 2025 else "2025-05-31"

    # Load Sentinel-2 composite (cloud-filtered median)
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(poly)
          .filterDate(f"{year}-04-01", end_date)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
          .sort("CLOUDY_PIXEL_PERCENTAGE")
          .limit(5)
          .median()
          .clip(poly))

    def mean(img, band):
        """Compute mean value of an image band over the polygon."""
        v = (img.reduceRegion(ee.Reducer.mean(), poly, 10, maxPixels=1e8)
               .get(band).getInfo())
        return round(float(v), 4) if v is not None else None

    # ── Vegetation indices ────────────────────────────────────────────────────
    # NDVI: standard vegetation index
    ndvi = mean(s2.normalizedDifference(["B8","B4"]).rename("nd"), "nd")

    # EVI: Enhanced Vegetation Index — reduces atmosphere and soil noise
    # EVI = 2.5 * (NIR-RED) / (NIR + 6*RED - 7.5*BLUE + 1)
    evi_img = s2.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {"NIR": s2.select("B8"), "RED": s2.select("B4"), "BLUE": s2.select("B2")}
    ).rename("evi")
    evi = mean(evi_img, "evi")

    # SAVI: Soil Adjusted Vegetation Index (L=0.5 for Afghan arid/semi-arid)
    # SAVI = ((NIR - RED) / (NIR + RED + L)) * (1 + L)
    savi_img = s2.expression(
        "((NIR - RED) / (NIR + RED + 0.5)) * 1.5",
        {"NIR": s2.select("B8"), "RED": s2.select("B4")}
    ).rename("savi")
    savi = mean(savi_img, "savi")

    # MNDWI: Modified Normalized Difference Water Index
    # MNDWI = (GREEN - SWIR1) / (GREEN + SWIR1)
    mndwi = mean(s2.normalizedDifference(["B3","B11"]).rename("nd"), "nd")

    # LSWI: Land Surface Water Index (sensitive to canopy + soil water)
    # LSWI = (NIR - SWIR1) / (NIR + SWIR1)
    lswi = mean(s2.normalizedDifference(["B8","B11"]).rename("nd"), "nd")

    # NDRE: Red Edge Normalized Difference — sensitive to chlorophyll
    # NDRE = (B8A - B5) / (B8A + B5)
    ndre = mean(s2.normalizedDifference(["B8A","B5"]).rename("nd"), "nd")

    # BSI: Bare Soil Index — detects bare/fallow land
    # BSI = ((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))
    bsi_img = s2.expression(
        "((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))",
        {"SWIR1": s2.select("B11"), "RED": s2.select("B4"),
         "NIR": s2.select("B8"),    "BLUE": s2.select("B2")}
    ).rename("bsi")
    bsi = mean(bsi_img, "bsi")

    # ── Rainfall (CHIRPS annual) ──────────────────────────────────────────────
    rain = mean(
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
          .filterBounds(poly)
          .filterDate(f"{year}-01-01", f"{year}-12-31")
          .select("precipitation")
          .sum()
          .clip(poly),
        "precipitation"
    )

    # ── 7-year NDVI trend ─────────────────────────────────────────────────────
    trend = {}
    for yr in range(2019, 2026):
        try:
            c2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(poly)
                  .filterDate(f"{yr}-05-01", f"{yr}-07-31")
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 25))
                  .median().clip(poly))
            v = (c2.normalizedDifference(["B8","B4"])
                   .reduceRegion(ee.Reducer.mean(), poly, 10, maxPixels=1e8)
                   .get("nd").getInfo())
            trend[yr] = round(float(v), 4) if v else None
        except:
            trend[yr] = None

    return {
        "ndvi": ndvi, "evi": evi, "savi": savi,
        "mndwi": mndwi, "water": mndwi, "lswi": lswi,
        "ndre": ndre, "bsi": bsi, "rain": rain,
        "trend": trend, "ndvi_trend": trend,
        "lat": round(clat, 5), "lon": round(clon, 5),
        "source": "gee_live",
        "image_date": f"{year}-05"
    }


# ════════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ════════════════════════════════════════════════════════════════════════════════

@app.route("/health")
def health():
    """Service health check — returns version, GEE status, AI status."""
    return jsonify({
        "status":  "ok",
        "version": "6.0",
        "gee":     gee_ok,
        "ai":      "gemini" if GEMINI_KEY else ("anthropic" if ANTHROPIC_KEY else "smart_only"),
        "indices": ["ndvi","evi","savi","mndwi","lswi","ndre","bsi"],
        "endpoints": ["/health","/analyse","/ask","/ndvi_tile","/crop_detect","/monthly_rain","/soil"]
    })


@app.route("/analyse", methods=["POST","OPTIONS"])
def analyse():
    """
    Full field analysis endpoint.
    Input:  {coords: [[lat,lon],...], year: int, label: str}
    Output: all satellite indices + trend + crop detection + season advice
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:
        data   = request.get_json(force=True)
        coords = data.get("coords", [])
        year   = int(data.get("year", datetime.now().year))
        label  = data.get("label", "Field")
        if len(coords) < 3:
            return jsonify({"error": "Need ≥3 coordinate points"}), 400

        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        clat = sum(lats) / len(lats)
        clon = sum(lons) / len(lons)
        area_ha    = calc_area_ha(coords)
        area_jereb = round(area_ha * 5, 1)
        month      = datetime.now().month

        # Try live GEE first
        if gee_ok:
            try:
                result = gee_analyse(coords, year, clat, clon)
                reg    = get_regional_data(clat, clon)  # for province name
                result.update({
                    "label": label, "area_ha": area_ha,
                    "area_jereb": area_jereb, "status": "success",
                    "province": reg["province"]
                })
                # Add crop detection + season advice
                result["crops"]  = detect_crop(
                    result["ndvi"], result["evi"], result["savi"],
                    result["mndwi"], result["lswi"], month, reg["province"])
                result["season"] = get_current_season_advice(
                    reg["province"], result["ndvi"], result["mndwi"])
                result["monthly_rain"] = get_monthly_rain(
                    result["rain"] or reg["rain"], reg["province"])
                result["soil"] = get_soil_data(clat, clon, reg["province"])
                # Compute VCI from 10-year trend if available
                if result.get("trend"):
                    tv = [v for v in result["trend"].values() if v]
                    if tv:
                        h_min = min(tv)
                        h_max = max(tv)
                        cur   = result["ndvi"] or 0
                        result["vci"] = round(
                            (cur - h_min) / (h_max - h_min + 0.001) * 100, 1
                        ) if h_max > h_min else None
                return jsonify(result)
            except Exception as e:
                log.error(f"GEE analysis failed: {e}")

        # Regional database fallback
        reg = get_regional_data(clat, clon)
        result = {
            "label": label, "status": "success",
            "source": reg["source"],
            "province": reg["province"],
            "ndvi":  reg["ndvi"],  "evi":   reg["evi"],
            "savi":  reg["savi"],  "mndwi": reg["mndwi"],
            "water": reg["mndwi"], "lswi":  reg["lswi"],
            "rain":  reg["rain"],
            "area_ha": area_ha, "area_jereb": area_jereb,
            "trend": reg["trend"], "ndvi_trend": reg["trend"],
            "year": year, "lat": round(clat,5), "lon": round(clon,5),
            "latest_date": f"{year}-05-15",
            "crops": detect_crop(
                reg["ndvi"], reg["evi"], reg["savi"],
                reg["mndwi"], reg["lswi"], month, reg["province"]),
            "season": get_current_season_advice(
                reg["province"], reg["ndvi"], reg["mndwi"]),
            "monthly_rain": get_monthly_rain(reg["rain"], reg["province"]),
        "soil": get_soil_data(clat, clon, reg["province"]),
        # Additional indices — estimated from NDVI when GEE unavailable
        "ndre":          round(reg["ndvi"] * 0.75, 4),
        "gndvi":         round(reg["ndvi"] * 0.88, 4),
        "ndmi":          round(reg["mndwi"] + 0.08, 4),
        "ndwi":          round(reg["mndwi"] + 0.05, 4),
        "vci":           None,  # VCI needs historical data — not available in regional DB
        "drought_index": round(reg["mndwi"] - reg["ndvi"], 4),
        }
        return jsonify(result)

    except Exception as e:
        log.error(f"/analyse error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST","OPTIONS"])
def ask():
    """
    AI question answering endpoint.
    Input:  {question, language, context, field_data}
    Output: {reply, model}
    Always returns an answer — uses smart fallback if AI APIs unavailable.
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:
        data     = request.get_json(force=True)
        question = data.get("question", "")
        language = data.get("language", "en")
        context  = data.get("context", "")
        fd       = data.get("field_data", {})
        if not question:
            return jsonify({"error": "No question provided"}), 400

        # Extract field parameters for smart fallback
        ndvi=0.28; water=-0.19; rain=240; area_j=5.0; province="Afghanistan"
        if isinstance(fd, dict) and fd:
            ndvi     = float(fd.get("ndvi", 0.28))
            water    = float(fd.get("mndwi", fd.get("water", -0.19)))
            rain     = float(fd.get("rain", 240))
            area_j   = float(fd.get("area_jereb", fd.get("area_ha", 1) * 5))
            province = fd.get("province", "Afghanistan")
            context  = (f"Field: NDVI={ndvi}, EVI={fd.get('evi','?')}, "
                       f"Water={water}, Rain={rain}mm, "
                       f"Area={fd.get('area_ha','?')}ha, Province={province}")
        elif context:
            # Parse values from context string
            import re
            for key, var in [("NDVI",None),("Water",None),("Rain",None)]:
                m = re.search(rf"{key}=([-\d.]+)", context)
                if m:
                    if key=="NDVI":   ndvi=float(m.group(1))
                    elif key=="Water": water=float(m.group(1))
                    elif key=="Rain":  rain=float(m.group(1))
            m = re.search(r"Area=([\d.]+)\s*jereb", context)
            if m: area_j = float(m.group(1))
        else:
            context = "No field data. Give general advice for Afghan smallholder farmers."

        lang_inst = {
            "fa": "Afghan Dari (دری). Use دهقان for farmer, جریب for land, تخم for seed. Use Eastern Arabic numerals ۱۲۳.",
            "ps": "Pashto (پښتو). Use proper Pashto farming terms. Use Eastern Arabic numerals.",
            "en": "English. Be concise and specific."
        }.get(language, "English.")

        prompt = (f"You are ZaminAI, an expert agricultural advisor for Afghan smallholder farmers.\n"
                  f"Satellite field data: {context}\n\n"
                  f"Respond ONLY in {lang_inst}\n"
                  f"Rules: exact amounts (kg/jereb, AFN, days). Under 90 words. "
                  f"Never say 'satellite' or 'AI'. Speak as a trusted local farming expert.\n\n"
                  f"Question: {question}")

        reply = call_gemini(prompt) or call_anthropic(prompt)

        # Smart fallback — always gives a useful answer
        if not reply or len(reply) < 8:
            log.warning("AI APIs returned empty — using smart fallback")
            reply = smart_fallback(question, ndvi, water, rain, area_j, language, province)
            model = "smart"
        else:
            model = "gemini" if GEMINI_KEY else "anthropic"

        return jsonify({"reply": reply, "answer": reply, "model": model})

    except Exception as e:
        log.error(f"/ask error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/crop_detect", methods=["POST","OPTIONS"])
def crop_detect():
    """
    Dedicated crop detection endpoint.
    Input:  {ndvi, evi, savi, mndwi, lswi, province, month (optional)}
    Output: {crops: [{crop, label_en, label_fa, label_ps, confidence, reason}]}
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:
        d        = request.get_json(force=True)
        ndvi     = float(d.get("ndvi", 0))
        evi      = float(d.get("evi",  0))
        savi     = float(d.get("savi", 0))
        mndwi    = float(d.get("mndwi", 0))
        lswi     = float(d.get("lswi", 0))
        province = d.get("province", "Afghanistan")
        month    = int(d.get("month", datetime.now().month))
        crops    = detect_crop(ndvi, evi, savi, mndwi, lswi, month, province)
        return jsonify({"status":"ok","crops":crops})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/monthly_rain", methods=["POST","OPTIONS"])
def monthly_rain():
    """
    Monthly rainfall breakdown endpoint.
    Input:  {annual_rain: float, province: str}
    Output: {months: [float x 12], labels: ['Jan'..'Dec']}
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:
        d        = request.get_json(force=True)
        annual   = float(d.get("annual_rain", 250))
        province = d.get("province", "Afghanistan")
        monthly  = get_monthly_rain(annual, province)
        return jsonify({
            "status":   "ok",
            "monthly":  monthly,
            "labels":   ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"],
            "province": province,
            "annual":   annual
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ndvi_tile", methods=["POST","OPTIONS"])
def ndvi_tile():
    """
    NDVI thumbnail image for mini-map display.
    Input:  {coords: [[lat,lon],...], year: int}
    Output: {status, tile_url}
    Requires GEE — returns error if not available.
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200
    if not gee_ok:
        return jsonify({"status": "error", "error": "GEE not available"}), 503
    try:
        import ee
        d      = request.get_json(force=True)
        coords = d.get("coords", [])
        year   = int(d.get("year", datetime.now().year))
        poly   = ee.Geometry.Polygon([[[c[1],c[0]] for c in coords]])
        ed     = f"{year}-07-31" if year < 2025 else "2025-05-31"
        col    = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(poly).filterDate(f"{year}-04-01", ed)
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
                  .median().clip(poly))
        url = col.normalizedDifference(["B8","B4"]).getThumbURL({
            "min": 0, "max": 0.7,
            "palette": ["#d73027","#fc8d59","#fee08b","#d9ef8b","#91cf60","#1a9850"],
            "dimensions": 512, "format": "png"
        })
        return jsonify({"status": "success", "tile_url": url})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500




@app.route("/soil", methods=["POST","OPTIONS"])
def soil():
    """
    Dedicated soil analysis endpoint.
    Input:  {lat: float, lon: float, province: str (optional)}
    Output: {ph, clay, sand, silt, soc, bulk_density,
             texture, recommendations, source, resolution}
    Calls SoilGrids API — free, no key needed.
    Falls back to Afghan provincial database if API unavailable.
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:
        d        = request.get_json(force=True)
        lat      = float(d.get("lat", 34.5))
        lon      = float(d.get("lon", 67.7))
        province = d.get("province", "Afghanistan")
        soil     = get_soil_data(lat, lon, province)
        soil["status"] = "ok"
        return jsonify(soil)
    except Exception as e:
        log.error(f"/soil error: {e}")
        return jsonify({"error": str(e)}), 500

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info(f"ZaminAI API v6.0 starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
