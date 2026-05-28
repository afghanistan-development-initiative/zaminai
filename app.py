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
SUPABASE_URL  = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY  = os.environ.get("SUPABASE_KEY", "")

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
    """
    Get all saved fields for a farmer.
    Returns: list of field dicts
    """
    if not sb_ok or not farmer_id:
        return []
    try:
        res = (sb.table("fields")
                 .select("*")
                 .eq("farmer_id", farmer_id)
                 .order("created_at", desc=True)
                 .execute())
        return res.data or []
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
        return res.data[0] if res.data else None
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
    except Exception as e:
        log.error(f"db_save_chat: {e}")


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
    ndvi  = round(max(0.08,min(0.45,0.06+lat*0.003+(lon-62)*0.002)),4)
    evi   = round(ndvi*0.72,4); savi=round(ndvi*0.85,4)
    rain  = max(80,min(480,int(lat*8)))
    mndwi = round(max(-0.38,min(0.05,-0.38+rain*0.001)),4)
    lswi  = round(mndwi+0.05,4)
    return {"province":"Afghanistan","ndvi":ndvi,"evi":evi,"savi":savi,
            "mndwi":mndwi,"lswi":lswi,"rain":rain,
            "trend":{2019:round(ndvi+0.07,4),2020:round(ndvi+0.05,4),
                     2021:round(ndvi+0.02,4),2022:round(ndvi-0.10,4),
                     2023:round(ndvi-0.04,4),2024:ndvi,2025:round(ndvi+0.02,4)},
            "source":"interpolated"}

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
    models=["gemini-1.5-flash","gemini-1.5-flash-latest","gemini-pro"]
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

def gee_analyse(coords,year,clat,clon):
    import ee
    poly=ee.Geometry.Polygon([[[c[1],c[0]] for c in coords]])
    end_date=f"{year}-07-31" if year<2025 else "2025-05-31"
    s2=(ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(poly).filterDate(f"{year}-04-01",end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",20))
        .sort("CLOUDY_PIXEL_PERCENTAGE").limit(5).median().clip(poly))
    def mean(img,band):
        v=(img.reduceRegion(ee.Reducer.mean(),poly,10,maxPixels=1e8).get(band).getInfo())
        return round(float(v),4) if v is not None else None
    ndvi=mean(s2.normalizedDifference(["B8","B4"]).rename("nd"),"nd")
    evi_img=s2.expression("2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))",
        {"NIR":s2.select("B8"),"RED":s2.select("B4"),"BLUE":s2.select("B2")}).rename("evi")
    evi=mean(evi_img,"evi")
    savi_img=s2.expression("((NIR-RED)/(NIR+RED+0.5))*1.5",
        {"NIR":s2.select("B8"),"RED":s2.select("B4")}).rename("savi")
    savi=mean(savi_img,"savi")
    mndwi=mean(s2.normalizedDifference(["B3","B11"]).rename("nd"),"nd")
    lswi=mean(s2.normalizedDifference(["B8","B11"]).rename("nd"),"nd")
    ndre=mean(s2.normalizedDifference(["B8A","B5"]).rename("nd"),"nd")
    bsi_img=s2.expression("((SWIR1+RED)-(NIR+BLUE))/((SWIR1+RED)+(NIR+BLUE))",
        {"SWIR1":s2.select("B11"),"RED":s2.select("B4"),"NIR":s2.select("B8"),"BLUE":s2.select("B2")}).rename("bsi")
    bsi=mean(bsi_img,"bsi")
    rain=mean(ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(poly)
              .filterDate(f"{year}-01-01",f"{year}-12-31").select("precipitation").sum().clip(poly),"precipitation")
    trend={}
    for yr in range(2019,2026):
        try:
            c2=(ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(poly)
                .filterDate(f"{yr}-05-01",f"{yr}-07-31")
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",25)).median().clip(poly))
            v=(c2.normalizedDifference(["B8","B4"])
               .reduceRegion(ee.Reducer.mean(),poly,10,maxPixels=1e8).get("nd").getInfo())
            trend[yr]=round(float(v),4) if v else None
        except: trend[yr]=None
    return {"ndvi":ndvi,"evi":evi,"savi":savi,"mndwi":mndwi,"water":mndwi,
            "lswi":lswi,"ndre":ndre,"bsi":bsi,"rain":rain,"trend":trend,
            "ndvi_trend":trend,"lat":round(clat,5),"lon":round(clon,5),
            "source":"gee_live","image_date":f"{year}-05"}


# ════════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ════════════════════════════════════════════════════════════════════════════════

@app.route("/health")
def health():
    return jsonify({
        "status":"ok","version":"7.0","gee":gee_ok,
        "database": sb_ok,
        "ai":"gemini" if GEMINI_KEY else "smart_only",
        "indices":["ndvi","evi","savi","mndwi","lswi","ndre","bsi"],
        "endpoints":["/health","/analyse","/ask","/ndvi_tile",
                     "/crop_detect","/monthly_rain","/soil",
                     "/db/farmer","/db/field/save","/db/fields/<id>",
                     "/db/analysis/save","/db/chat/save"]
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


# ── ANALYSIS ROUTE (unchanged logic, added db save) ───────────────────────────

@app.route("/analyse", methods=["POST","OPTIONS"])
def analyse():
    if request.method=="OPTIONS": return jsonify({}),200
    try:
        data      = request.get_json(force=True)
        coords    = data.get("coords",[])
        year      = int(data.get("year",datetime.now().year))
        label     = data.get("label","Field")
        farmer_id = data.get("farmer_id")   # optional — saves to DB if provided
        field_id  = data.get("field_id")    # optional
        if len(coords)<3:
            return jsonify({"error":"Need ≥3 coordinate points"}),400
        lats=[c[0] for c in coords]; lons=[c[1] for c in coords]
        clat=sum(lats)/len(lats); clon=sum(lons)/len(lons)
        area_ha=calc_area_ha(coords); area_jereb=round(area_ha*5,1)
        month=datetime.now().month
        if gee_ok:
            try:
                result=gee_analyse(coords,year,clat,clon)
                reg=get_regional_data(clat,clon)
                result.update({"label":label,"area_ha":area_ha,"area_jereb":area_jereb,
                               "status":"success","province":reg["province"]})
                result["crops"]=detect_crop(result["ndvi"],result["evi"],result["savi"],
                    result["mndwi"],result["lswi"],month,reg["province"])
                result["season"]=get_current_season_advice(reg["province"],result["ndvi"],result["mndwi"])
                result["monthly_rain"]=get_monthly_rain(result["rain"] or reg["rain"],reg["province"])
                result["soil"]=get_soil_data(clat,clon,reg["province"])
                if result.get("trend"):
                    tv=[v for v in result["trend"].values() if v]
                    if tv:
                        h_min=min(tv); h_max=max(tv); cur=result["ndvi"] or 0
                        result["vci"]=round((cur-h_min)/(h_max-h_min+0.001)*100,1) if h_max>h_min else None
                # Save to database if farmer_id provided
                if farmer_id and field_id:
                    db_save_analysis(field_id,farmer_id,result)
                return jsonify(result)
            except Exception as e:
                log.error(f"GEE failed: {e}")
        reg=get_regional_data(clat,clon)
        result={
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
            db_save_analysis(field_id,farmer_id,result)
        return jsonify(result)
    except Exception as e:
        log.error(f"/analyse: {e}"); return jsonify({"error":str(e)}),500


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
        prompt=(f"You are ZaminAI, expert agricultural advisor for Afghan smallholder farmers.\n"
                f"Satellite data: {context}\n\nRespond ONLY in {lang_inst}\n"
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


if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    log.info(f"ZaminAI API v7.0 starting on port {port}")
    app.run(host="0.0.0.0",port=port,debug=False)
