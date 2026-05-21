"""
ZaminAI - GEE Satellite + AI API
Credentials loaded from environment variables ONLY - never from code
"""
import os, ee, json, datetime, requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["*"])

# ── GEE INIT ──────────────────────────────────────────────────────────────────
def init_gee():
    try:
        sa  = os.environ.get("GEE_SERVICE_ACCOUNT","")
        key = os.environ.get("GEE_PRIVATE_KEY","").replace("\\n","\n")
        if not sa or not key:
            print("✗ GEE credentials missing"); return False
        creds = ee.ServiceAccountCredentials(sa, key_data=key)
        ee.Initialize(creds)
        print("✓ GEE connected"); return True
    except Exception as e:
        print(f"✗ GEE: {e}"); return False

GEE_OK = init_gee()
GEMINI_KEY    = os.environ.get("GEMINI_API_KEY","")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY","")

# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_mean(img, band, region, scale=100):
    try:
        v = img.select(band).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region, scale=scale, maxPixels=1e9
        ).getInfo()
        return round(float((v or {}).get(band) or 0), 4)
    except: return 0.0

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({"name":"ZaminAI API","status":"ok","gee":GEE_OK,
                    "ai":"gemini" if GEMINI_KEY else "anthropic" if ANTHROPIC_KEY else "none"})

@app.route("/health")
def health():
    return jsonify({"status":"ok","gee":GEE_OK,
                    "ai":"gemini" if GEMINI_KEY else "anthropic" if ANTHROPIC_KEY else "none"})

@app.route("/analyse", methods=["POST","OPTIONS"])
def analyse():
    if request.method=="OPTIONS": return jsonify({}), 200
    if not GEE_OK: return jsonify({"status":"error","message":"GEE not connected"}), 503
    try:
        data   = request.get_json(force=True)
        coords = data.get("coords",[])
        year   = int(data.get("year",2024))
        if len(coords)<3: return jsonify({"status":"error","message":"Need 3+ coords"}), 400

        gee_coords = [[float(c[1]),float(c[0])] for c in coords]
        region = ee.Geometry.Polygon([gee_coords])

        s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(region)
              .filterDate(f"{year}-04-01",f"{year}-08-31")
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",20))
              .median().clip(region))

        ndvi  = s2.normalizedDifference(["B8","B4"]).rename("NDVI")
        mndwi = s2.normalizedDifference(["B3","B11"]).rename("MNDWI")

        ndvi_val  = get_mean(ndvi,"NDVI",region,100)
        mndwi_val = get_mean(mndwi,"MNDWI",region,100)

        rain = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                .filterBounds(region)
                .filterDate(f"{year}-01-01",f"{year}-12-31")
                .select("precipitation").sum().clip(region))
        rain_val = get_mean(rain,"precipitation",region,5000)

        area_ha = round(region.area().getInfo()/10000,2)
        cen     = region.centroid().coordinates().getInfo()

        trend = {}
        for yr in [2019,2021,2022,2023,year]:
            try:
                s2yr = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                        .filterBounds(region)
                        .filterDate(f"{yr}-05-01",f"{yr}-07-31")
                        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",25))
                        .median().clip(region))
                trend[yr] = get_mean(s2yr.normalizedDifference(["B8","B4"]),"nd",region,100)
            except: trend[yr]=0.0

        return jsonify({"status":"success","ndvi":ndvi_val,"mndwi":mndwi_val,
                        "rain_mm":round(rain_val,1),"area_ha":area_ha,
                        "area_jereb":round(area_ha*5,1),"lat":round(cen[1],5),
                        "lon":round(cen[0],5),"ndvi_trend":trend,"year":year,
                        "image_date":f"{year}-07-01"})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500

@app.route("/ask", methods=["POST","OPTIONS"])
def ask_ai():
    """AI chat endpoint - keys stay safe on server"""
    if request.method=="OPTIONS": return jsonify({}), 200
    try:
        data     = request.get_json(force=True)
        question = data.get("question","")
        context  = data.get("context","")
        language = data.get("language","en")

        lang_instruction = ""
        if language=="fa": lang_instruction = "Respond in Dari (Afghan Dari, use دهقان not کشاورز)."
        elif language=="ps": lang_instruction = "Respond in Pashto."

        prompt = f"""You are ZaminAI, expert agricultural AI for Afghan smallholder farmers.
{context}
{lang_instruction}
Give specific practical advice. Use exact amounts (kg/jereb, AFN costs).
Keep response under 150 words. Use bullet points for action steps.

Farmer question: {question}"""

        reply = ""

        # Try Gemini first (free)
        if GEMINI_KEY:
            try:
                resp = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_KEY}",
                    json={"contents":[{"parts":[{"text":prompt}]}],
                          "generationConfig":{"maxOutputTokens":300}},
                    timeout=15
                )
                d = resp.json()
                reply = d.get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text","")
            except: pass

        # Fallback to Anthropic
        if not reply and ANTHROPIC_KEY:
            try:
                resp = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key":ANTHROPIC_KEY,"anthropic-version":"2023-06-01","content-type":"application/json"},
                    json={"model":"claude-haiku-4-5-20251001","max_tokens":300,
                          "messages":[{"role":"user","content":prompt}]},
                    timeout=15
                )
                d = resp.json()
                reply = d.get("content",[{}])[0].get("text","")
            except: pass

        if not reply:
            reply = "AI not connected. Please add GEMINI_API_KEY to Render environment variables."

        return jsonify({"status":"success","reply":reply})

    except Exception as e:
        return jsonify({"status":"error","reply":str(e)}), 500

if __name__=="__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port)
