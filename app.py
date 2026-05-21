"""
ZaminAI - GEE Satellite + AI API
Always returns LATEST available satellite data
Credentials from environment variables only
"""
import os, ee, json, requests
import datetime as dt
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["*"])

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

GEE_OK        = init_gee()
GEMINI_KEY    = os.environ.get("GEMINI_API_KEY","")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY","")

def get_mean(img, band, region, scale=100):
    try:
        v = img.select(band).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region, scale=scale, maxPixels=1e9
        ).getInfo()
        return round(float((v or {}).get(band) or 0), 4)
    except: return 0.0

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
        if len(coords)<3: return jsonify({"status":"error","message":"Need 3+ coords"}), 400

        gee_coords = [[float(c[1]),float(c[0])] for c in coords]
        region = ee.Geometry.Polygon([gee_coords])

        today     = dt.datetime.now()
        cur_year  = today.year
        end_date  = today.strftime("%Y-%m-%d")
        start_90  = (today - dt.timedelta(days=90)).strftime("%Y-%m-%d")

        # ── Get LATEST available image (last 90 days) ──
        latest_col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                      .filterBounds(region)
                      .filterDate(start_90, end_date)
                      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
                      .sort("system:time_start", False))

        col_size = latest_col.size().getInfo()

        if col_size > 0:
            s2 = latest_col.median().clip(region)
            img_ms = ee.Image(latest_col.first()).date().millis().getInfo()
            image_date = dt.datetime.fromtimestamp(img_ms/1000).strftime("%Y-%m-%d")
            data_year  = int(image_date[:4])
        else:
            # Fallback: current year
            s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(region)
                  .filterDate(f"{cur_year}-01-01", end_date)
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 25))
                  .median().clip(region))
            image_date = end_date
            data_year  = cur_year

        ndvi  = s2.normalizedDifference(["B8","B4"]).rename("NDVI")
        mndwi = s2.normalizedDifference(["B3","B11"]).rename("MNDWI")

        ndvi_val  = get_mean(ndvi,  "NDVI",  region, 100)
        mndwi_val = get_mean(mndwi, "MNDWI", region, 100)

        # Rainfall current year to today
        rain = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                .filterBounds(region)
                .filterDate(f"{cur_year}-01-01", end_date)
                .select("precipitation").sum().clip(region))
        rain_val = get_mean(rain, "precipitation", region, 5000)

        area_ha = round(region.area().getInfo()/10000, 2)
        cen     = region.centroid().coordinates().getInfo()

        # NDVI trend last 6 years
        trend = {}
        for yr in range(cur_year-5, cur_year+1):
            try:
                yr_end = min(today, dt.datetime(yr,12,31))
                s2yr = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                        .filterBounds(region)
                        .filterDate(f"{yr}-01-01", yr_end.strftime("%Y-%m-%d"))
                        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",25))
                        .median().clip(region))
                trend[yr] = get_mean(s2yr.normalizedDifference(["B8","B4"]),"nd",region,100)
            except: trend[yr]=0.0

        return jsonify({
            "status":     "success",
            "ndvi":       ndvi_val,
            "mndwi":      mndwi_val,
            "rain_mm":    round(rain_val,1),
            "area_ha":    area_ha,
            "area_jereb": round(area_ha*5,1),
            "lat":        round(cen[1],5),
            "lon":        round(cen[0],5),
            "ndvi_trend": trend,
            "year":       data_year,
            "image_date": image_date,
            "latest":     col_size > 0,
        })

    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500

@app.route("/ask", methods=["POST","OPTIONS"])
def ask_ai():
    if request.method=="OPTIONS": return jsonify({}), 200
    try:
        data     = request.get_json(force=True)
        question = data.get("question","")
        context  = data.get("context","")
        language = data.get("language","en")

        lang_instr = ""
        if language=="fa": lang_instr="Respond in Dari (Afghan Dari). Use دهقان not کشاورز, use جریب for land size."
        elif language=="ps": lang_instr="Respond in Pashto. Use دهقان, use جریب for land size."

        prompt = f"""You are ZaminAI, expert agricultural AI for Afghan smallholder farmers.
{context}
{lang_instr}
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
                    timeout=15)
                d = resp.json()
                reply = d.get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text","")
            except Exception as e:
                print(f"Gemini error: {e}")

        # Fallback Anthropic
        if not reply and ANTHROPIC_KEY:
            try:
                resp = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key":ANTHROPIC_KEY,
                             "anthropic-version":"2023-06-01",
                             "content-type":"application/json"},
                    json={"model":"claude-haiku-4-5-20251001","max_tokens":300,
                          "messages":[{"role":"user","content":prompt}]},
                    timeout=15)
                d = resp.json()
                reply = d.get("content",[{}])[0].get("text","")
            except Exception as e:
                print(f"Anthropic error: {e}")

        if not reply:
            reply = "AI not connected. Please add GEMINI_API_KEY to Render environment variables."

        return jsonify({"status":"success","reply":reply})

    except Exception as e:
        return jsonify({"status":"error","reply":str(e)}), 500

if __name__=="__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port)
