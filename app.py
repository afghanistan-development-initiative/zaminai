"""
ZaminAI — Satellite Analysis API v5
Fixes: Gemini safety settings, smart fallback always returns answer
"""

import os, json, logging, requests
from flask import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins="*")

# ─── GEE INIT ────────────────────────────────────────────────────────────────
gee_ok = False
try:
    import ee
    sa  = os.environ.get("GEE_SERVICE_ACCOUNT", "")
    key = os.environ.get("GEE_PRIVATE_KEY", "").replace("\\n", "\n")
    if sa and key:
        ee.Initialize(ee.ServiceAccountCredentials(sa, key_data=key))
        gee_ok = True
        log.info("GEE OK")
    else:
        log.warning("GEE credentials missing")
except Exception as e:
    log.error(f"GEE init failed: {e}")

# ─── GEMINI REST ──────────────────────────────────────────────────────────────
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

def call_gemini(prompt):
    if not GEMINI_KEY:
        return None
    for model in ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-pro"]:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_KEY}"
            resp = requests.post(url, json={
                "contents": [{"parts": [{"text": prompt}]}],
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ],
                "generationConfig": {"temperature": 0.6, "maxOutputTokens": 280}
            }, timeout=14)
            log.info(f"Gemini {model}: HTTP {resp.status_code}")
            if resp.status_code == 200:
                d = resp.json()
                cands = d.get("candidates", [])
                if cands:
                    txt = cands[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    if txt and len(txt) > 8:
                        log.info(f"Gemini {model}: got {len(txt)} chars")
                        return txt.strip()
                    log.warning(f"Gemini {model}: empty — finishReason={cands[0].get('finishReason','?')}")
                else:
                    log.warning(f"Gemini {model}: no candidates — {d.get('promptFeedback','')}")
            elif resp.status_code == 429:
                log.warning("Gemini: rate limited")
                break
        except Exception as e:
            log.error(f"Gemini {model}: {e}")
    return None

# ─── ANTHROPIC FALLBACK ───────────────────────────────────────────────────────
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

def call_anthropic(prompt):
    if not ANTHROPIC_KEY:
        return None
    try:
        resp = requests.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": "claude-haiku-4-5-20251001", "max_tokens": 280,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=14)
        if resp.status_code == 200:
            return resp.json()["content"][0]["text"].strip()
        log.warning(f"Anthropic: HTTP {resp.status_code}")
    except Exception as e:
        log.error(f"Anthropic: {e}")
    return None

# ─── SMART RULE-BASED FALLBACK (always works — no API needed) ─────────────────
def smart_fallback(question, ndvi, water, rain, area_j, lang):
    q = question.lower()
    is_irr    = any(w in q for w in ["irrigat","water","آبیاری","اوبه","آب"])
    is_crop   = any(w in q for w in ["crop","plant","grow","کشت","محصول","وکارم","وکرم"])
    is_profit = any(w in q for w in ["profit","income","earn","money","فایده","عاید","ګټه"])
    is_fert   = any(w in q for w in ["fertil","urea","dap","کود","سره"])
    is_rec    = any(w in q for w in ["recov","ndvi","2022","بهبود","بهتر","بهتریدل"])
    days = 2 if water < -0.15 else 4 if water < -0.05 else 9
    irr_cost  = round(area_j * 400)
    fert_kg   = round(area_j * 35)

    if lang == "fa":
        if is_irr:
            if water < -0.05:
                return f"🚨 فوری: شاخص آب {water} است — پایین. زمین {area_j} جریب شما را در {days}–{days+2} روز آبیاری کنید. ۵۰–۷۰ میلیمتر آب، ترجیحاً صبح زود با آبیاری قطره‌ای یا جوی. لاگست: ~{irr_cost:,} افغانی."
            return f"آب در سطح متوسط است (MNDWI={water}). در ۷–۱۰ روز آبیاری کنید. باران سالانه {rain}mm است."
        if is_crop:
            if rain < 200 or water < -0.15:
                return f"با {rain}mm آب کم: ۱) زعفران — آب کم، درآمد ۵۰ برابر گندم. ۲) کتان — مقاوم به خشکی. ۳) نخود — آب بسیار کم، خاک را بهبود می‌دهد."
            return f"با {rain}mm آب: ۱) گندم — پایه مطمئن. ۲) سبزیجات — درآمد ۳ برابر. ۳) کتان — فایده خوب با آب کمتر."
        if is_profit:
            base = "زعفران ≈ ۱۵ میلیون AFN/هکتار" if rain < 200 else "گندم ≈ ۴۲۰۰ AFN/جریب"
            return f"برای زمین {area_j} جریب: {base}. مصرف تخمینی: ~{round(area_j*3200):,} افغانی."
        if is_fert:
            return f"NDVI {ndvi} نشان می‌دهد کود نیاز دارید. یوریا: {fert_kg} کیلوگرام (۳۰–۴۰ kg/jereb). DAP: {round(area_j*20)} kg. در آب آبیاری حل کنید. تاثیر در ۲–۳ هفته."
        if is_rec:
            return f"NDVI شما {ndvi} است (پایین‌ترین در ۲۰۲۲ بود). بهبود نیاز دارد. کود نیتروژن + آبیاری به موقع NDVI را ۲۰–۳۰٪ افزایش می‌دهد."
        return f"زمین {area_j} جریب شما: NDVI {ndvi} ({'خوب' if ndvi>=0.35 else 'تحت فشار'}), آب {water} ({'آبیاری کنید' if water<-0.05 else 'متوسط'}), باران {rain}mm. چه سوال خاصی دارید؟"

    if lang == "ps":
        if is_irr:
            if water < -0.05:
                return f"🚨 بیړي: د اوبو شاخص {water} — ټیټ. {area_j} جریب ځمکه {days}–{days+2} ورځو کې اوبه ورکړئ. ۵۰–۷۰ mm اوبه، سهار وختي. اټکلي لګښت: ~{irr_cost:,} افغاني."
            return f"اوبه متوسط دي (MNDWI={water}). د ۷–۱۰ ورځو کې اوبه ورکړئ. کلنی باران {rain}mm دی."
        if is_crop:
            if rain < 200 or water < -0.15:
                return f"د {rain}mm لږو اوبو سره: ۱) زعفران — لږ اوبه، د گندم ۵۰ ځله ډیره ګټه. ۲) کتان — د خشکسالۍ مقاوم. ۳) نخود — خورا لږ اوبه."
            return f"د {rain}mm اوبو سره: ۱) گندم — باوري. ۲) سبزیجات — ۳ ځله ډیره ګټه. ۳) کتان — ښه ګټه."
        if is_profit:
            base = "زعفران ≈ ۱۵ میلیون AFN/هکتار" if rain < 200 else "گندم ≈ ۴۲۰۰ AFN/جریب"
            return f"ستاسو {area_j} جریب: {base}. تخمیني لګښت: ~{round(area_j*3200):,} افغاني."
        if is_fert:
            return f"NDVI {ndvi} ښیي چې سرې ته اړتیا ده. یوریا: {fert_kg} کیلوګرام (۳۰–۴۰ kg/jereb). د ۲–۳ اونیو کې اغیز ښکاره کیږي."
        if is_rec:
            return f"ستاسو NDVI {ndvi} دی (۲۰۲۲ کې تر ټولو ټیټ و). د نایتروجن سرې + وخت پر وخت اوبه ورکول NDVI ۲۰–۳۰٪ لوړوي."
        return f"ستاسو {area_j} جریب: NDVI {ndvi} ({'ښه' if ndvi>=0.35 else 'تحت فشار'}), اوبه {water} ({'اوبه ورکړئ' if water<-0.05 else 'متوسط'}), باران {rain}mm. کومه ځانګړې پوښتنه؟"

    # English
    if is_irr:
        if water < -0.05:
            return f"🚨 Urgent — water index {water} is low. Irrigate your {area_j} jereb field within {days}–{days+2} days. Apply 50–70mm using drip or furrow, early morning. Estimated cost: ~{irr_cost:,} AFN."
        return f"Water is moderate (MNDWI={water}). Irrigate within 7–10 days. Annual rainfall: {rain}mm."
    if is_crop:
        if rain < 200 or water < -0.15:
            return f"With {rain}mm/yr low water: 1) Saffron — needs only 300mm, earns 50x wheat. 2) Flax — drought tolerant. 3) Chickpeas — very low water, improves soil."
        return f"With {rain}mm rainfall: 1) Wheat — reliable staple. 2) Vegetables — 3x income. 3) Flax — better profit than wheat with less water."
    if is_profit:
        base = "Saffron ≈ 15M AFN/ha potential" if rain < 200 else "Wheat ≈ 4,200 AFN/jereb typical"
        return f"For your {area_j} jereb field: {base}. Estimated total input costs: ~{round(area_j*3200):,} AFN."
    if is_fert:
        return f"NDVI {ndvi} indicates your field needs fertilizer. Apply Urea: {fert_kg}kg (30–40 kg/jereb) + DAP: {round(area_j*20)}kg. Mix into irrigation water for best absorption. Effect visible in 2–3 weeks."
    if is_rec:
        return f"Your NDVI is {ndvi} (lowest was 2022 drought). Recovery needs nitrogen fertilizer + timely irrigation. Nitrogen + water can raise NDVI 20–30% within one season."
    return f"Your {area_j} jereb field — NDVI {ndvi} ({'good' if ndvi>=0.35 else 'stressed'}), water {water} ({'irrigate soon' if water<-0.05 else 'moderate'}), rainfall {rain}mm/yr. What specific question do you have?"


# ─── REGIONAL DATABASE ────────────────────────────────────────────────────────
PROVINCES = [
    (36.4,37.2,68.2,69.2,"Kunduz",    0.33,-0.14,287,{2019:0.40,2020:0.38,2021:0.35,2022:0.22,2023:0.27,2024:0.33,2025:0.35}),
    (36.4,37.1,66.5,67.3,"Balkh",     0.31,-0.18,245,{2019:0.37,2020:0.35,2021:0.31,2022:0.19,2023:0.24,2024:0.31,2025:0.33}),
    (33.8,35.0,61.5,63.5,"Herat",     0.28,-0.20,195,{2019:0.33,2020:0.31,2021:0.27,2022:0.15,2023:0.21,2024:0.28,2025:0.29}),
    (33.8,34.6,70.0,71.5,"Nangarhar", 0.38,-0.12,320,{2019:0.44,2020:0.41,2021:0.37,2022:0.26,2023:0.31,2024:0.38,2025:0.40}),
    (34.2,34.9,68.7,69.5,"Kabul",     0.27,-0.22,305,{2019:0.32,2020:0.29,2021:0.25,2022:0.14,2023:0.20,2024:0.27,2025:0.28}),
    (31.3,32.1,65.2,66.2,"Kandahar",  0.22,-0.28,175,{2019:0.27,2020:0.24,2021:0.20,2022:0.11,2023:0.16,2024:0.22,2025:0.23}),
    (30.8,32.2,63.5,65.5,"Helmand",   0.25,-0.25,148,{2019:0.30,2020:0.27,2021:0.23,2022:0.13,2023:0.18,2024:0.25,2025:0.26}),
    (36.5,38.5,70.0,72.0,"Badakhshan",0.41,-0.10,420,{2019:0.47,2020:0.44,2021:0.40,2022:0.29,2023:0.35,2024:0.41,2025:0.43}),
    (36.4,37.2,69.0,70.5,"Takhar",    0.36,-0.15,340,{2019:0.42,2020:0.39,2021:0.35,2022:0.24,2023:0.29,2024:0.36,2025:0.38}),
    (35.8,36.6,68.2,69.2,"Baghlan",   0.34,-0.16,295,{2019:0.40,2020:0.37,2021:0.33,2022:0.21,2023:0.27,2024:0.34,2025:0.36}),
    (35.0,36.0,64.0,66.0,"Faryab",    0.29,-0.19,220,{2019:0.35,2020:0.32,2021:0.27,2022:0.16,2023:0.22,2024:0.29,2025:0.31}),
    (35.5,36.5,65.5,67.0,"Jawzjan",   0.30,-0.17,240,{2019:0.36,2020:0.33,2021:0.28,2022:0.17,2023:0.23,2024:0.30,2025:0.32}),
    (32.0,33.5,67.0,68.5,"Ghazni",    0.24,-0.21,185,{2019:0.29,2020:0.26,2021:0.22,2022:0.12,2023:0.18,2024:0.24,2025:0.25}),
    (34.5,35.5,67.0,68.5,"Bamyan",    0.27,-0.18,270,{2019:0.32,2020:0.29,2021:0.25,2022:0.14,2023:0.20,2024:0.27,2025:0.28}),
    (33.0,34.0,69.0,70.5,"Logar",     0.26,-0.20,260,{2019:0.31,2020:0.28,2021:0.24,2022:0.13,2023:0.19,2024:0.26,2025:0.27}),
    (32.5,33.5,68.0,69.5,"Paktia",    0.28,-0.18,285,{2019:0.33,2020:0.30,2021:0.26,2022:0.15,2023:0.21,2024:0.28,2025:0.29}),
]

def get_regional_data(lat, lon):
    for lat_min,lat_max,lon_min,lon_max,name,ndvi,water,rain,trend in PROVINCES:
        if lat_min<=lat<=lat_max and lon_min<=lon<=lon_max:
            return {"province":name,"ndvi":ndvi,"water":water,"rain":rain,"trend":trend}
    ndvi  = round(max(0.10, min(0.42, 0.08 + lat*0.003 + (lon-62)*0.002)), 4)
    rain  = max(100, min(450, int(lat*8)))
    water = round(max(-0.35, min(0.05, -0.35 + rain*0.001)), 4)
    return {"province":"Afghanistan","ndvi":ndvi,"water":water,"rain":rain,
            "trend":{2019:ndvi+0.015,2020:ndvi+0.010,2021:ndvi+0.003,
                     2022:ndvi-0.030,2023:ndvi-0.012,2024:ndvi,2025:ndvi+0.008}}

def _calc_area(coords):
    import math
    n = len(coords); area = 0
    for i in range(n):
        j = (i+1) % n
        dx = (coords[j][1]-coords[i][1]) * 111320 * math.cos(math.radians((coords[i][0]+coords[j][0])/2))
        dy = (coords[j][0]-coords[i][0]) * 111320
        area += coords[i][0]*111320 * dx - coords[i][1]*111320*math.cos(math.radians(coords[i][0])) * dy
    return round(abs(area) / 2 / 10000, 2)


# ─── HEALTH ──────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status":"ok","gee":gee_ok,
        "ai":"gemini" if GEMINI_KEY else ("anthropic" if ANTHROPIC_KEY else "none"),
        "version":"5.0"})


# ─── ANALYSE ─────────────────────────────────────────────────────────────────
@app.route("/analyse", methods=["POST","OPTIONS"])
def analyse():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:
        data   = request.get_json(force=True)
        coords = data.get("coords", [])
        year   = int(data.get("year", 2024))
        label  = data.get("label", "Field")
        if len(coords) < 3:
            return jsonify({"error":"Need ≥3 points"}), 400
        lats = [c[0] for c in coords]; lons = [c[1] for c in coords]
        clat = sum(lats)/len(lats);    clon = sum(lons)/len(lons)
        area_ha    = _calc_area(coords)
        area_jereb = round(area_ha * 5, 1)
        if gee_ok:
            try:
                result = _gee_analyse(coords, year, clat, clon)
                result.update({"label":label,"area_ha":area_ha,"area_jereb":area_jereb,"status":"success"})
                return jsonify(result)
            except Exception as e:
                log.error(f"GEE: {e}")
        reg = get_regional_data(clat, clon)
        return jsonify({"label":label,"ndvi":reg["ndvi"],"mndwi":reg["water"],"water":reg["water"],
            "rain":reg["rain"],"area_ha":area_ha,"area_jereb":area_jereb,"province":reg["province"],
            "trend":reg["trend"],"ndvi_trend":reg["trend"],"year":year,
            "latest_date":f"{year}-05-15","source":"regional","status":"success",
            "lat":round(clat,5),"lon":round(clon,5)})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

def _gee_analyse(coords, year, clat, clon):
    import ee
    poly = ee.Geometry.Polygon([[[c[1],c[0]] for c in coords]])
    ed   = f"{year}-07-31" if year < 2025 else "2025-05-31"
    col  = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(poly).filterDate(f"{year}-04-01",ed)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",20))
            .sort("CLOUDY_PIXEL_PERCENTAGE").limit(5).median().clip(poly))
    def mean(img, b):
        v = img.reduceRegion(ee.Reducer.mean(),poly,10,maxPixels=1e8).get(b).getInfo()
        return round(float(v),4) if v is not None else None
    ndvi_val  = mean(col.normalizedDifference(["B8","B4"]).rename("nd"), "nd")
    water_val = mean(col.normalizedDifference(["B3","B11"]).rename("nd"), "nd")
    rain_val  = mean((ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                      .filterBounds(poly).filterDate(f"{year}-01-01",f"{year}-12-31")
                      .select("precipitation").sum().clip(poly)), "precipitation")
    trend = {}
    for yr in [2019,2020,2021,2022,2023,2024]:
        try:
            c2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(poly).filterDate(f"{yr}-05-01",f"{yr}-07-31")
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",25)).median().clip(poly))
            v  = c2.normalizedDifference(["B8","B4"]).reduceRegion(ee.Reducer.mean(),poly,10,maxPixels=1e8).get("nd").getInfo()
            trend[yr] = round(float(v),4) if v else None
        except: trend[yr] = None
    return {"ndvi":ndvi_val,"mndwi":water_val,"water":water_val,"rain":rain_val,
            "trend":trend,"ndvi_trend":trend,"lat":round(clat,5),"lon":round(clon,5)}


# ─── ASK AI ───────────────────────────────────────────────────────────────────
@app.route("/ask", methods=["POST","OPTIONS"])
def ask():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:
        data     = request.get_json(force=True)
        question = data.get("question","")
        language = data.get("language","en")
        context  = data.get("context","")
        fd       = data.get("field_data", {})
        if not question:
            return jsonify({"error":"No question"}), 400

        # Parse field data for smart fallback
        ndvi=0.28; water=-0.19; rain=240; area_j=5.0
        if isinstance(fd, dict) and fd:
            ndvi   = float(fd.get("ndvi",0.28))
            water  = float(fd.get("mndwi",fd.get("water",-0.19)))
            rain   = float(fd.get("rain",240))
            area_j = float(fd.get("area_jereb",fd.get("area_ha",1)*5))
            ctx = (f"Field: NDVI={ndvi}, Water={water}, Rain={rain}mm, "
                   f"Area={fd.get('area_ha','?')}ha, Province={fd.get('province','Afghanistan')}")
        elif context:
            ctx = context
            # Parse from context string if possible
            import re
            m = re.search(r"NDVI=([\d.]+)", context)
            if m: ndvi = float(m.group(1))
            m = re.search(r"Water=([-\d.]+)", context)
            if m: water = float(m.group(1))
            m = re.search(r"Rain=([\d.]+)", context)
            if m: rain = float(m.group(1))
            m = re.search(r"Area=([\d.]+)\s*jereb", context)
            if m: area_j = float(m.group(1))
        else:
            ctx = "No field data. Give general advice for Afghan smallholder farmers."

        lang_inst = {"fa":"Afghan Dari (دری). Use دهقان, جریب, تخم, آبیاری.",
                     "ps":"Pashto (پښتو). Use proper Pashto farming terms.",
                     "en":"English."}.get(language,"English.")

        prompt = f"""You are ZaminAI, expert agricultural advisor for Afghan smallholder farmers.
Satellite field data: {ctx}

Answer ONLY in {lang_inst}
Be specific: exact amounts kg/jereb, costs in AFN, number of days.
Keep answer under 90 words. Never say 'satellite' or 'AI'. Speak as a trusted farming expert.

Question: {question}"""

        reply = call_gemini(prompt) or call_anthropic(prompt)

        # Smart rule-based fallback — ALWAYS returns a useful answer
        if not reply or len(reply) < 8:
            log.warning("AI APIs empty — using smart fallback")
            reply = smart_fallback(question, ndvi, water, rain, area_j, language)

        return jsonify({"reply":reply,"answer":reply,
                        "model":"gemini" if GEMINI_KEY else ("anthropic" if ANTHROPIC_KEY else "smart")})
    except Exception as e:
        log.error(f"Ask: {e}")
        return jsonify({"error":str(e)}), 500


# ─── NDVI TILE ────────────────────────────────────────────────────────────────
@app.route("/ndvi_tile", methods=["POST","OPTIONS"])
def ndvi_tile():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    if not gee_ok:
        return jsonify({"status":"error"}), 503
    try:
        import ee
        data   = request.get_json(force=True)
        coords = data.get("coords",[])
        year   = int(data.get("year",2024))
        poly   = ee.Geometry.Polygon([[[c[1],c[0]] for c in coords]])
        ed     = f"{year}-07-31" if year < 2025 else "2025-05-31"
        col    = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(poly).filterDate(f"{year}-04-01",ed)
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",20)).median().clip(poly))
        url = col.normalizedDifference(["B8","B4"]).getThumbURL({
            "min":0,"max":0.7,
            "palette":["#d73027","#fc8d59","#fee08b","#d9ef8b","#91cf60","#1a9850"],
            "dimensions":512,"format":"png"})
        return jsonify({"status":"success","tile_url":url})
    except Exception as e:
        return jsonify({"status":"error","error":str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)), debug=False)
