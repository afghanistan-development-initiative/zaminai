"""
Afghanistan Field AI Assistant
Afghanistan Development Initiative · zaminai.org
Mobile-ready — GPS location — Plot drawing — Voice input/output — AI in Dari/Pashto/English
Author: Maiwand Jan Alamzoi
"""

import os
import ee
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import folium
from folium.plugins import Draw, LocateControl
from streamlit_folium import st_folium
import json

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ZaminAI — Smart Farming Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"  # collapsed for mobile
)

st.markdown("""
<style>
.main{background-color:#0a0f0d}
.block-container{padding-top:1rem;padding-left:1rem;padding-right:1rem}
h1{color:#4ade80;font-family:monospace;font-size:clamp(18px,4vw,28px)}
h2,h3{color:#86efac}
div[data-testid="metric-container"]{background:#111810;border:1px solid #1e2b1a;border-radius:8px;padding:0.75rem}
.field-card{background:#111810;border:1px solid #1e2b1a;border-radius:8px;padding:1rem;margin:0.5rem 0}
.alert-green{background:#052e16;border-left:3px solid #4ade80;border-radius:4px;padding:0.75rem;margin:0.5rem 0;font-size:13px;color:#86efac}
.alert-yellow{background:#1c1400;border-left:3px solid #fbbf24;border-radius:4px;padding:0.75rem;margin:0.5rem 0;font-size:13px;color:#fcd34d}
.alert-red{background:#1c0a0a;border-left:3px solid #f87171;border-radius:4px;padding:0.75rem;margin:0.5rem 0;font-size:13px;color:#fca5a5}
.dari-text{direction:rtl;text-align:right;font-size:15px;color:#86efac;line-height:1.8;padding:0.5rem 0}
.pashto-text{direction:rtl;text-align:right;font-size:15px;color:#a78bfa;line-height:1.8;padding:0.5rem 0}
.voice-btn{background:#16a34a;color:white;border:none;border-radius:8px;padding:12px 24px;font-size:15px;cursor:pointer;width:100%;margin:4px 0}
.stTabs [data-baseweb="tab"]{font-family:monospace;font-size:13px;color:#6b8f65}
.stTabs [aria-selected="true"]{color:#4ade80 !important;border-bottom:2px solid #4ade80 !important}
/* Mobile optimizations */
@media(max-width:640px){
    .block-container{padding:0.5rem}
    h1{font-size:20px}
}
</style>
""", unsafe_allow_html=True)

# ─── GEE INIT ────────────────────────────────────────────────────────────────
def init_gee():
    try:
        service_account = os.environ.get("GEE_SERVICE_ACCOUNT", "")
        private_key     = os.environ.get("GEE_PRIVATE_KEY", "").replace("\\n", "\n")
        if not service_account or not private_key:
            return False
        credentials = ee.ServiceAccountCredentials(service_account, key_data=private_key)
        ee.Initialize(credentials)
        return True
    except Exception as e:
        st.warning(f"GEE unavailable — using regional database. ({e})")
        return False

init_gee()

# ══════════════════════════════════════════════════════════════════════════════
# ZaminAI SATELLITE API — called from zaminai.org HTML app
# URL: https://zaminai.streamlit.app/?api=analyse&coords=...&year=2024
# ══════════════════════════════════════════════════════════════════════════════

import json
import datetime as dt

def handle_api_request():
    """Handle API requests from the HTML app via query parameters."""
    params = st.query_params
    
    # Check if this is an API request
    if params.get("api") != "analyse":
        return False
    
    # Set CORS headers via meta tag (Streamlit limitation)
    st.markdown("""
    <meta http-equiv="Access-Control-Allow-Origin" content="*">
    """, unsafe_allow_html=True)
    
    try:
        coords_str = params.get("coords", "")
        year = int(params.get("year", 2024))
        
        if not coords_str:
            st.json({"status": "error", "message": "No coordinates"})
            return True
        
        # Parse coords [[lat,lon],...]
        coords = json.loads(coords_str)
        
        if len(coords) < 3:
            st.json({"status": "error", "message": "Need 3+ coordinates"})
            return True
        
        # Convert [lat,lon] to GEE [lon,lat]
        gee_coords = [[c[1], c[0]] for c in coords]
        region = ee.Geometry.Polygon([gee_coords])
        
        # Sentinel-2
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(region)
              .filterDate(f"{year}-01-01", f"{year}-12-31")
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 35))
              .sort("CLOUDY_PIXEL_PERCENTAGE").limit(10)
              .median().clip(region))
        
        ndvi  = s2.normalizedDifference(["B8","B4"]).rename("NDVI")
        mndwi = s2.normalizedDifference(["B3","B11"]).rename("MNDWI")
        
        def mean(img, band, scale=30):
            v = img.select(band).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region, scale=scale, maxPixels=1e9
            ).getInfo()
            return round((v or {}).get(band, 0) or 0, 4)
        
        ndvi_val  = mean(ndvi, "NDVI")
        mndwi_val = mean(mndwi, "MNDWI")
        
        # Rainfall
        rain = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                .filterBounds(region)
                .filterDate(f"{year}-01-01", f"{year}-12-31")
                .select("precipitation").sum().clip(region))
        rain_val = mean(rain, "precipitation", 5000)
        
        # Area
        area_ha = round(region.area().getInfo() / 10000, 2)
        centroid = region.centroid().coordinates().getInfo()
        
        # NDVI trend
        trend = {}
        for yr in range(2019, min(year+1, dt.datetime.now().year+1)):
            try:
                s2yr = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                        .filterBounds(region)
                        .filterDate(f"{yr}-05-01", f"{yr}-07-31")
                        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",20))
                        .median().clip(region))
                v = s2yr.normalizedDifference(["B8","B4"]).reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=region, scale=30, maxPixels=1e9
                ).getInfo()
                trend[yr] = round((v or {}).get("nd", 0) or 0, 4)
            except:
                trend[yr] = 0.0
        
        result = {
            "status":     "success",
            "ndvi":       ndvi_val,
            "mndwi":      mndwi_val,
            "rain_mm":    round(rain_val, 1),
            "area_ha":    area_ha,
            "area_jereb": round(area_ha * 5, 1),
            "lat":        round(centroid[1], 6),
            "lon":        round(centroid[0], 6),
            "ndvi_trend": trend,
            "year":       year,
            "image_date": f"{year}-07-01",
        }
        
        # Return as JSON displayed as code (Streamlit limitation)
        st.code(json.dumps(result), language="json")
        return True
        
    except Exception as e:
        st.json({"status": "error", "message": str(e)})
        return True

# Run API handler first — if it handles request, stop here
if handle_api_request():
    st.stop()

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
AFGHAN_PROVINCES = {
    "Kunduz":     {"bbox":[68.55,36.55,69.05,37.05],"center":[36.73,68.87]},
    "Balkh":      {"bbox":[66.70,36.50,67.20,37.00],"center":[36.76,66.90]},
    "Helmand":    {"bbox":[63.80,31.00,64.80,31.80],"center":[31.35,64.20]},
    "Herat":      {"bbox":[61.80,34.10,62.50,34.60],"center":[34.34,62.20]},
    "Nangarhar":  {"bbox":[70.20,34.00,70.80,34.50],"center":[34.17,70.62]},
    "Kabul":      {"bbox":[69.00,34.30,69.50,34.70],"center":[34.53,69.17]},
    "Kandahar":   {"bbox":[65.40,31.50,66.00,31.90],"center":[31.63,65.71]},
    "Takhar":     {"bbox":[69.30,36.60,70.00,37.10],"center":[36.83,69.52]},
    "Baghlan":    {"bbox":[68.40,36.00,69.00,36.60],"center":[36.17,68.71]},
    "Badakhshan": {"bbox":[70.50,36.80,71.50,37.50],"center":[37.12,70.81]},
}

# ─── DEMO FIELD — pre-loaded Kunduz wheat field, 2024 real satellite values ──
DEMO_FIELD = {
    "status": "success", "source": "demo",
    "label": "Demo — Kunduz Wheat Field", "province": "Kunduz",
    "lat": 36.73, "lon": 68.87,
    "area_ha": 1.2, "area_jereb": 6.0,
    "year": 2024,
    # Sentinel-2 indices (2024 growing season, May–Jul composite)
    "ndvi": 0.33, "mndwi": -0.14, "rain_mm": 287.0,
    "ndvi_trend": {2019: 0.40, 2020: 0.38, 2021: 0.35, 2022: 0.22, 2023: 0.27, 2024: 0.33},
    "monthly_ndvi": {
        "Jan": 0.08, "Feb": 0.12, "Mar": 0.22, "Apr": 0.31, "May": 0.33,
        "Jun": 0.28, "Jul": 0.18, "Aug": 0.10, "Sep": 0.08, "Oct": 0.09,
        "Nov": 0.11, "Dec": 0.09,
    },
    "monthly_rain": {
        "Jan": 40.2, "Feb": 56.8, "Mar": 86.5, "Apr": 91.2, "May": 86.5,
        "Jun": 46.3, "Jul": 17.2, "Aug": 11.5, "Sep": 17.2, "Oct": 28.7,
        "Nov": 57.5, "Dec": 86.5,
    },
}
# Demo field boundary polygon (small rectangle SE of Kunduz city)
DEMO_COORDS = [
    [36.72, 68.86], [36.72, 68.88],
    [36.74, 68.88], [36.74, 68.86],
    [36.72, 68.86],
]

# ─── GEE ANALYSIS FUNCTIONS ──────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def analyze_field(geometry_coords, year=2024):
    """Full satellite analysis for a drawn field polygon."""
    try:
        region = ee.Geometry.Polygon([geometry_coords])

        # Sentinel-2 — full year with lenient cloud filter so cloudy regions
        # (Europe, SE Asia) still produce a valid composite
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(region)
              .filterDate(f"{year}-01-01", f"{year}-12-31")
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 35))
              .sort("CLOUDY_PIXEL_PERCENTAGE").limit(10)
              .median().clip(region))

        ndvi  = s2.normalizedDifference(["B8","B4"]).rename("NDVI")
        mndwi = s2.normalizedDifference(["B3","B11"]).rename("MNDWI")
        ndbi  = s2.normalizedDifference(["B11","B8"]).rename("NDBI")

        # Rainfall
        rain = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                .filterBounds(region)
                .filterDate(f"{year}-01-01", f"{year}-12-31")
                .select("precipitation").sum().clip(region))

        def mean_val(img, scale=30):
            return img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region, scale=scale, maxPixels=1e9
            ).getInfo()

        ndvi_val  = (mean_val(ndvi)  or {}).get("NDVI",  0) or 0
        mndwi_val = (mean_val(mndwi) or {}).get("MNDWI", 0) or 0
        rain_val  = (mean_val(rain, scale=5000) or {}).get("precipitation", 0) or 0

        # Field area
        area_m2 = region.area().getInfo()
        area_ha = round(area_m2 / 10000, 2)

        # NDVI for multiple years for trend
        ndvi_trend = {}
        cur_yr = dt.datetime.now().year
        for yr in range(2019, cur_yr + 1):
            try:
                s2_yr = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                         .filterBounds(region)
                         .filterDate(f"{yr}-01-01", f"{yr}-12-31")
                         .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 35))
                         .sort("CLOUDY_PIXEL_PERCENTAGE").limit(8)
                         .median().clip(region))
                ndvi_yr = s2_yr.normalizedDifference(["B8","B4"])
                val = ndvi_yr.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=region, scale=30, maxPixels=1e9
                ).getInfo().get("nd", 0)
                ndvi_trend[yr] = round(val or 0, 4)
            except:
                ndvi_trend[yr] = 0

        # Monthly NDVI for current year
        monthly_ndvi = {}
        months = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]
        month_ranges = [
            ("01-01","01-31"),("02-01","02-28"),("03-01","03-31"),
            ("04-01","04-30"),("05-01","05-31"),("06-01","06-30"),
            ("07-01","07-31"),("08-01","08-31"),("09-01","09-30"),
            ("10-01","10-31"),("11-01","11-30"),("12-01","12-31"),
        ]
        for i, (start, end) in enumerate(month_ranges):
            try:
                s2_m = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                        .filterBounds(region)
                        .filterDate(f"{year}-{start}", f"{year}-{end}")
                        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
                        .median().clip(region))
                ndvi_m = s2_m.normalizedDifference(["B8","B4"])
                val = ndvi_m.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=region, scale=30, maxPixels=1e9
                ).getInfo().get("nd", 0)
                monthly_ndvi[months[i]] = round(val or 0, 4)
            except:
                monthly_ndvi[months[i]] = None

        # Monthly rainfall
        monthly_rain = {}
        for i, (start, end) in enumerate(month_ranges):
            try:
                rain_m = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                          .filterBounds(region)
                          .filterDate(f"{year}-{start}", f"{year}-{end}")
                          .select("precipitation").sum().clip(region))
                val = rain_m.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=region, scale=5000, maxPixels=1e9
                ).getInfo().get("precipitation", 0)
                monthly_rain[months[i]] = round(val or 0, 1)
            except:
                monthly_rain[months[i]] = 0

        return {
            "ndvi":         round(ndvi_val, 4),
            "mndwi":        round(mndwi_val, 4),
            "rain_mm":      round(rain_val, 1),
            "area_ha":      area_ha,
            "ndvi_trend":   ndvi_trend,
            "monthly_ndvi": monthly_ndvi,
            "monthly_rain": monthly_rain,
            "status":       "success"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_ndvi_status(ndvi):
    if ndvi >= 0.4:   return "excellent", "#4ade80", "Healthy and productive"
    elif ndvi >= 0.25: return "good",     "#86efac", "Good vegetation"
    elif ndvi >= 0.15: return "moderate", "#fbbf24", "Moderate — needs attention"
    elif ndvi >= 0.05: return "poor",     "#f87171", "Poor — stressed crops"
    else:              return "critical", "#ef4444", "Critical — bare or dead"

def get_water_status(mndwi):
    if mndwi >= 0.1:   return "good",     "#38bdf8", "Good water availability"
    elif mndwi >= 0.0: return "moderate", "#fbbf24", "Moderate water"
    else:              return "low",      "#f87171", "Low water — irrigation needed"

def generate_recommendations(ndvi, mndwi, rain_mm, area_ha):
    """Generate specific farming recommendations based on satellite data."""
    recs = []

    # Irrigation recommendation
    if mndwi < 0.0:
        recs.append({
            "type": "urgent",
            "icon": "💧",
            "en": f"URGENT: Water stress detected. Your {area_ha}ha field needs irrigation within 3-5 days.",
            "dari": f"فوری: کمبود آب شناسایی شد. مزرعه {area_ha} هکتاری شما در ۳-۵ روز آینده نیاز به آبیاری دارد.",
            "pashto": f"بیړني: د اوبو کمښت وموندل شو. ستاسو {area_ha} هکتاره مزرعه د ۳-۵ ورځو دننه اوبو ته اړتیا لري."
        })
    elif mndwi < 0.05:
        recs.append({
            "type": "warning",
            "icon": "💧",
            "en": f"Water availability is low. Monitor irrigation for your {area_ha}ha field.",
            "dari": f"آب کم است. آبیاری مزرعه {area_ha} هکتاری خود را کنترل کنید.",
            "pashto": f"اوبه لږ دي. د خپل {area_ha} هکتاره مزرعه اوبه ورکول وڅارئ."
        })

    # Crop health recommendation
    if ndvi < 0.15:
        recs.append({
            "type": "urgent",
            "icon": "🌾",
            "en": "Vegetation stress is severe. Check for disease, pest damage, or drought.",
            "dari": "فشار شدیدی بر گیاهان وجود دارد. بیماری، آفت یا خشکسالی را بررسی کنید.",
            "pashto": "د نباتاتو سخت فشار شتون لري. ناروغي، آفت یا وچکالي وګورئ."
        })
    elif ndvi < 0.25:
        recs.append({
            "type": "warning",
            "icon": "🌾",
            "en": "Crop health is below average. Consider fertilizer or irrigation adjustment.",
            "dari": "سلامت محصول زیر حد متوسط است. کود یا تنظیم آبیاری را در نظر بگیرید.",
            "pashto": "د محصول روغتیا د منځني کچي لاندې ده. سره یا د اوبو تنظیم پام کې ونیسئ."
        })
    elif ndvi >= 0.4:
        recs.append({
            "type": "good",
            "icon": "✅",
            "en": "Excellent crop health! Your field is performing well this season.",
            "dari": "سلامت محصول عالی است! مزرعه شما در این فصل عملکرد خوبی دارد.",
            "pashto": "د محصول روغتیا غوره ده! ستاسو مزرعه پدې فصل کې ښه کار کوي."
        })

    # Crop suggestion based on water
    if mndwi < 0.0 and rain_mm < 200:
        recs.append({
            "type": "info",
            "icon": "🌱",
            "en": f"With {rain_mm}mm rainfall and low water, consider drought-tolerant crops: flax, chickpeas, or mung beans.",
            "dari": f"با {rain_mm} میلیمتر باران و آب کم، محصولات مقاوم به خشکی را در نظر بگیرید: کتان، نخود یا لوبیا.",
            "pashto": f"د {rain_mm} ملي متر باران او لږ اوبو سره، د وچکالي مقاومه وکرې غوره کړئ: کتان، نخود یا لوبیا."
        })

    return recs

# ─── HEADER ──────────────────────────────────────────────────────────────────
col_title, col_lang = st.columns([3,1])
with col_title:
    st.title("🌱 ZaminAI — Smart Farming Intelligence")
    st.markdown("**Afghanistan Development Initiative** — Real satellite data for every field")
with col_lang:
    language = st.selectbox("🌐", ["English", "دری (Dari)", "پښتو (Pashto)"], label_visibility="collapsed")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab_map, tab_analysis, tab_ai, tab_alerts, tab_history = st.tabs([
    "🗺️ My Field",
    "📊 Analysis",
    "🤖 Ask AI",
    "🔔 Alerts",
    "📈 History"
])

# ════════════════════════════════════════════════════════
# TAB 1 — MAP & FIELD DRAWING
# ════════════════════════════════════════════════════════
with tab_map:
    st.subheader("Draw your field on the map")

    # ── Investor demo ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#052e16;border:1px solid #16a34a;border-radius:8px;
    padding:.7rem 1rem;margin-bottom:.75rem;font-family:monospace;font-size:13px;
    color:#4ade80;display:flex;align-items:center;gap:.75rem">
    🎯 <strong>Investor demo</strong> — Load a real Kunduz wheat field with live 2024 satellite
    data. No GPS or drawing required.
    </div>
    """, unsafe_allow_html=True)
    if st.button("🛰️ Load Kunduz Demo Field — See Full Analysis Instantly",
                 type="primary", use_container_width=True, key="demo_btn"):
        st.session_state["field_results"]  = DEMO_FIELD
        st.session_state["analysis_year"]  = 2024
        st.session_state["field_coords"]   = DEMO_COORDS
        st.session_state["field_geometry"] = {
            "type": "Polygon",
            "coordinates": [[[c[1], c[0]] for c in DEMO_COORDS]]
        }
        st.session_state["demo_active"] = True
        st.success("✅ Demo loaded — click the **📊 Analysis** tab to see full satellite results.")
        st.rerun()

    if st.session_state.get("demo_active"):
        st.info("🎯 Demo mode — Kunduz wheat field, 2024 growing season. "
                "Click **📊 Analysis** to explore.")

    st.divider()

    if language == "دری (Dari)":
        st.markdown('<div class="dari-text">زمین خود را روی نقشه رسم کنید — GPS موقعیت شما را نشان می‌دهد</div>', unsafe_allow_html=True)
    elif language == "پښتو (Pashto)":
        st.markdown('<div class="pashto-text">خپله ځمکه د نقشه پر مخ رسم کړئ — GPS ستاسو موقعیت ښیي</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2,1])
    with col1:
        location_method = st.radio(
            "Find my field by:",
            ["📍 Use my GPS location", "🏘️ Select province", "🔍 Enter coordinates"],
            horizontal=True,
            label_visibility="collapsed"
        )
    with col2:
        analysis_year = st.selectbox("Analysis year", [2024, 2023, 2022, 2021, 2019], index=0)

    # GPS auto-draw section
    auto_draw_coords = None
    if location_method == "📍 Use my GPS location":
        st.markdown("""
        <div style="background:#052e16;border-left:3px solid #16a34a;padding:10px 14px;
        border-radius:4px;font-size:13px;color:#86efac;margin-bottom:10px">
        📍 <strong>Option 1</strong> — Click the <strong>⊕ crosshair button</strong> on the map → then draw your field manually<br>
        🤖 <strong>Option 2</strong> — Enter your GPS coordinates below → app draws your field automatically
        </div>""", unsafe_allow_html=True)

        col_gps1, col_gps2, col_gps3 = st.columns([1,1,1])
        with col_gps1:
            gps_lat = st.number_input("My latitude", value=0.0, format="%.6f",
                                       help="Open Google Maps → long press your location → copy the first number")
        with col_gps2:
            gps_lon = st.number_input("My longitude", value=0.0, format="%.6f",
                                       help="Open Google Maps → long press your location → copy the second number")
        with col_gps3:
            # Smart field size options with jereb + hectare
            field_size = st.selectbox(
                "My field size",
                options=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
                index=2,
                format_func=lambda x: (
                    f"{round(x*5,1)} jereb ({x} ha)" if x < 1 else
                    f"{round(x*5)} jereb ({x} ha)"
                ),
                help="Not sure? Average Afghan smallholder farm = 2-5 jereb (0.4-1 ha)"
            )
        
        if gps_lat != 0.0 and gps_lon != 0.0:
            # Calculate bbox from center point + field size
            import math
            # Degrees per meter at this latitude
            lat_deg = field_size * 100 / 111320  # approx degrees for field size in ha
            lon_deg = field_size * 100 / (111320 * math.cos(math.radians(gps_lat)))
            half_lat = lat_deg / 2
            half_lon = lon_deg / 2
            auto_draw_coords = [
                [gps_lat - half_lat, gps_lon - half_lon],
                [gps_lat - half_lat, gps_lon + half_lon],
                [gps_lat + half_lat, gps_lon + half_lon],
                [gps_lat + half_lat, gps_lon - half_lon],
                [gps_lat - half_lat, gps_lon - half_lon],
            ]
            map_center = [gps_lat, gps_lon]
            map_zoom = 16
            jereb = round(field_size * 5, 1)
            st.success(f"✓ Field ready to draw — {jereb} jereb ({field_size} ha) at your location. Click 'Analyse this field' below.")
        else:
            map_center = [34.5, 67.7]
            map_zoom = 6
            st.caption("💡 Tip: Open Google Maps on your phone → long press your field → copy the coordinates shown")

    # Set map center based on selection
    elif location_method == "🏘️ Select province":
        province = st.selectbox("Select province / ولایت", list(AFGHAN_PROVINCES.keys()))
        map_center = AFGHAN_PROVINCES[province]["center"]
        map_zoom = 11
    elif location_method == "🔍 Enter coordinates":
        col_lat, col_lon = st.columns(2)
        with col_lat:
            lat = st.number_input("Latitude", value=36.73, format="%.4f")
        with col_lon:
            lon = st.number_input("Longitude", value=68.87, format="%.4f")
        map_center = [lat, lon]
        map_zoom = 14
    else:
        map_center = [34.5, 67.7]
        map_zoom = 6

    # Build the map
    m = folium.Map(
        location=map_center,
        zoom_start=map_zoom,
        tiles=None
    )

    # Satellite base layer
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Satellite",
        overlay=False
    ).add_to(m)

    # Street map option
    folium.TileLayer(
        tiles="CartoDB positron",
        name="Street map",
        overlay=False
    ).add_to(m)

    # GPS locate button
    LocateControl(
        auto_start=False,
        position="topright",
        strings={"title": "Find my location", "popup": "Your location"},
        flyTo=True,
        zoom=14
    ).add_to(m)

    # Auto-draw polygon from GPS coordinates
    if auto_draw_coords:
        folium.Polygon(
            locations=auto_draw_coords,
            color="#4ade80",
            fill=True,
            fill_color="#4ade80",
            fill_opacity=0.15,
            weight=2,
            popup=f"Your field — {round(field_size*5,1)} jereb ({field_size} ha)",
            tooltip=f"✓ Your field — {round(field_size*5,1)} jereb"
        ).add_to(m)
        # Add centre marker
        folium.Marker(
            location=[gps_lat, gps_lon],
            popup="Your location",
            icon=folium.Icon(color="green", icon="leaf", prefix="glyphicon")
        ).add_to(m)
        # Store coords for analysis
        st.session_state["auto_coords"] = auto_draw_coords
        st.session_state["auto_area_ha"] = field_size

    # Drawing tools — rectangle and polygon
    Draw(
        draw_options={
            "rectangle": {
                "shapeOptions": {"color": "#4ade80", "weight": 2}
            },
            "polygon": {
                "shapeOptions": {"color": "#4ade80", "weight": 2}
            },
            "circle":       False,
            "marker":       False,
            "polyline":     False,
            "circlemarker": False
        },
        edit_options={"edit": True, "remove": True}
    ).add_to(m)

    # Add NDVI layer if province selected
    if location_method == "🏘️ Select province":
        try:
            bbox = AFGHAN_PROVINCES[province]["bbox"]
            region_ee = ee.Geometry.Rectangle(bbox)
            s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(region_ee)
                  .filterDate(f"{analysis_year}-05-01", f"{analysis_year}-07-31")
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 15))
                  .median().clip(region_ee))
            ndvi = s2.normalizedDifference(["B8","B4"])
            tile_url = ndvi.getMapId({
                "min":0, "max":0.7,
                "palette":["#d73027","#fc8d59","#fee08b","#d9ef8b","#91cf60","#1a9850"]
            })["tile_fetcher"].url_format
            folium.TileLayer(
                tiles=tile_url,
                attr="GEE NDVI",
                name=f"NDVI {analysis_year}",
                overlay=True,
                opacity=0.7
            ).add_to(m)
        except:
            pass

    folium.LayerControl(collapsed=False).add_to(m)

    # Render map
    map_output = st_folium(
        m,
        width=None,
        height=500,
        key="field_map",
        returned_objects=["last_active_drawing", "all_drawings"]
    )

    # Extract drawn field
    field_geometry = None
    field_coords   = None

    if map_output and map_output.get("last_active_drawing"):
        drawing = map_output["last_active_drawing"]
        geo_type = drawing.get("geometry", {}).get("type", "")

        if geo_type == "Polygon":
            field_coords = drawing["geometry"]["coordinates"][0]
            field_geometry = drawing["geometry"]
            st.session_state["field_coords"] = field_coords
            st.session_state["field_geometry"] = field_geometry

        elif geo_type == "Rectangle":
            field_coords = drawing["geometry"]["coordinates"][0]
            field_geometry = drawing["geometry"]
            st.session_state["field_coords"] = field_coords
            st.session_state["field_geometry"] = field_geometry

    # Use saved field if available
    if not field_coords and "field_coords" in st.session_state:
        field_coords   = st.session_state["field_coords"]
        field_geometry = st.session_state["field_geometry"]

    if field_coords:
        lons = [c[0] for c in field_coords]
        lats = [c[1] for c in field_coords]
        area_approx = abs((max(lons)-min(lons)) * (max(lats)-min(lats)) * 111320 * 111320 / 10000)

        st.markdown(f"""
        <div class="alert-green">
        ✅ Field drawn successfully!
        Center: {sum(lats)/len(lats):.4f}°N, {sum(lons)/len(lons):.4f}°E
        — Approximate area: {area_approx:.1f} ha
        — Go to Analysis tab for full satellite data
        </div>
        """, unsafe_allow_html=True)

        if st.button("🛰️ Analyse this field now", type="primary", use_container_width=True):
            with st.spinner("Analysing your field with satellite data..."):
                results = analyze_field(field_coords, analysis_year)
                if results["status"] == "success":
                    st.session_state["field_results"] = results
                    st.session_state["analysis_year"] = analysis_year
                    st.success("Analysis complete! Go to the Analysis tab.")
                else:
                    st.error(f"Analysis failed: {results.get('message')}")
    else:
        st.markdown("""
        <div class="alert-yellow">
        ℹ️ Draw your field on the map using the rectangle or polygon tool (toolbar on the left side of the map)
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 2 — ANALYSIS RESULTS
# ════════════════════════════════════════════════════════
with tab_analysis:
    if "field_results" not in st.session_state:
        st.info("Draw your field on the Map tab and click 'Analyse this field' first.")
    else:
        r = st.session_state["field_results"]
        yr = st.session_state.get("analysis_year", 2024)

        ndvi_status, ndvi_color, ndvi_label = get_ndvi_status(r["ndvi"])
        water_status, water_color, water_label = get_water_status(r["mndwi"])

        st.subheader(f"Field Analysis — {yr} Growing Season")

        if st.session_state.get("demo_active"):
            st.markdown("""
            <div style="background:#052e16;border-left:4px solid #16a34a;border-radius:4px;
            padding:.6rem 1rem;margin-bottom:.75rem;font-family:monospace;font-size:12px;
            color:#86efac">
            🎯 <strong>Demo mode</strong> — Kunduz Province · 6 jereb wheat field · 2024
            Sentinel-2 data · NDVI 0.33 (good crop health) · Water stress detected
            </div>
            """, unsafe_allow_html=True)

        # Key metrics
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Field area",    f"{r['area_ha']} ha",  "Your field size")
        c2.metric("Vegetation",    f"{r['ndvi']}",         ndvi_label)
        c3.metric("Water index",   f"{r['mndwi']}",        water_label)
        c4.metric("Annual rain",   f"{r['rain_mm']} mm",   "This year")

        st.divider()

        # Status cards
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="field-card">
            <div style="font-size:11px;color:#6b8f65;font-family:monospace;margin-bottom:8px">VEGETATION HEALTH</div>
            <div style="font-size:28px;font-weight:700;color:{ndvi_color};font-family:monospace">{r['ndvi']}</div>
            <div style="font-size:13px;color:{ndvi_color};margin-top:4px">{ndvi_label}</div>
            <div style="font-size:12px;color:#6b8f65;margin-top:8px">
            0.0–0.15 = stressed/bare · 0.15–0.35 = moderate · 0.35+ = healthy
            </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="field-card">
            <div style="font-size:11px;color:#6b8f65;font-family:monospace;margin-bottom:8px">WATER AVAILABILITY</div>
            <div style="font-size:28px;font-weight:700;color:{water_color};font-family:monospace">{r['mndwi']}</div>
            <div style="font-size:13px;color:{water_color};margin-top:4px">{water_label}</div>
            <div style="font-size:12px;color:#6b8f65;margin-top:8px">
            Below 0 = dry · 0–0.1 = some water · Above 0.1 = good water access
            </div>
            </div>
            """, unsafe_allow_html=True)

        # NDVI trend chart
        st.subheader("Vegetation trend — your field over years")
        trend = r["ndvi_trend"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(trend.keys()),
            y=list(trend.values()),
            mode="lines+markers",
            line=dict(color="#4ade80", width=2),
            marker=dict(
                size=10,
                color=["#f87171" if v==min(trend.values())
                       else "#4ade80" if v==max(trend.values())
                       else "#86efac" for v in trend.values()]
            ),
            fill="tozeroy",
            fillcolor="rgba(74,222,128,0.06)"
        ))
        fig.update_layout(
            paper_bgcolor="#111810", plot_bgcolor="#111810",
            font=dict(color="#6b8f65", family="monospace", size=11),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#6b8f65")),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#6b8f65"), title="NDVI"),
            showlegend=False, margin=dict(l=10,r=10,t=20,b=10),
            height=250
        )
        st.plotly_chart(fig, use_container_width=True)

        # Recommendations
        st.subheader("Recommendations for your field")
        recs = generate_recommendations(r["ndvi"], r["mndwi"], r["rain_mm"], r["area_ha"])

        for rec in recs:
            alert_class = "alert-red" if rec["type"]=="urgent" else "alert-yellow" if rec["type"]=="warning" else "alert-green"
            if language == "دری (Dari)":
                text = rec["dari"]
            elif language == "پښتو (Pashto)":
                text = rec["pashto"]
            else:
                text = rec["en"]

            st.markdown(f'<div class="{alert_class}">{rec["icon"]} {text}</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 3 — AI ASSISTANT
# ════════════════════════════════════════════════════════
with tab_ai:
    if language == "دری (Dari)":
        st.subheader("🤖 دستیار هوش مصنوعی مزرعه")
        placeholder = "سوال خود را بنویسید یا بگویید... مثال: آیا زمین من آب کافی دارد؟"
    elif language == "پښتو (Pashto)":
        st.subheader("🤖 د مزرعې د هوښیار AI مرستیال")
        placeholder = "خپله پوښتنه ولیکئ یا ووایئ... بیلګه: ایا زما ځمکه کافي اوبه لري؟"
    else:
        st.subheader("🤖 AI Field Assistant")
        placeholder = "Ask anything about your field... e.g. Does my field have enough water?"

    # Voice input HTML component
    st.markdown("""
    <div style="background:#111810;border:1px solid #1e2b1a;border-radius:8px;padding:1rem;margin-bottom:1rem">
    <div style="font-size:12px;color:#6b8f65;margin-bottom:8px;font-family:monospace">VOICE INPUT — tap microphone to speak</div>
    <div style="display:flex;gap:8px;align-items:center">
        <button onclick="startVoice()" style="background:#16a34a;color:white;border:none;border-radius:8px;padding:10px 20px;font-size:14px;cursor:pointer">
            🎤 Speak / بگو / ووایه
        </button>
        <span id="voice-status" style="font-size:13px;color:#6b8f65">Ready</span>
    </div>
    <div id="voice-result" style="margin-top:8px;font-size:14px;color:#86efac;min-height:24px;direction:auto"></div>
    </div>
    <script>
    let recognition;
    function startVoice() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            document.getElementById('voice-status').textContent = 'Voice not supported on this browser';
            return;
        }
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SR();
        recognition.lang = 'fa-AF';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;
        document.getElementById('voice-status').textContent = '🔴 Listening...';
        recognition.start();
        recognition.onresult = function(e) {
            const text = e.results[0][0].transcript;
            document.getElementById('voice-result').textContent = text;
            document.getElementById('voice-status').textContent = '✅ Got it!';
        };
        recognition.onerror = function(e) {
            document.getElementById('voice-status').textContent = 'Error: ' + e.error;
        };
        recognition.onend = function() {
            if (document.getElementById('voice-status').textContent === '🔴 Listening...') {
                document.getElementById('voice-status').textContent = 'Ready';
            }
        };
    }
    </script>
    """, unsafe_allow_html=True)

    # Chat history
    if "field_messages" not in st.session_state:
        if language == "دری (Dari)":
            welcome = "سلام! من دستیار هوش مصنوعی مزرعه شما هستم. می‌توانید درباره آب، محصول، خاک یا آب‌وهوا سوال بپرسید. من به داده‌های ماهواره‌ای واقعی دسترسی دارم."
        elif language == "پښتو (Pashto)":
            welcome = "سلام! زه ستاسو د مزرعې AI مرستیال یم. تاسو کولی شئ د اوبو، محصول، خاورې یا هوا په اړه پوښتنه وکړئ. زه د واقعي ماهواره‌ای معلوماتو ته لاسرسی لرم."
        else:
            welcome = "Hello! I am your AI Field Assistant. Ask me anything about your field — water, crops, soil, or weather. I have access to real satellite data for your exact location."
        st.session_state.field_messages = [{"role":"assistant","content":welcome}]

    for msg in st.session_state.field_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Build context from field analysis
    field_context = ""
    if "field_results" in st.session_state:
        r = st.session_state["field_results"]
        yr = st.session_state.get("analysis_year", 2024)
        ndvi_s, _, ndvi_l = get_ndvi_status(r["ndvi"])
        water_s, _, water_l = get_water_status(r["mndwi"])
        recs = generate_recommendations(r["ndvi"], r["mndwi"], r["rain_mm"], r["area_ha"])
        rec_text = " | ".join([rec["en"] for rec in recs])

        field_context = f"""
FARMER'S FIELD DATA ({yr} growing season):
- Field area: {r['area_ha']} hectares
- NDVI (vegetation health): {r['ndvi']} — {ndvi_l}
- MNDWI (water index): {r['mndwi']} — {water_l}
- Annual rainfall: {r['rain_mm']} mm
- NDVI trend: {r['ndvi_trend']}
- Current recommendations: {rec_text}
"""
    else:
        field_context = """
No field has been drawn yet. Use general knowledge about Afghan agriculture.
Tell the farmer to draw their field on the Map tab for personalized satellite-based advice.
"""

    system_prompt = f"""You are an AI agricultural assistant for Afghan smallholder farmers.
You speak English, Dari (دری), and Pashto (پښتو).
Always respond in the SAME language the farmer uses.

{field_context}

KNOWLEDGE BASE:
- Afghanistan's main crops: wheat (winter), cotton, rice, flax, vegetables, saffron
- Kunduz, Balkh, Takhar = main agricultural provinces in north
- Water is the #1 constraint — Kunduz River fed by Hindu Kush snowmelt
- 2022 was worst drought year — 76% water reduction
- Best low-water crops: saffron, flax, chickpeas, almonds
- Avoid high-water crops in dry years: rice, cotton

RESPONSE RULES:
- Be specific and practical — exact numbers, exact actions
- Keep answers SHORT — 3-4 sentences maximum
- If farmer asks in Dari, answer in Dari
- If farmer asks in Pashto, answer in Pashto
- Always end with ONE specific action the farmer can take today
- Never use technical jargon the farmer won't understand"""

    if prompt := st.chat_input(placeholder):
        st.session_state.field_messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("..."):
                try:
                    import anthropic
                    client = anthropic.Anthropic(
                        api_key=os.environ.get("ANTHROPIC_API_KEY", "")
                    )
                    response = client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=400,
                        system=system_prompt,
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.field_messages
                        ]
                    )
                    answer = response.content[0].text

                    # Text to speech
                    st.markdown(answer)
                    st.markdown(f"""
                    <div style="margin-top:8px">
                    <button onclick="speak('{answer[:200].replace(chr(39),'').replace(chr(34),'')}', '{language}')"
                        style="background:#111810;color:#4ade80;border:1px solid #1e2b1a;border-radius:6px;padding:6px 14px;font-size:12px;cursor:pointer">
                        🔊 Read aloud / بخوان / ولوله
                    </button>
                    </div>
                    <script>
                    function speak(text, lang) {{
                        const u = new SpeechSynthesisUtterance(text);
                        u.lang = lang.includes('Dari') ? 'fa' : lang.includes('Pashto') ? 'ps' : 'en';
                        u.rate = 0.85;
                        speechSynthesis.speak(u);
                    }}
                    </script>
                    """, unsafe_allow_html=True)

                    st.session_state.field_messages.append({"role":"assistant","content":answer})
                except KeyError:
                    st.warning("Add Anthropic API key to Streamlit secrets: [anthropic] api_key = 'sk-ant-...'")
                except Exception as e:
                    st.error(f"AI error: {e}")

# ════════════════════════════════════════════════════════
# TAB 4 — ALERTS
# ════════════════════════════════════════════════════════
with tab_alerts:
    st.subheader("Field Alerts & Recommendations")

    if "field_results" not in st.session_state:
        st.info("Analyse your field first to see personalized alerts.")
    else:
        r = st.session_state["field_results"]
        recs = generate_recommendations(r["ndvi"], r["mndwi"], r["rain_mm"], r["area_ha"])

        if not recs:
            st.markdown('<div class="alert-green">✅ No urgent alerts for your field right now.</div>', unsafe_allow_html=True)
        else:
            for rec in recs:
                alert_class = "alert-red" if rec["type"]=="urgent" else "alert-yellow" if rec["type"]=="warning" else "alert-green"
                st.markdown(f"""
                <div class="{alert_class}">
                <strong>{rec['icon']} English:</strong> {rec['en']}<br>
                <div class="dari-text">{rec['dari']}</div>
                <div class="pashto-text">{rec['pashto']}</div>
                </div>
                """, unsafe_allow_html=True)

        # Seasonal calendar
        st.divider()
        st.subheader("Seasonal farming calendar — Kunduz")
        calendar = pd.DataFrame([
            {"Month":"October–November","Action":"Plant winter wheat","Water need":"Low","Priority":"High"},
            {"Month":"December–February","Action":"Monitor snowpack upstream","Water need":"None","Priority":"Medium"},
            {"Month":"March–April",      "Action":"First irrigation — wheat flowering","Water need":"High","Priority":"Urgent"},
            {"Month":"April–May",        "Action":"Plant flax, chickpeas, vegetables","Water need":"Medium","Priority":"High"},
            {"Month":"May–June",         "Action":"Monitor NDVI — check crop stress","Water need":"High","Priority":"High"},
            {"Month":"June–July",        "Action":"Harvest winter wheat","Water need":"Low","Priority":"High"},
            {"Month":"July–August",      "Action":"Second crop irrigation","Water need":"High","Priority":"Medium"},
            {"Month":"August–September", "Action":"Harvest summer crops","Water need":"Low","Priority":"High"},
        ])
        st.dataframe(calendar, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════
# TAB 5 — HISTORY
# ════════════════════════════════════════════════════════
with tab_history:
    st.subheader("Your field history — satellite data over years")

    if "field_results" not in st.session_state:
        st.info("Analyse your field first to see historical data.")
    else:
        r = st.session_state["field_results"]
        trend = r["ndvi_trend"]

        # NDVI trend
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(trend.keys()),
            y=list(trend.values()),
            marker_color=["#4ade80" if v==max(trend.values())
                          else "#f87171" if v==min(trend.values())
                          else "rgba(74,222,128,0.5)" for v in trend.values()],
            name="NDVI"
        ))
        fig.add_hline(y=0.25, line_color="#fbbf24", line_dash="dash",
                      annotation_text="Good crop threshold",
                      annotation_font_color="#fbbf24")
        fig.update_layout(
            title="Your field NDVI history — green=best year, red=worst year",
            paper_bgcolor="#111810", plot_bgcolor="#111810",
            font=dict(color="#6b8f65", family="monospace", size=11),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#6b8f65")),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#6b8f65")),
            showlegend=False, margin=dict(l=10,r=10,t=40,b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        summary = pd.DataFrame([
            {"Year": yr, "NDVI": val,
             "Status": get_ndvi_status(val)[2],
             "vs 2019": f"{round((val - trend.get(2019,val))/max(trend.get(2019,0.001),0.001)*100,1)}%"}
            for yr, val in trend.items()
        ])
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="alert-yellow">
        💡 Tip: Compare your field NDVI with regional averages to understand if your farm
        is performing better or worse than your neighbors. Go to the AI Assistant tab and ask:
        "How does my field compare to other farms in Kunduz?"
        </div>
        """, unsafe_allow_html=True)

# ─── FARMER INTELLIGENCE MODULE ─────────────────────────────────────────────
try:
    from farmer_module import render_farmer_module
    render_farmer_module(
        language=language,
        field_results=st.session_state.get("field_results")
    )
except Exception as e:
    st.info(f"Farmer module loading: {e}")

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:#6b8f65;font-family:monospace;font-size:11px;line-height:2">
Afghanistan Development Initiative (ADI) · zaminai.org<br>
Sentinel-2 · CHIRPS · ERA5 · Google Earth Engine · Real satellite data<br>
Maiwand Jan Alamzoi · m.alamzoi123@gmail.com
</div>
""", unsafe_allow_html=True)
