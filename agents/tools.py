"""
Tool definitions and implementations for the ZaminAI agent system.
Each tool wraps an existing ZaminAI capability (GEE, Supabase, etc.)
so Claude can call them in a ReAct loop.
"""
import json, logging, math
from datetime import datetime, date

log = logging.getLogger(__name__)


# ── Claude tool schema definitions ────────────────────────────────────────────
TOOLS = [
    {
        "name": "query_satellite_data",
        "description": (
            "Query live satellite data from Google Earth Engine for a location or polygon. "
            "Returns NDVI, EVI, rainfall, SAR soil moisture, land cover classification, "
            "elevation, slope, and population. Use this as the first step for any "
            "field health or agricultural question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "coords": {
                    "type": "array",
                    "description": "List of [lat, lon] pairs forming the field/region polygon. Minimum 3 points.",
                    "items": {"type": "array"}
                },
                "year":     {"type": "integer", "description": "Analysis year (e.g. 2024). Defaults to current year."},
                "province": {"type": "string",  "description": "Province or region name for context"},
                "country":  {"type": "string",  "description": "Country name for context"},
            },
            "required": ["coords"]
        }
    },
    {
        "name": "get_ndvi_trend",
        "description": (
            "Get the multi-year NDVI vegetation trend for a location. "
            "Reveals whether land is degrading, recovering, or stable over time. "
            "Covers Landsat 2013–2018 and Sentinel-2 2019–present."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "coords":     {"type": "array",   "description": "List of [lat,lon] coordinate pairs"},
                "start_year": {"type": "integer", "description": "Start year (default: 2019)"},
                "end_year":   {"type": "integer", "description": "End year (default: current year)"},
            },
            "required": ["coords"]
        }
    },
    {
        "name": "get_land_cover",
        "description": (
            "Classify land cover using Dynamic World V1 (10m Sentinel-2). "
            "Returns percentage breakdown: crops, forest, water, bare soil, urban, grassland, etc. "
            "Use to confirm what is growing and detect unexpected changes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "coords": {"type": "array",   "description": "Polygon coordinates [lat,lon]"},
                "year":   {"type": "integer", "description": "Year for classification"},
            },
            "required": ["coords"]
        }
    },
    {
        "name": "get_monthly_rainfall",
        "description": (
            "Get monthly CHIRPS rainfall breakdown for a location and year. "
            "Use to diagnose irrigation need, drought stress, or waterlogging risk."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lat":  {"type": "number",  "description": "Latitude"},
                "lon":  {"type": "number",  "description": "Longitude"},
                "year": {"type": "integer", "description": "Year (default: current year)"},
            },
            "required": ["lat", "lon"]
        }
    },
    {
        "name": "get_soil_data",
        "description": (
            "Get soil composition, pH, organic carbon, and texture for a location "
            "from SoilGrids 250m. Use to give fertiliser and irrigation recommendations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Latitude"},
                "lon": {"type": "number", "description": "Longitude"},
            },
            "required": ["lat", "lon"]
        }
    },
    {
        "name": "get_crop_calendar",
        "description": (
            "Get the planting, growing, and harvest calendar for a specific crop "
            "and location. Use to time advice correctly to the growth stage."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "crop":     {"type": "string", "description": "Crop name (wheat, maize, rice, cotton, saffron, vegetables, etc.)"},
                "province": {"type": "string", "description": "Province or region"},
                "country":  {"type": "string", "description": "Country (default: Afghanistan)"},
            },
            "required": ["crop"]
        }
    },
    {
        "name": "compare_to_regional_average",
        "description": (
            "Compare a field's NDVI and rainfall to the regional average for the same province/district. "
            "Returns whether the field is above/below average and by how much. "
            "Use to contextualise individual field problems within the region."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "field_ndvi": {"type": "number", "description": "The field's current NDVI value"},
                "province":   {"type": "string", "description": "Province to compare against"},
                "country":    {"type": "string", "description": "Country"},
                "month":      {"type": "integer","description": "Month (1-12) for seasonal context"},
            },
            "required": ["field_ndvi", "province"]
        }
    },
    {
        "name": "detect_crop_type",
        "description": (
            "Estimate the crop type growing in a field using Sentinel-2 spectral indices. "
            "Returns the most likely crop and confidence level."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "coords": {"type": "array",   "description": "Field polygon coordinates [lat,lon]"},
                "year":   {"type": "integer", "description": "Year to analyse"},
                "month":  {"type": "integer", "description": "Peak growing month for the crop in question"},
            },
            "required": ["coords"]
        }
    },
    {
        "name": "get_farmer_fields",
        "description": (
            "Retrieve all registered fields for a farmer from the database. "
            "Returns field polygons, names, and last analysis results. "
            "Use to monitor all of a farmer's land in one agent call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "farmer_id": {"type": "string", "description": "Farmer UUID or phone number"},
            },
            "required": ["farmer_id"]
        }
    },
    {
        "name": "save_field_recommendation",
        "description": (
            "Save a recommendation or alert to the database for a specific farmer/field. "
            "Used to persist agent advice so farmers can review it later."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "farmer_id":  {"type": "string", "description": "Farmer UUID"},
                "field_id":   {"type": "string", "description": "Field UUID (optional)"},
                "message":    {"type": "string", "description": "The recommendation text"},
                "severity":   {"type": "string", "description": "info | warning | critical"},
                "language":   {"type": "string", "description": "en | fa | ps"},
            },
            "required": ["farmer_id", "message"]
        }
    },
    {
        "name": "get_agronomic_knowledge",
        "description": (
            "Look up agronomic knowledge: crop requirements, common diseases, "
            "Afghan soil management practices, fertiliser rates, pest identification. "
            "Use to enrich satellite data with agronomic context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic":    {"type": "string", "description": "Topic to look up (e.g. 'wheat iron deficiency', 'cotton water stress')"},
                "crop":     {"type": "string", "description": "Specific crop (optional)"},
                "language": {"type": "string", "description": "en | fa | ps"},
            },
            "required": ["topic"]
        }
    },
    {
        "name": "calculate_field_health_score",
        "description": (
            "Calculate a composite field health score (0–100) and risk level "
            "from multiple satellite indices. Combines NDVI, moisture, rainfall deficit, "
            "and trend to give a single health indicator."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ndvi":     {"type": "number", "description": "Current NDVI (0–1)"},
                "sar_vh":   {"type": "number", "description": "SAR VH backscatter (soil moisture proxy)"},
                "rain_mm":  {"type": "number", "description": "Annual rainfall mm"},
                "ndvi_trend":{"type": "number","description": "NDVI change vs previous year"},
                "month":    {"type": "integer","description": "Current month (1–12)"},
            },
            "required": ["ndvi"]
        }
    },
]


# ── Tool implementations ───────────────────────────────────────────────────────

def execute_tool(name: str, inputs: dict, app_context: dict) -> str:
    """Dispatch a tool call to its implementation. Returns a JSON string."""
    try:
        fn = _TOOL_MAP.get(name)
        if not fn:
            return json.dumps({"error": f"Unknown tool: {name}"})
        result = fn(inputs, app_context)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        log.error(f"Tool {name} failed: {e}")
        return json.dumps({"error": str(e), "tool": name})


def _query_satellite_data(inputs: dict, ctx: dict) -> dict:
    coords   = inputs.get("coords", [])
    year     = inputs.get("year", datetime.now().year)
    province = inputs.get("province", "")
    country  = inputs.get("country", "")

    gee_analyse = ctx.get("gee_analyse_officer")
    if not gee_analyse:
        return {"error": "GEE analysis not available", "fallback": True}
    try:
        result = gee_analyse(coords, year, province, country)
        # Summarise key fields for the agent
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        return {
            "ndvi":        result.get("ndvi"),
            "evi":         result.get("evi"),
            "rainfall_mm": result.get("rain"),
            "sar_vh":      result.get("sar", {}).get("VH"),
            "sar_vv":      result.get("sar", {}).get("VV"),
            "land_cover":  result.get("landcover"),
            "temperature_c": result.get("modis", {}).get("lst_day_c"),
            "frost_risk":  result.get("modis", {}).get("frost_risk"),
            "ndvi_trend":  result.get("ndvi_trend"),
            "area_km2":    result.get("area_km2"),
            "population":  result.get("population", {}).get("total"),
            "terrain":     result.get("terrain"),
            "source":      result.get("source", "gee"),
            "lat":         result.get("lat") or (round(sum(lats)/len(lats),5) if lats else None),
            "lon":         result.get("lon") or (round(sum(lons)/len(lons),5) if lons else None),
        }
    except Exception as e:
        return {"error": str(e)}


def _get_ndvi_trend(inputs: dict, ctx: dict) -> dict:
    coords     = inputs.get("coords", [])
    start_year = inputs.get("start_year", 2019)
    end_year   = inputs.get("end_year",   datetime.now().year)

    gee_analyse = ctx.get("gee_analyse_officer")
    if not gee_analyse:
        return {"error": "GEE not available"}
    try:
        trend = {}
        for yr in range(start_year, end_year + 1):
            r = gee_analyse(coords, yr, "", "")
            if r.get("ndvi"):
                trend[str(yr)] = round(float(r["ndvi"]), 3)
        years = list(trend.keys())
        vals  = list(trend.values())
        change = round(vals[-1] - vals[0], 3) if len(vals) >= 2 else 0
        return {
            "trend": trend,
            "change_over_period": change,
            "trend_direction": "improving" if change > 0.02 else "degrading" if change < -0.02 else "stable",
            "start_year": start_year, "end_year": end_year,
        }
    except Exception as e:
        return {"error": str(e)}


def _get_land_cover(inputs: dict, ctx: dict) -> dict:
    coords = inputs.get("coords", [])
    year   = inputs.get("year", datetime.now().year)
    gee_analyse = ctx.get("gee_analyse_officer")
    if not gee_analyse:
        return {"error": "GEE not available"}
    try:
        r = gee_analyse(coords, year, "", "")
        lc = r.get("landcover", {})
        return {
            "classes":    lc.get("classes", {}),
            "dominant":   lc.get("dominant"),
            "crop_pct":   lc.get("crop_pct"),
            "forest_pct": lc.get("forest_pct"),
            "water_pct":  lc.get("water_pct"),
            "bare_pct":   lc.get("bare_pct"),
        }
    except Exception as e:
        return {"error": str(e)}


def _get_monthly_rainfall(inputs: dict, ctx: dict) -> dict:
    lat  = inputs["lat"]
    lon  = inputs["lon"]
    year = inputs.get("year", datetime.now().year)
    # Use a 5-point tiny polygon around the point
    d = 0.05
    coords = [[lat-d,lon-d],[lat+d,lon-d],[lat+d,lon+d],[lat-d,lon+d],[lat-d,lon-d]]
    monthly_rain_fn = ctx.get("monthly_rain_fn")
    if not monthly_rain_fn:
        return {"error": "Rainfall function not available"}
    try:
        return monthly_rain_fn(lat, lon, year)
    except Exception as e:
        return {"error": str(e)}


def _get_soil_data(inputs: dict, ctx: dict) -> dict:
    lat = inputs["lat"]
    lon = inputs["lon"]
    soil_fn = ctx.get("soil_fn")
    if not soil_fn:
        return {"message": "SoilGrids lookup not available in this context"}
    try:
        return soil_fn(lat, lon)
    except Exception as e:
        return {"error": str(e)}


def _get_crop_calendar(inputs: dict, ctx: dict) -> dict:
    crop     = inputs.get("crop", "").lower()
    province = inputs.get("province", "")
    country  = inputs.get("country", "Afghanistan")

    CALENDARS = {
        "wheat": {"plant":"Nov–Dec","flowering":"Mar–Apr","harvest":"May–Jun",
                  "water_peak":"Feb–Apr","notes":"Winter wheat; needs frost during vernalisation"},
        "maize": {"plant":"Apr–May","flowering":"Jul","harvest":"Sep–Oct",
                  "water_peak":"Jun–Aug","notes":"Needs warm nights; sensitive to drought at silking"},
        "rice":  {"plant":"May","transplant":"Jun","harvest":"Sep",
                  "water_peak":"Jun–Aug","notes":"Flooded paddies; Kunduz/Baghlan specialty"},
        "cotton":{"plant":"Apr","boll_open":"Sep","harvest":"Oct–Nov",
                  "water_peak":"Jun–Aug","notes":"Very sensitive to water stress at flowering"},
        "saffron":{"plant":"Sep (corms)","flower":"Oct–Nov","dormant":"Dec–Mar",
                   "harvest":"Oct–Nov","notes":"Perennial; harvested at dawn in 2 weeks window"},
        "barley":{"plant":"Oct–Nov","harvest":"Apr–May",
                  "water_peak":"Feb–Mar","notes":"More drought-tolerant than wheat"},
        "potato":{"plant":"Mar–Apr (spring) or Jul (autumn)","harvest":"Jun–Jul or Oct",
                  "water_peak":"May–Jun","notes":"Needs well-drained sandy loam; 2 seasons possible"},
        "tomato":{"plant":"Mar–Apr","harvest":"Jun–Sep",
                  "water_peak":"Jun–Aug","notes":"High water requirement; support structures needed"},
        "almond":{"flower":"Feb–Mar","harvest":"Aug–Sep",
                  "notes":"Late frost risk during flowering is the main threat"},
        "grape": {"bud_break":"Apr","harvest":"Aug–Sep",
                  "notes":"Khorasan/Kandahar varieties; sun-drying for raisins (kishmish)"},
        "vegetables": {"plant":"Mar–Apr","harvest":"Jun–Sep",
                       "water_peak":"May–Aug","notes":"Mixed; irrigate every 3–5 days in summer"},
    }

    month = datetime.now().month
    cal   = CALENDARS.get(crop) or CALENDARS.get("wheat")
    return {
        "crop":     crop or "unknown",
        "province": province,
        "country":  country,
        "calendar": cal,
        "current_month": month,
        "current_stage": _current_stage(crop, month),
    }


def _current_stage(crop: str, month: int) -> str:
    stages = {
        "wheat":  {11:"planting",12:"germination",1:"tillering",2:"tillering",
                   3:"stem extension",4:"flowering",5:"grain fill",6:"harvest"},
        "maize":  {4:"planting",5:"germination",6:"vegetative",7:"tasseling",
                   8:"grain fill",9:"maturity",10:"harvest"},
        "rice":   {5:"nursery",6:"transplant",7:"vegetative",8:"heading",9:"harvest"},
        "cotton": {4:"planting",5:"germination",6:"vegetative",7:"squaring",
                   8:"flowering",9:"boll open",10:"harvest"},
    }
    crop_stages = stages.get(crop, stages["wheat"])
    return crop_stages.get(month, "dormant/off-season")


def _compare_to_regional_average(inputs: dict, ctx: dict) -> dict:
    field_ndvi = inputs["field_ndvi"]
    province   = inputs.get("province", "")
    country    = inputs.get("country", "Afghanistan")
    month      = inputs.get("month", datetime.now().month)

    # Regional NDVI averages by month (simplified from historical GEE data)
    REGIONAL_AVG = {
        "Kunduz":  {1:.12,2:.18,3:.28,4:.38,5:.42,6:.30,7:.22,8:.18,9:.15,10:.14,11:.13,12:.11},
        "Kabul":   {1:.08,2:.10,3:.18,4:.28,5:.32,6:.25,7:.20,8:.18,9:.15,10:.12,11:.10,12:.08},
        "Balkh":   {1:.10,2:.15,3:.25,4:.35,5:.40,6:.28,7:.20,8:.16,9:.14,10:.12,11:.10,12:.09},
        "default": {1:.10,2:.14,3:.22,4:.32,5:.36,6:.26,7:.20,8:.16,9:.14,10:.12,11:.10,12:.09},
    }
    avg_map    = REGIONAL_AVG.get(province, REGIONAL_AVG["default"])
    regional   = avg_map.get(month, 0.20)
    diff       = round(field_ndvi - regional, 3)
    pct_diff   = round(diff / regional * 100, 1) if regional else 0

    return {
        "field_ndvi":     field_ndvi,
        "regional_avg":   regional,
        "difference":     diff,
        "pct_vs_average": pct_diff,
        "status": "above average" if diff > 0.03 else
                  "below average" if diff < -0.03 else "near average",
        "province": province, "month": month,
    }


def _detect_crop_type(inputs: dict, ctx: dict) -> dict:
    coords = inputs.get("coords", [])
    year   = inputs.get("year", datetime.now().year)
    gee_analyse = ctx.get("gee_analyse_officer")
    if not gee_analyse:
        return {"error": "GEE not available"}
    try:
        r = gee_analyse(coords, year, "", "")
        detect_fn = ctx.get("detect_crop_fn")
        if detect_fn:
            crop = detect_fn(r.get("ndvi",0), r.get("evi",0), r.get("lswi",0),
                             r.get("savi",0), datetime.now().month)
        else:
            ndvi = r.get("ndvi", 0)
            crop = "wheat" if 0.25 < ndvi < 0.55 else \
                   "vegetables" if ndvi > 0.55 else \
                   "bare/fallow" if ndvi < 0.15 else "mixed crops"
        return {"estimated_crop": crop, "ndvi": r.get("ndvi"),
                "confidence": "moderate", "source": "spectral indices"}
    except Exception as e:
        return {"error": str(e)}


def _get_farmer_fields(inputs: dict, ctx: dict) -> dict:
    farmer_id = inputs["farmer_id"]
    db_fn = ctx.get("get_farmer_fields_fn")
    if not db_fn:
        return {"error": "Database not available"}
    try:
        fields = db_fn(farmer_id)
        return {"fields": fields, "count": len(fields)}
    except Exception as e:
        return {"error": str(e)}


def _save_field_recommendation(inputs: dict, ctx: dict) -> dict:
    save_fn = ctx.get("save_alert_fn")
    if not save_fn:
        return {"saved": False, "reason": "database not available"}
    try:
        save_fn(
            farmer_id = inputs["farmer_id"],
            field_id  = inputs.get("field_id"),
            message   = inputs["message"],
            severity  = inputs.get("severity", "info"),
            language  = inputs.get("language", "en"),
        )
        return {"saved": True, "farmer_id": inputs["farmer_id"]}
    except Exception as e:
        return {"saved": False, "error": str(e)}


def _get_agronomic_knowledge(inputs: dict, ctx: dict) -> dict:
    topic    = inputs.get("topic", "")
    crop     = inputs.get("crop", "")
    language = inputs.get("language", "en")

    KB = {
        "yellow leaves wheat":   "Common causes: nitrogen deficiency (pale yellow, older leaves first), iron deficiency (young leaves interveinal chlorosis), waterlogging (uniform yellowing), rust disease (yellow pustules). Check SAR for waterlogging, NDVI for severity.",
        "water stress cotton":   "Cotton is most sensitive to water stress during squaring (60–80 days after planting) and flowering. NDVI drop > 0.10 in 2 weeks during this stage indicates critical stress. Irrigate immediately — delay >5 days risks 20–30% yield loss.",
        "wheat rust":            "Wheat rust (yellow, brown, black stripe) spreads rapidly in warm+humid conditions. NDVI drop is fast (< 1 week). Fungicide must be applied at first sign. Scout fields after rain + wind from Pakistan/Iran direction.",
        "saffron harvest":       "Saffron flowers open for only 2–4 weeks (Oct–Nov). Each flower lasts <24h. Harvest must happen at dawn before petals open. Stigmas are 3 per flower — remove by hand. Yield: 150–200 flowers per gram dried saffron.",
        "nitrogen deficiency":   "Symptoms: pale/yellow leaves starting with older growth, reduced tillering in wheat. Apply urea (46%N) at 50–100 kg/ha split in two applications. First application at tillering, second at stem extension.",
        "drip irrigation":       "Drip irrigation saves 40–50% water vs flood irrigation. Critical for cotton, vegetables, orchard crops. Investment cost recovers in 2–3 seasons from water and fertiliser savings.",
        "soil salinity":         "High soil salinity (EC > 4 dS/m) shows as white crust, sparse patchy growth in satellite images. Leach with excess irrigation if drainage is available. Plant salt-tolerant crops (barley, cotton) while remediating.",
        "frost risk":            "Afghanistan frost risk: Oct–Apr in northern provinces (Kunduz, Balkh). Spring frost (Mar–Apr) is most damaging — catches wheat at flowering. MODIS LST < 0°C triggers frost risk alert. Row covers for vegetables; nothing can be done for field crops.",
        "default":               "Satellite data shows the 'what' — field visits confirm the 'why'. Use NDVI trend + seasonal calendar + local knowledge for diagnosis. When uncertain, recommend field scout followed by soil/tissue test.",
    }
    # Find best matching knowledge
    topic_lower = topic.lower()
    answer = next((v for k,v in KB.items() if any(w in topic_lower for w in k.split())),
                  KB["default"])
    return {"topic": topic, "knowledge": answer, "crop": crop}


def _calculate_field_health_score(inputs: dict, ctx: dict) -> dict:
    ndvi      = inputs.get("ndvi", 0)
    sar_vh    = inputs.get("sar_vh", -15)
    rain_mm   = inputs.get("rain_mm", 250)
    ndvi_trend= inputs.get("ndvi_trend", 0)
    month     = inputs.get("month", datetime.now().month)

    # Weighted score
    ndvi_score  = min(100, int(ndvi * 150))        # NDVI 0→0, 0.67→100
    moist_score = min(100, int((sar_vh + 25) * 4)) # VH -25→0, 0→100
    rain_score  = min(100, int(rain_mm / 5))       # 500mm→100
    trend_score = min(100, max(0, 50 + int(ndvi_trend * 200)))

    score = int(0.45*ndvi_score + 0.25*moist_score + 0.15*rain_score + 0.15*trend_score)
    risk  = "LOW" if score >= 70 else "MEDIUM" if score >= 45 else "HIGH" if score >= 25 else "CRITICAL"

    return {
        "health_score":  score,
        "risk_level":    risk,
        "components":    {"ndvi": ndvi_score, "moisture": moist_score,
                          "rainfall": rain_score, "trend": trend_score},
        "interpretation": (
            "Excellent — field is healthy and productive." if score >= 80 else
            "Good — minor concerns, monitor closely." if score >= 65 else
            "Moderate — investigate irrigation or nutrient issues." if score >= 45 else
            "Poor — significant stress detected, immediate action needed." if score >= 25 else
            "Critical — severe degradation or crop failure risk."
        )
    }


_TOOL_MAP = {
    "query_satellite_data":         _query_satellite_data,
    "get_ndvi_trend":               _get_ndvi_trend,
    "get_land_cover":               _get_land_cover,
    "get_monthly_rainfall":         _get_monthly_rainfall,
    "get_soil_data":                _get_soil_data,
    "get_crop_calendar":            _get_crop_calendar,
    "compare_to_regional_average":  _compare_to_regional_average,
    "detect_crop_type":             _detect_crop_type,
    "get_farmer_fields":            _get_farmer_fields,
    "save_field_recommendation":    _save_field_recommendation,
    "get_agronomic_knowledge":      _get_agronomic_knowledge,
    "calculate_field_health_score": _calculate_field_health_score,
}
