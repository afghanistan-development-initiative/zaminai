# ZaminAI — Complete Technical Reference

**Version 8.0** · Flask API + PWA Frontend · Production: https://zaminai.onrender.com  
**Organization:** Afghanistan Development Initiative (ADI)  
**Author:** Maiwand Jan Alamzoi

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Satellite Data Sources](#4-satellite-data-sources)
5. [Remote Sensing Indices & Formulas](#5-remote-sensing-indices--formulas)
6. [AI & Machine Learning Models](#6-ai--machine-learning-models)
7. [RAG / Vector Search System](#7-rag--vector-search-system)
8. [Computer Vision Pipeline (4-Mode)](#8-computer-vision-pipeline-4-mode)
9. [Multi-Agent System](#9-multi-agent-system)
10. [Regenerative Agriculture Module](#10-regenerative-agriculture-module)
11. [Afghan-Specific Data & Knowledge](#11-afghan-specific-data--knowledge)
12. [API Endpoints Reference](#12-api-endpoints-reference)
13. [Database Schema (Supabase/PostgreSQL)](#13-database-schema-supabasepostgresql)
14. [Multilingual Support](#14-multilingual-support)
15. [Frontend Architecture (PWA)](#15-frontend-architecture-pwa)
16. [Graceful Degradation & Fallback Strategy](#16-graceful-degradation--fallback-strategy)
17. [Deployment & Infrastructure](#17-deployment--infrastructure)
18. [Key Calculations Explained](#18-key-calculations-explained)

---

## 1. System Overview

ZaminAI is a **satellite farming intelligence platform** designed specifically for Afghan smallholder farmers. It combines space-based remote sensing, large language models (LLMs), computer vision, and a mobile-first web interface to deliver agricultural advice in Dari, Pashto, and English — languages accessible to farmers who may be illiterate or semi-literate.

### Core Problem Solved

Afghan smallholder farmers (average 1.5–2.5 ha / 7–12 jeribs) lack access to agronomic expertise, timely weather data, and soil information. ZaminAI delivers satellite-powered field intelligence without the farmer needing internet infrastructure beyond basic mobile data.

### Key Capabilities

| Capability | Technology Used |
|---|---|
| Draw field boundary on map → get satellite NDVI, moisture, rainfall | Google Earth Engine + Sentinel-2 |
| Ask farming questions in Dari/Pashto/English | Gemini LLM + RAG knowledge base |
| Upload crop photo → disease/pest/yield/soil diagnosis | Claude Haiku Vision + YOLOv8m |
| Historical NDVI trend (2013–present) | Sentinel-2 + Landsat 8/9 |
| 7-day weather forecast | Open-Meteo API (free, no key) |
| Regenerative crop rotation recommendations | Rule engine + cultivation history |
| Telegram push alerts for NDVI drops, drought | Telegram Bot API |
| Field officer regional dashboard | Google Earth Engine officer analysis |
| Global admin boundary mapping | FAO/GAUL via GEE + GADM fallback |
| Agentic satellite Q&A (multi-tool Claude) | Claude Sonnet 4.6 + tools system |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (PWA)                           │
│  index.html — Farmer App (Leaflet map, draw, diagnose)      │
│  officer.html — Field Officer Dashboard                     │
│  agent.html — Agentic Chat UI                               │
│  (No build step — all vanilla JS + CSS)                     │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP/REST
┌────────────────────▼────────────────────────────────────────┐
│               Flask API (app.py — single file v8.0)         │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Data Layer   │  │  AI Layer    │  │  Database Layer  │  │
│  │              │  │              │  │                  │  │
│  │ gee_analyse()│  │ call_gemini()│  │ db_* functions  │  │
│  │ GEE API      │  │ Claude Haiku │  │ Supabase/PG     │  │
│  │ Sentinel-2   │  │ YOLOv8m      │  │ pgvector RAG    │  │
│  │ Landsat 8/9  │  │ smart_fallbck│  │ cultivation_    │  │
│  │ Sentinel-1   │  │ RAG/pgvector │  │ history         │  │
│  │ MODIS        │  │ Agents SDK   │  │                 │  │
│  │ CHIRPS rain  │  │              │  │                 │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Async Task Pattern (all GEE calls)                 │   │
│  │  POST → returns {task_id} → GET polls until done    │   │
│  │  (Prevents Render's 30s proxy timeout)              │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │
     ┌───────────────┼────────────────────────────┐
     ▼               ▼                            ▼
 Google Earth    Supabase              Telegram / Open-Meteo
 Engine API   (PostgreSQL+pgvector)    / SoilGrids REST
```

### Async Task Pattern

Every Google Earth Engine call runs in a background thread because GEE processing takes 30–120 seconds (longer than Render's proxy timeout). The pattern:

1. `POST /analyse` → immediately returns `{"task_id": "uuid"}`
2. Background thread runs GEE analysis
3. Client polls `GET /analyse-result/<task_id>` every 3 seconds
4. When complete, returns full satellite data

This applies to: `/analyse`, `/officer/analyse`, `/officer/detect-fields`, `/officer/layer/*`, `/gadm/*`, `/agent/chat`.

---

## 3. Technology Stack

### Python Dependencies (`requirements.txt`)

| Package | Version | Purpose |
|---|---|---|
| `flask` | 3.0.0 | Web framework |
| `flask-cors` | 4.0.0 | Cross-origin requests |
| `gunicorn` | 22.0.0 | Production WSGI server |
| `earthengine-api` | 0.1.390 | Google Earth Engine Python client |
| `google-generativeai` | ≥0.7.0 | Gemini LLM API |
| `anthropic` | ≥0.40.0 | Claude API (vision + agents) |
| `supabase` | ≥2.6.0 | PostgreSQL / vector DB client |
| `numpy` | 1.26.4 | Vector math for cosine similarity |
| `scikit-learn` | 1.4.0 | ML utilities |
| `xgboost` | 2.0.3 | Gradient boosting (optional) |
| `ultralytics` | ≥8.3.0 | YOLOv8 plant disease detection |
| `Pillow` | ≥10.2.0 | Image processing |
| `huggingface-hub` | ≥0.24.0 | Model weights download |
| `requests` | 2.31.0 | HTTP client |
| `pandas` | 2.2.0 | Data manipulation |
| `streamlit` | ≥1.35.0 | Field assistant UI |
| `python-dotenv` | ≥1.0.0 | Environment variable loading |

### Environment Variables Required

| Variable | Description |
|---|---|
| `GEE_SERVICE_ACCOUNT` | Google Earth Engine service account email |
| `GEE_PRIVATE_KEY` | GEE private key JSON (newlines as `\n`) |
| `GEMINI_API_KEY` | Google Gemini API (LLM + embeddings) |
| `ANTHROPIC_API_KEY` | Claude API (vision diagnosis + agents) |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase service role key |
| `TELEGRAM_BOT_TOKEN` | Telegram bot for push alerts (optional) |
| `DISABLE_YOLO` | Set to `1` to skip YOLOv8 loading (for 512MB hosts) |

---

## 4. Satellite Data Sources

ZaminAI queries four separate satellite instruments in sequence, each providing different information:

### 4.1 Sentinel-2 SR (Primary Optical)

- **Satellite:** ESA Copernicus Sentinel-2A/2B twin satellites
- **GEE Collection:** `COPERNICUS/S2_SR_HARMONIZED`
- **Resolution:** 10 m (visible/NIR), 20 m (SWIR/red-edge)
- **Revisit time:** 5 days (twin satellites)
- **Coverage:** 2017–present
- **Cloud filter:** `CLOUDY_PIXEL_PERCENTAGE < 20%`
- **Processing:** Median composite of best images in growing season
- **What it provides:** NDVI, EVI, SAVI, MNDWI, LSWI, NDRE, BSI
- **Bands used:**

| Band | Name | Wavelength | Resolution |
|---|---|---|---|
| B2 | Blue | 490 nm | 10 m |
| B3 | Green | 560 nm | 10 m |
| B4 | Red | 665 nm | 10 m |
| B5 | Red Edge 1 | 705 nm | 20 m |
| B8 | NIR | 842 nm | 10 m |
| B8A | Red Edge 4 | 865 nm | 20 m |
| B11 | SWIR 1 | 1610 nm | 20 m |

### 4.2 Landsat 8/9 (Historical Optical)

- **Satellite:** NASA/USGS Landsat 8 (2013–) and Landsat 9 (2021–)
- **GEE Collection:** `LANDSAT/LC08/C02/T1_L2` (Collection 2 Level-2)
- **Resolution:** 30 m
- **Revisit time:** 16 days
- **Coverage:** 2013–2018 (fills pre-Sentinel era for trend analysis)
- **Scale factor:** `pixel_value × 0.0000275 + (−0.2)` (Collection 2 SR scaling)
- **Bands used:** SR_B2 (Blue), SR_B3 (Green), SR_B4 (Red), SR_B5 (NIR), SR_B6 (SWIR1)

### 4.3 Sentinel-1 SAR (Radar, Cloud-Free)

- **Satellite:** ESA Copernicus Sentinel-1A/1B
- **GEE Collection:** `COPERNICUS/S1_GRD`
- **Resolution:** 10 m
- **Mode:** IW (Interferometric Wide Swath)
- **Polarization:** VV + VH dual-pol
- **Key advantage:** C-band radar penetrates clouds and smoke — works year-round
- **What it provides:** Soil moisture proxy (VV/VH backscatter in dB)
- **Filter:** `instrumentMode = IW`, both VV and VH channels required

### 4.4 MODIS MOD11A2 (Land Surface Temperature)

- **Satellite:** NASA Terra MODIS
- **GEE Collection:** `MODIS/061/MOD11A2`
- **Resolution:** 1 km
- **Temporal resolution:** 8-day composite
- **Scale factor:** `pixel_value × 0.02 − 273.15` → result in °C
- **Bands:** `LST_Day_1km`, `LST_Night_1km`
- **What it provides:** Summer/winter day+night temperatures, frost risk detection

### 4.5 CHIRPS Daily Precipitation

- **Source:** UC Santa Barbara Climate Hazards Group
- **GEE Collection:** `UCSB-CHG/CHIRPS/DAILY`
- **Resolution:** ~5 km
- **Coverage:** 1981–present
- **What it provides:** Annual rainfall sum (mm) from daily precipitation accumulation
- **Calculation:** `.sum()` of daily precipitation over the year

### 4.6 Additional Data Sources (Officer Dashboard)

| Source | GEE Collection | What it provides |
|---|---|---|
| WorldPop | `WorldPop/GP/100m/pop` | Population count and density (100m, 2000–2020) |
| Dynamic World | `GOOGLE/DYNAMICWORLD/V1` | Land cover classification: 9 classes at 10m |
| SRTM Terrain | `USGS/SRTMGL1_003` | Elevation (m) and slope (degrees) at 30m |
| JRC Surface Water | `JRC/GSW1_4/GlobalSurfaceWater` | Permanent water body percentage |
| FAO/GAUL 2015 | `FAO/GAUL/2015/level1/2/3` | Global admin boundaries |

---

## 5. Remote Sensing Indices & Formulas

All indices are computed by Google Earth Engine using `reduceRegion(ee.Reducer.mean())` over the field polygon and returned as scalar values.

### 5.1 NDVI — Normalized Difference Vegetation Index

```
NDVI = (NIR − RED) / (NIR + RED)
     = (B8 − B4) / (B8 + B4)
```

**Range:** −1 to +1  
**Interpretation:**
| Value | Meaning |
|---|---|
| < 0.10 | Bare soil / rock — no active vegetation |
| 0.10–0.20 | Very sparse — severe drought or seedling stage |
| 0.20–0.40 | Moderate vegetation — dryland cereals in semi-arid regions |
| 0.40–0.60 | Healthy dense crop — well-irrigated or established orchard |
| > 0.60 | Very dense canopy — tropical or heavily irrigated |

**Source:** Tucker (1979), Remote Sensing of Environment; NASA MODIS Land Team

### 5.2 EVI — Enhanced Vegetation Index

```
EVI = 2.5 × (NIR − RED) / (NIR + 6×RED − 7.5×BLUE + 1)
    = 2.5 × (B8 − B4) / (B8 + 6×B4 − 7.5×B2 + 1)
```

**Advantage over NDVI:** Corrects for atmospheric aerosols and soil background signal. Does not saturate at high biomass (NDVI saturates > 0.8). Better for dense tropical crops.

### 5.3 SAVI — Soil-Adjusted Vegetation Index

```
SAVI = ((NIR − RED) / (NIR + RED + 0.5)) × 1.5
     = ((B8 − B4) / (B8 + B4 + 0.5)) × 1.5
```

**Purpose:** Corrects for soil brightness effects in semi-arid and sparse vegetation areas (like Afghanistan). L=0.5 is used as the soil correction factor.

### 5.4 MNDWI — Modified Normalized Difference Water Index

```
MNDWI = (GREEN − SWIR1) / (GREEN + SWIR1)
       = (B3 − B11) / (B3 + B11)
```

**Range:** −1 to +1  
**Interpretation:**
| Value | Meaning |
|---|---|
| < −0.25 | Severe soil water deficit — irrigate within 2–3 days |
| −0.25 to −0.10 | Moderate water stress — irrigate within 4–7 days |
| −0.10 to 0.00 | Mild stress — irrigate within 7–10 days |
| > 0.00 | Adequate soil moisture or recent rainfall |

**Source:** Xu (2006), International Journal of Remote Sensing

### 5.5 LSWI — Land Surface Water Index

```
LSWI = (NIR − SWIR1) / (NIR + SWIR1)
     = (B8 − B11) / (B8 + B11)
```

**Purpose:** Sensitive to leaf water content and canopy moisture. High LSWI indicates high leaf water content. Used in crop type detection (discriminates rice from other crops).

### 5.6 NDRE — Normalized Difference Red Edge

```
NDRE = (RedEdge4 − RedEdge1) / (RedEdge4 + RedEdge1)
     = (B8A − B5) / (B8A + B5)
```

**Purpose:** Chlorophyll content indicator. More sensitive than NDVI for detecting early stress before visible symptoms appear. Useful for precision nitrogen management.

### 5.7 BSI — Bare Soil Index

```
BSI = ((SWIR1 + RED) − (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))
    = ((B11 + B4) − (B8 + B2)) / ((B11 + B4) + (B8 + B2))
```

**Purpose:** Detects exposed bare soil. Positive values = bare/eroded soil. Used in land degradation monitoring and desertification detection.

### 5.8 VCI — Vegetation Condition Index

```
VCI = (NDVI_current − NDVI_min) / (NDVI_max − NDVI_min) × 100
```

Computed in Python from the multi-year NDVI trend data:
```python
vci = round((cur - h_min) / (h_max - h_min + 0.001) * 100, 1)
```

**Purpose:** Drought early warning. Normalizes NDVI against the pixel's historical range so it accounts for natural seasonal variability.

**Interpretation (NOAA/NESDIS, FAO GIEWS):**
| VCI | Drought Status |
|---|---|
| < 10 | Extreme drought — crop failure likely |
| 10–35 | Severe drought — 40–70% yield loss |
| 35–50 | Moderate drought — 20–40% yield loss |
| 50–75 | Near-normal conditions |
| > 75 | Above-average vegetation |

### 5.9 SAR Backscatter (Sentinel-1)

| Parameter | Formula | Meaning |
|---|---|---|
| VV (dB) | Direct measurement | Primary soil moisture proxy |
| VH (dB) | Direct measurement | Vegetation volume scattering |
| VH/VV ratio | `VH_db − VV_db` | Canopy roughness, vegetation density |

**SAR soil moisture thresholds (Wagner et al. 1999):**
- VV < −15 dB: dry soil, high water deficit
- VV −15 to −8 dB: moderate soil moisture
- VV > −8 dB: wet soil or near-surface water

### 5.10 MODIS Land Surface Temperature

```
LST_Celsius = pixel_value × 0.02 − 273.15
```

**Frost risk thresholds for winter wheat (Porter & Gawith 1999):**
- Tillering: < −5°C for 2+ hours → leaf damage
- Stem extension: < −2°C → significant tiller death
- Anthesis (flowering): < 0°C for 2 hours → sterility, poor grain set

---

## 6. AI & Machine Learning Models

### 6.1 Claude Haiku 4.5 Vision (Primary Crop Diagnosis)

- **Model ID:** `claude-haiku-4-5-20251001`
- **Provider:** Anthropic
- **Use case:** 4-mode crop photo analysis (Disease / Pest / Yield / Soil)
- **Input:** Base64-encoded image + structured prompt
- **Output:** Structured 5-section diagnosis (problem name, severity, steps, treatment, prevention)
- **Max tokens:** 600
- **Languages:** English, Dari (Afghan Persian), Pashto
- **Why Haiku:** Fastest vision model; sufficient for structured diagnosis tasks; low latency for mobile

### 6.2 Gemini LLM Cascade (Farming Q&A)

- **Function:** `call_gemini(prompt)` — tries 3 models in sequence:
  1. `gemini-1.5-flash` (fastest, free tier)
  2. `gemini-2.0-flash` (better reasoning)
  3. `gemini-2.5-flash` (most capable)
- **Use case:** `/ask` endpoint — farmer questions in Dari/Pashto/English
- **Fallback:** If all models fail, `smart_fallback()` provides rule-based responses
- **RAG augmentation:** Top-4 knowledge chunks prepended to prompt before Gemini call

### 6.3 Gemini Vision Fallback (Crop Diagnosis Fallback)

- **Use case:** `/diagnose` fallback if Anthropic key unavailable
- **Models tried:** `gemini-2.0-flash` → `gemini-2.5-flash` → `gemini-flash-latest`
- **Same structured prompt** as Claude Haiku

### 6.4 Claude Sonnet 4.6 (Agentic System)

- **Model:** `claude-sonnet-4-6` (in agents/orchestrator)
- **Use case:** `/agent/chat` — multi-step satellite analysis with tool use
- **Capability:** Orchestrates satellite queries, soil lookups, field analysis, alert writing
- **Tools available:** 12 tools including GEE analysis, CHIRPS rain, SoilGrids, field lookup
- **Fallback:** `run_gemini_agent()` if no Anthropic key

### 6.5 Gemini Embedding Model (RAG / Vector Search)

- **Model:** `gemini-embedding-2` (also tried: `text-embedding-004`)
- **Embedding dimension:** 768
- **Use case:** Encode farming knowledge chunks + farmer questions for semantic retrieval
- **Storage:** Supabase PostgreSQL with `pgvector` extension
- **Similarity:** Cosine similarity computed in Python (numpy)

### 6.6 YOLOv8m Plant Disease Detection

- **Model:** YOLOv8m (medium variant) fine-tuned on plant disease dataset
- **Classes:** 38 disease classes
- **Input:** Resized + encoded crop image
- **Output:** Bounding boxes + class labels + confidence scores
- **Status:** Disabled on Render via `DISABLE_YOLO=1` (512MB RAM limit)
- **Local:** Runs when `DISABLE_YOLO` is not set and `crop_vision.py` module is present

### 6.7 Rule-Based Crop Type Detection (`detect_crop()`)

```python
# Pixel-level classification rules (applied as raster in GEE officer mode)
is_bare  = ndvi < 0.12
is_wheat = (0.25 ≤ ndvi ≤ 0.60) AND (evi < 0.38) AND (month in 3-7)
is_veg   = (ndvi ≥ 0.38) AND (evi ≥ 0.28) AND (lswi ≥ -0.10)
is_orch  = (ndvi ≥ 0.42) AND (evi ≥ 0.30)
```

The same thresholds are applied both as scalar classification (farmer field) and as pixel-level raster maps (officer village-crops layer).

### 6.8 `smart_fallback()` — Rule-Based AI Fallback

When no AI keys are available, `smart_fallback()` generates responses based on keyword detection in farmer questions:

- **Irrigation questions:** Responds based on MNDWI threshold
- **Crop choice questions:** Responds based on province type and season
- **Fertilizer questions:** Province-type and crop-specific advice
- **Language:** Responds in Dari, Pashto, or English based on `language` parameter

---

## 7. RAG / Vector Search System

RAG = Retrieval-Augmented Generation. ZaminAI embeds farming knowledge into a vector database so the LLM can retrieve precise facts before answering.

### Architecture

```
Farmer question → embed_text() → 768-dim vector
                                      ↓
                              Supabase pgvector table
                              knowledge_chunks (vector index)
                                      ↓
                         _cosine_similarity() top-k=4
                         threshold ≥ 0.50
                                      ↓
                    Inject top-k chunks into Gemini/Claude prompt
                                      ↓
                         LLM answers with verified facts
```

### Technical Details

| Parameter | Value |
|---|---|
| Embedding model | `gemini-embedding-2` |
| Embedding dimension | 768 |
| Similarity metric | Cosine similarity |
| Retrieval threshold | 0.50 (minimum cosine similarity) |
| Top-k chunks retrieved | 4 (Q&A) / 2 (diagnosis) |
| Similarity computation | Python/numpy (not SQL) |
| Storage | Supabase PostgreSQL + `pgvector` extension |
| Index type | IVFFlat with `vector_cosine_ops`, 100 lists |

### Knowledge Base Content (`_RAG_SEED_DOCS`)

The seed knowledge base contains 28+ carefully curated documents from peer-reviewed sources:

| Category | Sources |
|---|---|
| NDVI thresholds | Tucker 1979; NASA MODIS; WUR GIS |
| VCI drought monitoring | Kogan 1990 NOAA; FAO GIEWS; FEWS NET |
| MNDWI irrigation scheduling | Xu 2006; WUR; ESA |
| Sentinel-1 SAR soil moisture | ESA; Ulaby 1978; Wagner 1999 TU Wien |
| MODIS LST frost/heat | NASA; WMO; Porter & Gawith 1999 |
| Afghan wheat production | CIMMYT/ICARDA; FAO 2022; WFP ADAM |
| Wheat sowing calendar | FEWS NET; FAO GIEWS; ICARDA |
| Fertilizer recommendations | ICARDA; CIMMYT; FAO |
| Wheat rust diseases | CIMMYT; ICARDA 2010–2023; APS Press |
| Afghan soil characteristics | ISRIC SoilGrids v2.0; WUR; FAO-UNESCO |
| Soil carbon improvement | ISRIC/WUR; FAO 2017; Minasny et al. 2017 |
| Irrigation systems | FAO AQUASTAT; World Bank; ADB |
| Rainfall climatology | CHIRPS v2.0; FEWS NET; World Bank |
| Crop calendar | FEWS NET 2023; FAO GIEWS; MAIL |
| Saffron cultivation | FAO 2016; USDA GAIN 2019; Gresta 2008 |
| Pomegranate & grape | FAO 2013; USAID VIPA; Sarkhosh 2021 |
| WUR yield gap research | WUR Food Systems; GYGA; van Ittersum 2013 |
| Food security context | WFP FSMS; FAO/WFP CFSAM 2022–2023; IPC |
| Crop rotation systems | ICARDA; FAO 2018; WUR Farming Systems |
| Legume crops | ICARDA; FAO 2016; WUR Plant Sciences |
| Regenerative agriculture | FAO 2017; WUR Agroecology; Rodale; ICARDA |
| Conservation tillage | ICARDA WANA; FAO 2014; WUR Soil Physics |
| Agroforestry | FAO 2017; WUR Forest Ecology; ICARDA |
| IPM | FAO IFS; ICARDA; WUR Entomology |

### Key Functions

```python
embed_text(text: str) → list[float]  # 768-dim embedding via Gemini
rag_store(text, source, metadata) → bool  # store chunk in Supabase
rag_retrieve(query, top_k=4, threshold=0.50) → list[str]  # retrieve top-k
_cosine_similarity(a, b) → float  # numpy dot product similarity
```

---

## 8. Computer Vision Pipeline (4-Mode)

The `/diagnose` endpoint supports four analysis modes selected by the farmer:

```
Mode: disease | pest | yield | soil
```

### Pipeline

```
1. Receive base64 image + mode + language + crop hint

2. Decode base64 → raw bytes

3. Stage 1 (optional): YOLOv8m fast detection
   - 38 disease classes → bounding boxes + confidence
   - Disabled on Render (DISABLE_YOLO=1)

4. Build mode-specific prompt:
   Disease: "Examine this crop photo... 1.Disease name 2.Severity 3.Steps now 4.Product/dose 5.Prevention"
   Pest:    "...1.Pest name+count 2.Infestation level 3.Steps now 4.Pesticide+dose 5.Prevention"
   Yield:   "...1.Crop+growth stage 2.Days until harvest 3.Pre-harvest checks 4.Quality issues 5.Yield tip"
   Soil:    "...1.Soil type+colour 2.Moisture 3.Compaction/erosion signs 4.Amendments+dose/jerib 5.Best crops"

5. RAG retrieval (top-k=2, threshold=0.45)
   - Query: "{crop} {yolo_label or mode} Afghanistan"
   - Inject relevant knowledge chunks into prompt

6. Stage 2a: Claude Haiku 4.5 Vision (preferred)
   - model: claude-haiku-4-5-20251001
   - max_tokens: 600
   - Input: image + structured prompt

7. Stage 2b: Gemini Vision (fallback if no Anthropic key)
   - gemini-2.0-flash → gemini-2.5-flash → gemini-flash-latest

8. Return structured JSON:
   {ok, detections, top_detection, is_healthy, yolo_ok, diagnosis, model, language}
```

### Frontend Parsing (`ydExtractSections()`)

The frontend parses the AI response into 5 numbered sections:

```javascript
// Regex matches "1.", "**1.**", "1:", "1)"
const re = /(?:^|\n)\s*(\d+)[.):]\s*/g
```

| Section | Displayed as |
|---|---|
| Section 1 | Disease/pest name (in result card header) |
| Section 2 | Severity badge (mild/moderate/severe) |
| Section 3 | "Do now" step list |
| Section 4 | Treatment / product box |
| Section 5 | Prevention text |

---

## 9. Multi-Agent System

Located in `agents/` directory. Provides a conversational interface where Claude orchestrates multiple tool calls to answer complex farming questions.

### Architecture

```
User question → /agent/chat (async)
                    ↓
         Load ORCHESTRATOR_PROMPT or OFFICER_AGENT_PROMPT
                    ↓
         RAG retrieval (top-k=3) → inject into system prompt
                    ↓
         Claude Sonnet 4.6 with 12 tools
                    ↓
         Tool calls: query_satellite_data, get_soil_info,
                     get_monthly_rain, detect_crops,
                     get_all_fields, get_farmer_fields,
                     save_alert, ...
                    ↓
         Multi-step reasoning loop (up to N iterations)
                    ↓
         Return final answer + tool_calls log + map_data
```

### Available Tools (12)

| Tool | Description |
|---|---|
| `query_satellite_data` | GEE officer analysis for any polygon |
| `get_monthly_rain` | CHIRPS monthly rainfall for lat/lon/year |
| `get_soil_info` | SoilGrids pH, SOC, clay% |
| `detect_crops` | Rule-based crop classification from indices |
| `get_all_fields` | List all registered farmer fields |
| `get_farmer_fields` | List one farmer's fields |
| `save_alert` | Create a farmer alert in database |
| + 5 more | Province lookup, area calculation, etc. |

### Agentic Routes

| Endpoint | Description |
|---|---|
| `POST /agent/chat` | Async agent chat |
| `GET /agent/result/<task_id>` | Poll for agent result |
| `POST /agent/monitor` | Autonomous field monitoring loop (cron) |
| `POST /agent/weekly-report` | Weekly province intelligence report |
| `GET /agent/history/<session_id>` | Conversation history (last 10 turns) |
| `GET /agent/status` | Agent system status + backend info |

### Session Management

Conversation history is stored in memory (`_agent_sessions` dict), keyed by `session_id`. Last 20 messages (10 turns) are kept. Saved to Supabase `conversations` table if farmer is registered.

---

## 10. Regenerative Agriculture Module

The Regen module lets farmers log their cultivation history and receive science-based crop rotation recommendations to improve soil health over time.

### Key Data Structures

#### `CROP_ROTATION_RULES` (12 crops)

For each crop: `next_best`, `next_good`, `avoid`, and trilingual explanations.

Example (wheat):
```python
"wheat": {
    "next_best": ["chickpea", "mung_bean", "lentil"],
    "next_good": ["vegetables", "sunflower"],
    "avoid": ["wheat"],
    "regen_reason": "Legumes fix nitrogen after wheat exhausts soil N — reduces fertilizer need 30–40%."
}
```

#### `CROP_VALUE_TABLE` (11 crops)

For each crop: income (USD/ha), yield (kg/ha), water requirement, soil benefit, market, regen score (1–5).

| Crop | Income (USD/ha) | Water | Soil Benefit | Regen Score |
|---|---|---|---|---|
| Saffron | 3,000–8,000 | Low | Medium | 4/5 |
| Chickpea | 250–500 | Low | High (N-fix) | 5/5 |
| Lentil | 200–400 | Low | High (N-fix) | 5/5 |
| Mung Bean | 300–500 | Low | High (N-fix) | 5/5 |
| Vegetables | 800–2,000 | High | Low | 2/5 |
| Pomegranate | 500–1,500 | Medium | Medium | 3/5 |
| Wheat | 300–600 | Medium | Low | 2/5 |
| Cotton | 400–700 | High | Depletes | 1/5 |
| Rice | 400–900 | Very High | Low | 1/5 |

#### `regen_build_recommendation()` Logic

1. Get last crop from cultivation history (or current_crop parameter)
2. Look up rotation rules for that crop
3. Filter recommendations by province type (north/south/east/west/central)
4. If annual rain < 150mm: sort by water requirement (lowest first)
5. Return top-3 "best" + top-2 "good" crops with full value table data

### Database Table: `cultivation_history`

```sql
CREATE TABLE cultivation_history (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  farmer_id    uuid,
  field_id     uuid,
  crop         TEXT NOT NULL,
  season       TEXT,
  year         INTEGER,
  notes        TEXT,
  source       TEXT DEFAULT 'farmer_reported',
  ndvi_at_peak FLOAT,
  province     TEXT,
  created_at   TIMESTAMP DEFAULT NOW()
);
```

### Regen API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/regen/log` | POST | Log a crop for a field/year |
| `/regen/history/<farmer_id>` | GET | Get cultivation history |
| `/regen/recommend` | POST | Get crop rotation recommendations |
| `/regen/setup` | GET | SQL migration for cultivation_history table |

---

## 11. Afghan-Specific Data & Knowledge

### 11.1 PROVINCES — 16 Afghan Provinces Database

Each province entry contains:
- Bounding box (`lat_min`, `lat_max`, `lon_min`, `lon_max`)
- Baseline indices: `ndvi`, `evi`, `savi`, `mndwi`, `lswi`
- Annual rainfall (`rain` mm)
- Historical NDVI trend dict: 2019–2025 (year → scalar)
- Province type: `north`, `south`, `east`, `west`, `central`

Provinces covered: Kunduz, Baghlan, Balkh, Herat, Nangarhar, Kabul, Kandahar, Helmand, Takhar, Badakhshan, Bamyan, Ghazni, Faryab, Logar, Wardak, Zabul.

### 11.2 AFGHAN_SOILS — Province-Level Soil Database

```python
AFGHAN_SOILS = {
    "Kunduz": {"texture":"silty loam","ph":7.2,"organic_matter":1.1,...},
    "Kandahar": {"texture":"sandy loam","ph":7.8,"organic_matter":0.5,...},
    # ... 16 provinces
}
```

Each entry: soil texture, pH, organic matter %, sand/silt/clay %, fertility rating, dominant soil type, and management recommendations.

### 11.3 CROP_CALENDAR — Planting & Harvest by Province Type

```python
CROP_CALENDAR = {
    "north": {
        "wheat": {
            "sow": ["Oct", "Nov"],
            "harvest": ["Jun", "Jul"],
            "irrigation_events": 4
        },
        ...
    },
    "south": {...},
    "east": {...},
    "west": {...},
    "central": {...}
}
```

### 11.4 MONTHLY_RAIN_FRACTION — Seasonal Distribution

Province-zone-specific monthly rainfall fractions (sums to 1.0). Multiplied by annual rain from CHIRPS to give monthly breakdown. Afghan pattern: 75–85% falls October–April (Mediterranean winter regime).

### 11.5 Land Unit Conversion

```python
# 1 jerib = 0.2 hectares (Afghan standard land unit)
area_jereb = round(area_ha * 5, 1)
```

### 11.6 Crop Calendar (National Summary)

| Crop | Region | Sow | Harvest |
|---|---|---|---|
| Winter wheat | North | Oct–Nov | Jun–Jul |
| Winter wheat | South/West | Nov–Dec | Apr–May |
| Winter wheat | East | Oct–Nov | May–Jun |
| Spring barley | Highlands >1500m | Mar–Apr | Aug–Sep |
| Maize | Nangarhar, Laghman | Apr–May | Aug–Sep |
| Rice | Kunduz, Baghlan, Nangarhar | May–Jun | Sep–Oct |
| Saffron | Herat, Ghor, Farah | Aug–Sep (corms) | Oct–Nov (flowers) |
| Potato | Lowlands | Apr–May | Aug–Sep |
| Cotton | Kunduz, Baghlan, Balkh | Apr–May | Oct–Nov |
| Chickpea | All | Feb–Mar | May–Jun |
| Pomegranate | Kandahar, Logar | Established | Aug–Oct |

---

## 12. API Endpoints Reference

### Farmer App Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service status, feature flags (gee_ok, ai_ok, db_ok) |
| `/analyse` | POST | Start async field satellite analysis → returns `{task_id}` |
| `/analyse-result/<task_id>` | GET | Poll for analysis result |
| `/ask` | POST | AI Q&A with RAG (Dari/Pashto/English) |
| `/diagnose` | POST | 4-mode crop photo diagnosis |
| `/crop_detect` | POST | Rule-based crop type from satellite indices |
| `/monthly_rain` | POST | Monthly rainfall breakdown for province |
| `/ndvi_tile` | POST | NDVI thumbnail PNG URL from GEE |
| `/soil` | POST | SoilGrids + Afghan soil DB query |
| `/weather` | POST | 7-day forecast via Open-Meteo (free, no key) |

### Database / Farmer Registration

| Endpoint | Method | Description |
|---|---|---|
| `/db/farmer` | POST | Register or retrieve farmer by phone |
| `/db/field/save` | POST | Save drawn field polygon |
| `/db/field/delete` | POST | Delete a field |
| `/db/fields/<farmer_id>` | GET | List farmer's fields with latest analyses |
| `/db/analysis/save` | POST | Save a satellite analysis record |
| `/db/chat/save` | POST | Save a conversation turn |

### Alerts & Telegram

| Endpoint | Method | Description |
|---|---|---|
| `/alerts/save` | POST | Create NDVI/moisture/rain threshold alert |
| `/alerts/<farmer_id>` | GET | List active alerts |
| `/alerts/delete` | POST | Remove an alert |
| `/alerts/check` | POST | Check which alerts fire against current readings |
| `/alerts/daily` | POST/GET | Run daily alert checks (cron trigger) |
| `/telegram/webhook` | POST | Receive Telegram messages |
| `/telegram/setup` | GET | Register webhook URL with Telegram |

### Regenerative Agriculture

| Endpoint | Method | Description |
|---|---|---|
| `/regen/log` | POST | Log crop for a field/year |
| `/regen/history/<farmer_id>` | GET | Cultivation history |
| `/regen/recommend` | POST | Crop rotation recommendations |
| `/regen/setup` | GET | SQL migration SQL |

### RAG / Knowledge Base

| Endpoint | Method | Description |
|---|---|---|
| `/rag/setup` | GET | Returns pgvector migration SQL |
| `/rag/seed` | POST | Load built-in Afghan farming knowledge |
| `/rag/ingest` | POST | Add custom knowledge chunks |
| `/rag/search` | POST | Test semantic retrieval |
| `/rag/stats` | GET | DB stats (chunk count, source breakdown) |

### Officer Dashboard

| Endpoint | Method | Description |
|---|---|---|
| `/officer/analyse` | POST | Start async regional GEE analysis |
| `/officer/analyse-result/<task_id>` | GET | Poll for result |
| `/officer/detect-fields` | POST | Start async field boundary detection |
| `/officer/detect-fields/<task_id>` | GET | Poll for GeoJSON polygons |
| `/officer/layer/<layer>` | POST | Start async satellite layer (ndvi/water/baresoil/croptype/forest) |
| `/officer/layer-result/<task_id>` | GET | Poll for layer GeoJSON |
| `/officer/farmers` | GET | List registered farmers for a province |
| `/officer/fields` | GET | List all fields with latest analyses for province |
| `/officer/village-crops` | POST | High-res 10m crop type map (<50km²) |
| `/officer/parcel-thumbnail` | POST | Download satellite image PNG |
| `/officer/proxy-image` | GET | CORS proxy for GEE thumbnail URLs |

### Admin Boundaries

| Endpoint | Method | Description |
|---|---|---|
| `/gadm/<iso>/<level>` | GET | Admin boundaries (FAO/GAUL via GEE, async) |
| `/gadm-result/<task_id>` | GET | Poll for boundary GeoJSON |

### Multi-Agent

| Endpoint | Method | Description |
|---|---|---|
| `/agent/chat` | POST | Agentic conversational Q&A (async) |
| `/agent/result/<task_id>` | GET | Poll for agent answer |
| `/agent/monitor` | POST | Autonomous field monitoring loop |
| `/agent/weekly-report` | POST | Weekly province intelligence report |
| `/agent/history/<session_id>` | GET | Conversation history |
| `/agent/status` | GET | Agent system status |

---

## 13. Database Schema (Supabase/PostgreSQL)

### `farmers`

| Column | Type | Description |
|---|---|---|
| id | uuid (PK) | Auto-generated |
| phone | text | Phone number (identifier) |
| language | text | `en`, `fa`, or `ps` |
| province | text | Afghan province name |
| telegram_chat_id | text | Telegram chat ID (nullable) |
| created_at | timestamp | Registration time |

### `fields`

| Column | Type | Description |
|---|---|---|
| id | uuid (PK) | Auto-generated |
| farmer_id | uuid (FK→farmers) | Owner |
| label | text | Field name |
| coords | jsonb | Array of [lat, lon] pairs |
| province | text | Province name |
| area_ha | float | Hectares |
| area_jereb | float | Jeribs (1 ha = 5 jeribs) |
| created_at | timestamp | Save time |

### `analyses`

| Column | Type | Description |
|---|---|---|
| id | uuid (PK) | Auto-generated |
| field_id | uuid (FK→fields) | Field analysed |
| farmer_id | uuid (FK→farmers) | Owner |
| ndvi | float | NDVI scalar |
| evi | float | EVI scalar |
| savi | float | SAVI scalar |
| mndwi | float | MNDWI scalar |
| lswi | float | LSWI scalar |
| rain | float | Annual rainfall (mm) |
| province | text | Province |
| analysed_at | timestamp | Analysis time |
| data | jsonb | Full analysis JSON |

### `conversations`

| Column | Type | Description |
|---|---|---|
| id | uuid (PK) | Auto-generated |
| farmer_id | uuid (FK→farmers) | Farmer |
| field_id | uuid (FK→fields, nullable) | Optional field context |
| question | text | Farmer's question |
| answer | text | AI answer |
| language | text | `en`, `fa`, `ps` |
| tool_calls | jsonb | Agent tool call log |
| created_at | timestamp | Time |

### `farmer_alerts`

| Column | Type | Description |
|---|---|---|
| id | uuid (PK) | Auto-generated |
| farmer_id | uuid (FK→farmers) | Owner |
| field_id | uuid (nullable) | Optional field |
| alert_type | text | `ndvi_drop`, `drought`, `rain_below`, etc. |
| threshold | float | Trigger threshold value |
| crop | text | Crop name |
| province | text | Province |
| created_at | timestamp | Creation time |

### `knowledge_chunks` (RAG / pgvector)

```sql
CREATE TABLE knowledge_chunks (
    id         uuid    PRIMARY KEY DEFAULT gen_random_uuid(),
    content    text    NOT NULL,
    embedding  vector(768),
    source     text    DEFAULT 'manual',
    metadata   jsonb   DEFAULT '{}',
    created_at timestamp DEFAULT NOW()
);
CREATE INDEX knowledge_chunks_embedding_idx
    ON knowledge_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
```

### `cultivation_history` (Regen Module)

```sql
CREATE TABLE cultivation_history (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  farmer_id    uuid,
  field_id     uuid,
  crop         TEXT NOT NULL,
  season       TEXT,
  year         INTEGER,
  notes        TEXT,
  source       TEXT DEFAULT 'farmer_reported',
  ndvi_at_peak FLOAT,
  province     TEXT,
  created_at   TIMESTAMP DEFAULT NOW()
);
```

---

## 14. Multilingual Support

ZaminAI fully supports three languages with bidirectional text rendering:

| Language | Code | Script | Direction | Numerals |
|---|---|---|---|---|
| English | `en` | Latin | LTR | 0–9 |
| Dari (Afghan Persian) | `fa` | Perso-Arabic | RTL | ۰–۹ (Eastern Arabic) |
| Pashto | `ps` | Perso-Arabic | RTL | ۰–۹ (Eastern Arabic) |

### Implementation Details

- **Backend:** `lang_inst` dict in `/ask` and `/diagnose` routes selects the language instruction appended to AI prompts
- **Frontend:** `_ydLang` state variable; `yd-rtl` CSS class applied to right-to-left text blocks
- **Telegram bot:** Detects farmer's stored language preference and responds in that language
- **smart_fallback():** Hardcoded Dari and Pashto responses for common question types
- **AI prompts:** Explicit instruction at end of prompt: "Respond ONLY in Dari / Pashto / English"
- **Dari specifics:** "دهقان" for farmer, "جریب" for land unit, Eastern Arabic numerals

---

## 15. Frontend Architecture (PWA)

### `index.html` — Farmer App

- **Type:** Progressive Web App (PWA), single HTML file, no build step
- **Map:** Leaflet.js v1.9 with `leaflet-draw` plugin for polygon drawing
- **Design:** Mobile-first, dark theme, custom `yd-*` CSS design system
- **Size:** ~3,650 lines of self-contained HTML/CSS/JS

### Key JavaScript Architecture

```javascript
// Core state
let farmerId, fieldData, pgCurrentLang;
const API  = "https://zaminai.onrender.com";  // Main API
const YAPI = "https://zaminai.onrender.com";  // Vision API

// Disease detection state
let _ydLang = 'en';
let _ydMode = 'disease';  // disease | pest | yield | soil

// Mode configuration
const _YD_MODE_CFG = {
  disease: {icon:'📸', title:'Upload crop photo', eyebrow:'Disease detected', ...},
  pest:    {icon:'🐛', title:'Upload pest photo', eyebrow:'Pest identified', ...},
  yield:   {icon:'🌾', title:'Upload crop head photo', eyebrow:'Yield estimate', ...},
  soil:    {icon:'🌱', title:'Upload soil photo', eyebrow:'Soil assessment', ...},
};
```

### Key Functions

| Function | Purpose |
|---|---|
| `ydSetLang(l, btn)` | Switch disease card language |
| `ydSetMode(m, btn)` | Switch diagnosis mode (disease/pest/yield/soil) |
| `yResizeAndEncode(file)` | Resize image to max 1024px, encode to base64 |
| `yOnFile(file)` | Handle file upload → preview → diagnose |
| `ydExtractSections(text)` | Parse numbered sections from AI response |
| `ydStripMarkdown(t)` | Remove **bold** and *italic* markers |
| `ydExtractDiseaseName(sec1)` | Extract disease name from section 1 text |
| `ydBuildSteps(text)` | Build step-card HTML from text |
| `ydRenderResult(d, lang, mode)` | Render full result card |
| `yDiagnose()` | Main diagnosis flow (upload → API → render) |
| `openRegenPanel()` | Open regenerative agriculture panel |
| `regenLoadHistory()` | Load cultivation history from API |
| `regenLoadRecommendations()` | Load crop rotation recommendations |

### `officer.html` — Field Officer Dashboard

- Regional analysis across districts and provinces
- Satellite layer overlays: NDVI, water, bare soil, crop type, forest, land cover
- Dynamic World land cover classification (9 classes)
- Terrain analysis (elevation, slope via SRTM)
- Population density (WorldPop)
- Surface water bodies (JRC)
- Monthly NDVI profile chart
- Admin boundary selector (FAO/GAUL — 190+ countries)
- Farmer registry per province

---

## 16. Graceful Degradation & Fallback Strategy

ZaminAI is designed to work at reduced capability when external services are unavailable.

```
GEE available?
    YES → full satellite analysis (NDVI, EVI, SAR, MODIS, CHIRPS trend)
    NO  → get_regional_data() → hardcoded province-level values for 16 provinces
              ↓
          outside known provinces?
              → get_climate_zone_fallback() → climate-zone estimate

Supabase available?
    YES → save analyses, conversations, alerts, cultivation history
    NO  → stateless mode (no persistence, all features still work)

Anthropic API available?
    YES → Claude Haiku for /diagnose, Claude Sonnet for /agent/chat
    NO  → Gemini Vision fallback for /diagnose

Gemini API available?
    YES → LLM answers in /ask, Gemini Vision fallback for /diagnose
    NO  → smart_fallback() → rule-based Dari/Pashto/English responses

YOLO available?
    YES → fast bounding-box disease detection (Stage 1)
    NO  → Vision LLM only (Stage 2 runs regardless)

Telegram configured?
    YES → push alert notifications
    NO  → silent (alerts still saved to DB)
```

### `get_regional_data(lat, lon)` Logic

1. Check if `(lat, lon)` falls inside any of the 16 province bounding boxes
2. If match found: return province hardcoded index values + historical trend
3. If no match: call `get_climate_zone_fallback(lat, lon)` — returns values based on estimated climate zone (arid/semi-arid/humid)

---

## 17. Deployment & Infrastructure

### Production

- **Host:** Render.com (Free Tier → Hobby)
- **URL:** https://zaminai.onrender.com
- **WSGI:** Gunicorn — `gunicorn app:app --workers 2 --timeout 300 --bind 0.0.0.0:$PORT`
- **RAM:** 512 MB (YOLO disabled with `DISABLE_YOLO=1`)
- **Cold start:** 30–60 seconds on free tier (Render spins down when idle)

### Local Development

```bash
cd zaminai
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
copy .env.example .env      # fill in real values
python app.py               # Flask dev server on :5000
```

### GEE Authentication (Service Account)

```python
credentials = ee.ServiceAccountCredentials(
    GEE_SERVICE_ACCOUNT,  # email
    key_data=GEE_PRIVATE_KEY  # private key JSON as string
)
ee.Initialize(credentials)
```

### Auto-Seeding on Startup

```python
# Runs in background thread at startup if RAG DB is empty
if rag_ok and GEMINI_KEY:
    threading.Thread(target=_auto_seed_rag, daemon=True).start()
```

This ensures the knowledge base is populated on fresh Render deploys without manual intervention.

---

## 18. Key Calculations Explained

### 18.1 Field Area Calculation (`calc_area_ha()`)

Uses the spherical excess formula (Haversine-based polygon area):

```python
def calc_area_ha(coords):
    """Haversine polygon area in hectares."""
    R = 6371000  # Earth radius in metres
    n = len(coords)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        lat1, lon1 = math.radians(coords[i][0]), math.radians(coords[i][1])
        lat2, lon2 = math.radians(coords[j][0]), math.radians(coords[j][1])
        area += (lon2 - lon1) * (2 + math.sin(lat1) + math.sin(lat2))
    area = abs(area) * R * R / 2
    return round(area / 10000, 4)  # m² → hectares
```

Result also converted to jeribs: `area_jereb = area_ha × 5`

### 18.2 Landsat SR Scale Factor

```python
# Collection 2 Level-2 radiometric scaling
scaled = pixel_value × 0.0000275 + (−0.2)
```

Raw DN values from C02/T1_L2 must be scaled to surface reflectance (0–1 range) before computing indices.

### 18.3 MODIS LST Kelvin → Celsius

```python
lst_celsius = pixel_value × 0.02 − 273.15
```

MODIS stores LST as 16-bit integer in Kelvin × 50 (scale factor 0.02). Subtract 273.15 to convert Kelvin to Celsius.

### 18.4 Adaptive Analysis Scale

GEE analysis resolution adapts to polygon size to stay within compute budgets:

```python
if area_km2 < 1:       scale = 10    # single field
elif area_km2 < 10:    scale = 20    # village
elif area_km2 < 50:    scale = 30    # small district
elif area_km2 < 200:   scale = 50    # district
elif area_km2 < 1000:  scale = 100   # province
elif area_km2 < 5000:  scale = 300
elif area_km2 < 20000: scale = 500
else:                   scale = 1000  # country-level
```

### 18.5 Cosine Similarity (RAG Retrieval)

```python
def _cosine_similarity(a, b):
    a = np.array(a); b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

Applied to compare query embedding against all stored `knowledge_chunks` embeddings. Chunks with similarity ≥ 0.50 are returned (top-k=4).

### 18.6 Open-Meteo Weather Alert Thresholds

```python
if rain_mm >= 20:    alert("heavy_rain")
if temp_min <= 0:    alert("frost")
if temp_max >= 40:   alert("extreme_heat")
if wind_kmh >= 50:   alert("high_wind")
```

---

## Summary: What Makes ZaminAI Technically Unique

1. **Multi-sensor fusion:** Combines optical (Sentinel-2), SAR radar (Sentinel-1), thermal (MODIS), and precipitation (CHIRPS) — four independent data streams — to build a complete picture of field conditions without requiring any ground sensors.

2. **Afghan-first design:** Every calculation, fallback value, soil database, crop calendar, and AI prompt is calibrated for Afghan conditions, provinces, crops, and languages — not a generic global tool.

3. **Graceful degradation at every layer:** The system works without GEE, without AI keys, without a database. Each missing service has a defined fallback, down to hardcoded province-level baseline values.

4. **RAG-grounded AI:** LLM answers are augmented with verified agronomic knowledge (ICARDA, WUR, FAO, CIMMYT sources) via vector retrieval — reducing hallucination and providing citable responses.

5. **Mobile-first PWA with zero build step:** The entire frontend is a single HTML file served by Flask. No Node.js, no webpack, no CDN dependencies beyond Leaflet. Works on basic Android phones over 3G.

6. **Async task architecture:** All satellite computations run in background threads with polling, so the UI never blocks and Render's 30-second proxy timeout is never hit.

7. **Cultivation memory + regen module:** ZaminAI learns from farmers as they record what they grew, and gives progressively better rotation advice as the history grows — moving from generic rules to field-specific recommendations.

---

*ZaminAI Technical Reference — Generated 2026 · Afghanistan Development Initiative*
