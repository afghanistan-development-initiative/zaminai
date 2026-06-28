# ZaminAI — Platform Brief

---

## What is ZaminAI

ZaminAI is a satellite-powered farming intelligence platform that gives smallholder farmers access to field-level data they could not previously afford or access. A farmer draws their field on a map; the platform analyses it across four satellite sources and returns crop health, soil moisture, temperature risk, rainfall trends, and actionable planting advice — in the farmer's own language, on a phone, within seconds.

---

## The Problem

Smallholder farmers — who produce roughly 70% of the food consumed in low-income countries — make planting, irrigation, and harvest decisions with almost no data. They cannot afford soil tests, agronomists, or remote sensing services. Climate stress is intensifying: erratic rainfall, late frosts, and drought are compressing growing windows in ways that historical knowledge alone cannot anticipate. The result is yield loss, food insecurity, and economic fragility at household scale, compounding into national food system risk.

In Afghanistan specifically, where the platform was built, 80% of the population depends on agriculture and average farm size is 0.2–0.6 hectares. A 20% yield improvement on that scale is the difference between subsistence and surplus.

---

## What ZaminAI Does

**Satellite field analysis** — The farmer draws a polygon on a map. The platform queries four satellite datasets and returns within 10–30 seconds:

- **Crop health index (NDVI 0–1)** with threshold interpretation (stressed / adequate / healthy / dense)
- **Vegetation indices**: EVI, SAVI (low-vegetation correction), NDRE (nitrogen stress), BSI (bare soil exposure)
- **Soil moisture** from Sentinel-1 SAR radar — works through cloud cover, day or night
- **Land surface temperature** and frost risk from MODIS thermal
- **Monthly rainfall** trends and seasonal anomalies
- **Historical NDVI trend** from 2013 to present (Landsat 2013–2018, Sentinel-2 2019+)
- **Planting and harvest calendar** calibrated to the detected crop and region
- **Soil type** from SoilGrids global database (texture, pH, organic carbon)

**Computer vision (photo upload)** — Four analysis modes from a single photo:

1. **Disease detection** — identifies wheat rust, powdery mildew, leaf blight, and 35+ other diseases; returns severity classification (mild / moderate / severe), numbered treatment steps, and product name with dose
2. **Pest identification** — names the pest, estimates infestation level, recommends pesticide product, dose per hectare, and application timing
3. **Yield estimation** — reads crop growth stage, estimates days to optimal harvest, flags visible quality issues
4. **Soil health** — identifies soil type and moisture from a handful photo, recommends organic and chemical amendments with quantities

**AI-powered advisory** — Farmers type questions in their own language. The system retrieves relevant agronomic knowledge from a vector database (RAG / pgvector) and passes it to a large language model to generate grounded, locally-appropriate answers. Responses are generated in the farmer's language without requiring translation.

**Regenerative agriculture module** — Farmers log cultivation history (what crop, which season, which field). The platform tracks this and recommends crop rotations based on soil depletion patterns, water stress, and local market value — with explicit soil benefit scoring for each recommended crop.

---

## Technology

| Layer | Component |
|---|---|
| Satellite data | Google Earth Engine — Sentinel-2 SR (10 m), Landsat 8/9 (30 m), Sentinel-1 SAR (10 m), MODIS MOD11A2 (1 km) |
| Computer vision | Claude Haiku 4.5 Vision (Anthropic) — primary; Gemini 2.0/2.5 Flash (Google) — fallback |
| Object detection | YOLOv8m trained on 38 plant disease classes |
| Language model | Gemini 2.0/2.5 Flash for multilingual advisory Q&A |
| Knowledge retrieval | pgvector (PostgreSQL) with Gemini text-embedding-2; cosine similarity search across 44 agronomic knowledge chunks |
| Database | Supabase (PostgreSQL) — farmer profiles, field polygons, analysis history, alert thresholds, conversation logs |
| Backend | Python / Flask — single-file API with graceful degradation when any service is unavailable |
| Frontend | Vanilla JS progressive web app (PWA) — Leaflet map, Leaflet Draw plugin, installable on Android/iOS |
| Hosting | Render (API), Supabase (database), Google Earth Engine (satellite compute) |

The platform degrades gracefully: if GEE is unavailable it uses a regional database fallback; if the AI key is missing it uses rule-based responses; if the database is offline it operates statelessly.

---

## Global Applicability

The satellite infrastructure, computer vision pipeline, and AI advisory layer operate on any coordinate on Earth. Google Earth Engine covers global Sentinel-2 and Landsat archives; SoilGrids covers 250 m global soil data; Claude and Gemini Vision work for crops on any continent.

The current version is being extended with:

- **Country auto-detection** from GPS coordinates via reverse geocoding — AI prompts and advice automatically adapt to the detected country and region
- **Hemisphere-aware crop calendars** — planting and harvest windows flip correctly for Southern Hemisphere locations
- **Local land units** — switches from Afghan jerib (0.2 ha) to hectares or acres based on detected country
- **12 farming languages** — expanding from English, Dari, and Pashto to include Spanish, French, Arabic, Hindi, Swahili, Bengali, Portuguese, Amharic, and Hausa

No architectural changes are required to support new geographies. A deployment for Kenya, Peru, or Morocco requires only updated knowledge base content and language configuration.

---

## Impact Potential

**Who benefits directly:** Smallholder farmers in data-scarce environments — primarily Sub-Saharan Africa, South and Central Asia, and Latin America — who currently make decisions with no access to agronomic data or professional advice.

**Scale:** The FAO estimates 500 million smallholder farm households globally. ZaminAI's architecture is built to serve this scale: satellite data is free at point of use (ESA Copernicus program), the AI inference cost per analysis is under $0.01, and the PWA frontend works on low-end Android devices on 2G/3G connections.

**Measurable outcomes the platform enables:**
- Earlier disease detection → reduced crop loss (wheat rust untreated = 70% yield loss within 2 weeks)
- Precision irrigation scheduling from soil moisture data → 20–40% water saving in water-stressed regions
- Crop rotation recommendations → soil carbon recovery and reduced fertiliser dependency over 3–5 year cycles
- Historical NDVI trend data → farmers and extension workers can identify field degradation before it becomes irreversible

**Current deployment:** Afghanistan (production), with global expansion in development. Built by the Afghanistan Development Initiative in collaboration with Wageningen University & Research and FAO.

---

*ZaminAI is open to partnership with agricultural development organisations, extension services, and impact investors working on food security and climate adaptation at smallholder scale.*

*Technical contact: m.alamzoi123@gmail.com*
*Repository: github.com/afghanistan-development-initiative/zaminai*
