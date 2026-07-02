# ZaminAI — Task Backlog

This file drives autonomous work. When the user says "pick a task" or "work on ZaminAI",
Claude Code reads this file, picks the highest-priority OPEN task, implements it,
pushes, verifies production, then marks it DONE.

## Status legend
- `[ ]` OPEN — ready to pick up
- `[~]` IN PROGRESS — being worked on now
- `[x]` DONE — shipped to production
- `[!]` BLOCKED — needs user input before starting

---

## Priority 1 — Core reliability

- [x] **YOLO11 upgrade** — crop_vision.py upgraded to support YOLOv8 + YOLO11 via env vars.
        YOLO_MODEL_REPO/FILE/PATH/CONF/IOU all configurable. Expanded CLASS_NAMES to 56 classes
        (wheat, rice, cassava, coffee, sorghum added). YOLO11 classification probs path supported.
- [x] Fix /diagnose 502 timeout — add 55s Claude timeout + warm-up ping
- [x] Globalise: 12 languages, country detection, dynamic land units
- [x] Fix disease name parser ("Analysis complete" bug)
- [x] 4-mode vision card: Disease / Pest / Yield / Soil
- [x] **Cache GEE results** — save satellite analysis per field_id + date in Supabase
        `analyses` table already exists; add 24h cache check before calling GEE.
        Avoids repeated API calls for same field. Fallback: call GEE if no cached row.
- [x] **Retry on GEE failure** — currently returns regional fallback immediately.
        Try Sentinel-2 → Landsat → Sentinel-1 → MODIS → regional in sequence.
        Log which source succeeded to `analyses.data_source` column.

## Priority 2 — Farmer experience

- [x] **7-day weather forecast** — call Open-Meteo API (free, no key needed)
        `https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&daily=precipitation_sum,temperature_2m_max`
        Return in /analyse response as `weather_forecast: [{day, rain_mm, temp_max}]`
        Show in index.html as a mini 7-day strip below the field stats card.

- [x] **Field report PDF export** — add GET /report/<field_id> endpoint
        Returns HTML (print-ready) with: field map thumbnail, NDVI value,
        satellite date, AI advice, disease results if any.
        Frontend: "Download Report" button on the analysis result panel.

- [x] **Offline cache** — store last field analysis in localStorage (IndexedDB).
        On page load with no network, show cached data with "Last updated: X days ago" banner.
        Key: `zaminai_field_{field_id}`, value: full JSON response + timestamp.

- [x] **Crop calendar overlay** — on the Leaflet map, show a colour-coded band
        at the top of the sidebar showing current season stage for detected crop
        (e.g. "Wheat — Grain filling" in green, "Harvest in ~18 days" in amber).
        Data: use existing CROP_CALENDAR dict in app.py.
        Implemented: get_season_stage() in app.py, makeSeasonBand() in index.html.
        Supports wheat/saffron/vegetables/orchard with days-to-harvest countdown.

## Priority 3 — AI improvements

- [x] **Confidence score on disease detection** — in /diagnose response add
        `confidence: "high" | "medium" | "low"` based on Claude's response certainty.
        Parse: if Claude says "likely", "possibly", "may be" → medium/low.
        Show as a small badge next to the severity pill in the UI.

- [x] **Multi-language /ask smart fallback** — current smart_fallback() only returns English.
        Add translations for irrigation, fertiliser, and crop-choice answers in
        FA, PS, AR, HI, SW so offline fallback still works for farmers without data.
        Extended to all 12 languages: AR, UR, HI, BN, SW, ES, FR, PT, AM added.

- [x] **RAG expansion** — seed 20 more knowledge chunks covering:
        - East Africa (Kenya, Ethiopia, Tanzania) crop calendars
        - South Asia (Bangladesh, Pakistan) flooding / waterlogging advice
        - Central America (Guatemala, Honduras) coffee / maize disease guide
        Added 21 chunks; total _RAG_SEED_DOCS = 45. Auto-seeded on next cold start.

## Priority 4 — Platform

- [!] **Field sharing** — ENDPOINTS IMPLEMENTED, needs Supabase schema migration.
        POST /db/field/share → generates 8-char token, saves to fields.share_token
        GET /field/<token> → read-only HTML page with NDVI, metrics, seasonal advice
        REQUIRED SQL (run in Supabase dashboard → SQL Editor):
          ALTER TABLE fields ADD COLUMN IF NOT EXISTS share_token VARCHAR(8);
          CREATE UNIQUE INDEX IF NOT EXISTS fields_share_token_idx ON fields(share_token);
        After running SQL, the endpoints go live automatically.

- [x] **Telegram alert improvements** — expanded check_alerts_fire() with two new types:
        disease_detected (fires when latest /diagnose stored severe/high disease in analyses.full_data
        within past 7 days); rain_deficit improved to compare current month's satellite rain against
        60% of province seasonal average (MONTHLY_RAIN_FRACTION). /diagnose now persists
        disease_name, disease_severity, disease_diagnosed_at to analyses.full_data when field_id
        provided. build_alert_message() expanded with FA/PS translations for all 5 alert types.

- [ ] **Drone integration endpoint** — POST /drone/mission
        Accepts field_id, returns GeoJSON waypoints for the ZaminAI Agri Pro drone:
        GPS coordinates of stress spots (NDVI < 0.3 zones) with nozzle-on/off flags.
        Format: `{waypoints: [{lat, lng, spray: true/false, ndvi: 0.22}]}`

---

## How Claude Code picks a task

1. Read this file
2. Find first `[ ]` task in priority order
3. Mark it `[~]` and push
4. Implement fully (backend + frontend if needed)
5. Run: `python -c "import ast; ast.parse(open('app.py').read())"` to verify syntax
6. Commit with clear message and push
7. Wait ~2 min, check `https://zaminai.onrender.com/health`
8. Mark task `[x]` and push updated TASKS.md
9. Report to user: what was done, what changed, what's next

## Rules for autonomous work

- Never break existing endpoints — run syntax check before every push
- Never commit .env or credentials
- Keep commits small and focused (one task per commit)
- If a task needs Supabase schema changes, describe the SQL but don't run it — flag `[!]`
- Test the golden path mentally before pushing: does the change work end-to-end?
