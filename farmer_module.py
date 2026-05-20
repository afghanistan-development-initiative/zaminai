"""
ZaminAI — Complete Afghan Farmer Intelligence Module
Covers: Profile | Irrigation | Fertilizer | Machinery | Seeds | Market | Programs | Photo | Feedback
Author: Maiwand Jan Alamzoi — Afghanistan Development Initiative · zaminai.org
"""

import streamlit as st
import json
import datetime
import base64
import anthropic

# ═══════════════════════════════════════════════════════════════════════
# DATABASES
# ═══════════════════════════════════════════════════════════════════════

MARKET_DATABASE = {
    "Kunduz":   {"main_market":"Kunduz City Central Market","buyers":["Local traders","WFP procurement"],"export_crops":["Flax","Dried fruits","Saffron"],"cold_storage":"Kunduz Cooperative Storage","avg_prices":{"Wheat":"12-15 AFN/kg","Flax":"25-30 AFN/kg","Vegetables":"8-15 AFN/kg"}},
    "Balkh":    {"main_market":"Mazar-i-Sharif Market","buyers":["Local traders","Export to Central Asia"],"export_crops":["Cotton","Dried fruits","Wheat"],"cold_storage":"Balkh Cold Storage","avg_prices":{"Wheat":"11-14 AFN/kg","Cotton":"18-22 AFN/kg"}},
    "Herat":    {"main_market":"Herat Central Market","buyers":["Iran export agents","Local traders"],"export_crops":["Saffron","Dried fruits","Pistachios"],"cold_storage":"Herat Agricultural Hub","avg_prices":{"Saffron":"2000-4000 AFN/gram","Wheat":"12-16 AFN/kg"}},
    "Nangarhar":{"main_market":"Jalalabad Market","buyers":["Pakistan export agents","Local traders"],"export_crops":["Citrus","Vegetables","Sugar cane"],"cold_storage":"Nangarhar Farmers Association","avg_prices":{"Vegetables":"6-12 AFN/kg"}},
    "Helmand":  {"main_market":"Lashkargah Market","buyers":["Local traders","WFP"],"export_crops":["Wheat","Cotton"],"cold_storage":"Helmand Agricultural Center","avg_prices":{"Wheat":"11-14 AFN/kg"}},
    "Kabul":    {"main_market":"Kabul Central Market","buyers":["Local traders","NGO procurement"],"export_crops":["Dried fruits","Vegetables"],"cold_storage":"Kabul Cold Storage","avg_prices":{"Wheat":"13-16 AFN/kg","Vegetables":"10-20 AFN/kg"}},
    "Kandahar": {"main_market":"Kandahar City Market","buyers":["Pakistan export agents","Local traders"],"export_crops":["Pomegranates","Grapes","Dried fruits"],"cold_storage":"Kandahar Fruit Processing Center","avg_prices":{"Pomegranates":"30-50 AFN/kg"}},
    "Takhar":   {"main_market":"Taloqan Market","buyers":["Local traders"],"export_crops":["Wheat","Vegetables"],"cold_storage":"Contact provincial agriculture","avg_prices":{"Wheat":"11-15 AFN/kg"}},
    "Baghlan":  {"main_market":"Pul-e-Khumri Market","buyers":["Local traders"],"export_crops":["Sugar beet","Wheat"],"cold_storage":"Contact provincial agriculture","avg_prices":{"Wheat":"11-15 AFN/kg"}},
    "Badakhshan":{"main_market":"Faizabad Market","buyers":["Local traders"],"export_crops":["Dried fruits","Lapis lazuli"],"cold_storage":"Contact provincial agriculture","avg_prices":{"Wheat":"12-16 AFN/kg"}},
    "Other":    {"main_market":"Provincial central market","buyers":["Local traders"],"export_crops":["Wheat","Vegetables"],"cold_storage":"Contact provincial agriculture department","avg_prices":{"Wheat":"11-15 AFN/kg","Vegetables":"8-15 AFN/kg"}},
}

SEED_DATABASE = {
    "Wheat": {
        "varieties": ["Mazar-99", "Herat-99", "Roshan"],
        "water_mm": 450, "value_usd_ha": 400,
        "best_for": "Cold winters, low-medium water",
        "plant": "October–November", "harvest": "June–July",
        "where_to_buy": "Ministry of Agriculture offices, local cooperatives"
    },
    "Flax": {
        "varieties": ["Local Afghan flax", "Certified imported"],
        "water_mm": 350, "value_usd_ha": 800,
        "best_for": "Very low water, sandy soil",
        "plant": "April–May", "harvest": "August–September",
        "where_to_buy": "FAO distribution centers, provincial agriculture departments"
    },
    "Saffron": {
        "varieties": ["Afghan Red Gold"],
        "water_mm": 300, "value_usd_ha": 15000,
        "best_for": "Dry climate, well-drained soil",
        "plant": "August–September", "harvest": "October–November",
        "where_to_buy": "Herat and Balkh agricultural centers"
    },
    "Chickpeas": {
        "varieties": ["Kabuli", "Desi"],
        "water_mm": 300, "value_usd_ha": 600,
        "best_for": "Drought tolerant, low input",
        "plant": "March–April", "harvest": "July–August",
        "where_to_buy": "Local markets, agricultural cooperatives"
    },
    "Vegetables": {
        "varieties": ["Tomato", "Onion", "Potato", "Cucumber"],
        "water_mm": 500, "value_usd_ha": 2000,
        "best_for": "High value, local market",
        "plant": "April–May", "harvest": "July–October",
        "where_to_buy": "Local seed shops"
    },
    "Cotton": {
        "varieties": ["Upland cotton", "Local variety"],
        "water_mm": 700, "value_usd_ha": 600,
        "best_for": "High water availability only",
        "plant": "April–May", "harvest": "September–October",
        "where_to_buy": "Cotton processing companies"
    },
    "Almonds": {
        "varieties": ["Mamra almond", "Sangi almond"],
        "water_mm": 400, "value_usd_ha": 3000,
        "best_for": "Long term investment, 3yr establishment",
        "plant": "February–March", "harvest": "August–September",
        "where_to_buy": "Herat and Kandahar nurseries"
    },
    "Rice": {
        "varieties": ["Afghan long grain", "Basmati"],
        "water_mm": 1200, "value_usd_ha": 500,
        "best_for": "Irrigated lowlands only — high water",
        "plant": "May–June", "harvest": "September–October",
        "where_to_buy": "Local seed shops"
    },
}

IRRIGATION_SYSTEMS = {
    "Flood irrigation (كاريز/جوی)": {
        "icon": "🌊",
        "description": "Traditional method — water flows by gravity through channels to flood the field",
        "water_efficiency": "30-40% — most water is lost to evaporation and runoff",
        "best_for": "Rice, large wheat fields, low-lying land near river",
        "cost": "Very low — mainly labour",
        "problems": ["Wastes large amounts of water", "Causes soil erosion", "Uneven water distribution", "Salt buildup over time"],
        "recommendation": "Suitable only where water is abundant. Switch to furrow or drip if water is limited.",
        "dari": "آبیاری غرقابی — آب از طریق جوی‌ها به مزرعه می‌رسد. پرمصرف‌ترین روش آبیاری",
        "pashto": "د سیلابي اوبو ورکول — اوبه د کانالونو له لارې مزرعې ته رسیږي"
    },
    "Furrow irrigation (دندانه‌دار)": {
        "icon": "〰️",
        "description": "Water flows through small channels (furrows) between crop rows",
        "water_efficiency": "50-60% — better than flood but still significant losses",
        "best_for": "Vegetables, cotton, maize, row crops",
        "cost": "Low — needs tractor or hand tools to make furrows",
        "problems": ["Still loses water to evaporation", "Needs relatively flat land", "Labour intensive to maintain"],
        "recommendation": "Good improvement over flood irrigation. Use for vegetables and row crops.",
        "dari": "آبیاری جوی‌بندی — آب از طریق شیارهای بین ردیف‌ها جریان می‌یابد",
        "pashto": "د کنډوو اوبه ورکول — اوبه د کرونو ترمنځ کنډوو له لارې ځي"
    },
    "Drip irrigation (قطره‌ای)": {
        "icon": "💧",
        "description": "Water delivered directly to plant roots through pipes and drippers",
        "water_efficiency": "85-95% — saves 40-60% water vs flood irrigation",
        "best_for": "Vegetables, fruit trees, saffron, high-value crops",
        "cost": "Medium — 15,000-50,000 AFN per jereb to install",
        "problems": ["Higher upfront cost", "Drippers can clog — needs clean water", "Requires maintenance knowledge"],
        "recommendation": "BEST choice for water-stressed areas. Pays back in 1-2 seasons through water savings and higher yield.",
        "dari": "آبیاری قطره‌ای — آب مستقیماً به ریشه گیاه می‌رسد. بهترین روش برای صرفه‌جویی در آب",
        "pashto": "د څاڅکو اوبه ورکول — اوبه مستقیماً د نبات ریښو ته رسیږي. د اوبو سپمول غوره لار"
    },
    "Sprinkler irrigation (بارانی)": {
        "icon": "🌧️",
        "description": "Water sprayed over crops like rainfall through rotating sprinklers",
        "water_efficiency": "70-80% — good efficiency especially for germination",
        "best_for": "Wheat, vegetables, young seedlings, pasture",
        "cost": "Medium-high — 25,000-80,000 AFN per jereb",
        "problems": ["Wind can reduce efficiency", "Higher energy cost", "Not suitable for all crops"],
        "recommendation": "Good for wheat and vegetables. Very effective for germination and young crops.",
        "dari": "آبیاری بارانی — آب مانند باران روی محصول پاشیده می‌شود",
        "pashto": "د باران غوندې اوبه ورکول — اوبه د باران غوندې د محصول پر سر شیندل کیږي"
    },
    "Karez (قنات/كاريز)": {
        "icon": "🏔️",
        "description": "Traditional underground channel bringing groundwater from mountains — ancient Afghan system",
        "water_efficiency": "60-70% — good if well maintained",
        "best_for": "Traditional farming areas, where rivers are not available",
        "cost": "High to build — but free water once built. Community maintained.",
        "problems": ["Requires community cooperation", "Maintenance needed after earthquakes/floods", "Fixed water supply — cannot increase"],
        "recommendation": "Excellent sustainable system. Maintain and protect existing karez. UNESCO heritage.",
        "dari": "قنات/كاريز — سیستم سنتی آبیاری زیرزمینی افغانستان. میراث فرهنگی",
        "pashto": "کاریز — د افغانستان دودیز لاندیني اوبه لیږد سیستم"
    }
}

FERTILIZER_DATABASE = {
    "Urea (یوریا)": {
        "icon": "🧪",
        "type": "Nitrogen (N)",
        "npk": "46-0-0",
        "what_it_does": "Makes plants grow faster and greener — increases leaf and stem growth",
        "when_to_use": "At planting and 4-6 weeks after germination",
        "how_much": "50-80 kg per jereb for wheat, 30-50 kg for vegetables",
        "cost": "1,200-1,500 AFN per 50kg bag",
        "warning": "Too much causes burning and pollution of water. Do not use more than recommended.",
        "organic_alternative": "Animal manure, compost",
        "where_to_buy": "Agricultural input shops, Ministry of Agriculture",
        "dari": "اوره — کود نیتروژنی که رشد گیاه را تسریع می‌کند",
        "pashto": "یوریا — د نایتروجن سره کود چې د نبات وده ګړندۍ کوي"
    },
    "DAP (دی‌آمونیوم فسفات)": {
        "icon": "🔵",
        "type": "Phosphorus + Nitrogen (P+N)",
        "npk": "18-46-0",
        "what_it_does": "Strengthens roots, helps flowering and fruit development, disease resistance",
        "when_to_use": "At planting — mix into soil before sowing",
        "how_much": "25-40 kg per jereb",
        "cost": "2,500-3,000 AFN per 50kg bag",
        "warning": "Apply at planting only — less effective if applied after",
        "organic_alternative": "Bone meal, rock phosphate",
        "where_to_buy": "Agricultural input shops, cooperatives",
        "dari": "DAP — کود فسفری که ریشه را تقویت می‌کند",
        "pashto": "DAP — د فاسفور کود چې ریښه پیاوړې کوي"
    },
    "Potassium (پتاس)": {
        "icon": "🟡",
        "type": "Potassium (K)",
        "npk": "0-0-60",
        "what_it_does": "Improves fruit quality, water use efficiency, disease resistance, drought tolerance",
        "when_to_use": "Before planting or early growth stage",
        "how_much": "15-25 kg per jereb",
        "cost": "1,800-2,200 AFN per 50kg bag",
        "warning": "Often overlooked in Afghanistan — but important for fruit and vegetable quality",
        "organic_alternative": "Wood ash, compost",
        "where_to_buy": "Agricultural input shops",
        "dari": "پتاس — کیفیت میوه را بهبود می‌بخشد و مقاومت به خشکی را افزایش می‌دهد",
        "pashto": "پتاس — د میوې کیفیت ښه کوي او د وچکالي مقاومت زیاتوي"
    },
    "Compost (کود دامی/کمپوست)": {
        "icon": "🌿",
        "type": "Organic NPK",
        "npk": "Variable — 1-3% N, 0.5-1% P, 1-2% K",
        "what_it_does": "Improves soil structure, water retention, adds beneficial microorganisms, long-term soil health",
        "when_to_use": "Before planting — mix deeply into soil",
        "how_much": "500-1000 kg per jereb (more is better)",
        "cost": "Free if made from farm waste. 500-1000 AFN if purchased.",
        "warning": "Must be fully decomposed — fresh manure can burn crops",
        "organic_alternative": "This IS the organic option",
        "where_to_buy": "Make from your own animal waste and crop residues",
        "dari": "کود دامی — بهترین کود برای سلامت طولانی مدت خاک",
        "pashto": "د حیواناتو سره کود — د خاورې د اوږدمهاله روغتیا لپاره غوره کود"
    },
    "Zinc sulfate (زینک)": {
        "icon": "⚗️",
        "type": "Micronutrient",
        "npk": "Zn 21%",
        "what_it_does": "Fixes zinc deficiency — common in Afghan soils. White stripes on wheat leaves = zinc deficiency",
        "when_to_use": "At planting every 2-3 years",
        "how_much": "3-5 kg per jereb",
        "cost": "800-1200 AFN per 25kg bag",
        "warning": "Check for zinc deficiency first — look for white or yellow stripes on young leaves",
        "organic_alternative": "Compost can provide some zinc",
        "where_to_buy": "Agricultural input shops",
        "dari": "زینک — کمبود زینک در خاک افغانستان بسیار رایج است. خطوط سفید روی برگ گندم نشانه کمبود زینک است",
        "pashto": "زینک — د افغانستان خاورو کې د زینک کمبود ډیر عام دی"
    }
}

MACHINERY_DATABASE = {
    "Tractor (تراکتور)": {
        "icon": "🚜",
        "operations": [
            "Plowing — breaks and turns soil to 20-30cm depth before planting",
            "Harrowing — breaks soil clumps, levels field surface",
            "Seed drilling — plants wheat and other crops in straight rows",
            "Fertilizer spreading — applies fertilizer evenly across field",
            "Transportation — moves harvest from field to storage or market",
            "Subsoiling — breaks hardpan layer that blocks root growth"
        ],
        "fuel": "Diesel — 8-12 liters per hour of work",
        "cost_rent": "800-1,500 AFN per jereb per operation",
        "cost_buy_used": "400,000-800,000 AFN",
        "cost_buy_new": "1,500,000-2,500,000 AFN",
        "best_for": "Fields larger than 5 jereb (1 hectare)",
        "not_suitable": "Very small plots, steep hillside fields, waterlogged fields",
        "maintenance": [
            "Check engine oil every 50 hours",
            "Change oil filter every 200 hours",
            "Check tire pressure weekly",
            "Clean air filter after dusty work",
            "Service before planting season"
        ],
        "where_to_rent": "Agricultural cooperative, neighboring farmer, provincial agriculture department",
        "tip": "Book tractor 2-3 weeks before planting season — high demand in October",
        "dari": "تراکتور — برای شخم، کاشت، کود پاشی و حمل محصول",
        "pashto": "ټراکتور — د ځمکې وهلو، کرلو، سره اچولو او محصول لیږدولو لپاره"
    },
    "Water pump (پمپ آب)": {
        "icon": "💧",
        "operations": [
            "River pumping — lifts water from river or canal to field level",
            "Well pumping — extracts groundwater for irrigation",
            "Drip irrigation — pressurizes water for drip system",
            "Sprinkler — feeds sprinkler irrigation system",
            "Drainage — removes excess water from flooded fields"
        ],
        "fuel": "Diesel 2-4 L/hr or Electric 2-5 kWh/hr",
        "cost_rent": "300-500 AFN per day",
        "cost_buy_small": "15,000-35,000 AFN (small, 1-3 jereb)",
        "cost_buy_large": "50,000-150,000 AFN (large, 10+ jereb)",
        "best_for": "Any farm without reliable gravity irrigation",
        "maintenance": [
            "Check fuel/oil before each use",
            "Clean strainer/filter weekly",
            "Winterize before cold season — drain water",
            "Check impeller annually"
        ],
        "where_to_rent": "Hardware shops, agricultural equipment rental",
        "tip": "Solar-powered pumps now available — no fuel cost after purchase. Ask FAO about subsidies.",
        "dari": "پمپ آب — برای آبیاری از رودخانه، چاه یا سیستم قطره‌ای",
        "pashto": "د اوبو پمپ — د سیند، کوهي یا د قطرو سیستم لپاره اوبه ورکول"
    },
    "Thresher (خرمنکوب)": {
        "icon": "⚙️",
        "operations": [
            "Separates wheat grain from straw — 10x faster than manual threshing",
            "Threshes barley, rice, and other grains",
            "Reduces grain losses during harvest by 30-50%",
            "Produces clean straw for animal feed"
        ],
        "fuel": "Diesel or tractor PTO connection",
        "cost_rent": "400-700 AFN per jereb",
        "cost_buy": "80,000-250,000 AFN",
        "best_for": "Wheat, barley, and rice at harvest time",
        "maintenance": [
            "Clean thoroughly after each use — prevents grain loss and fire risk",
            "Check and replace worn threshing bars",
            "Lubricate bearings after each season",
            "Store covered and dry"
        ],
        "where_to_rent": "Mobile threshers travel village to village at harvest — ask local cooperative",
        "tip": "Book early — threshers are in very high demand during June-July harvest",
        "dari": "خرمنکوب — گندم را از ساقه جدا می‌کند. 10 برابر سریع‌تر از خرمن‌کوبی دستی",
        "pashto": "خرمنکوب — گندم د ساقې نه جلا کوي. د لاسي کولو نه 10 ګنې ګړندی"
    },
    "Cultivator/rotavator": {
        "icon": "🔧",
        "operations": [
            "Shallow tillage — mixes top 10-15cm of soil",
            "Weed control — kills weeds between crop rows",
            "Incorporates fertilizer or compost into soil",
            "Seed bed preparation — creates fine, even soil for germination"
        ],
        "fuel": "Tractor PTO or hand tractor (2-wheel)",
        "cost_rent": "500-900 AFN per jereb",
        "cost_buy": "25,000-80,000 AFN (attachment), 150,000-300,000 AFN (2-wheel tractor)",
        "best_for": "Vegetables, orchards, inter-row cultivation",
        "maintenance": [
            "Check tines/blades for wear — replace when worn",
            "Clean soil and crop residue after use",
            "Oil gearbox annually"
        ],
        "where_to_rent": "Agricultural cooperative, equipment rental shops",
        "tip": "2-wheel hand tractor is affordable alternative for small farmers — one person can operate",
        "dari": "کولتیواتور — برای کنترل علف‌های هرز و آماده‌سازی بستر بذر",
        "pashto": "کولتیواتور — د بوټیو کنترول او د تخم د ځای چمتو کولو لپاره"
    },
    "Sprayer (سمپاش)": {
        "icon": "🌿",
        "operations": [
            "Pesticide application — controls insects and disease",
            "Herbicide spraying — kills weeds without damaging crops",
            "Foliar fertilizer — sprays liquid fertilizer directly on leaves",
            "Fungicide — prevents and treats fungal diseases"
        ],
        "fuel": "Manual (knapsack) or engine powered or drone",
        "cost_rent": "200-400 AFN per jereb",
        "cost_buy_manual": "2,000-5,000 AFN (knapsack sprayer)",
        "cost_buy_engine": "15,000-40,000 AFN",
        "best_for": "All crops — essential for pest and disease management",
        "safety": [
            "ALWAYS wear gloves and mask when spraying chemicals",
            "Never spray near water sources",
            "Read label — use correct dose",
            "Wash hands and equipment after use",
            "Keep children away from sprayed fields for 24-48 hours"
        ],
        "maintenance": [
            "Clean thoroughly after every use",
            "Check nozzles for blockage",
            "Replace worn nozzles for even application"
        ],
        "where_to_rent": "Agricultural input shops, cooperative",
        "tip": "Spray early morning or evening — avoid midday heat which reduces effectiveness",
        "dari": "سمپاش — برای کنترل آفات، بیماری‌ها و علف‌های هرز",
        "pashto": "سمپاش — د آفتونو، ناروغیو او بوټیو کنترول لپاره"
    },
    "Seed drill (بذرکار)": {
        "icon": "🌱",
        "operations": [
            "Plants wheat, barley, and other seeds in straight rows at correct depth",
            "Controls seed spacing — avoids wasting seed",
            "Places seed at correct depth — better germination",
            "Can combine seeding with fertilizer application"
        ],
        "fuel": "Tractor PTO",
        "cost_rent": "400-600 AFN per jereb",
        "cost_buy": "80,000-200,000 AFN",
        "best_for": "Wheat, barley, flax, chickpeas — any small grain",
        "maintenance": [
            "Clean seed tubes before use",
            "Check seed rate calibration",
            "Lubricate moving parts"
        ],
        "where_to_rent": "Agricultural cooperative",
        "tip": "Drill-seeded wheat yields 15-20% more than broadcast seeding — worth the cost",
        "dari": "بذرکار — بذر را در ردیف‌های منظم کشت می‌کند. عملکرد 15-20% بهتر از کاشت دستی",
        "pashto": "تخم کار — تخم د منظمو کرونو کې کري. د لاسي کرلو نه 15-20% ښه حاصل"
    },
    "Solar pump (پمپ خورشیدی)": {
        "icon": "☀️",
        "operations": [
            "Pumps water using solar energy — no fuel cost",
            "Works best midday when sun is strongest — ideal irrigation timing",
            "Can pump from 10-50m depth wells",
            "Stores energy in battery for cloudy days"
        ],
        "fuel": "Solar — FREE after installation",
        "cost_buy": "80,000-250,000 AFN depending on size",
        "payback": "2-3 years through fuel savings",
        "best_for": "Any farm — especially remote areas far from fuel supply",
        "maintenance": [
            "Clean solar panels monthly — dust reduces output 20-30%",
            "Check connections annually",
            "Replace battery every 5-7 years"
        ],
        "where_to_buy": "FAO and NGO programs sometimes subsidize — ask provincial agriculture office",
        "tip": "FAO and USAID have solar pump subsidy programs in some provinces — ask about eligibility",
        "dari": "پمپ خورشیدی — پس از نصب هزینه سوخت ندارد. در 2-3 سال خود را جبران می‌کند",
        "pashto": "د لمر پمپ — له نصبولو وروسته د تیلو لګښت نشته. 2-3 کلونو کې ځان بیرته ورکوي"
    },
    "Drone (پهپاد)": {
        "icon": "🚁",
        "operations": [
            "Aerial spraying — covers 10 jereb per hour vs 1 jereb by foot",
            "Field mapping — creates precise map of your field",
            "NDVI imaging — shows which parts of field are healthy or stressed",
            "Seeding — spreads seeds over difficult terrain"
        ],
        "fuel": "Electric — battery powered",
        "cost_rent": "1,500-3,000 AFN per jereb (service)",
        "cost_buy": "500,000-2,000,000 AFN agricultural drone",
        "best_for": "Large farms 50+ jereb, difficult terrain, precision spraying",
        "availability": "Limited in Afghanistan — available through some NGO programs",
        "tip": "Drone spraying reduces chemical use by 30-40% — better for environment and lower cost",
        "dari": "پهپاد — سمپاشی هوایی 10 برابر سریع‌تر. استفاده از کمتر 30-40% مواد شیمیایی",
        "pashto": "ډرون — د هوایي سمپاشۍ 10 ګنې ګړندۍ. 30-40% لږ کیمیاوي موادو کارول"
    }
}

DEVELOPMENT_PROGRAMS = {
    "FAO Afghanistan": {
        "icon": "🌍",
        "programs": [
            "Emergency seed and fertilizer distribution",
            "Irrigation rehabilitation and construction",
            "Agricultural extension training for farmers",
            "Livestock vaccination and support",
            "Food security monitoring"
        ],
        "contact": "fao-af@fao.org | Kabul office",
        "provinces": "All provinces",
        "how_to_apply": "Contact provincial agriculture department or FAO field office"
    },
    "WFP Afghanistan": {
        "icon": "🍞",
        "programs": [
            "Food for Assets — work on irrigation in exchange for food",
            "Cash for Work — cash payment for land rehabilitation",
            "School feeding programme",
            "Local food purchase from Afghan farmers"
        ],
        "contact": "wfp.afghanistan@wfp.org",
        "provinces": "All provinces",
        "how_to_apply": "Through community development councils (CDCs)"
    },
    "USAID Afghanistan": {
        "icon": "🇺🇸",
        "programs": [
            "Agricultural inputs support",
            "Market linkage and value chain",
            "Water management training",
            "Women in agriculture programmes"
        ],
        "contact": "Through local implementing partners",
        "provinces": "Kunduz, Balkh, Helmand, Herat, Nangarhar",
        "how_to_apply": "Through local NGO partners"
    },
    "AKDN (Aga Khan)": {
        "icon": "💎",
        "programs": [
            "Mountain agriculture support",
            "Microfinance loans for farmers",
            "Rural infrastructure construction",
            "Horticulture and high-value crops"
        ],
        "contact": "akdn.org/afghanistan",
        "provinces": "Badakhshan, Takhar, Baghlan, Bamyan",
        "how_to_apply": "Contact local AKDN office"
    },
    "Ministry of Agriculture (MAIL)": {
        "icon": "🏛️",
        "programs": [
            "Free or subsidised seeds distribution",
            "Agricultural extension officers — free advice",
            "Veterinary services for livestock",
            "Farmer training programmes",
            "Market price information"
        ],
        "contact": "Provincial agriculture departments in every province",
        "provinces": "All provinces",
        "how_to_apply": "Visit your provincial agriculture department office"
    }
}

CROP_CALENDAR = {
    "Wheat (گندم)": {
        "operations": [
            {"month": "Sep–Oct", "action": "Soil preparation — plow and harrow field", "priority": "high"},
            {"month": "Oct–Nov", "action": "Apply DAP fertilizer before planting", "priority": "high"},
            {"month": "Oct–Nov", "action": "Plant wheat — drill or broadcast at 15-20 kg/jereb", "priority": "urgent"},
            {"month": "Nov–Dec", "action": "First irrigation if no rain", "priority": "medium"},
            {"month": "Feb–Mar", "action": "Apply urea fertilizer — top dressing", "priority": "high"},
            {"month": "Mar–Apr", "action": "Second irrigation — critical flowering stage", "priority": "urgent"},
            {"month": "Apr–May", "action": "Watch for rust disease — spray if needed", "priority": "high"},
            {"month": "May–Jun", "action": "Final irrigation before harvest", "priority": "medium"},
            {"month": "Jun–Jul", "action": "Harvest when golden — do not delay", "priority": "urgent"},
            {"month": "Jul–Aug", "action": "Thresh, dry, and store in cool dry place", "priority": "high"},
        ]
    },
    "Vegetables (سبزیجات)": {
        "operations": [
            {"month": "Mar–Apr", "action": "Prepare soil — add compost/manure", "priority": "high"},
            {"month": "Apr", "action": "Plant seedlings or direct seed in rows", "priority": "urgent"},
            {"month": "Apr–May", "action": "Set up drip irrigation if available", "priority": "high"},
            {"month": "Apr–Oct", "action": "Water every 2-3 days in hot weather", "priority": "urgent"},
            {"month": "May", "action": "Apply urea fertilizer — first application", "priority": "high"},
            {"month": "May–Sep", "action": "Monitor for pests — aphids, whitefly", "priority": "high"},
            {"month": "Jun–Jul", "action": "Apply potassium fertilizer for fruit quality", "priority": "medium"},
            {"month": "Jul–Oct", "action": "Harvest regularly — do not let overripen", "priority": "urgent"},
            {"month": "Aug–Sep", "action": "Plant second crop — autumn vegetables", "priority": "medium"},
        ]
    },
    "Saffron (زعفران)": {
        "operations": [
            {"month": "Jul–Aug", "action": "Prepare well-drained soil — add compost", "priority": "high"},
            {"month": "Aug–Sep", "action": "Plant corms 10-15cm deep, 15cm apart", "priority": "urgent"},
            {"month": "Sep–Oct", "action": "Light irrigation after planting", "priority": "high"},
            {"month": "Oct–Nov", "action": "HARVEST — pick flowers at sunrise, every morning", "priority": "urgent"},
            {"month": "Oct–Nov", "action": "Separate stigmas same day — dry immediately", "priority": "urgent"},
            {"month": "Nov–Dec", "action": "Weed between rows — manual only", "priority": "medium"},
            {"month": "Feb–Mar", "action": "Apply potassium fertilizer — improves quality", "priority": "high"},
            {"month": "Mar–Apr", "action": "Leaves grow and die — normal process", "priority": "low"},
            {"month": "Jun–Jul", "action": "Divide corms every 3 years to maintain yield", "priority": "medium"},
        ]
    }
}

# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def save_farmer_data(farmer_id, data):
    if "farmer_database" not in st.session_state:
        st.session_state.farmer_database = {}
    st.session_state.farmer_database[farmer_id] = {
        **data,
        "timestamp": datetime.datetime.now().isoformat()
    }

def get_ai_response(system, messages, max_tokens=400):
    try:
        client = anthropic.Anthropic(api_key=st.secrets["anthropic"]["api_key"])
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=max_tokens,
            system=system,
            messages=messages
        )
        return response.content[0].text
    except Exception as e:
        return f"AI error: {e}"

def card(content, color="#4ade80"):
    st.markdown(f"""
    <div style="background:#111810;border-left:3px solid {color};
    border-radius:6px;padding:1rem;margin:6px 0;font-size:13px;
    color:#e8f5e4;line-height:1.7">{content}</div>
    """, unsafe_allow_html=True)

def rtl(text, color="#86efac"):
    st.markdown(f'<div style="direction:rtl;text-align:right;font-size:14px;color:{color};line-height:1.8;padding:4px 0">{text}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# MAIN MODULE
# ═══════════════════════════════════════════════════════════════════════

def render_farmer_module(language="English", field_results=None):

    st.divider()
    st.markdown("## 🌾 ZaminAI Farmer Intelligence Center")

    tabs = st.tabs([
        "👤 Profile",
        "💧 Irrigation",
        "🧪 Fertilizer",
        "🚜 Machinery",
        "🌱 Seeds",
        "🏪 Market",
        "🏛️ Programs",
        "📅 Calendar",
        "📸 Photo",
        "⭐ Feedback",
        "📊 Data"
    ])

    province = st.session_state.get("farmer_profile", {}).get("province", "Kunduz")
    field_ndvi  = (field_results or {}).get("ndvi", 0) or 0
    field_water = (field_results or {}).get("mndwi", 0) or 0
    field_rain  = (field_results or {}).get("rain_mm", 0) or 0
    field_area  = (field_results or {}).get("area_ha", 1) or 1

    # ═══════ TAB 1 — PROFILE ═══════
    with tabs[0]:
        st.subheader("👤 Farmer Profile")

        method = st.radio(
            "How do you want to share your info?",
            ["💬 Talk to AI", "📝 Fill form"],
            horizontal=True
        )

        if method == "📝 Fill form":
            col1, col2 = st.columns(2)
            with col1:
                name     = st.text_input("Name (optional)", placeholder="Anonymous", key="profile_name")
                province_sel = st.selectbox("Province", list(MARKET_DATABASE.keys()) + ["Kabul","Kandahar","Takhar","Baghlan","Badakhshan","Other"], key="profile_province")
                district = st.text_input("District", key="profile_district")
                village  = st.text_input("Village", key="profile_village")
            with col2:
                land_unit = st.selectbox("Land unit", ["Jereb جریب","Hectare ha","Acre"], key="profile_land_unit")
                land_size = st.number_input("Land size", min_value=0.1, value=2.0, step=0.5, key="profile_land_size")
                land_ha   = round(land_size * (0.2 if "Jereb" in land_unit else 0.4 if "Acre" in land_unit else 1), 2)
                land_jereb= round(land_size * (1 if "Jereb" in land_unit else 5 if "Hectare" in land_unit else 2.5), 1)
                st.caption(f"= {land_ha} ha = {land_jereb} jereb")

                main_crop = st.selectbox("Main crop", list(SEED_DATABASE.keys()), key="profile_main_crop")
                has_irr   = st.checkbox("Has irrigation", key="profile_has_irr")
                has_mkt   = st.checkbox("Has market access", key="profile_has_mkt")

            problems = st.multiselect("Problems faced", [
                "Water shortage","Crop disease","Pests","No seeds",
                "No fertilizer","Cannot sell crops","Flooding",
                "Drought","No machinery","Soil problems","Other"
            ], key="profile_problems")
            wants = st.text_input("Wants to grow next season", key="profile_wants")

            if st.button("💾 Save Profile", type="primary", use_container_width=True):
                st.session_state.farmer_profile = {
                    "name": name or "Anonymous",
                    "province": province_sel,
                    "district": district,
                    "village": village,
                    "land_ha": land_ha,
                    "land_jereb": land_jereb,
                    "main_crop": main_crop,
                    "has_irrigation": has_irr,
                    "has_market_access": has_mkt,
                    "problems": problems,
                    "wants_to_grow": wants
                }
                save_farmer_data(f"farmer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}", st.session_state.farmer_profile)
                st.success("✅ Profile saved!")
                st.balloons()

        else:
            if "onboard_msgs" not in st.session_state:
                st.session_state.onboard_msgs = []

            for msg in st.session_state.onboard_msgs:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if not st.session_state.onboard_msgs:
                if st.button("🌱 Start conversation with ZaminAI", type="primary", use_container_width=True):
                    reply = get_ai_response(
                        """You are ZaminAI — friendly agricultural AI for Afghan farmers.
                        Collect: name, province, district, village, land size (accept jereb/ha/acre),
                        main crop, problems, wants to grow, has irrigation, has market access.
                        Ask ONE question at a time. Be warm. Accept Dari/Pashto/English.
                        Convert units: 1 jereb=0.2ha. Say 'profile saved' when done.""",
                        [{"role":"user","content":"Hello I want to register"}]
                    )
                    st.session_state.onboard_msgs.append({"role":"assistant","content":reply})
                    st.rerun()

            if st.session_state.onboard_msgs:
                if prompt := st.chat_input("Tell ZaminAI about your farm..."):
                    st.session_state.onboard_msgs.append({"role":"user","content":prompt})
                    reply = get_ai_response(
                        """You are ZaminAI — collect farmer profile through conversation.
                        Ask ONE question at a time. Accept any language. Convert jereb to ha.
                        When all info collected say 'Thank you, your profile is saved'.""",
                        st.session_state.onboard_msgs
                    )
                    st.session_state.onboard_msgs.append({"role":"assistant","content":reply})

                    # Extract profile
                    if "profile is saved" in reply.lower() or "ممنون" in reply or "مننه" in reply:
                        extract = get_ai_response(
                            "Extract farmer info as JSON only. Fields: name, province, district, village, land_ha, land_jereb, main_crop, has_irrigation(bool), has_market_access(bool), problems(list), wants_to_grow",
                            [{"role":"user","content":f"Conversation: {json.dumps(st.session_state.onboard_msgs)}"}],
                            max_tokens=400
                        )
                        try:
                            profile = json.loads(extract)
                            st.session_state.farmer_profile = profile
                            save_farmer_data(f"farmer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}", profile)
                        except:
                            pass
                    st.rerun()

        # Show saved profile
        if "farmer_profile" in st.session_state:
            p = st.session_state.farmer_profile
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                card(f"👤 {p.get('name','Anonymous')}<br>📍 {p.get('province','—')} · {p.get('district','—')}<br>🏘️ {p.get('village','—')}<br>📐 {p.get('land_ha','—')} ha / {p.get('land_jereb','—')} jereb")
            with col2:
                card(f"🌾 {p.get('main_crop','—')}<br>💧 Irrigation: {'Yes ✓' if p.get('has_irrigation') else 'No ✗'}<br>🏪 Market: {'Yes ✓' if p.get('has_market_access') else 'No ✗'}<br>🎯 Wants: {p.get('wants_to_grow','—')}")
            if p.get("problems"):
                card(f"⚠️ Problems: {', '.join(p.get('problems',[]))}", color="#f87171")

    # ═══════ TAB 2 — IRRIGATION ═══════
    with tabs[1]:
        st.subheader("💧 Irrigation Systems — Which is Best for Your Field?")

        # Recommendation based on satellite data
        if field_water < -0.1:
            rec_system = "Drip irrigation (قطره‌ای)"
            rec_reason = f"Your field shows severe water stress (MNDWI={field_water:.2f}). Drip irrigation saves 40-60% water."
        elif field_water < 0:
            rec_system = "Furrow irrigation (دندانه‌دار)"
            rec_reason = f"Your field has low water availability. Furrow irrigation is more efficient than flood."
        else:
            rec_system = "Sprinkler irrigation (بارانی)"
            rec_reason = f"Water is available. Sprinkler gives good efficiency for your crops."

        card(f"🛰️ Satellite recommendation: <strong>{rec_system}</strong><br>{rec_reason}", color="#38bdf8")

        for name, info in IRRIGATION_SYSTEMS.items():
            is_recommended = name == rec_system
            with st.expander(f"{info['icon']} {name} {'⭐ RECOMMENDED' if is_recommended else ''} — Efficiency: {info['water_efficiency']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**What it is:** {info['description']}")
                    st.markdown(f"**Water efficiency:** `{info['water_efficiency']}`")
                    st.markdown(f"**Best for:** {info['best_for']}")
                    st.markdown(f"**Cost:** {info['cost']}")
                with col2:
                    st.markdown("**Problems:**")
                    for p in info["problems"]:
                        st.markdown(f"⚠️ {p}")
                    st.markdown(f"**Recommendation:** {info['recommendation']}")

                if language == "دری (Dari)":
                    rtl(info["dari"])
                elif language == "پښتو (Pashto)":
                    rtl(info["pashto"], "#a78bfa")

        st.divider()
        if q := st.chat_input("Ask about irrigation for your field..."):
            sys = f"""Irrigation expert for Afghan farmers.
            Field data: NDVI={field_ndvi}, Water={field_water}, Rain={field_rain}mm, Area={field_area}ha
            Farmer province: {province}. Answer in farmer's language. Be specific and practical."""
            ans = get_ai_response(sys, [{"role":"user","content":q}])
            card(f"💧 {ans}", color="#38bdf8")

    # ═══════ TAB 3 — FERTILIZER ═══════
    with tabs[2]:
        st.subheader("🧪 Fertilizer Guide — What, When, How Much")

        # Recommendation based on NDVI
        if field_ndvi < 0.15:
            card("⚠️ Low NDVI detected — your soil likely needs both nitrogen (Urea) and phosphorus (DAP). Start with DAP at planting and Urea 4 weeks later.", color="#f87171")
        elif field_ndvi < 0.25:
            card("🟡 Moderate NDVI — apply Urea top dressing now to boost vegetation health. Check for zinc deficiency (white stripes on leaves).", color="#fbbf24")
        else:
            card("✅ Good vegetation health — maintain with balanced fertilizer. Focus on potassium for quality improvement.", color="#4ade80")

        for name, info in FERTILIZER_DATABASE.items():
            with st.expander(f"{info['icon']} {name} — {info['type']} — NPK: {info['npk']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**What it does:** {info['what_it_does']}")
                    st.markdown(f"**When to use:** {info['when_to_use']}")
                    st.markdown(f"**How much:** {info['how_much']}")
                    st.markdown(f"**Cost:** {info['cost']}")
                with col2:
                    st.markdown(f"**⚠️ Warning:** {info['warning']}")
                    st.markdown(f"**Organic alternative:** {info['organic_alternative']}")
                    st.markdown(f"**Where to buy:** {info['where_to_buy']}")

                if language == "دری (Dari)":
                    rtl(info["dari"])
                elif language == "پښتو (Pashto)":
                    rtl(info["pashto"], "#a78bfa")

        st.divider()
        main_crop_sel = st.selectbox("Your crop", list(SEED_DATABASE.keys()), key="fertilizer_crop_select")
        if st.button("📋 Get fertilizer plan for my field", type="primary"):
            sys = f"""Fertilizer expert for Afghan farmers.
            Field: {field_area}ha, NDVI={field_ndvi}, Water={field_water}, Crop={main_crop_sel}
            Fertilizer database: {json.dumps(FERTILIZER_DATABASE)}
            Give specific fertilizer plan: what to buy, how much total for {field_area}ha, when to apply, total cost estimate.
            Answer in farmer's language."""
            ans = get_ai_response(sys, [{"role":"user","content":f"Give me fertilizer plan for {field_area}ha of {main_crop_sel}"}], max_tokens=500)
            card(f"🧪 Fertilizer plan for your {field_area}ha field:<br><br>{ans.replace(chr(10),'<br>')}", color="#fbbf24")

        if q := st.chat_input("Ask about fertilizer..."):
            sys = f"Fertilizer expert for Afghan farmers. Field NDVI={field_ndvi}, Area={field_area}ha. Answer in farmer's language."
            card(f"🧪 {get_ai_response(sys, [{'role':'user','content':q}])}", color="#fbbf24")

    # ═══════ TAB 4 — MACHINERY ═══════
    with tabs[3]:
        st.subheader("🚜 Machinery & Equipment Guide")

        card(f"📐 Your field size: {field_area}ha = {round(field_area*5,1)} jereb — {'Tractor recommended' if field_area > 1 else 'Hand tools or 2-wheel tractor suitable'}", color="#86efac")

        for name, info in MACHINERY_DATABASE.items():
            with st.expander(f"{info['icon']} {name}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Operations:**")
                    for op in info["operations"]:
                        st.markdown(f"✅ {op}")
                    st.markdown(f"**Fuel:** {info['fuel']}")
                    st.markdown(f"**Rent:** {info['cost_rent']}")

                with col2:
                    if "cost_buy_used" in info:
                        st.markdown(f"**Buy (used):** {info['cost_buy_used']}")
                        st.markdown(f"**Buy (new):** {info['cost_buy_new']}")
                    elif "cost_buy" in info:
                        st.markdown(f"**Buy:** {info['cost_buy']}")
                    st.markdown(f"**Best for:** {info['best_for']}")
                    st.markdown(f"**Where to rent:** {info['where_to_rent']}")
                    st.markdown(f"💡 **Tip:** {info['tip']}")

                    if "safety" in info:
                        st.markdown("**⚠️ Safety:**")
                        for s in info["safety"]:
                            st.markdown(f"🔴 {s}")

                if "maintenance" in info:
                    st.markdown("**🔧 Maintenance:**")
                    for m in info["maintenance"]:
                        st.markdown(f"🔧 {m}")

                if language == "دری (Dari)" and "dari" in info:
                    rtl(info["dari"])
                elif language == "پښتو (Pashto)" and "pashto" in info:
                    rtl(info["pashto"], "#a78bfa")

        st.divider()
        if q := st.chat_input("Ask about machinery..."):
            sys = f"Agricultural machinery expert for Afghan farmers. Field size={field_area}ha. Answer in farmer's language. Be specific about costs and where to find equipment."
            card(f"🚜 {get_ai_response(sys, [{'role':'user','content':q}])}", color="#86efac")

    # ═══════ TAB 5 — SEEDS ═══════
    with tabs[4]:
        st.subheader("🌱 Seeds & Varieties — What to Plant")

        # Recommend based on water
        if field_water < -0.1 or field_rain < 200:
            recommended = ["Saffron","Flax","Chickpeas","Wheat"]
            card(f"💧 Low water detected — drought-tolerant crops recommended", color="#38bdf8")
        else:
            recommended = ["Wheat","Vegetables","Saffron","Almonds"]
            card(f"✅ Water available — most crops suitable", color="#4ade80")

        st.markdown("### ⭐ Recommended for your field")
        for crop in recommended:
            if crop in SEED_DATABASE:
                info = SEED_DATABASE[crop]
                with st.expander(f"🌾 {crop} — {info['value_usd_ha']} USD/ha · {info['water_mm']}mm water needed"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Varieties:** {', '.join(info['varieties'])}")
                        st.markdown(f"**Best for:** {info['best_for']}")
                        st.markdown(f"**Water needed:** {info['water_mm']} mm/year")
                        st.markdown(f"**Value:** {info['value_usd_ha']} USD/ha")
                    with col2:
                        st.markdown(f"**Plant:** {info['plant']}")
                        st.markdown(f"**Harvest:** {info['harvest']}")
                        st.markdown(f"**Where to buy:** {info['where_to_buy']}")

        st.divider()
        if q := st.chat_input("Ask about seeds, varieties, planting..."):
            sys = f"""Seed expert for Afghan farmers. 
            Field: NDVI={field_ndvi}, Water={field_water}, Rain={field_rain}mm, Area={field_area}ha
            Seed database: {json.dumps(SEED_DATABASE)}
            Answer in farmer's language. Give specific variety names and where to buy."""
            card(f"🌱 {get_ai_response(sys, [{'role':'user','content':q}])}", color="#4ade80")

    # ═══════ TAB 6 — MARKET ═══════
    with tabs[5]:
        st.subheader("🏪 Market Access — Where to Sell")

        mkt = MARKET_DATABASE.get(province, MARKET_DATABASE["Kunduz"])
        card(f"🏪 <strong>{mkt['main_market']}</strong><br>👥 Buyers: {', '.join(mkt['buyers'])}<br>📦 Cold storage: {mkt['cold_storage']}", color="#38bdf8")

        st.markdown("### 💰 Current prices")
        for crop, price in mkt["avg_prices"].items():
            col1, col2 = st.columns([2,1])
            col1.markdown(f"**{crop}**")
            col2.markdown(f"`{price}`")

        st.markdown("### 🚀 Export opportunities")
        for crop in mkt["export_crops"]:
            st.markdown(f"✅ **{crop}** — export potential")

        st.divider()
        if q := st.chat_input("Ask about selling your crops..."):
            sys = f"""Market expert for Afghan farmers.
            Province: {province}. Market: {json.dumps(mkt)}
            Answer in farmer's language. Give specific advice on where to sell and how to get best price."""
            card(f"🏪 {get_ai_response(sys, [{'role':'user','content':q}])}", color="#38bdf8")

    # ═══════ TAB 7 — PROGRAMS ═══════
    with tabs[6]:
        st.subheader("🏛️ Development Programs in Your Area")

        for org, info in DEVELOPMENT_PROGRAMS.items():
            with st.expander(f"{info['icon']} {org} — {info['provinces']}"):
                st.markdown("**Programs:**")
                for prog in info["programs"]:
                    st.markdown(f"✅ {prog}")
                st.markdown(f"**Contact:** {info['contact']}")
                st.markdown(f"**How to apply:** {info['how_to_apply']}")

        st.divider()
        if q := st.chat_input("Ask about programs and support..."):
            sys = f"""Expert on development programs for Afghan farmers.
            Province: {province}. Programs: {json.dumps(DEVELOPMENT_PROGRAMS)}
            Answer in farmer's language. Be specific about how to apply."""
            card(f"🏛️ {get_ai_response(sys, [{'role':'user','content':q}])}", color="#a78bfa")

    # ═══════ TAB 8 — CALENDAR ═══════
    with tabs[7]:
        st.subheader("📅 Crop Calendar — What to Do and When")

        crop_sel = st.selectbox("Select your crop", list(CROP_CALENDAR.keys()), key="calendar_crop_sel")

        if crop_sel in CROP_CALENDAR:
            ops = CROP_CALENDAR[crop_sel]["operations"]
            for op in ops:
                color = "#f87171" if op["priority"] == "urgent" else "#fbbf24" if op["priority"] == "high" else "#4ade80"
                icon  = "🔴" if op["priority"] == "urgent" else "🟡" if op["priority"] == "high" else "🟢"
                card(f"{icon} <strong>{op['month']}</strong> — {op['action']}", color=color)

        st.divider()
        current_month = datetime.datetime.now().strftime("%B")
        if st.button(f"📅 What should I do in {current_month}?", type="primary", use_container_width=True):
            p_str = st.session_state.get("farmer_profile", {})
            sys = f"""Agricultural calendar expert for Afghanistan.
            Current month: {current_month}. Farmer crop: {p_str.get('main_crop','wheat')}
            Province: {province}. Field NDVI: {field_ndvi}
            Tell farmer exactly what 3-5 most important actions to take THIS month.
            Answer in farmer's language. Be specific."""
            card(f"📅 {get_ai_response(sys, [{'role':'user','content':f'What should I do in {current_month}?'}])}", color="#4ade80")

    # ═══════ TAB 9 — PHOTO ═══════
    with tabs[8]:
        st.subheader("📸 Upload Photos — AI Will Analyse")

        photo_type = st.selectbox("What are you uploading?", [
            "🌾 My field", "🍂 Sick or yellow plant",
            "🌱 Seedlings", "💧 Irrigation system",
            "🌍 Soil", "📦 Harvest", "🐛 Pest or insect"
        ], key="photo_type_sel")

        uploaded = st.file_uploader(
            "Upload 1-3 photos (JPG or PNG)",
            type=["jpg","jpeg","png","webp"],
            accept_multiple_files=True
        )

        extra = st.text_area(
            "Describe what you see (any language)",
            placeholder="e.g. The leaves are turning yellow since last week...",
            height=80,
            key="photo_extra_desc"
        )

        if uploaded and st.button("🔍 Analyse photos", type="primary", use_container_width=True):
            with st.spinner("AI is analysing your photos..."):
                try:
                    client = anthropic.Anthropic(api_key=st.secrets["anthropic"]["api_key"])
                    content = []

                    for photo in uploaded[:3]:
                        img_b64 = base64.b64encode(photo.read()).decode()
                        ext     = photo.name.split(".")[-1].lower()
                        media   = f"image/{'jpeg' if ext in ['jpg','jpeg'] else ext}"
                        content.append({
                            "type": "image",
                            "source": {"type":"base64","media_type":media,"data":img_b64}
                        })

                    content.append({"type":"text","text":f"""You are ZaminAI — agricultural AI expert for Afghan farmers.
Photo type: {photo_type}
Farmer description: {extra}
Satellite field data: NDVI={field_ndvi}, Water={field_water}, Area={field_area}ha

Analyse and provide:
1. What you see in the photo
2. Diagnosis — disease/pest/condition identification
3. Severity — how serious is the problem (1-5)
4. Action TODAY — what to do immediately
5. Action THIS WEEK — follow-up steps
6. Prevention — how to avoid next season

Be specific. Respond in the language the farmer used in their description."""})

                    response = client.messages.create(
                        model="claude-sonnet-4-5",
                        max_tokens=700,
                        messages=[{"role":"user","content":content}]
                    )
                    analysis = response.content[0].text
                    card(f"🔍 <strong>Photo Analysis Result:</strong><br><br>{analysis.replace(chr(10),'<br>')}", color="#4ade80")

                    if "photo_analyses" not in st.session_state:
                        st.session_state.photo_analyses = []
                    st.session_state.photo_analyses.append({
                        "type": photo_type,
                        "analysis": analysis,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                except Exception as e:
                    st.error(f"Photo analysis error: {e}")

        if "photo_analyses" in st.session_state and st.session_state.photo_analyses:
            st.divider()
            st.markdown("### Previous analyses")
            for a in st.session_state.photo_analyses[-3:]:
                with st.expander(f"{a['type']} — {a['timestamp'][:10]}"):
                    st.markdown(a["analysis"])

    # ═══════ TAB 10 — FEEDBACK ═══════
    with tabs[9]:
        st.subheader("⭐ Feedback — Did Our Recommendations Work?")

        col1, col2 = st.columns(2)
        with col1:
            rec_type    = st.selectbox("Which recommendation?", ["Irrigation","Fertilizer","Crop selection","Seed","Market advice","Machinery","Weather alert","Other"], key="fb_rec_type")
            worked      = st.radio("Did it work?", ["Yes ✅","Partly 🟡","No ❌"], key="fb_worked")
            yield_change= st.select_slider("Yield change", ["Much worse","Worse","Same","Better","Much better"], key="fb_yield")
        with col2:
            fb_text  = st.text_area("Tell us more (any language)", height=120, key="fb_text")
            rating   = st.slider("Overall rating", 1, 5, 4, key="fb_rating")

        if st.button("📤 Submit Feedback", type="primary", use_container_width=True):
            if "feedback_db" not in st.session_state:
                st.session_state.feedback_db = []
            st.session_state.feedback_db.append({
                "type":rec_type,"worked":worked,"yield":yield_change,
                "text":fb_text,"rating":rating,
                "province":province,"ndvi":field_ndvi,
                "timestamp":datetime.datetime.now().isoformat()
            })
            st.success("✅ Thank you! Your feedback helps improve ZaminAI for all Afghan farmers.")

            if fb_text:
                sys = "You are ZaminAI. Respond to farmer feedback warmly. Acknowledge what worked or not and give improved advice. Answer in farmer's language."
                card(f"🤖 {get_ai_response(sys, [{'role':'user','content':f'Feedback: {fb_text}. Rating: {rating}/5. Worked: {worked}'}])}", color="#4ade80")

    # ═══════ TAB 11 — DATA ═══════
    with tabs[10]:
        st.subheader("📊 ZaminAI Data Dashboard")
        st.markdown("*Anonymous aggregated data from all farmers*")

        db = st.session_state.get("farmer_database", {})

        if db:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Farmers", len(db))
            c2.metric("Total land", f"{sum(float(v.get('land_ha',0) or 0) for v in db.values()):.1f} ha")
            c3.metric("Provinces", len(set(v.get('province','') for v in db.values())))
            c4.metric("Feedbacks", len(st.session_state.get("feedback_db",[])))

            provs  = {}
            crops  = {}
            issues = {}
            for v in db.values():
                p = v.get("province","Unknown")
                provs[p] = provs.get(p,0)+1
                c = v.get("main_crop","Unknown")
                crops[c] = crops.get(c,0)+1
                for issue in v.get("problems",[]):
                    issues[issue] = issues.get(issue,0)+1

            col1,col2 = st.columns(2)
            with col1:
                st.markdown("**By province:**")
                for p,n in sorted(provs.items(),key=lambda x:-x[1]):
                    st.markdown(f"📍 {p}: **{n}**")
            with col2:
                st.markdown("**By crop:**")
                for c,n in sorted(crops.items(),key=lambda x:-x[1]):
                    st.markdown(f"🌾 {c}: **{n}**")

            if issues:
                st.markdown("**Most reported problems:**")
                for issue,n in sorted(issues.items(),key=lambda x:-x[1])[:5]:
                    st.markdown(f"⚠️ {issue}: **{n}** farmers")

            if st.session_state.get("feedback_db"):
                fb = st.session_state.feedback_db
                avg = sum(f["rating"] for f in fb)/len(fb)
                worked_pct = len([f for f in fb if "Yes" in f["worked"]])/len(fb)*100
                st.divider()
                c1,c2,c3 = st.columns(3)
                c1.metric("Feedbacks",len(fb))
                c2.metric("Avg rating",f"{avg:.1f}/5")
                c3.metric("Worked",f"{worked_pct:.0f}%")
        else:
            card("No farmer data yet. Farmers need to complete their profile first.<br><br>This dashboard will show total farmers, provinces, crops, problems, and recommendation success rates — valuable data for FAO, WFP, and researchers.", color="#6b8f65")
