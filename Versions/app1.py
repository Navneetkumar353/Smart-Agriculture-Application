# app.py
# Streamlit Multi-Module Demo (Spray + SeedMatch Advisor)
# Run: pip install streamlit joblib scikit-learn xgboost && streamlit run app.py

import os
import json
import numpy as np
import joblib
import streamlit as st

# ----------------------------
# Global App Config & Routing
# ----------------------------
st.set_page_config(page_title="Project Demo", page_icon="🧪", layout="centered")

if "route" not in st.session_state:
    st.session_state.route = "home"

def goto(route: str):
    st.session_state.route = route
    st.rerun()

def back_to_home(key: str):
    if st.button("↩️  Back to Home", key=key):
        goto("home")

# ----------------------------
# Shared Styles (cards/buttons)
# ----------------------------
CARD_CSS = """
<style>
.card {
  padding: 1.0rem 1.2rem;
  border: 1px solid rgba(49,51,63,0.2);
  border-radius: 14px;
  margin-bottom: 0.7rem;
  background: rgba(250,250,252,0.65);
}
.card h3 { margin: 0 0 0.35rem 0; }
.badge {
  display:inline-block; padding:0.15rem 0.5rem; border-radius:999px;
  background:#eef; font-size:0.8rem; margin-left:0.4rem;
}
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)

# ----------------------------
# Module 1: Spray Chemical
# ----------------------------
def page_spray():
    back_to_home("back_home_spray")
    st.title("🧪 Spray Chemical Module")
    st.caption("Adjust thresholds for soil nutrients. If a reading is below its threshold, a colored spray button appears.")

    UNITS = "mg/kg"
    CHEM_INFO = {
        "Nitrogen (N)":   {"color_name": "blue",   "emoji": "🟦"},
        "Phosphorus (P)": {"color_name": "orange", "emoji": "🟧"},
        "Potassium (K)":  {"color_name": "green",  "emoji": "🟩"},
    }

    # Sidebar thresholds
    st.sidebar.header("Thresholds (spray triggers)")
    thresholds = {}
    for chem in CHEM_INFO:
        thresholds[chem] = st.sidebar.number_input(
            f"{chem} ({UNITS})", min_value=0.0, max_value=2000.0,
            value=50.0, step=1.0, key=f"thr_{chem}"
        )

    st.subheader("Soil Sensor Readings")
    cols = st.columns(3)
    readings = {}
    defaults = [35.0, 60.0, 40.0]  # demo defaults
    for i, chem in enumerate(CHEM_INFO):
        with cols[i]:
            readings[chem] = st.number_input(
                f"{chem} ({UNITS})",
                min_value=0.0, max_value=2000.0, value=defaults[i], step=1.0, key=f"val_{chem}"
            )

    st.divider()
    st.subheader("Spray Recommendations")

    needs_spray = []
    for chem, info in CHEM_INFO.items():
        curr, thr = readings[chem], thresholds[chem]
        color, emoji = info["color_name"], info["emoji"]

        with st.container(border=True):
            if curr < thr:
                needs_spray.append(chem)
                st.markdown(f"**{chem}** is **below threshold** → {curr} < {thr} {UNITS}")
                if st.button(f"{emoji} Spray {chem}", key=f"spray_{chem}"):
                    st.write(f"Spray color: **{color}**")
                    st.toast(f"{chem}: spray color is {color}", icon="✅")
                    # TODO: Integrate drone command here
                    # send_command_to_drone(chemical=chem, color=color)
            else:
                st.markdown(f"**{chem}** is OK ✅ → {curr} ≥ {thr} {UNITS}")

    st.divider()
    if needs_spray:
        st.warning("Needs spraying: " + ", ".join(needs_spray))
    else:
        st.success("All chemicals meet or exceed thresholds. No spray needed.")

    with st.expander("Debug / Last Action (optional)"):
        st.write("Use this area to show logs or last actions once you wire up your drone controls.")

# ----------------------------
# SeedMatch Advisor: helpers (artifacts + i18n + predict)
# ----------------------------
FEATURE_ORDER = ['N','P','K','temperature','humidity','ph','rainfall']
DEFAULTS = {'N': 90.0, 'P': 42.0, 'K': 43.0, 'temperature': 25.0, 'humidity': 70.0, 'ph': 6.5, 'rainfall': 150.0}

@st.cache_resource
def load_artifacts():
    model = joblib.load("artifacts/model_best.joblib")
    scaler = joblib.load("artifacts/scaler.joblib")
    classes = joblib.load("artifacts/classes.joblib")
    return model, scaler, classes

def predict_topk(model, scaler, classes, sample_dict, k=3):
    x = np.array([[sample_dict[kf] for kf in FEATURE_ORDER]], dtype=np.float32)
    x = scaler.transform(x)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
    else:
        pred = model.predict(x)[0]
        proba = np.zeros(len(classes), dtype=float)
        proba[pred] = 1.0
    idx = np.argsort(proba)[-k:][::-1]
    return [(classes[i], float(proba[i])) for i in idx], proba

LANGS = {
    "en": {
        "title": "🌱 SeedMatch Advisor",
        "mode": "Choose mode",
        "farmer": "Farmer Mode (Guided)",
        "pro": "Pro Mode (Experts)",
        "recommend": "Recommend",
        "result": "Top suggestions",
        "details": "Show numeric details used",
        "district": "District (optional)",
        "state": "State (optional)",
        "rain_prompt": "How is the rain recently?",
        "rain_low": "Low (dry)",
        "rain_norm": "Normal",
        "rain_high": "Heavy (lots of rain)",
        "soil_ph": "Soil reaction",
        "ph_acid": "Acidic",
        "ph_neutral": "Neutral",
        "ph_alk": "Alkaline",
        "fert": "Fertilizer usage recently",
        "fert_none": "None",
        "fert_some": "Some",
        "fert_lots": "High",
        "temp": "Temperature (°C)",
        "hum": "Humidity (%)",
        "pro_note": "Enter exact values from soil test / sensors:",
        "units_hint": "Units: N,P,K (dataset scale), rainfall (mm)"
    },
    "hi": {
        "title": "🌱 बीज सलाहकार",
        "mode": "मोड चुनें",
        "farmer": "किसान मोड (मार्गदर्शित)",
        "pro": "विशेषज्ञ मोड",
        "recommend": "सलाह दें",
        "result": "सर्वोत्तम सुझाव",
        "details": "प्रयुक्त संख्याएँ देखें",
        "district": "ज़िला (वैकल्पिक)",
        "state": "राज्य (वैकल्पिक)",
        "rain_prompt": "हाल में वर्षा कैसी है?",
        "rain_low": "कम (सूखा)",
        "rain_norm": "सामान्य",
        "rain_high": "अधिक (भारी वर्षा)",
        "soil_ph": "मिट्टी की प्रकृति",
        "ph_acid": "अम्लीय",
        "ph_neutral": "तटस्थ",
        "ph_alk": "क्षारीय",
        "fert": "हाल में खाद का उपयोग",
        "fert_none": "नहीं",
        "fert_some": "थोड़ा",
        "fert_lots": "ज्यादा",
        "temp": "तापमान (°C)",
        "hum": "आर्द्रता (%)",
        "pro_note": "मिट्टी/सेंसर से सटीक मान दर्ज करें:",
        "units_hint": "इकाइयाँ: N,P,K (डेटासेट), वर्षा (mm)"
    }
}
def t(key, lang):
    return LANGS.get(lang, LANGS["en"]).get(key, LANGS["en"].get(key, key))

# ----------------------------
# Module 2: SeedMatch Advisor (dual-mode UI)
# ----------------------------
def page_seed():
    back_to_home("back_home_seed")
    st.title("🌱 SeedMatch Advisor")

    # Try loading artifacts
    try:
        model, scaler, classes = load_artifacts()
    except Exception as e:
        st.error("Model artifacts not found. Make sure these files exist:\n"
                 "• artifacts/model_best.joblib\n• artifacts/scaler.joblib\n• artifacts/classes.joblib")
        st.stop()

    # Language + mode (kept inside page body to avoid sidebar conflicts with Spray module)
    lang = st.selectbox("Language / भाषा", ["en","hi"], format_func=lambda x: {"en":"English","hi":"हिन्दी"}.get(x,x))
    mode = st.radio(t("mode", lang), [t("farmer", lang), t("pro", lang)], horizontal=True)

    # Optional region (future use with IMD)
    colr1, colr2 = st.columns(2)
    with colr1:
        district = st.text_input(t("district", lang))
    with colr2:
        state = st.text_input(t("state", lang))

    st.caption(t("units_hint", lang))
    st.divider()

    if mode == t("farmer", lang):
        # Guided mapping
        rain = st.select_slider(
            t("rain_prompt", lang),
            options=[t("rain_low", lang), t("rain_norm", lang), t("rain_high", lang)],
            value=t("rain_norm", lang)
        )
        if rain == t("rain_low", lang):
            rainfall_mm = 60.0
        elif rain == t("rain_high", lang):
            rainfall_mm = 300.0
        else:
            rainfall_mm = 150.0

        soil = st.select_slider(
            t("soil_ph", lang),
            options=[t("ph_acid", lang), t("ph_neutral", lang), t("ph_alk", lang)],
            value=t("ph_neutral", lang)
        )
        ph_val = 5.8 if soil == t("ph_acid", lang) else (8.0 if soil == t("ph_alk", lang) else 6.8)

        fert = st.select_slider(
            t("fert", lang),
            options=[t("fert_none", lang), t("fert_some", lang), t("fert_lots", lang)],
            value=t("fert_some", lang)
        )
        if fert == t("fert_none", lang):
            N_val, P_val, K_val = 60.0, 30.0, 30.0
        elif fert == t("fert_lots", lang):
            N_val, P_val, K_val = 120.0, 60.0, 60.0
        else:
            N_val, P_val, K_val = DEFAULTS['N'], DEFAULTS['P'], DEFAULTS['K']

        col1, col2 = st.columns(2)
        with col1:
            temp_c = st.slider(t("temp", lang), min_value=10.0, max_value=45.0, value=DEFAULTS['temperature'], step=0.5)
        with col2:
            hum = st.slider(t("hum", lang),  min_value=20.0, max_value=100.0, value=DEFAULTS['humidity'], step=1.0)

        if st.button(t("recommend", lang), use_container_width=True):
            features = {
                'N': N_val, 'P': P_val, 'K': K_val,
                'temperature': temp_c, 'humidity': hum,
                'ph': ph_val, 'rainfall': rainfall_mm
            }
            topk, proba = predict_topk(model, scaler, classes, features, k=3)
            st.subheader(t("result", lang))
            for name, p in topk:
                st.progress(min(max(p,0.0),1.0), text=f"{name}: {p:.2f}")
            with st.expander(t("details", lang)):
                st.json(features)

    else:
        # Pro Mode: direct numeric inputs
        st.caption(t("pro_note", lang))
        c1, c2, c3 = st.columns(3)
        with c1:
            N_val = st.number_input("N", min_value=0.0, max_value=200.0, value=DEFAULTS['N'], step=1.0)
            temp_c = st.number_input("temperature (°C)", min_value=-10.0, max_value=60.0, value=DEFAULTS['temperature'], step=0.5)
            ph_val = st.number_input("ph", min_value=3.5, max_value=9.5, value=DEFAULTS['ph'], step=0.1)
        with c2:
            P_val = st.number_input("P", min_value=0.0, max_value=200.0, value=DEFAULTS['P'], step=1.0)
            hum = st.number_input("humidity (%)", min_value=0.0, max_value=100.0, value=DEFAULTS['humidity'], step=1.0)
            rainfall_mm = st.number_input("rainfall (mm)", min_value=0.0, max_value=500.0, value=DEFAULTS['rainfall'], step=1.0)
        with c3:
            K_val = st.number_input("K", min_value=0.0, max_value=200.0, value=DEFAULTS['K'], step=1.0)

        if st.button(t("recommend", lang), use_container_width=True):
            features = {
                'N': N_val, 'P': P_val, 'K': K_val,
                'temperature': temp_c, 'humidity': hum,
                'ph': ph_val, 'rainfall': rainfall_mm
            }
            topk, proba = predict_topk(model, scaler, classes, features, k=3)
            st.subheader(t("result", lang))
            for name, p in topk:
                st.progress(min(max(p,0.0),1.0), text=f"{name}: {p:.2f}")
            with st.expander(t("details", lang)):
                st.json(features)

# ----------------------------
# Home Page
# ----------------------------
def page_home():
    st.title("🚜 Project Demo")
    st.write(
        "This demo showcases two modules for smart farming:\n"
        "1) **Spray Chemical Module** — threshold-based control for Nitrogen, Phosphorus, and Potassium.\n"
        "2) **SeedMatch Advisor** — weather/soil-based seed recommendations with a farmer-friendly mode."
    )

    # Spray card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧪 Spray Chemical Module <span class='badge'>N-P-K</span>", unsafe_allow_html=True)
    st.write(
        "Set nutrient **thresholds** and enter **soil readings**. If a value is below its threshold, "
        "you’ll see a **colored spray button** (prints the color now; later it will trigger your drone)."
    )
    if st.button("Open Spray Module"):
        goto("spray")
    st.markdown("</div>", unsafe_allow_html=True)

    # Seed card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🌱 SeedMatch Advisor <span class='badge'>Dual-Mode</span>", unsafe_allow_html=True)
    st.write("Choose **Farmer Mode** (guided) or **Pro Mode** (direct inputs) to get seed recommendations.")
    if st.button("Open Seed Advisor"):
        goto("seed")
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption("Tip: Use the Back button on module pages to return here.")

# ----------------------------
# Router
# ----------------------------
ROUTES = {
    "home": page_home,
    "spray": page_spray,
    "seed": page_seed,
}
ROUTES[st.session_state.route]()
