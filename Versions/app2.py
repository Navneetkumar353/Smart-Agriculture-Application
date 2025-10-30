# app.py
# Streamlit Multi-Module Demo (Spray + SeedMatch Advisor + Voice: WebRTC w/o STUN + Simple Recorder)
# Run:
#   pip install streamlit streamlit-webrtc faster-whisper edge-tts av soundfile streamlit-mic-recorder joblib scikit-learn==1.7.2 xgboost
#   (Install FFmpeg and ensure `ffmpeg -version` works)
#   streamlit run app.py

import os, json, tempfile, re, sys, asyncio, io
import numpy as np
import joblib
import streamlit as st

# ============================
# Windows + aiortc stability
# ============================
try:
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except Exception:
    pass

# ============================
# Global App Config & Routing
# ============================
st.set_page_config(page_title="Project Demo", page_icon="🧪", layout="centered")

if "route" not in st.session_state:
    st.session_state.route = "home"

def goto(route: str):
    st.session_state.route = route
    st.rerun()

def back_to_home(key: str):
    if st.button("↩️  Back to Home", key=key):
        goto("home")

# ============================
# Shared Styles (cards/buttons)
# ============================
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

# ============================
# Module 1: Spray Chemical
# ============================
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
            else:
                st.markdown(f"**{chem}** is OK ✅ → {curr} ≥ {thr} {UNITS}")

    st.divider()
    if needs_spray:
        st.warning("Needs spraying: " + ", ".join(needs_spray))
    else:
        st.success("All chemicals meet or exceed thresholds. No spray needed.")

# ===============================================
# SeedMatch Advisor: artifacts + i18n + predict
# ===============================================
FEATURE_ORDER = ['N','P','K','temperature','humidity','ph','rainfall']
DEFAULTS = {'N': 90.0, 'P': 42.0, 'K': 43.0, 'temperature': 25.0, 'humidity': 70.0, 'ph': 6.8, 'rainfall': 150.0}

@st.cache_resource
def load_artifacts():
    model = joblib.load("artifacts/model_best.joblib")
    scaler = joblib.load("artifacts/scaler.joblib")
    classes = joblib.load("artifacts/classes.joblib")
    return model, scaler, classes

def predict_topk(model, scaler, classes, sample_dict, k=3):
    x = np.array([[sample_dict[kf] for kf in FEATURE_ORDER]], dtype=np.float32)
    x = scaler.transform(x)
    proba = model.predict_proba(x)[0] if hasattr(model, "predict_proba") else None
    if proba is None:
        pred = model.predict(x)[0]; proba = np.zeros(len(classes)); proba[pred] = 1.0
    idx = np.argsort(proba)[-k:][::-1]
    return [(classes[i], float(proba[i])) for i in idx], proba

LANGS = {
    "en": {
        "title": "🌱 SeedMatch Advisor", "mode": "Choose mode",
        "farmer": "Farmer Mode (Guided)", "voice": "Voice Mode (Ask & Answer)", "pro": "Pro Mode (Experts)",
        "recommend": "Recommend", "result": "Top suggestions", "details": "Show numeric details used",
        "district": "District (optional)", "state": "State (optional)",
        "rain_prompt": "How is the rain recently?", "rain_low": "Low (dry)", "rain_norm": "Normal", "rain_high": "Heavy (lots of rain)",
        "soil_ph": "Soil reaction", "ph_acid": "Acidic", "ph_neutral": "Neutral", "ph_alk": "Alkaline",
        "fert": "Fertilizer usage recently", "fert_none": "None", "fert_some": "Some", "fert_lots": "High",
        "temp": "Temperature (°C)", "hum": "Humidity (%)",
        "pro_note": "Enter exact values from soil test / sensors:", "units_hint": "Units: N,P,K (dataset scale), rainfall (mm)"
    },
    "hi": {
        "title": "🌱 बीज सलाहकार", "mode": "मोड चुनें",
        "farmer": "किसान मोड (मार्गदर्शित)", "voice": "वॉइस मोड (सवाल-जवाब)", "pro": "विशेषज्ञ मोड",
        "recommend": "सलाह दें", "result": "सर्वोत्तम सुझाव", "details": "प्रयुक्त संख्याएँ देखें",
        "district": "ज़िला (वैकल्पिक)", "state": "राज्य (वैकल्पिक)",
        "rain_prompt": "हाल में बारिश कैसी है?", "rain_low": "कम (सूखा)", "rain_norm": "सामान्य", "rain_high": "अधिक (भारी वर्षा)",
        "soil_ph": "मिट्टी की प्रकृति", "ph_acid": "अम्लीय", "ph_neutral": "तटस्थ", "ph_alk": "क्षारीय",
        "fert": "हाल में खाद का उपयोग", "fert_none": "नहीं", "fert_some": "थोड़ा", "fert_lots": "ज्यादा",
        "temp": "तापमान (°C)", "hum": "आर्द्रता (%)",
        "pro_note": "मिट्टी/सेंसर से सटीक मान दर्ज करें:", "units_hint": "इकाइयाँ: N,P,K (डेटासेट), वर्षा (mm)"
    }
}
def tr(key, lang):
    return LANGS.get(lang, LANGS["en"]).get(key, LANGS["en"].get(key, key))

# ===============================================
# Voice helpers (TTS + ASR + resample, robust I/O)
# ===============================================
VOICE_READY = True
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
    from faster_whisper import WhisperModel
    import av
except Exception:
    VOICE_READY = False

AUDIO_MAX_SECS = 20

@st.cache_resource
def get_whisper():
    return WhisperModel("small", device="cpu", compute_type="int8")

VOICE_MAP = {"en": "en-US-JennyNeural", "hi": "hi-IN-MadhurNeural"}

# IMPORTANT: Disable STUN/TURN (host candidates only) to avoid UDP/STUN errors on restricted networks.
RTC_CONFIGURATION = {"iceServers": []}

async def tts_bytes(text, lang):
    import edge_tts
    voice = VOICE_MAP.get(lang, VOICE_MAP["en"])
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        out = f.name
    await edge_tts.Communicate(text, voice=voice).save(out)
    with open(out, "rb") as r:
        return r.read()

def _resample_any(x: np.ndarray, sr_src: int, sr_dst: int) -> np.ndarray:
    if sr_src == sr_dst or len(x) == 0:
        return x.astype(np.float32)
    new_len = max(1, int(round(len(x) * sr_dst / sr_src)))
    t_old = np.linspace(0, 1, num=len(x), endpoint=False)
    t_new = np.linspace(0, 1, num=new_len, endpoint=False)
    return np.interp(t_new, t_old, x).astype(np.float32)

def _resample_48k_to_16k(x: np.ndarray) -> np.ndarray:
    return _resample_any(x, 48000, 16000)

class AudioBuffer(AudioProcessorBase):
    """Collect mono float32 audio into session_state['audio']; update level meter."""
    def __init__(self) -> None:
        self.level = 0.0

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        arr = frame.to_ndarray()  # shape may be (samples, channels) or (channels, samples)
        if arr.ndim == 2:
            if arr.shape[0] <= 8:  # treat as (channels, samples)
                arr = arr.mean(axis=0)
            else:                  # treat as (samples, channels)
                arr = arr.mean(axis=1)
        arr = arr.astype(np.float32, copy=False)
        maxabs = float(np.max(np.abs(arr))) if arr.size else 0.0
        if maxabs > 1.5:  # likely int16 scale
            arr = arr / 32768.0

        keep = int(48000 * AUDIO_MAX_SECS)        # store last ~20s @48k
        buf = st.session_state.get("audio", np.zeros(0, dtype=np.float32))
        buf = np.concatenate([buf, arr])[-keep:]
        st.session_state["audio"] = buf

        self.level = float(np.sqrt((arr**2).mean())) if arr.size else 0.0
        st.session_state["vu"] = self.level
        return frame

# ---------- Robust audio decoding + transcription (any format) ----------
def _decode_with_soundfile(data: bytes):
    import soundfile as sf
    y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if hasattr(y, "ndim") and y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float32), int(sr)

def _decode_with_av(data: bytes):
    import av, av.audio
    buf = io.BytesIO(data)
    container = av.open(buf, mode="r")
    astreams = [s for s in container.streams if s.type == "audio"]
    if not astreams:
        raise RuntimeError("No audio stream found")
    stream = astreams[0]
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks = []
    for packet in container.demux(stream):
        for frame in packet.decode():
            frame16 = resampler.resample(frame)
            arr = frame16.to_ndarray()
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            x = arr.astype(np.float32) / 32768.0
            chunks.append(x)
    container.close()
    if not chunks:
        raise RuntimeError("No decodable audio frames")
    y = np.concatenate(chunks)
    return y, 16000

def transcribe_audio_bytes(data: bytes) -> str:
    try:
        y, sr = _decode_with_soundfile(data)
    except Exception:
        y, sr = _decode_with_av(data)
    y16 = _resample_any(y, sr, 16000)
    segments, info = get_whisper().transcribe(y16, language=None)
    return " ".join([seg.text for seg in segments]).strip()

# ===============================================
# Module 2: SeedMatch Advisor (Farmer + Voice + Pro)
# ===============================================
def page_seed():
    back_to_home("back_home_seed")
    st.title("🌱 SeedMatch Advisor")

    try:
        model, scaler, classes = load_artifacts()
    except Exception:
        st.error("Model artifacts not found.\nPlace:\n• artifacts/model_best.joblib\n• artifacts/scaler.joblib\n• artifacts/classes.joblib")
        st.stop()

    lang = st.selectbox("Language / भाषा", ["en", "hi"], format_func=lambda x: {"en":"English","hi":"हिन्दी"}.get(x,x))
    mode = st.radio(tr("mode", lang), [tr("farmer", lang), tr("voice", lang), tr("pro", lang)], horizontal=True)

    colr1, colr2 = st.columns(2)
    with colr1: st.text_input(tr("district", lang))
    with colr2: st.text_input(tr("state", lang))

    st.caption(tr("units_hint", lang))
    st.divider()

    # ---------- Farmer Mode ----------
    if mode == tr("farmer", lang):
        rain = st.select_slider(tr("rain_prompt", lang),
            options=[tr("rain_low", lang), tr("rain_norm", lang), tr("rain_high", lang)],
            value=tr("rain_norm", lang))
        rainfall_mm = 60.0 if rain == tr("rain_low", lang) else (300.0 if rain == tr("rain_high", lang) else 150.0)

        soil = st.select_slider(tr("soil_ph", lang),
            options=[tr("ph_acid", lang), tr("ph_neutral", lang), tr("ph_alk", lang)],
            value=tr("ph_neutral", lang))
        ph_val = 5.8 if soil == tr("ph_acid", lang) else (8.0 if soil == tr("ph_alk", lang) else 6.8)

        fert = st.select_slider(tr("fert", lang),
            options=[tr("fert_none", lang), tr("fert_some", lang), tr("fert_lots", lang)],
            value=tr("fert_some", lang))
        if fert == tr("fert_none", lang):
            N_val, P_val, K_val = 60.0, 30.0, 30.0
        elif fert == tr("fert_lots", lang):
            N_val, P_val, K_val = 120.0, 60.0, 60.0
        else:
            N_val, P_val, K_val = DEFAULTS['N'], DEFAULTS['P'], DEFAULTS['K']

        col1, col2 = st.columns(2)
        with col1:  temp_c = st.slider(tr("temp", lang), 10.0, 45.0, DEFAULTS['temperature'], 0.5)
        with col2:  hum    = st.slider(tr("hum", lang), 20.0, 100.0, DEFAULTS['humidity'], 1.0)

        if st.button(tr("recommend", lang), use_container_width=True):
            features = {'N': N_val, 'P': P_val, 'K': K_val, 'temperature': temp_c, 'humidity': hum, 'ph': ph_val, 'rainfall': rainfall_mm}
            topk, proba = predict_topk(model, scaler, classes, features, k=3)
            st.subheader(tr("result", lang))
            for name, p in topk: st.progress(min(max(p,0.0),1.0), text=f"{name}: {p:.2f}")
            with st.expander(tr("details", lang)): st.json(features)

    # ---------- Voice Mode ----------
    elif mode == tr("voice", lang):
        if "voice_step" not in st.session_state:
            st.session_state.voice_step = 1
            st.session_state.collected = {}
            st.session_state.audio = np.zeros(0, dtype=np.float32)
            st.session_state.tts = None

        QUESTIONS = {
            1: {"en": "How is the rain recently? Say: low, normal, or heavy.",
                "hi": "हाल में बारिश कैसी है? बोलें: कम, सामान्य, या अधिक।"},
            2: {"en": "What is the soil reaction? Say: acidic, neutral, or alkaline.",
                "hi": "मिट्टी कैसी है? बोलें: अम्लीय, तटस्थ, या क्षारीय।"},
            3: {"en": "How much fertilizer did you use recently? Say: none, some, or high.",
                "hi": "हाल में खाद का उपयोग कितना किया? बोलें: नहीं, थोड़ा, या ज्यादा।"},
            4: {"en": "What is the temperature in degrees Celsius?",
                "hi": "तापमान कितने डिग्री सेल्सियस है?"},
            5: {"en": "What is the humidity percentage?",
                "hi": "आर्द्रता कितने प्रतिशत है?"}
        }
        q_text = QUESTIONS[st.session_state.voice_step].get(lang, QUESTIONS[st.session_state.voice_step]["en"])
        colq1, colq2 = st.columns(2)
        if colq1.button("🔊 Play Question"):
            st.session_state.tts = asyncio.run(tts_bytes(q_text, lang))
        if st.session_state.get("tts"):
            st.audio(st.session_state.tts, format="audio/mp3")

        tabs = st.tabs(["🎧 Live (WebRTC)", "🎙️ Simple Recorder"])

        # --- Tab 1: WebRTC (SENDONLY, no STUN/TURN) ---
        with tabs[0]:
            if not VOICE_READY:
                st.info("WebRTC not available. Use the Simple Recorder tab.")
            else:
                st.write("1) Click **Start** and allow mic. 2) Speak. 3) Click **Transcribe**.")
                try:
                    from streamlit_webrtc import webrtc_streamer, WebRtcMode
                    ctx = webrtc_streamer(
                        key="stt",
                        mode=WebRtcMode.SENDONLY,
                        rtc_configuration=RTC_CONFIGURATION,          # <- no STUN/TURN
                        audio_processor_factory=AudioBuffer,
                        media_stream_constraints={"audio": True, "video": False},
                        async_processing=True,
                    )
                except Exception as e:
                    st.warning(f"WebRTC init failed: {e}")
                    ctx = None

                vu = float(st.session_state.get("vu", 0.0))
                buflen = int(len(st.session_state.get("audio", np.zeros(0))))
                state = getattr(ctx, "state", None)
                st.caption(f"WebRTC state: {state}  |  Mic level: {vu:.3f}  |  buffer: {buflen} samples")
                st.progress(min(max(vu*5, 0.0), 1.0))

                colb1, colb2 = st.columns(2)
                if colb1.button("🧹 Reset Recording"):
                    st.session_state.audio = np.zeros(0, dtype=np.float32)

                can_transcribe = bool(state and getattr(state, "playing", False) and buflen >= 4000)
                if colb2.button("📝 Transcribe", disabled=not can_transcribe):
                    audio = st.session_state.get("audio", np.zeros(0, dtype=np.float32))
                    audio_16k = _resample_48k_to_16k(audio)
                    segments, info = get_whisper().transcribe(audio_16k, language=None)
                    text = " ".join([seg.text for seg in segments]).strip()
                    st.write("You said:", text if text else "(empty)")
                    _interpret_and_advance(text, lang)
                if not can_transcribe:
                    st.info("Press **Start**, speak for 2–3 seconds, then Transcribe.")

        # --- Tab 2: Simple Recorder (no WebRTC needed) ---
        with tabs[1]:
            st.write("Click **Start recording**, speak, then **Stop**. Works even if WebRTC is blocked.")
            try:
                from streamlit_mic_recorder import mic_recorder
                rec = mic_recorder(start_prompt="Start recording", stop_prompt="Stop",
                                   key="simple_recorder_v1", use_container_width=True)
                if rec and rec.get("bytes"):
                    st.audio(rec["bytes"])
                    st.caption(f"MIME: {rec.get('mime_type','unknown')}")
                    if st.button("📝 Transcribe (Simple Recorder)"):
                        try:
                            text = transcribe_audio_bytes(rec["bytes"])
                            st.write("You said:", text if text else "(empty)")
                            _interpret_and_advance(text, lang)
                        except Exception as e:
                            st.exception(e)
            except Exception:
                st.warning("Install the recorder: pip install streamlit-mic-recorder")

        # Fallback UI per step
        step = st.session_state.voice_step
        if step == 1:
            cols = st.columns(3)
            if cols[0].button(tr("rain_low", lang)):  st.session_state.collected['rainfall']=60.0; st.session_state.voice_step=2
            if cols[1].button(tr("rain_norm", lang)): st.session_state.collected['rainfall']=150.0; st.session_state.voice_step=2
            if cols[2].button(tr("rain_high", lang)): st.session_state.collected['rainfall']=300.0; st.session_state.voice_step=2
        elif step == 2:
            cols = st.columns(3)
            if cols[0].button(tr("ph_acid", lang)):    st.session_state.collected['ph']=5.8; st.session_state.voice_step=3
            if cols[1].button(tr("ph_neutral", lang)): st.session_state.collected['ph']=6.8; st.session_state.voice_step=3
            if cols[2].button(tr("ph_alk", lang)):     st.session_state.collected['ph']=8.0; st.session_state.voice_step=3
        elif step == 3:
            cols = st.columns(3)
            if cols[0].button(tr("fert_none", lang)):  st.session_state.collected.update({'N':60.0,'P':30.0,'K':30.0}); st.session_state.voice_step=4
            if cols[1].button(tr("fert_some", lang)):  st.session_state.collected.update({'N':DEFAULTS['N'],'P':DEFAULTS['P'],'K':DEFAULTS['K']}); st.session_state.voice_step=4
            if cols[2].button(tr("fert_lots", lang)):  st.session_state.collected.update({'N':120.0,'P':60.0,'K':60.0}); st.session_state.voice_step=4
        elif step == 4:
            val = st.slider(tr("temp", lang), 10.0, 45.0, DEFAULTS['temperature'], 0.5)
            if st.button("Next ➜"): st.session_state.collected['temperature']=val; st.session_state.voice_step=5
        elif step == 5:
            val = st.slider(tr("hum", lang), 20.0, 100.0, DEFAULTS['humidity'], 1.0)
            if st.button("Get Advice ✅"):
                st.session_state.collected['humidity']=val
                data = {**DEFAULTS, **st.session_state.collected}
                topk, proba = predict_topk(model, scaler, classes, data, k=3)
                st.subheader(tr("result", lang))
                for name, p in topk:
                    st.progress(min(max(p,0.0),1.0), text=f"{name}: {p:.2f}")
                with st.expander(tr("details", lang)): st.json(data)
                msg = (f"Recommended seed: {topk[0][0]}" if lang=="en" else f"अनुशंसित बीज: {topk[0][0]}")
                st.audio(asyncio.run(tts_bytes(msg, lang)), format="audio/mp3")
                if st.button("Start Over"):
                    for k in ["voice_step","collected","audio","vu","tts"]:
                        st.session_state.pop(k, None)

    # ---------- Pro Mode ----------
    else:
        st.caption(tr("pro_note", lang))
        c1, c2, c3 = st.columns(3)
        with c1:
            N_val = st.number_input("N", 0.0, 200.0, DEFAULTS['N'], 1.0)
            temp_c = st.number_input("temperature (°C)", -10.0, 60.0, DEFAULTS['temperature'], 0.5)
            ph_val = st.number_input("ph", 3.5, 9.5, DEFAULTS['ph'], 0.1)
        with c2:
            P_val = st.number_input("P", 0.0, 200.0, DEFAULTS['P'], 1.0)
            hum = st.number_input("humidity (%)", 0.0, 100.0, DEFAULTS['humidity'], 1.0)
            rainfall_mm = st.number_input("rainfall (mm)", 0.0, 500.0, DEFAULTS['rainfall'], 1.0)
        with c3:
            K_val = st.number_input("K", 0.0, 200.0, DEFAULTS['K'], 1.0)

        if st.button(tr("recommend", lang), use_container_width=True):
            features = {'N': N_val, 'P': P_val, 'K': K_val,
                        'temperature': temp_c, 'humidity': hum,
                        'ph': ph_val, 'rainfall': rainfall_mm}
            topk, proba = predict_topk(model, scaler, classes, features, k=3)
            st.subheader(tr("result", lang))
            for name, p in topk: st.progress(min(max(p,0.0),1.0), text=f"{name}: {p:.2f}")
            with st.expander(tr("details", lang)): st.json(features)

# ---- Helper to interpret voice text and advance the step ----
def _interpret_and_advance(text: str, lang: str):
    step = st.session_state.voice_step
    ok = False
    if step == 1:
        low  = {"low","कम"}; high = {"heavy","अधिक","ज्यादा"}; txt = text.lower()
        if any(w in txt for w in low):   st.session_state.collected['rainfall']=60.0; ok=True
        elif any(w in txt for w in high): st.session_state.collected['rainfall']=300.0; ok=True
        elif "normal" in txt or "सामान्य" in text: st.session_state.collected['rainfall']=150.0; ok=True
    elif step == 2:
        txt = text.lower()
        if "acid" in txt or "अम्ल" in text: st.session_state.collected['ph']=5.8; ok=True
        elif "alk" in txt or "क्षार" in text: st.session_state.collected['ph']=8.0; ok=True
        elif "neutral" in txt or "तटस्थ" in text: st.session_state.collected['ph']=6.8; ok=True
    elif step == 3:
        txt = text.lower()
        if "none" in txt or "नहीं" in text:
            st.session_state.collected.update({'N':60.0,'P':30.0,'K':30.0}); ok=True
        elif "high" in txt or "ज्यादा" in text or "अधिक" in text:
            st.session_state.collected.update({'N':120.0,'P':60.0,'K':60.0}); ok=True
        elif "some" in txt or "थोड़ा" in text:
            st.session_state.collected.update({'N':DEFAULTS['N'],'P':DEFAULTS['P'],'K':DEFAULTS['K']}); ok=True
    elif step == 4:
        m = re.search(r"(-?\d+(?:\.\d+)?)", text)
        if m: st.session_state.collected['temperature']=float(m.group(1)); ok=True
    elif step == 5:
        m = re.search(r"(-?\d+(?:\.\d+)?)", text)
        if m: st.session_state.collected['humidity']=float(m.group(1)); ok=True

    if ok:
        st.success("Got it.")
        st.session_state.voice_step += 1
    else:
        st.warning("Couldn't understand. Use the buttons below.")

# ============================
# Home Page
# ============================
def page_home():
    st.title("🚜 Project Demo")
    st.write(
        "This demo showcases two modules for smart farming:\n"
        "1) **Spray Chemical Module** — threshold-based control for Nitrogen, Phosphorus, and Potassium.\n"
        "2) **SeedMatch Advisor** — farmer-friendly (voice) and expert modes."
    )
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧪 Spray Chemical Module <span class='badge'>N-P-K</span>", unsafe_allow_html=True)
    st.write("Set nutrient thresholds and enter soil readings. If a value is below its threshold, a colored spray button appears.")
    if st.button("Open Spray Module"): goto("spray")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🌱 SeedMatch Advisor <span class='badge'>Dual-Mode + Voice</span>", unsafe_allow_html=True)
    st.write("Use **Farmer Mode** (simple choices), **Voice Mode** (ask & answer), or **Pro Mode** (direct inputs).")
    if st.button("Open Seed Advisor"): goto("seed")
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption("Tip: Use the Back button on module pages to return here.")

# ============================
# Router
# ============================
ROUTES = {"home": page_home, "spray": page_spray, "seed": page_seed}
ROUTES[st.session_state.route]()
