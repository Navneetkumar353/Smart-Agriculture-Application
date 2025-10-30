# app.py — Spray + SeedMatch (Farmer / Voice: Ask→Record→Auto-Transcribe / Pro)
# Install:
#   pip install streamlit streamlit-mic-recorder faster-whisper soundfile av joblib scikit-learn==1.7.2 xgboost numpy
# Artifacts:
#   ./artifacts/model_best.joblib  ./artifacts/scaler.joblib  ./artifacts/classes.joblib
# Run:
#   streamlit run app.py

import io, re, json, asyncio
import numpy as np
import streamlit as st
import joblib
import streamlit.components.v1 as components
import unicodedata
from typing import Optional

# -----------------------------------
# App setup + router
# -----------------------------------
st.set_page_config(page_title="Project Demo", page_icon="🧪", layout="centered")
if "route" not in st.session_state:
    st.session_state.route = "home"

def goto(route: str):
    st.session_state.route = route
    try: st.rerun()
    except Exception: st.experimental_rerun()

def back_to_home(key: str):
    if st.button("↩️  Back to Home", key=key):
        goto("home")

# -----------------------------------
# Styles
# -----------------------------------
st.markdown("""
<style>
.card{padding:1.0rem 1.2rem;border:1px solid rgba(49,51,63,0.2);border-radius:14px;margin-bottom:0.9rem;background:rgba(250,250,252,0.65)}
.card h3{margin:0 0 0.35rem 0}
.badge{display:inline-block;padding:0.15rem 0.5rem;border-radius:999px;background:#eef;font-size:0.8rem;margin-left:0.4rem}
.row{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}
.btn{padding:.5rem .8rem;border-radius:8px;border:0;background:#e2e8f0;cursor:pointer}
small.mono{font-family:ui-monospace,Menlo,Consolas,monospace;opacity:.75}

@media (prefers-color-scheme:dark){
 .card{background:rgba(30,41,59,0.55);border-color:rgba(148,163,184,0.25)}
 .badge{background:#334155;color:#e2e8f0}
 .btn{background:#334155;color:#e2e8f0}
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# Spray Chemical Module
# -----------------------------------
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
    defaults = [35.0, 60.0, 40.0]
    for i, chem in enumerate(CHEM_INFO):
        with cols[i]:
            readings[chem] = st.number_input(
                f"{chem} ({UNITS})", min_value=0.0, max_value=2000.0,
                value=defaults[i], step=1.0, key=f"val_{chem}"
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
        st.success("All chemicals meet or exceed thresholds.")

# -----------------------------------
# SeedMatch artifacts & helpers
# -----------------------------------
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

# ---------- Whisper decode utilities ----------
@st.cache_resource
def get_whisper():
    from faster_whisper import WhisperModel
    return WhisperModel("small", device="cpu", compute_type="int8")

def _decode_with_soundfile(data: bytes):
    import soundfile as sf
    y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if hasattr(y, "ndim") and y.ndim > 1: y = y.mean(axis=1)
    return y.astype(np.float32), int(sr)

def _decode_with_av(data: bytes):
    import av, av.audio
    buf = io.BytesIO(data)
    with av.open(buf, mode="r") as container:
        astreams = [s for s in container.streams if s.type == "audio"]
        if not astreams: raise RuntimeError("No audio stream found")
        stream = astreams[0]
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
        chunks = []
        for packet in container.demux(stream):
            for frame in packet.decode():
                out = resampler.resample(frame)
                frames = out if isinstance(out, list) else [out]
                for f in frames:
                    if f is None: continue
                    arr = f.to_ndarray()
                    if arr.ndim == 2: arr = arr.mean(axis=0 if arr.shape[0] <= 8 else 1)
                    chunks.append(arr.astype(np.float32)/32768.0)
        if not chunks: raise RuntimeError("No decodable audio frames")
    y = np.concatenate(chunks)
    return y, 16000

def _resample_any(x: np.ndarray, sr_src: int, sr_dst: int) -> np.ndarray:
    if sr_src == sr_dst or len(x) == 0: return x.astype(np.float32)
    new_len = max(1, int(round(len(x) * sr_dst / sr_src)))
    t_old = np.linspace(0, 1, num=len(x), endpoint=False)
    t_new = np.linspace(0, 1, num=new_len, endpoint=False)
    return np.interp(t_new, t_old, x).astype(np.float32)

def transcribe_audio_bytes(data: bytes, lang_code: Optional[str] = None) -> str:
    try:
        y, sr = _decode_with_soundfile(data)
    except Exception:
        y, sr = _decode_with_av(data)
    y16 = _resample_any(y, sr, 16000)
    with st.spinner("Transcribing…"):
        segments, _ = get_whisper().transcribe(y16, language=lang_code)
    return " ".join([seg.text for seg in segments]).strip()

# -----------------------------------
# Browser TTS "Ask" button (Web Speech)
# -----------------------------------
def ask_button_widget(question_text: str, lang_code: str, key: str):
    """Plays the question using browser speechSynthesis."""
    js_lang = "en-IN" if lang_code == "en" else "hi-IN"
    html = f"""
<div class="row">
  <button id="ask{key}" class="btn">▶ Ask (play question)</button>
  <small id="s{key}" class="mono">Idle</small>
</div>
<script>
(function(){{
  const btn = document.getElementById("ask{key}");
  const st  = document.getElementById("s{key}");
  const SS  = window.speechSynthesis;
  const q   = {json.dumps(question_text)};
  const lang= {json.dumps(js_lang)};
  btn.onclick = () => {{
    try {{
      const u = new SpeechSynthesisUtterance(q);
      u.lang = lang; u.onstart = () => st.textContent="Speaking…"; u.onend=() => st.textContent="Done.";
      SS.cancel(); SS.speak(u);
    }} catch(e) {{
      st.textContent = "TTS error";
    }}
  }};
}})();
</script>
"""
    components.html(html, height=40, scrolling=False)

# -----------------------------------
# Robust numeric extractor (EN + Hindi digits/words)
# -----------------------------------


# 1) Convert Devanagari digits → ASCII digits
_INDIC_DIGIT_MAP = str.maketrans("०१२३४५६७८९", "0123456789")

def _normalize_indic_digits(text: str) -> str:
    return (text or "").translate(_INDIC_DIGIT_MAP)

# 2) Quick coverage of common Hindi number words (0–100 range + tens)
_HI_NUM_WORDS = {
    "शून्य":0, "zero":0,
    "एक":1, "दो":2, "तीन":3, "चार":4, "पांच":5, "पाँच":5, "छह":6, "सात":7, "आठ":8, "नौ":9,
    "दस":10, "ग्यारह":11, "बारह":12, "तेरह":13, "चौदह":14, "पंद्रह":15, "सोलह":16, "सत्रह":17, "अठारह":18, "उन्नीस":19,
    "बीस":20, "इक्कीस":21, "बाईस":22, "तेईस":23, "चौबीस":24, "पच्चीस":25, "छब्बीस":26, "सत्ताईस":27, "अट्ठाईस":28, "उनतीस":29,
    "तीस":30, "चालीस":40, "पचास":50, "साठ":60, "सत्तर":70, "अस्सी":80, "नब्बे":90, "सौ":100
}

def _extract_hindi_word_number(text: str):
    if not text:
        return None
    t = text.replace("°", " ").replace("%", " ").strip()
    # Exact word match first
    for w, n in _HI_NUM_WORDS.items():
        if w in t:
            return float(n)
    # Simple two-token composite like "नब्बे नौ" (90+9) → 99
    tokens = t.split()
    if len(tokens) >= 2:
        tens = next(( _HI_NUM_WORDS.get(tok) for tok in tokens if tok in _HI_NUM_WORDS and _HI_NUM_WORDS[tok] % 10 == 0 and _HI_NUM_WORDS[tok] >= 20 ), None)
        unit = next(( _HI_NUM_WORDS.get(tok) for tok in tokens if tok in _HI_NUM_WORDS and _HI_NUM_WORDS[tok] < 10 ), None)
        if tens is not None and unit is not None:
            return float(tens + unit)
    return None

# Regex for numbers like 20, 20.5, 20,5, also after Indic→ASCII normalization
NUM_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")

def extract_number(txt: str):
    # Normalize Indic digits first (e.g., "२०" → "20")
    t = _normalize_indic_digits((txt or "").replace("°", " ").replace("%", " "))
    m = NUM_RE.search(t)
    if m:
        try:
            return float(m.group(0).replace(",", "."))
        except Exception:
            pass
    # Fallback: Hindi number words (बीस, पच्चीस, सत्तर, सौ, etc.)
    val = _extract_hindi_word_number(txt)
    return val

# --- Helpers for Hindi/English keyword matching (add above INTERVIEW_STEPS) ---
def _prep(txt: str):
    t = (txt or "").strip().lower()
    # ASCII fallback so transliterated Hindi like "samanya/zyada/nahi/thoda" works
    t_ascii = t.encode("ascii", "ignore").decode("ascii")
    return t, t_ascii

def _has_any(text: str, text_ascii: str, terms: set[str]) -> bool:
    return any(w in text for w in terms) or any(w in text_ascii for w in terms)


# -----------------------------------
# Interview steps & parsers
# -----------------------------------
INTERVIEW_STEPS = {
    1: {
        "key": "rainfall",
        "prompt": {"en": "How is the rain recently? Say: low, normal, or heavy.",
                   "hi": "हाल में बारिश कैसी है? बोलें: कम, सामान्य, या अधिक।"},
        "options": [("Low (dry)", 60.0), ("Normal", 150.0), ("Heavy (lots of rain)", 300.0)],
        "parser": lambda txt: (lambda t, ta:
            60.0  if _has_any(t, ta, {"low","कम","kam","kum"}) else
            300.0 if _has_any(t, ta, {"heavy","ज्यादा","अधिक","zyada","jyada","adhik","bhari"}) else
            150.0 if _has_any(t, ta, {"normal","सामान्य","सामान्या","सामान्या","samanya","samaanya"}) else None
        )(*_prep(txt)),
    },
    2: {
        "key": "ph",
        "prompt": {"en": "What is the soil reaction? Say: acidic, neutral, or alkaline.",
                   "hi": "मिट्टी कैसी है? बोलें: अम्लीय, तटस्थ, या क्षारीय।"},
        "options": [("Acidic", 5.8), ("Neutral", 6.8), ("Alkaline", 8.0)],
        "parser": lambda txt: (lambda t, ta:
            5.8 if _has_any(t, ta, {"acid","अम्ल","अम्लीय","amli","amliya"}) else
            8.0 if _has_any(t, ta, {"alk","क्षार","क्षारीय","kshar","kshariya","alkaline"}) else
            6.8 if _has_any(t, ta, {"neutral","तटस्थ","तटस्त","ता तस्ट","tatstha","tatsath"}) else None
        )(*_prep(txt)),
    },
    3: {
        "key": "npk",
        "prompt": {"en": "How much fertilizer did you use? Say: none, some, or high.",
                   "hi": "खाद का उपयोग कितना किया? बोलें: नहीं, थोड़ा, या ज्यादा।"},
        "options": [("None", {"N":60.0,"P":30.0,"K":30.0}),
                    ("Some", {"N":DEFAULTS['N'],"P":DEFAULTS['P'],"K":DEFAULTS['K']}),
                    ("High", {"N":120.0,"P":60.0,"K":60.0})],
        "parser": lambda txt: (lambda t, ta:
            {"N":60.0,"P":30.0,"K":30.0} if _has_any(t, ta, {"none","नहीं","nahin","nahi"}) else
            {"N":120.0,"P":60.0,"K":60.0} if _has_any(t, ta, {"high","ज्यादा","अधिक","zyada","jyada","adhik"}) else
            {"N":DEFAULTS['N'],"P":DEFAULTS['P'],"K":DEFAULTS['K']} if _has_any(t, ta, {"some","थोड़ा","तोड़ा","thoda","thodi"}) else
            None
        )(*_prep(txt)),
    },
   
    # ---- FIXED numeric parsers below ----
    4: {"key": "temperature",
        "prompt": {"en": "What is the temperature in degrees Celsius?", "hi": "तापमान कितने डिग्री सेल्सियस है?"},
        "options": None,
        "parser": lambda txt: extract_number(txt),
    },
    5: {"key": "humidity",
        "prompt": {"en": "What is the humidity percentage?", "hi": "आर्द्रता कितने प्रतिशत है?"},
        "options": None,
        "parser": lambda txt: extract_number(txt),
    },
}

def _commit_option_value(step: int, value):
    if step == 1:
        st.session_state.collected['rainfall'] = float(value)
    elif step == 2:
        st.session_state.collected['ph'] = float(value)
    elif step == 3 and isinstance(value, dict):
        st.session_state.collected.update(value)
    elif step == 4:
        st.session_state.collected['temperature'] = float(value)
    elif step == 5:
        st.session_state.collected['humidity'] = float(value)

def _parse_and_advance(text: str, lang: str):
    step = st.session_state.voice_step
    parser = INTERVIEW_STEPS[step]['parser']
    print(f"[VOICE] step={step} text={text}")  # terminal log
    try:
        val = parser(text or "")
    except Exception:
        val = None
    if val is None:
        st.warning("Couldn't understand. Use the quick options below or try again.")
        return
    # log Q/A
    if "qa_log" not in st.session_state: st.session_state.qa_log = []
    st.session_state.qa_log.append({
        "q": INTERVIEW_STEPS[step]["prompt"].get(lang, INTERVIEW_STEPS[step]["prompt"]["en"]),
        "a": text
    })
    _commit_option_value(step, val)
    if step < len(INTERVIEW_STEPS):
        st.session_state.voice_step += 1
    else:
        st.session_state.voice_done = True
    st.rerun()

# -----------------------------------
# SeedMatch page
# -----------------------------------
def page_seed():
    back_to_home("back_home_seed")
    st.title("🌱 SeedMatch Advisor")

    try:
        model, scaler, classes = load_artifacts()
    except Exception:
        st.error("Model artifacts not found.\nPlace files in ./artifacts/: model_best.joblib, scaler.joblib, classes.joblib")
        st.stop()

    lang = st.selectbox("Language / भाषा", ["en", "hi"], format_func=lambda x: {"en":"English","hi":"हिन्दी"}[x])
    mode = st.radio("Choose mode", ["Farmer Mode (Guided)", "Voice Mode (Ask → Record)", "Pro Mode (Experts)"], horizontal=True)

    colr1, colr2 = st.columns(2)
    with colr1: st.text_input("District (optional)" if lang=="en" else "ज़िला (वैकल्पिक)")
    with colr2: st.text_input("State (optional)" if lang=="en" else "राज्य (वैकल्पिक)")
    st.caption("Units: N,P,K (dataset scale), rainfall (mm)")
    st.divider()

    # ---------- Farmer ----------
    if mode.startswith("Farmer"):
        rain = st.select_slider("How is the rain recently?" if lang=="en" else "हाल में बारिश कैसी है?",
                                options=["Low (dry)" if lang=="en" else "कम (सूखा)",
                                         "Normal" if lang=="en" else "सामान्य",
                                         "Heavy (lots of rain)" if lang=="en" else "अधिक (भारी वर्षा)"],
                                value="Normal" if lang=="en" else "सामान्य")
        rainfall_mm = 60.0 if ("Low" in rain or "कम" in rain) else (300.0 if ("Heavy" in rain or "अधिक" in rain) else 150.0)

        soil = st.select_slider("Soil reaction" if lang=="en" else "मिट्टी की प्रकृति",
                                options=["Acidic" if lang=="en" else "अम्लीय",
                                         "Neutral" if lang=="en" else "तटस्थ",
                                         "Alkaline" if lang=="en" else "क्षारीय"],
                                value="Neutral" if lang=="en" else "तटस्थ")
        ph_val = 5.8 if ("Acidic" in soil or "अम्लीय" in soil) else (8.0 if ("Alkaline" in soil or "क्षारीय" in soil) else 6.8)

        fert = st.select_slider("Fertilizer usage recently" if lang=="en" else "हाल में खाद का उपयोग",
                                options=["None" if lang=="en" else "नहीं",
                                         "Some" if lang=="en" else "थोड़ा",
                                         "High" if lang=="en" else "ज्यादा"],
                                value="Some" if lang=="en" else "थोड़ा")
        if ("None" in fert) or ("नहीं" in fert):
            N_val, P_val, K_val = 60.0, 30.0, 30.0
        elif ("High" in fert) or ("ज्यादा" in fert):
            N_val, P_val, K_val = 120.0, 60.0, 60.0
        else:
            N_val, P_val, K_val = DEFAULTS['N'], DEFAULTS['P'], DEFAULTS['K']

        col1, col2 = st.columns(2)
        with col1:  temp_c = st.slider("Temperature (°C)" if lang=="en" else "तापमान (°C)", 10.0, 45.0, 25.0, 0.5)
        with col2:  hum    = st.slider("Humidity (%)" if lang=="en" else "आर्द्रता (%)", 20.0, 100.0, 70.0, 1.0)

        if st.button("Recommend" if lang=="en" else "सलाह दें", use_container_width=True):
            feats = {'N': N_val, 'P': P_val, 'K': K_val,
                     'temperature': temp_c, 'humidity': hum, 'ph': ph_val, 'rainfall': rainfall_mm}
            topk, _ = predict_topk(model, scaler, classes, feats, k=3)
            st.subheader("Top suggestions" if lang=="en" else "सर्वोत्तम सुझाव")
            for name, p in topk: st.progress(min(max(p,0.0),1.0), text=f"{name}: {p:.2f}")
            with st.expander("Show numeric details used" if lang=="en" else "प्रयुक्त संख्याएँ देखें"): st.json(feats)

    # ---------- Voice (Ask → Record → Auto-Transcribe) ----------
    elif mode.startswith("Voice"):
        if "voice_step" not in st.session_state:
            st.session_state.voice_step = 1
            st.session_state.collected = {}
            st.session_state.voice_done = False
            st.session_state.qa_log = []
            st.session_state.last_audio_step = None
            st.session_state.last_audio_len = None

        step = st.session_state.voice_step
        step_def = INTERVIEW_STEPS[step]
        q_text = step_def["prompt"].get(lang, step_def["prompt"]["en"])

        # Show past Q/A lines
        if st.session_state.qa_log:
            st.subheader("Interview log")
            for i, qa in enumerate(st.session_state.qa_log, start=1):
                st.markdown(f"**Q{i}.** {qa['q']}  \n**A.** {qa['a']}")

        st.progress(step / len(INTERVIEW_STEPS))
        st.write(f"**Q{step}. {q_text}**")

        # 1) Ask (play the question)
        ask_button_widget(q_text, lang_code=lang, key=f"speak_{step}")

        # 2) Record → Stop → Auto-transcribe & advance (using streamlit-mic-recorder)
        try:
            from streamlit_mic_recorder import mic_recorder
            rec = mic_recorder(start_prompt="🎙️ Record", stop_prompt="■ Stop",
                               key=f"rec_s{step}", use_container_width=True)
            if rec and rec.get("bytes"):
                # Avoid double-processing the same audio across reruns
                audio_len = len(rec["bytes"])
                if st.session_state.get("last_audio_step") != step or st.session_state.get("last_audio_len") != audio_len:
                    st.audio(rec["bytes"])
                    st.session_state["last_audio_step"] = step
                    st.session_state["last_audio_len"] = audio_len
                    text = transcribe_audio_bytes(rec["bytes"], lang_code=lang)  # 'en' or 'hi'
                    st.success(f"You said: “{text or '(empty)'}”")
                    _parse_and_advance(text, lang)
        except Exception:
            st.info("Install the recorder: pip install streamlit-mic-recorder")

        # Quick option buttons (tap instead of speaking)
        st.divider()
        if step_def.get("options"):
            cols = st.columns(len(step_def["options"]))
            for i, (label, value) in enumerate(step_def["options"]):
                if cols[i].button(label):
                    if "qa_log" not in st.session_state: st.session_state.qa_log = []
                    st.session_state.qa_log.append({"q": q_text, "a": label})
                    _commit_option_value(step, value)
                    if step < len(INTERVIEW_STEPS): st.session_state.voice_step += 1
                    else: st.session_state.voice_done = True
                    st.rerun()

        # Final recommendation
        if st.session_state.voice_done:
            data = {**DEFAULTS, **st.session_state.collected}
            data['temperature'] = float(np.clip(data.get('temperature', DEFAULTS['temperature']),  -10, 60))
            data['humidity']    = float(np.clip(data.get('humidity',    DEFAULTS['humidity']),      0, 100))
            data['ph']          = float(np.clip(data.get('ph',          DEFAULTS['ph']),          3.5, 9.5))
            data['rainfall']    = float(np.clip(data.get('rainfall',    DEFAULTS['rainfall']),      0, 500))
            for k in ['N','P','K']: data[k] = float(np.clip(data.get(k, DEFAULTS[k]), 0, 200))
            print("[VOICE] DONE collected:", data)

            topk, _ = predict_topk(model, scaler, classes, data, k=3)
            st.subheader("Top suggestions" if lang=="en" else "सर्वोत्तम सुझाव")
            for name, p in topk: st.progress(min(max(p,0.0),1.0), text=f"{name}: {p:.2f}")
            with st.expander("Show numeric details used" if lang=="en" else "प्रयुक्त संख्याएँ देखें"):
                st.json(data)

            if st.button("Start Over"):
                for k in ["voice_step","collected","voice_done","qa_log","last_audio_step","last_audio_len"]:
                    st.session_state.pop(k, None)
                st.rerun()

    # ---------- Pro ----------
    else:
        st.caption("Enter exact values from soil test / sensors:" if lang=="en" else "मिट्टी/सेंसर से सटीक मान दर्ज करें:")
        c1, c2, c3 = st.columns(3)
        with c1:
            N_val = st.number_input("N", 0.0, 200.0, 90.0, 1.0)
            temp_c = st.number_input("temperature (°C)" if lang=="en" else "तापमान (°C)", -10.0, 60.0, 25.0, 0.5)
            ph_val = st.number_input("ph", 3.5, 9.5, 6.8, 0.1)
        with c2:
            P_val = st.number_input("P", 0.0, 200.0, 42.0, 1.0)
            hum = st.number_input("humidity (%)" if lang=="en" else "आर्द्रता (%)", 0.0, 100.0, 70.0, 1.0)
            rainfall_mm = st.number_input("rainfall (mm)", 0.0, 500.0, 150.0, 1.0)
        with c3:
            K_val = st.number_input("K", 0.0, 200.0, 43.0, 1.0)

        if st.button("Recommend" if lang=="en" else "सलाह दें", use_container_width=True):
            features = {'N': N_val, 'P': P_val, 'K': K_val,
                        'temperature': temp_c, 'humidity': hum,
                        'ph': ph_val, 'rainfall': rainfall_mm}
            topk, _ = predict_topk(model, scaler, classes, features, k=3)
            st.subheader("Top suggestions" if lang=="en" else "सर्वोत्तम सुझाव")
            for name, p in topk: st.progress(min(max(p,0.0),1.0), text=f"{name}: {p:.2f}")
            with st.expander("Show numeric details used" if lang=="en" else "प्रयुक्त संख्याएँ देखें"):
                st.json(features)

# -----------------------------------
# Home
# -----------------------------------
def page_home():
    st.title("🚜 Project Demo")
    st.write(
        "This demo showcases two modules for smart farming:\n"
        "1) **Spray Chemical Module** — threshold-based control for Nitrogen, Phosphorus, and Potassium.\n"
        "2) **SeedMatch Advisor** — AI-powered recommendation system using soil and weather conditions to suggest the most suitable crops for smart agriculture."
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧪 Spray Chemical Module <span class='badge'>N-P-K</span>", unsafe_allow_html=True)
    st.write("Set nutrient **thresholds** and enter **soil readings**. If a value is below its threshold, you’ll see a **colored spray button**.")
    if st.button("Open Spray Module"):
        goto("spray")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🌱 SeedMatch Advisor <span class='badge'>Voice + Pro</span>", unsafe_allow_html=True)
    st.write(
        "The **SeedMatch Advisor** helps farmers identify the right crop based on local **N, P, K, pH, rainfall, temperature, and humidity** data. "
        "It offers three modes — **Farmer Mode (guided sliders)**, **Voice Mode (speech-based Q&A)**, and **Pro Mode (manual expert inputs)** — making agriculture smarter, data-driven, and accessible for everyone."
    )
    if st.button("Open Seed Advisor"):
        goto("seed")
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption("Tip: Use the Back button on module pages to return here.")

# -----------------------------------
# Router
# -----------------------------------
ROUTES = {"home": page_home, "spray": page_spray, "seed": page_seed}
ROUTES[st.session_state.route]()
