# app.py
# Smart Agri demo: Spray module + SeedMatch (Farmer / Voice / Pro)
# Voice: Web Speech API (no WebRTC / no audio files). Prints transcripts to your terminal.

import json
import numpy as np
import streamlit as st
import joblib
import streamlit.components.v1 as components

# ----------------------------
# Setup / Router
# ----------------------------
st.set_page_config(page_title="Project Demo", page_icon="🧪", layout="centered")

if "route" not in st.session_state:
    st.session_state.route = "home"

def goto(route: str):
    st.session_state.route = route
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

def back_to_home(key: str):
    if st.button("↩️  Back to Home", key=key):
        goto("home")

# ----------------------------
# Shared CSS (card style + dark mode)
# ----------------------------
st.markdown("""
<style>
.card {
  padding: 1.0rem 1.2rem;
  border: 1px solid rgba(49,51,63,0.2);
  border-radius: 14px;
  margin-bottom: 0.9rem;
  background: rgba(250,250,252,0.65);
}
.card h3 { margin: 0 0 0.35rem 0; }
.badge {
  display:inline-block; padding:0.15rem 0.5rem; border-radius:999px;
  background:#eef; font-size:0.8rem; margin-left:0.4rem;
}
@media (prefers-color-scheme: dark) {
  .card { background: rgba(30,41,59,0.55); border-color: rgba(148,163,184,0.25); }
  .badge { background:#334155; color:#e2e8f0; }
}
</style>
""", unsafe_allow_html=True)

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
    st.warning("Needs spraying: " + ", ".join(needs_spray)) if needs_spray else st.success("All chemicals meet or exceed thresholds.")

# ----------------------------
# SeedMatch: artifacts & helpers
# ----------------------------
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

# ----------------------------
# Voice component (dual-protocol + terminal logging)
# ----------------------------
def voice_interview_component(lang="en"):
    """
    Returns dicts when:
      - a final phrase is recognized: {"partial": "...", "step": int, "ts": ...}
      - interview finishes: {"done": true, "collected": {...}, "ts": ...}
    Returns None if no new message this run.
    """
    QUESTIONS = {
        1: {"en":"How is the rain recently? Say low, normal, or heavy.",
            "hi":"हाल में बारिश कैसी है? बोलें: कम, सामान्य, या अधिक।"},
        2: {"en":"What is the soil reaction? Say acidic, neutral, or alkaline.",
            "hi":"मिट्टी कैसी है? बोलें: अम्लीय, तटस्थ, या क्षारीय।"},
        3: {"en":"How much fertilizer did you use recently? Say none, some, or high.",
            "hi":"हाल में खाद का उपयोग कितना किया? बोलें: नहीं, थोड़ा, या ज्यादा।"},
        4: {"en":"What is the temperature in degrees Celsius?",
            "hi":"तापमान कितने डिग्री सेल्सियस है?"},
        5: {"en":"What is the humidity percentage?",
            "hi":"आर्द्रता कितने प्रतिशत है?"}
    }
    js_lang = "en-IN" if lang == "en" else "hi-IN"
    qdata = {str(i): QUESTIONS[i][lang] for i in range(1,6)}

    html = """
<style>
:root { color-scheme: light dark; }
html,body{background:transparent;margin:0;padding:0}
body{font:14px system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#111}
.box{border:1px solid #4443;border-radius:12px;padding:12px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.btn{background:#efefef;border:0;padding:8px 12px;border-radius:8px;cursor:pointer}
.btn.pri{background:#ffd9d9}
#st,#q,#out{color:#111}
#out{background:#fafafa;border:1px dashed #aaa7;min-height:42px;padding:8px;border-radius:8px;margin-top:8px;color:#111}
@media (prefers-color-scheme:dark){
  body{color:#e5e7eb} #st,#q,#out{color:#e5e7eb!important} #out{background:#0f172a;border-color:#475569}
  .btn{background:#334155;color:#e2e8f0}.btn.pri{background:#7f1d1d;color:#f8fafc}
}
</style>
<div class="box">
  <div class="row">
    <button id="b_begin" class="btn pri">▶ Begin Interview</button>
    <button id="b_stop" class="btn">■ Stop</button>
    <span id="st">Idle</span>
  </div>
  <div id="q" style="margin-top:6px;"></div>
  <div id="out"></div>
</div>
<script>
(function(){
  // --- Dual-protocol sender: modern API + legacy postMessage fallback ---
  function sendVal(val){
    let payload = JSON.stringify(val);
    // modern (if available)
    try { if (window.Streamlit && Streamlit.setComponentValue) { Streamlit.setComponentValue(payload); } } catch(e){}
    // legacy fallback
    try { window.parent.postMessage({isStreamlitMessage:true,type:"streamlit:setComponentValue",value:payload},"*"); } catch(e){}
    // nudge height + rerun (legacy)
    try { window.parent.postMessage({isStreamlitMessage:true,type:"streamlit:setFrameHeight",height:document.body.scrollHeight},"*"); } catch(e){}
    try { window.parent.postMessage({isStreamlitMessage:true,type:"streamlit:rerun"},"*"); } catch(e){}
  }
  function ready(){
    // modern
    try { if (window.Streamlit && Streamlit.setComponentReady) { Streamlit.setComponentReady(); } } catch(e){}
    try { if (window.Streamlit && Streamlit.setFrameHeight) { Streamlit.setFrameHeight(document.body.scrollHeight); } } catch(e){}
    // legacy announce
    try { window.parent.postMessage({isStreamlitMessage:true,type:"streamlit:componentReady",apiVersion:1},"*"); } catch(e){}
    try { window.parent.postMessage({isStreamlitMessage:true,type:"streamlit:setFrameHeight",height:document.body.scrollHeight},"*"); } catch(e){}
  }
  function onRender(){ try { if (window.Streamlit && Streamlit.setFrameHeight) { Streamlit.setFrameHeight(document.body.scrollHeight);} } catch(e){} }
  if (window.Streamlit && window.Streamlit.events) {
    window.Streamlit.events.addEventListener(window.Streamlit.RENDER_EVENT, onRender);
  }
  ready();

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const SS = window.speechSynthesis;
  const st = document.getElementById("st"), q = document.getElementById("q"), out = document.getElementById("out");
  const bBegin = document.getElementById("b_begin"), bStop = document.getElementById("b_stop");
  if (!SR) { st.innerText = "Not supported (use Chrome/Edge)."; sendVal({error:"unsupported", ts: Date.now()}); return; }

  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({audio:true}).then(s=>{if(s&&s.getTracks)s.getTracks().forEach(t=>t.stop())}).catch(()=>{});
  }

  const rec = new SR(); rec.lang = "%%LANG%%"; rec.interimResults = true; rec.continuous = true; rec.maxAlternatives = 1;
  let step = 1, running = false; const Q = %%QDATA%%;

  function speak(text, cb){ try{ const u = new SpeechSynthesisUtterance(text); u.lang="%%LANG%%"; u.onend=()=>cb&&cb(); SS.cancel(); SS.speak(u);}catch(e){ if(cb) cb(); } }
  function showQ(){ q.innerText = "Q" + step + ": " + (Q[String(step)] || ""); }
  function sendMany(obj){ let n=0; const t=setInterval(()=>{ obj.ts = Date.now(); sendVal(obj); n++; if(n>10) clearInterval(t); }, 250); }
  function finish(col){ try{ rec.stop(); }catch(e){} st.innerText = "Finished"; sendMany({done:true, collected: col}); }

  function interpret(t){
    t=(t||"").toLowerCase(); window.__col = window.__col || {};
    if (step===1){const low=["low","कम"], high=["heavy","अधिक","ज्यादा"];
      if(low.some(w=>t.includes(w))){window.__col.rainfall=60;return true;}
      if(high.some(w=>t.includes(w))){window.__col.rainfall=300;return true;}
      if(t.includes("normal")||t.includes("सामान्य")){window.__col.rainfall=150;return true;} return false;}
    if (step===2){ if(t.includes("acid")||t.includes("अम्ल")){window.__col.ph=5.8;return true;}
                   if(t.includes("alk")||t.includes("क्षार")){window.__col.ph=8.0;return true;}
                   if(t.includes("neutral")||t.includes("तटस्थ")){window.__col.ph=6.8;return true;} return false;}
    if (step===3){ if(t.includes("none")||t.includes("नहीं")){window.__col.N=60;window.__col.P=30;window.__col.K=30;return true;}
                   if(t.includes("high")||t.includes("ज्यादा")||t.includes("अधिक")){window.__col.N=120;window.__col.P=60;window.__col.K=60;return true;}
                   if(t.includes("some")||t.includes("थोड़ा")){window.__col.N=90;window.__col.P=42;window.__col.K=43;return true;} return false;}
    if (step===4){const m=t.match(/-?\\d+(?:\\.\\d+)?/); if(m){window.__col.temperature=parseFloat(m[0]);return true;} return false;}
    if (step===5){const m=t.match(/-?\\d+(?:\\.\\d+)?/); if(m){window.__col.humidity=parseFloat(m[0]);return true;} return false;}
    return false;
  }

  function askNext(){
    if (!running) return;
    if (step > 5) { finish(window.__col||{}); return; }
    showQ(); try{ rec.stop(); }catch(e){} st.innerText = "Asking…";
    speak(Q[String(step)], ()=>{ try{ rec.start(); st.innerText = "Listening…"; }catch(e){} });
  }

  rec.onresult = (ev) => {
    for (let i=ev.resultIndex;i<ev.results.length;i++){
      const txt = ev.results[i][0].transcript;
      if (ev.results[i].isFinal){
        const finalT = (txt||"").trim(); out.innerText = finalT;
        sendVal({partial: finalT, step: step, ts: Date.now()});   // Python prints this
        if (interpret(finalT)){ step++; askNext(); } else { st.innerText = "Didn't catch. Please repeat."; }
      }
    }
  };
  rec.onerror = (e) => { st.innerText = "Error: " + e.error; };
  rec.onend   = () => { /* controlled */ };

  bBegin.onclick = () => { if (running) return; running = true; window.__col = {}; step = 1; out.innerText = ""; st.innerText = "Starting…"; askNext(); };
  bStop.onclick  = () => { running = false; try{ rec.stop(); }catch(e){} st.innerText = "Stopped by user."; sendMany({done:true, collected:(window.__col||{})}); };
})();
</script>
"""
    html = html.replace("%%LANG%%", js_lang).replace("%%QDATA%%", json.dumps(qdata))
    # Older Streamlit builds don't accept 'key' here
    val = components.html(html, height=320, scrolling=False)
    if isinstance(val, str) and val.strip():
        try:
            return json.loads(val)
        except Exception:
            return {"error": "parse"}
    return val  # None if no new message this run

# ----------------------------
# SeedMatch page
# ----------------------------
def page_seed():
    back_to_home("back_home_seed")
    st.title("🌱 SeedMatch Advisor")

    # Load model artifacts
    try:
        model, scaler, classes = load_artifacts()
    except Exception:
        st.error("Model artifacts not found.\nPlace files in ./artifacts/: model_best.joblib, scaler.joblib, classes.joblib")
        st.stop()

    lang = st.selectbox("Language / भाषा", ["en", "hi"], format_func=lambda x: {"en":"English","hi":"हिन्दी"}[x])
    mode = st.radio("Choose mode", ["Farmer Mode (Guided)", "Voice Mode (Auto Interview)", "Pro Mode (Experts)"], horizontal=True)

    colr1, colr2 = st.columns(2)
    with colr1: st.text_input("District (optional)" if lang=="en" else "ज़िला (वैकल्पिक)")
    with colr2: st.text_input("State (optional)" if lang=="en" else "राज्य (वैकल्पिक)")
    st.caption("Units: N,P,K (dataset scale), rainfall (mm)")
    st.divider()

    # Farmer mode
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
            N_val, P_val, K_val = 90.0, 42.0, 43.0

        col1, col2 = st.columns(2)
        with col1:  temp_c = st.slider("Temperature (°C)" if lang=="en" else "तापमान (°C)", 10.0, 45.0, 25.0, 0.5)
        with col2:  hum    = st.slider("Humidity (%)" if lang=="en" else "आर्द्रता (%)", 20.0, 100.0, 70.0, 1.0)

        if st.button("Recommend" if lang=="en" else "सलाह दें", use_container_width=True):
            features = {'N': N_val, 'P': P_val, 'K': K_val,
                        'temperature': temp_c, 'humidity': hum,
                        'ph': ph_val, 'rainfall': rainfall_mm}
            topk, _ = predict_topk(model, scaler, classes, features, k=3)
            st.subheader("Top suggestions" if lang=="en" else "सर्वोत्तम सुझाव")
            for name, p in topk:
                st.progress(min(max(p,0.0),1.0), text=f"{name}: {p:.2f}")
            with st.expander("Show numeric details used" if lang=="en" else "प्रयुक्त संख्याएँ देखें"):
                st.json(features)

    # Voice mode
    elif mode.startswith("Voice"):
        st.info("Click **Begin Interview**. It will speak, listen, auto-advance, then send data for recommendation. Works in Chrome/Edge.")

        # Persist last payload + transcript log
        if "voice_payload" not in st.session_state:
            st.session_state.voice_payload = None
        if "voice_log" not in st.session_state:
            st.session_state.voice_log = []

        res = voice_interview_component(lang=lang)
        st.button("⟳ Refresh result (if needed)")

        # Process messages from iframe
        if isinstance(res, dict):
            # Partial transcript -> print to terminal & remember
            if "partial" in res:
                txt = (res.get("partial") or "").strip()
                step = res.get("step")
                if txt and (not st.session_state.voice_log or st.session_state.voice_log[-1] != txt):
                    st.session_state.voice_log.append(txt)
                    print(f"[VOICE] step={step} text={txt}")  # <-- appears in your terminal

            # Final payload -> save & rerun to render
            if res.get("done"):
                st.session_state.voice_payload = res
                print("[VOICE] DONE payload:", res)  # <-- final object in terminal
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()

        # Small transcript viewer
        if st.session_state.voice_log:
            with st.expander("Transcript log (latest first)"):
                for t in reversed(st.session_state.voice_log[-10:]):
                    st.write("• ", t)

        # If payload exists, compute recommendation
        payload = st.session_state.voice_payload
        if isinstance(payload, dict) and payload.get("done"):
            data = {**DEFAULTS, **(payload.get("collected") or {})}
            # Clamp ranges
            data['temperature'] = float(np.clip(data['temperature'],  -10, 60))
            data['humidity']    = float(np.clip(data['humidity'],      0, 100))
            data['ph']          = float(np.clip(data['ph'],          3.5, 9.5))
            data['rainfall']    = float(np.clip(data['rainfall'],      0, 500))
            data['N']           = float(np.clip(data['N'],              0, 200))
            data['P']           = float(np.clip(data['P'],              0, 200))
            data['K']           = float(np.clip(data['K'],              0, 200))

            topk, _ = predict_topk(model, scaler, classes, data, k=3)
            st.subheader("Top suggestions" if lang=="en" else "सर्वोत्तम सुझाव")
            for name, p in topk:
                st.progress(min(max(p,0.0),1.0), text=f"{name}: {p:.2f}")
            with st.expander("Show numeric details used" if lang=="en" else "प्रयुक्त संख्याएँ देखें"):
                st.json(data)

            # Utility: clear
            if st.button("Clear last voice result"):
                st.session_state.voice_payload = None
                st.experimental_rerun()

    # Pro mode
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
            for name, p in topk:
                st.progress(min(max(p,0.0),1.0), text=f"{name}: {p:.2f}")
            with st.expander("Show numeric details used" if lang=="en" else "प्रयुक्त संख्याएँ देखें"):
                st.json(features)

# ----------------------------
# Home (card layout)
# ----------------------------
def page_home():
    st.title("🚜 Project Demo")
    st.write(
        "This demo showcases two modules for smart farming:\n"
        "1) **Spray Chemical Module** — threshold-based control for Nitrogen, Phosphorus, and Potassium.\n"
        "2) **SeedMatch Advisor** — weather/field informed seed recommendations with **Voice interview** and **Pro** modes."
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧪 Spray Chemical Module <span class='badge'>N-P-K</span>", unsafe_allow_html=True)
    st.write("Set nutrient **thresholds** and enter **soil readings**. If a value is below its threshold, you’ll see a **colored spray button**.")
    if st.button("Open Spray Module"):
        goto("spray")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🌱 SeedMatch Advisor <span class='badge'>Voice + Pro</span>", unsafe_allow_html=True)
    st.write("Use **Farmer Mode** (simple choices), **Voice Mode** (auto interview), or **Pro Mode** (direct inputs).")
    if st.button("Open Seed Advisor"):
        goto("seed")
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption("Tip: Use the Back button on module pages to return here.")

# ----------------------------
# Router
# ----------------------------
ROUTES = {"home": page_home, "spray": page_spray, "seed": page_seed}
ROUTES[st.session_state.route]()
