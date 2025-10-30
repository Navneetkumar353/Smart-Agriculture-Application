# app.py
# Streamlit Multi-Module Demo (fixed back button + Seed page placeholder)
# Run: pip install streamlit && streamlit run app.py

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
    # Use a plain button to control the custom router
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

    # ---- (Optional) st.session_state log viewer ----
    with st.expander("Debug / Last Action (optional)"):
        st.write("Use this area to show logs or last actions once you wire up your drone controls.")

# ----------------------------
# Module 2: SeedMatch Advisor (Placeholder)
# ----------------------------
def page_seed():
    back_to_home("back_home_seed")
    st.title("🌱 SeedMatch Advisor")
    st.info("Coming soon…")
    st.caption("This module will use weather and field conditions to suggest seeds best suited for the season.")

# ----------------------------
# Home Page
# ----------------------------
def page_home():
    st.title("🚜 Project Demo")
    st.write(
        "This demo showcases two modules for smart farming:\n"
        "1) **Spray Chemical Module** — threshold-based control for Nitrogen, Phosphorus, and Potassium.\n"
        "2) **SeedMatch Advisor** — weather-informed seed recommendations (coming soon)."
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
    st.markdown("### 🌱 SeedMatch Advisor <span class='badge'>Prototype</span>", unsafe_allow_html=True)
    st.write("Weather-informed seed recommendations. **Coming soon…**")
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
