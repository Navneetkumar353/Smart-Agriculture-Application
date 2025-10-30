# Smart Agriculture — Streamlit 
An interactive **Streamlit-based Smart Farming app** with two modules:

* **🧪 Spray Chemical Module** – Checks soil nutrient thresholds (N, P, K) and gives color-coded spray recommendations.
* **🌱 SeedMatch Advisor** – Predicts suitable crops using soil and weather data (N, P, K, pH, rainfall, temperature, humidity).
  Includes multilingual (English + Hindi) and voice-enabled modes for an easy user experience.

---

This project demonstrates an **AI-powered Smart Agriculture System** built with Streamlit.
It helps farmers analyze soil nutrients and receive intelligent crop recommendations.

---

## 🌾 Features
- **🧪 Spray Chemical Module:** Threshold-based control for N, P, and K nutrients.
- **🌱 SeedMatch Advisor:** ML-based crop prediction (Farmer / Voice / Pro modes).
- **Multilingual support:** English and Hindi.
- **Voice Mode:** Works with your system microphone (no external FFmpeg dependency).

---

## 🧱 1. Prerequisites (Windows)

- **Python 3.10 or newer**  
  Download from [python.org](https://www.python.org/downloads/)
- **Pip** (comes with Python)
- (Optional) **Git** for cloning repositories

---

## ⚙️ 2. Setup Virtual Environment

Open PowerShell in your project folder and run:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
````

## 📦 3. Install Required Packages

Install all dependencies using:

```powershell
pip install -r requirements.txt
```

Your `requirements.txt` already includes:

* Streamlit core
* Mic recorder widget
* Faster Whisper (for voice transcription)
* Sound and audio processing libs
* ML stack (scikit-learn, xgboost, numpy, joblib)

---

## 📁 4. Folder Structure

```
repo/
├─ final.py
├─ requirements.txt
├─ README.md
└─ artifacts/
   ├─ model_best.joblib
   ├─ scaler.joblib
   └─ classes.joblib
```

> ⚠️ Make sure all `.joblib` model files are inside the `artifacts` folder.

---

## ▶️ 5. Run the Application

```powershell
streamlit run final.py
```

or (if Streamlit isn’t on PATH):

```powershell
py -m streamlit run final.py
```

Then open the browser link displayed in the terminal
(default: [http://localhost:8501](http://localhost:8501))

---

## 🧠 6. How to Use

### 🧪 Spray Chemical Module

* Set thresholds in the sidebar.
* Input soil readings (N, P, K).
* The app shows which chemicals need spraying.

### 🌱 SeedMatch Advisor

Choose mode:

* **Farmer Mode:** Guided slider inputs.
* **Voice Mode:** Speak answers directly in English or Hindi.
* **Pro Mode:** Enter all numeric inputs manually.

Click **Recommend** to see the top 3 suggested crops.

---

## ❗ Troubleshooting

| Issue                              | Solution                                        |
| ---------------------------------- | ----------------------------------------------- |
| “Model artifacts not found”        | Place `.joblib` files inside `artifacts/`.      |
| “streamlit-mic-recorder not found” | Run `pip install -r requirements.txt` again.    |
| “Voice not recording”              | Allow browser microphone access (Chrome/Edge).  |
| “Port already in use”              | Run `streamlit run final.py --server.port 8502` |

---

## 🧩 7. About

Developed as a **smart agriculture demo** combining AI, Streamlit, and machine learning
for practical decision-making in crop selection and soil management.


