# Smart Agriculture — Streamlit (Windows Setup)

Two modules:
- **🧪 Spray Chemical Module**: Threshold-based N, P, K spray hints.
- **🌱 SeedMatch Advisor**: AI crop suggestions (Farmer / Voice / Pro modes).

> **Artifacts required** (place these in a folder named `artifacts` in the repo root):
> - `artifacts/model_best.joblib`
> - `artifacts/scaler.joblib`
> - `artifacts/classes.joblib`

---

## 1) Prerequisites (Windows 10/11)

- **Python 3.10+** (install from https://www.python.org/ if needed)
- **Git** (optional, for cloning)
- **FFmpeg (optional but recommended for robust audio decoding)**  
  - With **winget**: `winget install Gyan.FFmpeg`  
  - With **Chocolatey** (admin PowerShell): `choco install ffmpeg -y`

---

## 2) Create & Activate a Virtual Environment

Open **PowerShell** in your project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
