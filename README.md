# MRV Verification & Insights Portal (Streamlit)

Professional, read-only MRV visualization portal built with Streamlit. Includes: Baseline, Monitoring, Map, Data Explorer, QC Status, Version History/Audit, Verification Dashboard, and Downloads.

## Run locally

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1) Push this repo to GitHub (see below).
2) Go to `https://share.streamlit.io` and sign in.
3) Click New app → pick your GitHub repo, choose branch `main`, and set the file to `app.py`.
4) Advanced settings:
   - Python dependencies: `requirements.txt` (auto-detected)
   - Optional secrets (Settings → Secrets):
     ```
     MAPBOX_API_KEY = "your_mapbox_token"   # if map tiles require it
     ```
5) Deploy.

## Push to GitHub

If you haven't created a GitHub repo yet:

1) Create an empty repo on GitHub (e.g. `MRV_Visuals_Angola`).
2) In PowerShell from the project folder:

```bash
git init
git branch -M main
git add .
git commit -m "Initial commit: MRV Streamlit portal"
git remote add origin https://github.com/<your-github-username>/MRV_Visuals_Angola.git
git push -u origin main
```

Update paths if your repo name differs.

## Configuration

- App entry point: `app.py`
- Dependencies: `requirements.txt`
- Optional theme: `.streamlit/config.toml`
- Optional tokens: `.streamlit/secrets.toml` (not committed)

## Notes

- The app loads from `./data/` if present, otherwise uses realistic sample data.
- For evidence documents, place files in `./docs/` (not committed by default).


