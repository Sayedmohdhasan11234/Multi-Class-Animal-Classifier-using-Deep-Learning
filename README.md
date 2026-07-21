# SpeciesID — Render-ready (lightweight) build

This version is built to fit Render's free tier (512MB RAM). The trick:
it uses **TFLite + `ai-edge-litert`** for inference instead of full
TensorFlow. Full TensorFlow alone is 400-600MB and will crash a 512MB
instance on its own — this build avoids installing it entirely at runtime.

## Folder structure
```
animal_classifier_render/
├── app.py                    # lightweight Flask app (no TensorFlow)
├── convert_to_tflite.py      # one-time conversion script (needs full TF)
├── animal_classifier.tflite  # <-- YOU generate this, see Step 1
├── requirements.txt
├── Procfile
├── render.yaml
├── templates/
│   └── index.html
```

## Step 1 — convert your model to TFLite (one-time, do this NOT on Render)

Run this wherever your `.keras` model already lives with full TensorFlow
installed — Google Colab, Kaggle, or your local training machine:

1. Put `convert_to_tflite.py` next to your `animal_classifier_final.keras`
   file.
2. Run:
   ```bash
   python convert_to_tflite.py
   ```
3. This produces `animal_classifier.tflite` (typically 4-6x smaller than
   the original `.keras` file thanks to quantization).
4. Copy `animal_classifier.tflite` into this project folder, next to
   `app.py`. You do NOT need the original `.keras` file for deployment —
   only the `.tflite` file.

## Step 2 — test locally (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Visit http://127.0.0.1:5000 and confirm predictions still work correctly
after the TFLite conversion.

## Step 3 — push to GitHub

Render deploys from a Git repository, so this folder needs to be a repo:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
Make sure `animal_classifier.tflite` is committed too (check it's not
accidentally in a `.gitignore`).

## Step 4 — deploy on Render

**Option A — using render.yaml (recommended):**
1. Go to https://dashboard.render.com → New → Blueprint
2. Connect your GitHub repo
3. Render reads `render.yaml` automatically and configures everything

**Option B — manual setup:**
1. Go to https://dashboard.render.com → New → Web Service
2. Connect your GitHub repo
3. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** Free
4. Click Create Web Service

Render will build and give you a live URL like
`https://species-id.onrender.com`.

## Why this fits under 512MB

| Component            | Full TensorFlow build | This lightweight build |
|----------------------|------------------------|-------------------------|
| ML runtime            | ~450-600MB            | ~10-20MB (ai-edge-litert) |
| Model file             | .keras (uncompressed) | .tflite (quantized, smaller) |
| Typical RAM at runtime | 700MB-1GB+            | ~150-250MB              |

## Notes

- Render's free tier spins down after inactivity, so the first request
  after idle time will be slow (cold start) — this is normal.
- If accuracy drops noticeably after quantization, open
  `convert_to_tflite.py` and remove the `converter.optimizations` line to
  export a full-precision (larger) `.tflite` file instead — it'll still
  be far smaller than shipping full TensorFlow.
- The same technique (TFLite + ai-edge-litert) works for your Pill
  Identification project too, if you want to deploy that one to Render
  as well — just ask and I'll build that version too.
