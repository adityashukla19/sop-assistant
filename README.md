
# 🧭 SOP Assistant — Streamlit Cloud Ready

Turn SOP files into clear, numbered, step-by-step instructions.  
**This repo is ready to deploy on Streamlit Community Cloud.**

## Files
- `app.py` — Streamlit app (reads `OPENAI_API_KEY` from Streamlit **Secrets**)
- `requirements.txt` — Python deps for Cloud build
- `.streamlit/config.toml` — dark theme
- `.env.example` / `.env` — local dev only (do **not** commit real keys)
- `sops/` — sample SOPs

---

## 1) Local quickstart (optional)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

# Set your key (either)
# A) create .env from example and paste your key
# B) export OPENAI_API_KEY in your shell

streamlit run app.py
```

---

## 2) Deploy to Streamlit Community Cloud (shareable link)
1. Create a new **GitHub repository** (public or private).
2. Upload all files from this folder.
3. Go to **https://share.streamlit.io** and sign in with GitHub.
4. Click **New app** → select your repo + branch → set **Main file path** to `app.py` → **Deploy**.
5. After first build, open **App → Settings → Secrets** and add:
   ```toml
   OPENAI_API_KEY="sk-your-real-key"
   OPENAI_MODEL="gpt-4o-mini"
   OPENAI_EMBED_MODEL="text-embedding-3-small"
   ```
6. Click **Rerun**. Your app URL looks like: `https://your-app-name.streamlit.app` — share it with anyone.

### Optional hardening
- **Permissions**: App → Settings → **Share** → invite specific emails (viewer auth).
- **Usage limits**: Consider cheaper model and limit file size/count.
- **Rotate key**: Update Secrets any time; redeploy is automatic.

---

## 3) PyCharm (local dev) — API key setup
1. **Open project** in PyCharm.
2. Create/select interpreter (venv).
3. `pip install -r requirements.txt` in the PyCharm terminal.
4. Add `.env` (from `.env.example`) and paste your key.
5. Run: right‑click `app.py` → **Run** (or `streamlit run app.py`).

---

## 4) Notes
- Streamlit Cloud cannot run local models like Ollama — the app uses OpenAI only when hosted.
- All uploads are processed on the Streamlit Cloud server. Avoid sensitive data unless you trust the hosting.
