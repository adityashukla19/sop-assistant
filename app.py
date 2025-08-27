
import os
import base64
from typing import List, Dict, Tuple
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# Optional readers
try:
    from PyPDF2 import PdfReader
    HAS_PDF = True
except Exception:
    HAS_PDF = False

try:
    import docx  # python-docx
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

# ---------- Config & Theme ----------
st.set_page_config(page_title="SOP Assistant", page_icon="🧭", layout="wide")

# Load .env for local dev (Streamlit Cloud uses st.secrets)
load_dotenv()

# Secrets → then env → fallback empty
def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)

OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "")
OPENAI_MODEL = get_secret("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = get_secret("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Import OpenAI only after we read the key
from openai import OpenAI
client = None
if OPENAI_API_KEY:
    try:
        client = OpenAI()  # reads key from env/secrets automatically
    except Exception:
        client = None

# ---------- Styles ----------
st.markdown(
    """
    <style>
    .block-container {max-width: 1100px; padding-top: 1.2rem;}
    .section-card {background: #101827; padding: 18px 20px; border-radius: 18px; box-shadow: 0 6px 18px rgba(0,0,0,0.25); border: 1px solid rgba(255,255,255,0.06);}
    h1, h2, h3 { color: #e6e8ee; }
    .muted { color: #a4a8b7; }
    .hr {height:1px; background:linear-gradient(90deg, transparent, #243b6b, transparent); border:none; margin:18px 0;}
    .kpi {background:#101827; padding:10px 14px; border-radius:12px; display:inline-block; margin-right:10px; border:1px solid rgba(255,255,255,0.06);}
    a.download {background:#0e1426; padding:6px 12px; border-radius:10px; border:1px solid #243b6b; text-decoration:none;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("# 🧭 SOP Assistant")
st.write("Upload your SOPs and get clear, numbered, step-by-step instructions you can share.")

# ---------- Helpers ----------

def read_pdf(file) -> str:
    if not HAS_PDF:
        return ""
    text = []
    try:
        reader = PdfReader(file)
        for page in reader.pages:
            text.append(page.extract_text() or "")
    except Exception:
        return ""
    return "\n".join(text)

def read_docx(file) -> str:
    if not HAS_DOCX:
        return ""
    try:
        document = docx.Document(file)
        return "\n".join([p.text for p in document.paragraphs])
    except Exception:
        return ""

def read_text(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return file_bytes.decode("latin-1")
        except Exception:
            return ""

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return [c.strip() for c in chunks if c.strip()]

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a_norm, b_norm))

def embed_texts(client, texts: List[str]) -> List[np.ndarray]:
    embs = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=batch)
        for d in resp.data:
            embs.append(np.array(d.embedding, dtype="float32"))
    return embs

def build_index(client, docs: List[Tuple[str, str]]) -> Dict:
    all_chunks, sources = [], []
    for src_name, text in docs:
        for ch in chunk_text(text):
            all_chunks.append(ch)
            sources.append(src_name)
    embeddings = embed_texts(client, all_chunks) if all_chunks else []
    return {"chunks": all_chunks, "sources": sources, "embeddings": embeddings}

def retrieve(index: Dict, query_vec: np.ndarray, k: int = 5):
    scored = []
    for ch, src, emb in zip(index["chunks"], index["sources"], index["embeddings"]):
        scored.append((ch, src, cosine_similarity(query_vec, emb)))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:k]

def build_messages(query: str, context_blocks: List[str]) -> List[Dict]:
    system = (
        "You are a meticulous process instructor. Using the provided SOP context only, "
        "produce a clear, numbered, step-by-step guide. Each step must have a short "
        "bold title followed by the detailed instruction. If information is missing "
        "or ambiguous, explicitly state assumptions and safe best-practices. Keep the tone professional. "
        "End with a short checklist of critical points."
    )
    context = "\n\n---\n\n".join(context_blocks)
    user = f"Query: {query}\n\nRelevant SOP context:\n{context}"
    return [{"role":"system","content":system},{"role":"user","content":user}]

def download_button(label: str, data: bytes, file_name: str):
    b64 = base64.b64encode(data).decode()
    href = f'<a class="download" href="data:application/octet-stream;base64,{b64}" download="{file_name}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    if OPENAI_API_KEY:
        st.success("OpenAI key is set (via Secrets or environment).")
    else:
        st.error("OpenAI key missing. On Streamlit Cloud, add it in **App → Settings → Secrets**.")
    st.markdown(f'<div class="kpi">Model: <b>{OPENAI_MODEL}</b></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi">Embeddings: <b>{OPENAI_EMBED_MODEL}</b></div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.write("Tips: Upload 1–10 SOP files (PDF, DOCX, TXT, MD). Ask task-focused questions.")

# ---------- Main ----------
c1, c2 = st.columns([1,1.5])
with c1:
    files = st.file_uploader("Upload SOPs (PDF/DOCX/TXT/MD)", type=["pdf","docx","txt","md"], accept_multiple_files=True)
with c2:
    query = st.text_area("What do you need instructions for?", placeholder="e.g., 'How do I provision a new employee laptop?'", height=120)
go = st.button("Generate Steps ▶", type="primary", use_container_width=True)

# Load docs
docs = []
if files:
    for f in files:
        name = f.name
        ext = name.lower().split(".")[-1]
        content = ""
        try:
            if ext == "pdf":
                content = read_pdf(f)
            elif ext == "docx":
                content = read_docx(f)
            else:
                content = read_text(f.getvalue())
        except Exception:
            content = ""
        if content:
            docs.append((name, content))

# Fallback to bundled samples (nice for first run)
if not docs:
    sample_dir = os.path.join(os.path.dirname(__file__), "sops")
    if os.path.isdir(sample_dir):
        for fname in os.listdir(sample_dir):
            if fname.endswith((".md", ".txt")):
                with open(os.path.join(sample_dir, fname), "r", encoding="utf-8") as fh:
                    docs.append((fname, fh.read()))

if go:
    if not OPENAI_API_KEY:
        st.error("No OpenAI key configured. On Streamlit Cloud, open **App → Settings → Secrets** and add OPENAI_API_KEY.")
    elif not docs:
        st.warning("Please upload at least one SOP (or keep the samples).")
    elif not query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Indexing SOPs..."):
                index = build_index(client, docs)
            with st.spinner("Retrieving and generating instructions..."):
                q_emb = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=[query]).data[0].embedding
                q_vec = np.array(q_emb, dtype="float32")
                top = retrieve(index, q_vec, k=6)
                context_blocks = [t[0] for t in top]
                msg = build_messages(query, context_blocks)
                completion = client.chat.completions.create(model=OPENAI_MODEL, messages=msg, temperature=0.2)
                answer = completion.choices[0].message.content
            st.markdown("### ✅ Generated Instructions")
            st.markdown(f'<div class="section-card">{answer}</div>', unsafe_allow_html=True)
            with st.expander("Show context sources"):
                for i, (chunk, src, score) in enumerate(top, start=1):
                    st.markdown(f"**{i}. {src}** (similarity: {score:.3f})")
                    st.code(chunk[:1200] + ("..." if len(chunk) > 1200 else ""), language="markdown")
            download_button("Download as Markdown", answer.encode("utf-8"), "instructions.md")
        except Exception as e:
            st.error(f"Error from OpenAI API: {e}")
            st.info("If this is a quota (429) error, add billing to your OpenAI account or switch to a cheaper model in secrets.")
