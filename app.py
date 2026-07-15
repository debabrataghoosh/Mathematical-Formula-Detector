
import streamlit as st
import cv2 
import numpy as np
import Inference_Math_Detection as MD
import Recog_MathForm as RM
import formula_extraction as FE
from PIL import Image
import pdf2image
import os
import zipfile
import io
import re

def download_models():
    mathdetector = './Models/MathDetector.ts'
    mathrecog = './Models/MathRecog.pth'
    
    if not os.path.exists(mathdetector):
        detector_url = 'gdown -O '+mathdetector+' https://drive.google.com/uc?id=1AGZTIRbx-KmLQ7bSEAcxUWWtdSrYucFz'
        with st.spinner('done!\nmodel weights were not found, downloading them...'):
            os.system(detector_url)
    else:
        print("Detector Model is here")

    if not os.path.exists(mathrecog):
        detector_url = 'gdown -O '+mathrecog+' https://drive.google.com/uc?id=1oR7eNBOC_3TBhFQ1KTzuWSl7-fet4cYh'
        with st.spinner('done!\nmodel weights were not found, downloading them...'):
            os.system(detector_url)
    else:
        print("Reconizer Model is here")

def draw_rectangles (image, preds):
    for each_pred in preds:
        cv2.rectangle(image, (int(each_pred[0]),int(each_pred[1])), (int(each_pred[2]),int(each_pred[3])),(255,0,0),2)


def _normalize_latex_for_katex(s: str) -> str:
    r"""Normalize common non-standard macros to KaTeX-safe equivalents.

    Examples handled:
    - \cal X -> \mathcal{X}
    - \bf X  -> \mathbf{X}; \bf\nabla -> \boldsymbol{\nabla}
    - \it X  -> \mathit{X}; \rm X -> \mathrm{X}
    - \simLambda -> \tilde{\Lambda} (heuristic)
    - \stackrel{a}{b} -> \overset{a}{b} (more robust in KaTeX)
    """
    if not s:
        return s
    t = s
    # \cal -> \mathcal{}
    t = re.sub(r"\\cal\s*([A-Za-z])", r"\\mathcal{\1}", t)
    # \bf token forms
    t = re.sub(r"\\bf\s*([A-Za-z])", r"\\mathbf{\1}", t)
    t = re.sub(r"\\bf\s*\{([^}]*)\}", r"\\mathbf{\1}", t)
    t = t.replace("\\bf\\nabla", "\\boldsymbol{\\nabla}")
    # \it, \rm
    t = re.sub(r"\\it\s*([A-Za-z])", r"\\mathit{\1}", t)
    t = re.sub(r"\\rm\s*([A-Za-z])", r"\\mathrm{\1}", t)
    # \simX -> \tilde{X} (heuristic for recognized tokens like \simLambda)
    t = re.sub(r"\\sim([A-Za-z])", r"\\tilde{\\\1}", t)
    # stackrel -> overset (KaTeX supports both; overset is often safer)
    t = t.replace("\\stackrel", "\\overset")
    # Minor whitespace cleanup
    t = re.sub(r"\s+", " ", t).strip()
    return t

def render_latex_block(latex_text):
    """Render LaTeX with safe fallback to keep layout aligned."""
    if latex_text is None or str(latex_text).strip() == "":
        st.info("No LaTeX available for this formula.")
        return
    normalized = _normalize_latex_for_katex(str(latex_text))
    try:
        # st.latex centers the formula and avoids overflowing raw text blocks
        st.latex(normalized)
    except Exception:
        st.warning("Could not render LaTeX; showing raw text instead.")
        st.code(normalized, language='latex')

if __name__ == '__main__':
    st.set_page_config(page_title="Math Formula Detection", page_icon="∑", layout="wide")
    download_models()

    # ── CSS ──────────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg:        #08101a;
        --surface:   rgba(255,255,255,0.04);
        --border:    rgba(255,255,255,0.08);
        --border-hi: rgba(29,211,176,0.5);
        --accent:    #1dd3b0;
        --accent2:   #00c6ff;
        --text:      #e8f0f8;
        --muted:     #7a90a8;
        --radius:    14px;
    }

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, sans-serif !important;
        color: var(--text) !important;
    }
    .main, section.main {
        background: radial-gradient(ellipse 80% 50% at 20% -10%, rgba(0,198,255,.10) 0%, transparent 60%),
                    radial-gradient(ellipse 60% 40% at 80% 110%, rgba(29,211,176,.08) 0%, transparent 60%),
                    #08101a !important;
    }
    .block-container {
        padding: 2rem 3rem 4rem !important;
        max-width: none !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #060e18 !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }
    [data-testid="stSidebar"] .stMarkdown p {
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        color: var(--muted) !important;
        margin-bottom: 0.5rem !important;
    }

    /* ── Headings ── */
    h1,h2,h3,h4 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        color: var(--text) !important;
    }

    /* ── Hero ── */
    .hero {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1.6rem 1.8rem;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 20px;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(12px);
    }
    .hero-icon {
        width: 48px; height: 48px;
        background: linear-gradient(135deg, var(--accent), var(--accent2));
        border-radius: 13px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.4rem; font-weight: 800; color: #07101a;
        flex-shrink: 0;
    }
    .hero-title {
        font-size: 1.6rem; font-weight: 700;
        background: linear-gradient(135deg, #fff 0%, #a8c4e0 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0; line-height: 1.2;
    }
    .hero-sub {
        font-size: 0.9rem; color: var(--muted); margin: 0.2rem 0 0;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
        color: #07101a !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.55rem 1.4rem !important;
        width: 100% !important;
        letter-spacing: 0.01em !important;
        transition: opacity .2s, transform .2s !important;
        box-shadow: 0 4px 18px rgba(29,211,176,.2) !important;
    }
    .stButton > button:hover {
        opacity: .88 !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active { transform: none !important; }

    /* ── Download buttons ── */
    .stDownloadButton > button {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        width: 100% !important;
        transition: border-color .2s, color .2s !important;
    }
    .stDownloadButton > button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: var(--surface) !important;
        border: 1px dashed var(--border) !important;
        border-radius: var(--radius) !important;
        transition: border-color .2s !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent) !important;
    }

    /* ── Selectbox ── */
    [data-baseweb="select"] > div {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
    }
    [data-baseweb="select"] > div:hover { border-color: var(--accent) !important; }

    /* ── Number input ── */
    [data-testid="stNumberInput"] input {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
    }
    [data-testid="stNumberInput"] input:focus { border-color: var(--accent) !important; }

    /* ── Expanders ── */
    div[data-testid="stExpander"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        margin-bottom: 0.75rem !important;
        transition: border-color .25s, box-shadow .25s !important;
        overflow: hidden !important;
    }
    div[data-testid="stExpander"]:hover {
        border-color: rgba(29,211,176,.25) !important;
        box-shadow: 0 4px 20px rgba(29,211,176,.06) !important;
    }
    div[data-testid="stExpander"] > details > summary {
        padding: 1rem 1.2rem !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        color: var(--text) !important;
        background: transparent !important;
    }
    div[data-testid="stExpander"] > details > summary:hover { color: var(--accent) !important; }
    div[data-testid="stExpander"] > details[open] > summary {
        border-bottom: 1px solid var(--border) !important;
    }

    /* ── Alerts ── */
    div[data-testid="stAlert"] {
        border-radius: var(--radius) !important;
        border: 1px solid rgba(29,211,176,.2) !important;
        background: rgba(29,211,176,.05) !important;
    }

    /* ── Images in expander ── */
    div[data-testid="stExpander"] [data-testid="stImage"] img {
        border-radius: 10px !important;
        border: 1px solid var(--border) !important;
        background: #fff !important;
        padding: 6px !important;
    }

    /* ── Formula description callout ── */
    .formula-desc {
        background: rgba(29,211,176,.05);
        border-left: 3px solid var(--accent);
        border-radius: 0 10px 10px 0;
        padding: 0.65rem 0.9rem;
        font-size: 0.88rem;
        color: var(--muted);
        line-height: 1.5;
        margin-top: 0.5rem;
    }

    /* ── Code ── */
    code {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
        background: rgba(0,0,0,.3) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        color: var(--accent2) !important;
    }
    pre code { background: transparent !important; border: none !important; }

    /* ── Section label ── */
    .section-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--muted);
        margin: 1.5rem 0 0.6rem;
        font-weight: 600;
    }

    /* ── Hint banner ── */
    .hint {
        background: var(--surface);
        border: 1px dashed var(--border);
        border-radius: var(--radius);
        padding: 0.9rem 1.2rem;
        color: var(--muted);
        font-size: 0.88rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Session state ─────────────────────────────────────────────────────────────
    for key, default in [
        ('extraction_done', False), ('extracted_formulas', None),
        ('extracted_crops', None),  ('output_dir', None),
        ('detection_done',  False), ('results_boxes', None),
        ('opencv_image',    None),  ('pdf_pages', None),
        ('pdf_file_name',   None),  ('pdf_active_page', None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    math_model = MD.initialize_model("./Models/MathDetector.ts")
    mathargs, *mathobjs = RM.initialize()

    # Hard-coded detection defaults (sliders removed for clean UI)
    DET_CONF     = 0.40
    DET_NMS      = 0.60
    DET_DUP      = 0.45
    DET_MIN_AREA = 0.00008
    LATEX_API_FIRST  = False
    LATEX_MAX_CALLS  = 8
    DESC_MAX_CALLS   = 3

    # ── Hero ──────────────────────────────────────────────────────────────────────
    st.markdown("""
        <div class="hero">
            <div class="hero-icon">∑</div>
            <div>
                <div class="hero-title">Math Formula Detector</div>
                <div class="hero-sub">Detect, extract &amp; render LaTeX from images or PDFs — powered by YOLOv5 + Transformer OCR.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("**Input type**")
        inf_style = st.selectbox("", ("Image", "PDF"), label_visibility="collapsed")

        st.markdown("**Accuracy mode**")
        extraction_mode = st.selectbox("", ("Fast", "Balanced", "High Accuracy"), index=1, label_visibility="collapsed")
        if extraction_mode == "Fast":
            LATEX_MAX_CALLS, DESC_MAX_CALLS, LATEX_API_FIRST = 4,  0,  False
        elif extraction_mode == "Balanced":
            LATEX_MAX_CALLS, DESC_MAX_CALLS, LATEX_API_FIRST = 8,  3,  False
        else:
            LATEX_MAX_CALLS, DESC_MAX_CALLS, LATEX_API_FIRST = 50, 10, True

        st.markdown("---")

        if inf_style == "Image":
            uploaded_file = st.file_uploader("Upload image", type=["png", "jpeg", "jpg"])
        else:
            uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    # ─────────────────────────────── IMAGE FLOW ───────────────────────────────────
    if inf_style == "Image":
        if uploaded_file is None:
            st.markdown('<div class="hint">Upload an image from the sidebar to get started.</div>', unsafe_allow_html=True)
        else:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            st.session_state.opencv_image = opencv_image

            col_img, col_ctrl = st.columns([3, 1], gap="large")
            with col_img:
                preview = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
                if st.session_state.detection_done and st.session_state.results_boxes is not None:
                    annotated = st.session_state.opencv_image.copy()
                    draw_rectangles(annotated, st.session_state.results_boxes)
                    st.image(annotated, use_container_width=True)
                else:
                    st.image(preview, use_container_width=True)

            with col_ctrl:
                st.markdown('<div class="section-label">Detection</div>', unsafe_allow_html=True)
                if st.button("Run Detection", key="img_detect"):
                    with st.spinner("Detecting…"):
                        boxes = MD.predict_formulas(
                            opencv_image, math_model,
                            conf_thres=DET_CONF, nms_iou_thres=DET_NMS,
                            duplicate_iou_thres=DET_DUP, min_area_ratio=DET_MIN_AREA,
                        )
                        st.session_state.results_boxes = boxes
                        st.session_state.detection_done = True
                        st.session_state.extraction_done = False
                        st.session_state.extracted_formulas = None
                        st.rerun()

                if st.session_state.detection_done and st.session_state.results_boxes is not None:
                    n = len(st.session_state.results_boxes)
                    if n > 0:
                        st.success(f"{n} formula{'s' if n!=1 else ''} found")
                        st.markdown('<div class="section-label">Export</div>', unsafe_allow_html=True)
                        if st.button("Extract & Save", key="img_extract"):
                            from datetime import datetime
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            out = f"extracted_output_{ts}"
                            os.makedirs(out, exist_ok=True)
                            n_formulas = len(st.session_state.results_boxes)

                            progress_bar = st.progress(0)
                            status_box   = st.empty()

                            def _step(pct, icon, msg):
                                progress_bar.progress(pct)
                                status_box.markdown(
                                    f'<div style="padding:10px 14px;background:rgba(29,211,176,0.08);'
                                    f'border-left:3px solid #1dd3b0;border-radius:8px;'
                                    f'font-size:0.92rem;color:#e8f0f8;">'
                                    f'{icon} &nbsp;<b>{msg}</b></div>',
                                    unsafe_allow_html=True)

                            _step(5,  '🔍', 'Cropping detected formula regions from image…')
                            crops = FE.extract_formula_crops(st.session_state.opencv_image, st.session_state.results_boxes)

                            _step(15, '🧠', f'Loading transformer OCR model — recognising {n_formulas} formula{"s" if n_formulas!=1 else ""}…')
                            formulas = FE.recognize_formulas(crops, mathargs, mathobjs)

                            _step(45, '✨', 'Refining LaTeX with Gemini vision model…')
                            try:
                                formulas = FE.refine_formulas_latex_with_gemini(
                                    formulas, crops,
                                    max_calls=min(len(crops), LATEX_MAX_CALLS),
                                    api_first=LATEX_API_FIRST, provider='gemini')
                            except Exception:
                                pass

                            _step(65, '📝', 'Generating semantic descriptions for each formula…')
                            formulas = FE.enrich_formulas_with_descriptions(formulas, max_api_calls=DESC_MAX_CALLS)

                            _step(75, '🖼️', 'Saving annotated image and formula crops…')
                            st.session_state.extracted_crops   = crops
                            st.session_state.extracted_formulas = formulas
                            FE.save_pdf_report(formulas, extracted_crops=crops,
                                output_path=os.path.join(out,'formulas_report.pdf'),
                                original_image=st.session_state.opencv_image)
                            FE.save_annotated_image(st.session_state.opencv_image, formulas,
                                os.path.join(out,'annotated_image.png'))
                            fd = os.path.join(out,'formula_images'); os.makedirs(fd, exist_ok=True)
                            for i,c in enumerate(crops):
                                cv2.imwrite(os.path.join(fd,f'formula_{i+1:04d}.png'), c['image'])

                            _step(90, '📦', 'Packaging all files into ZIP archive…')
                            zp = os.path.join(out,'extracted_formulas.zip')
                            with zipfile.ZipFile(zp,'w') as zf:
                                for r,_,fs in os.walk(out):
                                    for f in fs:
                                        if not f.endswith('.zip'):
                                            zf.write(os.path.join(r,f), os.path.relpath(os.path.join(r,f),out))

                            _step(100, '✅', f'Done! {n_formulas} formula{"s" if n_formulas!=1 else ""} extracted and saved to {out}')
                            st.session_state.output_dir    = out
                            st.session_state.extraction_done = True

                        if st.button("View Formulas", key="img_view"):
                            n_formulas = len(st.session_state.results_boxes)
                            progress_bar2 = st.progress(0)
                            status_box2   = st.empty()

                            def _step2(pct, icon, msg):
                                progress_bar2.progress(pct)
                                status_box2.markdown(
                                    f'<div style="padding:10px 14px;background:rgba(29,211,176,0.08);'
                                    f'border-left:3px solid #1dd3b0;border-radius:8px;'
                                    f'font-size:0.92rem;color:#e8f0f8;">'
                                    f'{icon} &nbsp;<b>{msg}</b></div>',
                                    unsafe_allow_html=True)

                            _step2(5,  '🔍', 'Cropping detected formula regions from image…')
                            crops = FE.extract_formula_crops(st.session_state.opencv_image, st.session_state.results_boxes)

                            _step2(20, '🧠', f'Running transformer OCR on {n_formulas} formula crop{"s" if n_formulas!=1 else ""}…')
                            formulas = FE.recognize_formulas(crops, mathargs, mathobjs)

                            _step2(55, '✨', 'Refining LaTeX expressions with Gemini vision model…')
                            try:
                                formulas = FE.refine_formulas_latex_with_gemini(
                                    formulas, crops,
                                    max_calls=min(len(crops), LATEX_MAX_CALLS),
                                    api_first=LATEX_API_FIRST, provider='gemini')
                            except Exception:
                                pass

                            _step2(80, '📝', 'Generating semantic descriptions…')
                            formulas = FE.enrich_formulas_with_descriptions(formulas, max_api_calls=DESC_MAX_CALLS)

                            _step2(100, '✅', f'Recognised {n_formulas} formula{"s" if n_formulas!=1 else ""} — rendering results…')
                            st.session_state.extracted_crops    = crops
                            st.session_state.extracted_formulas = formulas
                            st.session_state.extraction_done    = 'view'
                    else:
                        st.warning("No formulas detected.")

            # ── Download row ──────────────────────────────────────────────────────
            if st.session_state.extraction_done is True and st.session_state.output_dir:
                out = st.session_state.output_dir
                st.markdown('<div class="section-label">Downloads</div>', unsafe_allow_html=True)
                dc1, dc2 = st.columns(2)
                with dc1:
                    pf = os.path.join(out,'formulas_report.pdf')
                    if os.path.exists(pf):
                        with open(pf,'rb') as f: st.download_button("PDF report", f.read(), "formulas_report.pdf", "application/pdf")
                with dc2:
                    zf = os.path.join(out,'extracted_formulas.zip')
                    if os.path.exists(zf):
                        with open(zf,'rb') as f: st.download_button("ZIP package", f.read(), "extracted_formulas.zip", "application/zip")

            # ── Formula cards ─────────────────────────────────────────────────────
            if st.session_state.extraction_done in (True, 'view') and st.session_state.extracted_formulas:
                formulas = st.session_state.extracted_formulas
                st.markdown(f'<div class="section-label">{len(formulas)} Formulas</div>', unsafe_allow_html=True)
                for formula in formulas:
                    conf_pct = f"{formula['confidence']*100:.1f}%"
                    with st.expander(f"Formula #{formula['id']} · {conf_pct} confidence", expanded=False):
                        fc1, fc2 = st.columns([1, 1], gap="medium")
                        with fc1:
                            coords = formula['coordinates']
                            crop_img = st.session_state.opencv_image[coords[1]:coords[3], coords[0]:coords[2]]
                            st.image(crop_img, use_container_width=True)
                            if formula.get('description'):
                                st.markdown(f'<div class="formula-desc">{formula["description"]}</div>', unsafe_allow_html=True)
                        with fc2:
                            render_latex_block(formula['latex'])
                            st.code(formula['latex'], language='latex')

    # ─────────────────────────────── PDF FLOW ────────────────────────────────────
    else:
        if uploaded_file is None:
            st.markdown('<div class="hint">Upload a PDF from the sidebar to get started.</div>', unsafe_allow_html=True)
        else:
            if st.session_state.pdf_file_name != uploaded_file.name:
                pdf_bytes = uploaded_file.read()
                st.session_state.pdf_pages = pdf2image.convert_from_bytes(pdf_bytes)
                st.session_state.pdf_file_name = uploaded_file.name
                for k in ('detection_done','extraction_done','results_boxes','extracted_formulas','extracted_crops','output_dir'):
                    st.session_state[k] = False if k in ('detection_done','extraction_done') else None

            if st.session_state.pdf_pages:
                col_pg, col_ctrl = st.columns([3, 1], gap="large")
                with col_ctrl:
                    st.markdown('<div class="section-label">Page</div>', unsafe_allow_html=True)
                    page_idx = st.number_input("", min_value=1, max_value=len(st.session_state.pdf_pages), value=1, step=1, label_visibility="collapsed")
                    if st.session_state.pdf_active_page != page_idx:
                        for k in ('detection_done','extraction_done','results_boxes','extracted_formulas','extracted_crops','output_dir'):
                            st.session_state[k] = False if k in ('detection_done','extraction_done') else None
                        st.session_state.pdf_active_page = page_idx

                page_image = st.session_state.pdf_pages[int(page_idx)-1]
                opencv_image = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
                st.session_state.opencv_image = opencv_image

                with col_pg:
                    if st.session_state.detection_done and st.session_state.results_boxes is not None:
                        ann = opencv_image.copy()
                        draw_rectangles(ann, st.session_state.results_boxes)
                        st.image(ann, caption=f"Page {page_idx}", use_container_width=True)
                    else:
                        st.image(page_image, caption=f"Page {page_idx}", use_container_width=True)

                with col_ctrl:
                    st.markdown('<div class="section-label">Detection</div>', unsafe_allow_html=True)
                    if st.button("Run Detection", key="pdf_detect"):
                        with st.spinner("Detecting…"):
                            boxes = MD.predict_formulas(
                                opencv_image, math_model,
                                conf_thres=DET_CONF, nms_iou_thres=DET_NMS,
                                duplicate_iou_thres=DET_DUP, min_area_ratio=DET_MIN_AREA,
                            )
                            st.session_state.results_boxes = boxes
                            st.session_state.detection_done = True
                            st.session_state.extraction_done = False
                            st.session_state.extracted_formulas = None
                            st.rerun()

                    if st.session_state.detection_done and st.session_state.results_boxes is not None:
                        n = len(st.session_state.results_boxes)
                        if n > 0:
                            st.success(f"{n} formula{'s' if n!=1 else ''} found")
                            st.markdown('<div class="section-label">Export</div>', unsafe_allow_html=True)
                            if st.button("Extract & Save", key="pdf_extract"):
                                from datetime import datetime
                                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                out = f"extracted_output_pdf_p{page_idx}_{ts}"
                                os.makedirs(out, exist_ok=True)
                                n_formulas = len(st.session_state.results_boxes)

                                pb = st.progress(0)
                                sb = st.empty()
                                def _pstep(pct, icon, msg):
                                    pb.progress(pct)
                                    sb.markdown(
                                        f'<div style="padding:10px 14px;background:rgba(29,211,176,0.08);'
                                        f'border-left:3px solid #1dd3b0;border-radius:8px;'
                                        f'font-size:0.92rem;color:#e8f0f8;">'
                                        f'{icon} &nbsp;<b>{msg}</b></div>',
                                        unsafe_allow_html=True)

                                _pstep(5,  '🔍', f'Cropping {n_formulas} formula region{"s" if n_formulas!=1 else ""} from PDF page…')
                                crops = FE.extract_formula_crops(st.session_state.opencv_image, st.session_state.results_boxes)

                                _pstep(18, '🧠', f'Running transformer OCR on {n_formulas} crop{"s" if n_formulas!=1 else ""}…')
                                formulas = FE.recognize_formulas(crops, mathargs, mathobjs)

                                _pstep(45, '✨', 'Refining LaTeX with Gemini vision model…')
                                try:
                                    formulas = FE.refine_formulas_latex_with_gemini(
                                        formulas, crops,
                                        max_calls=min(len(crops), LATEX_MAX_CALLS),
                                        api_first=LATEX_API_FIRST, provider='auto')
                                except Exception:
                                    pass

                                _pstep(65, '📝', 'Generating semantic descriptions…')
                                formulas = FE.enrich_formulas_with_descriptions(formulas, max_api_calls=DESC_MAX_CALLS)

                                _pstep(75, '🖼️', 'Saving annotated image and formula crops…')
                                st.session_state.extracted_crops   = crops
                                st.session_state.extracted_formulas = formulas
                                FE.save_pdf_report(formulas, extracted_crops=crops,
                                    output_path=os.path.join(out,'formulas_report.pdf'),
                                    original_image=st.session_state.opencv_image)
                                FE.save_annotated_image(st.session_state.opencv_image, formulas,
                                    os.path.join(out,'annotated_image.png'))
                                fd = os.path.join(out,'formula_images'); os.makedirs(fd, exist_ok=True)
                                for i,c in enumerate(crops):
                                    cv2.imwrite(os.path.join(fd,f'formula_{i+1:04d}.png'), c['image'])

                                _pstep(90, '📦', 'Packaging into ZIP archive…')
                                zp = os.path.join(out,'extracted_formulas.zip')
                                with zipfile.ZipFile(zp,'w') as zf:
                                    for r,_,fs in os.walk(out):
                                        for f in fs:
                                            if not f.endswith('.zip'):
                                                zf.write(os.path.join(r,f), os.path.relpath(os.path.join(r,f),out))

                                _pstep(100, '✅', f'Done! {n_formulas} formula{"s" if n_formulas!=1 else ""} saved to {out}')
                                st.session_state.output_dir      = out
                                st.session_state.extraction_done = True

                            if st.button("View Formulas", key="pdf_view"):
                                n_formulas = len(st.session_state.results_boxes)
                                pb2 = st.progress(0)
                                sb2 = st.empty()
                                def _pstep2(pct, icon, msg):
                                    pb2.progress(pct)
                                    sb2.markdown(
                                        f'<div style="padding:10px 14px;background:rgba(29,211,176,0.08);'
                                        f'border-left:3px solid #1dd3b0;border-radius:8px;'
                                        f'font-size:0.92rem;color:#e8f0f8;">'
                                        f'{icon} &nbsp;<b>{msg}</b></div>',
                                        unsafe_allow_html=True)

                                _pstep2(5,  '🔍', f'Cropping {n_formulas} formula region{"s" if n_formulas!=1 else ""} from PDF page…')
                                crops = FE.extract_formula_crops(st.session_state.opencv_image, st.session_state.results_boxes)

                                _pstep2(20, '🧠', f'Running transformer OCR on {n_formulas} crop{"s" if n_formulas!=1 else ""}…')
                                formulas = FE.recognize_formulas(crops, mathargs, mathobjs)

                                _pstep2(55, '✨', 'Refining LaTeX with Gemini vision model…')
                                try:
                                    formulas = FE.refine_formulas_latex_with_gemini(
                                        formulas, crops,
                                        max_calls=min(len(crops), LATEX_MAX_CALLS),
                                        api_first=LATEX_API_FIRST, provider='auto')
                                except Exception:
                                    pass

                                _pstep2(80, '📝', 'Generating semantic descriptions…')
                                formulas = FE.enrich_formulas_with_descriptions(formulas, max_api_calls=DESC_MAX_CALLS)

                                _pstep2(100, '✅', f'Recognised {n_formulas} formula{"s" if n_formulas!=1 else ""} — rendering results…')
                                st.session_state.extracted_crops    = crops
                                st.session_state.extracted_formulas = formulas
                                st.session_state.extraction_done    = 'view'
                        else:
                            st.warning("No formulas detected.")

                # ── Download row ──────────────────────────────────────────────────
                if st.session_state.extraction_done is True and st.session_state.output_dir:
                    out = st.session_state.output_dir
                    st.markdown('<div class="section-label">Downloads</div>', unsafe_allow_html=True)
                    dc1, dc2 = st.columns(2)
                    with dc1:
                        pf = os.path.join(out,'formulas_report.pdf')
                        if os.path.exists(pf):
                            with open(pf,'rb') as f: st.download_button("PDF report", f.read(), "formulas_report.pdf", "application/pdf")
                    with dc2:
                        zf = os.path.join(out,'extracted_formulas.zip')
                        if os.path.exists(zf):
                            with open(zf,'rb') as f: st.download_button("ZIP package", f.read(), "extracted_formulas.zip", "application/zip")

                # ── Formula cards ──────────────────────────────────────────────────
                if st.session_state.extraction_done in (True, 'view') and st.session_state.extracted_formulas:
                    formulas = st.session_state.extracted_formulas
                    st.markdown(f'<div class="section-label">{len(formulas)} Formulas</div>', unsafe_allow_html=True)
                    for formula in formulas:
                        conf_pct = f"{formula['confidence']*100:.1f}%"
                        with st.expander(f"Formula #{formula['id']} · {conf_pct} confidence", expanded=False):
                            fc1, fc2 = st.columns([1, 1], gap="medium")
                            with fc1:
                                coords = formula['coordinates']
                                crop_img = st.session_state.opencv_image[coords[1]:coords[3], coords[0]:coords[2]]
                                st.image(crop_img, use_container_width=True)
                                if formula.get('description'):
                                    st.markdown(f'<div class="formula-desc">{formula["description"]}</div>', unsafe_allow_html=True)
                            with fc2:
                                render_latex_block(formula['latex'])
                                st.code(formula['latex'], language='latex')


            st.markdown("""
            <style>
            --bg-1: #0b1220;
            --panel: rgba(13, 22, 38, 0.75);
            --panel-hover: rgba(18, 30, 51, 0.85);
            --panel-border: rgba(67, 178, 255, 0.18);
            --panel-border-hover: rgba(29, 211, 176, 0.4);
            --text-main: #f0f5ff;
            --text-muted: #a0b2c6;
            --mint: #1dd3b0;
            --mint-glow: rgba(29, 211, 176, 0.25);
            --cyan: #00f2fe;
            --cyan-glow: rgba(0, 242, 254, 0.25);
            --gold: #ffb255;
            --purple: #7f5af0;
        }

        html, body, [class*="css"] { 
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; 
            color: var(--text-main); 
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em !important;
            color: var(--text-main) !important;
        }

        .main {
            background: 
                radial-gradient(circle at 10% 20%, rgba(67, 178, 255, 0.12), transparent 40%),
                radial-gradient(circle at 90% 10%, rgba(127, 90, 240, 0.1), transparent 40%),
                radial-gradient(circle at 50% 80%, rgba(29, 211, 176, 0.08), transparent 50%),
                linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 100%) !important;
        }

        .block-container { 
            padding-top: 1.5rem; 
            padding-bottom: 3rem; 
            max-width: none; 
            padding-left: 3rem; 
            padding-right: 3rem; 
        }

        [data-testid="stSidebar"] { 
            background: linear-gradient(180deg, #050b14 0%, #08111e 100%) !important; 
            border-right: 1px solid var(--panel-border) !important; 
        }

        .hero {
            border: 1px solid var(--panel-border) !important;
            border-radius: 20px !important;
            padding: 1.5rem !important;
            background: linear-gradient(135deg, rgba(67, 178, 255, 0.08), rgba(127, 90, 240, 0.08)) !important;
            background-color: var(--panel) !important;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
            margin-bottom: 1.5rem !important;
        }
        
        .hero:hover {
            border-color: rgba(67, 178, 255, 0.3) !important;
            box-shadow: 0 20px 40px rgba(67, 178, 255, 0.08) !important;
        }

        .hero-title { 
            font-size: 2.2rem !important; 
            font-weight: 700 !important; 
            line-height: 1.2; 
            background: linear-gradient(135deg, #ffffff 30%, var(--text-muted) 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            margin-bottom: 0.4rem !important; 
        }
        
        .hero-subtitle { 
            color: var(--text-muted); 
            font-size: 1.05rem; 
            margin-bottom: 1rem; 
        }

        .chip-row { display: flex; flex-wrap: wrap; gap: 0.6rem; }
        .chip {
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            background: rgba(255, 255, 255, 0.04) !important;
            color: var(--text-muted) !important;
            border-radius: 999px !important;
            padding: 0.3rem 0.8rem !important;
            font-size: 0.8rem !important;
            font-weight: 500 !important;
            letter-spacing: 0.02em !important;
            transition: all 0.25s ease !important;
        }
        .chip:hover {
            border-color: var(--mint) !important;
            color: var(--mint) !important;
            background: rgba(29, 211, 176, 0.05) !important;
        }

        .stButton > button {
            border-radius: 12px !important;
            border: 1px solid rgba(29, 211, 176, 0.4) !important;
            color: #050b14 !important;
            background: linear-gradient(135deg, var(--mint) 0%, var(--cyan) 100%) !important;
            box-shadow: 0 8px 20px rgba(0, 242, 254, 0.15) !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            letter-spacing: 0.02em !important;
            padding: 0.6rem 1.5rem !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
            width: 100% !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 12px 25px rgba(29, 211, 176, 0.35) !important;
            border-color: var(--mint) !important;
            color: #050b14 !important;
        }
        .stButton > button:active {
            transform: translateY(0px) !important;
        }

        .stDownloadButton > button {
            border-radius: 12px !important;
            background: rgba(13, 22, 38, 0.6) !important;
            border: 1px solid var(--panel-border) !important;
            color: var(--text-main) !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 600 !important;
            padding: 0.6rem 1.5rem !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
            width: 100% !important;
        }
        .stDownloadButton > button:hover { 
            border-color: var(--mint) !important; 
            color: var(--mint) !important; 
            background: rgba(29, 211, 176, 0.05) !important;
            box-shadow: 0 0 15px var(--mint-glow) !important;
            transform: translateY(-2px) !important;
        }
        .stDownloadButton > button:active {
            transform: translateY(0px) !important;
        }

        /* Streamlit Accordion Cards */
        div[data-testid="stExpander"] {
            border-radius: 16px !important;
            border: 1px solid var(--panel-border) !important;
            background-color: var(--panel) !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            margin-bottom: 1rem !important;
            overflow: hidden !important;
        }
        div[data-testid="stExpander"]:hover {
            border-color: var(--panel-border-hover) !important;
            box-shadow: 0 8px 25px rgba(29, 211, 176, 0.08) !important;
            transform: translateY(-2px) !important;
        }
        div[data-testid="stExpander"] > details > summary {
            background-color: transparent !important;
            padding: 1rem 1.2rem !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 600 !important;
            font-size: 1.05rem !important;
            color: var(--text-main) !important;
            transition: background-color 0.2s ease, color 0.2s ease !important;
        }
        div[data-testid="stExpander"] > details > summary:hover {
            background-color: rgba(255, 255, 255, 0.03) !important;
            color: var(--mint) !important;
        }
        div[data-testid="stExpander"] > details[open] > summary {
            border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
        }
        div[data-testid="stExpander"] > details > summary > span {
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 600 !important;
        }

        /* Alerts Styling */
        div[data-testid="stAlert"] { 
            border-radius: 14px !important; 
            border: 1px solid rgba(29, 211, 176, 0.2) !important; 
            background-color: rgba(13, 22, 38, 0.6) !important;
            color: var(--text-main) !important;
        }

        /* File Uploader Dropzone Styling */
        [data-testid="stFileUploader"] {
            background: rgba(13, 22, 38, 0.4) !important;
            border: 2px dashed var(--panel-border) !important;
            border-radius: 16px !important;
            padding: 1.5rem !important;
            transition: all 0.3s ease !important;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: var(--mint) !important;
            background: rgba(13, 22, 38, 0.6) !important;
            box-shadow: 0 0 15px var(--mint-glow) !important;
        }

        /* Slider overrides */
        .stSlider [data-testid="stThumbValue"] {
            color: var(--mint) !important;
            font-weight: 600 !important;
        }
        .stSlider [data-baseweb="slider"] > div {
            background-color: rgba(29, 211, 176, 0.2) !important;
        }
        .stSlider [data-baseweb="slider"] [role="slider"] {
            background-color: var(--mint) !important;
            box-shadow: 0 0 10px var(--mint-glow) !important;
        }

        /* Selectbox overrides */
        [data-baseweb="select"] > div {
            background-color: rgba(13, 22, 38, 0.6) !important;
            border: 1px solid var(--panel-border) !important;
            color: var(--text-main) !important;
            border-radius: 10px !important;
            transition: all 0.2s ease !important;
        }
        [data-baseweb="select"] > div:hover {
            border-color: var(--mint) !important;
        }

        /* Number Input overrides */
        [data-testid="stNumberInput"] input {
            background-color: rgba(13, 22, 38, 0.6) !important;
            border: 1px solid var(--panel-border) !important;
            color: var(--text-main) !important;
            border-radius: 10px !important;
            transition: all 0.2s ease !important;
        }
        [data-testid="stNumberInput"] input:hover {
            border-color: var(--mint) !important;
        }

        .empty-state {
            margin-top: 1rem !important;
            border: 1px dashed var(--panel-border) !important;
            background: rgba(13, 22, 38, 0.3) !important;
            border-radius: 16px !important;
            padding: 1.2rem 1.5rem !important;
            color: var(--text-muted) !important;
            font-size: 0.95rem !important;
            text-align: center !important;
        }

        code {
            font-family: 'JetBrains Mono', 'IBM Plex Mono', monospace !important;
            background-color: rgba(5, 11, 20, 0.8) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 8px !important;
            padding: 0.15rem 0.4rem !important;
            color: var(--cyan) !important;
            font-size: 0.9rem !important;
        }
        
        pre code {
            background-color: transparent !important;
            border: none !important;
            padding: 0 !important;
        }

        /* Formula Crop Image Styling */
        div[data-testid="stExpander"] img {
            border: 1px solid var(--panel-border) !important;
            border-radius: 8px !important;
            padding: 8px !important;
            background-color: #ffffff !important;
            box-shadow: inset 0 0 10px rgba(0,0,0,0.05) !important;
            transition: all 0.25s ease !important;
        }
        div[data-testid="stExpander"] img:hover {
            border-color: var(--mint) !important;
            transform: scale(1.01) !important;
        }

        /* Description Styling */
        .formula-desc {
            background-color: rgba(29, 211, 176, 0.05) !important;
            border-left: 3px solid var(--mint) !important;
            padding: 0.8rem 1rem !important;
            border-radius: 0 10px 10px 0 !important;
            font-size: 0.9rem !important;
            color: var(--text-main) !important;
            margin-top: 0.5rem !important;
            line-height: 1.4 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Initialize session state
    if 'extraction_done' not in st.session_state:
        st.session_state.extraction_done = False
    if 'extracted_formulas' not in st.session_state:
        st.session_state.extracted_formulas = None
    if 'extracted_crops' not in st.session_state:
        st.session_state.extracted_crops = None
    if 'output_dir' not in st.session_state:
        st.session_state.output_dir = None
    if 'detection_done' not in st.session_state:
        st.session_state.detection_done = False
    if 'results_boxes' not in st.session_state:
        st.session_state.results_boxes = None
    if 'opencv_image' not in st.session_state:
        st.session_state.opencv_image = None
    if 'pdf_pages' not in st.session_state:
        st.session_state.pdf_pages = None
    if 'pdf_file_name' not in st.session_state:
        st.session_state.pdf_file_name = None
    if 'pdf_active_page' not in st.session_state:
        st.session_state.pdf_active_page = None
    
    math_model = MD.initialize_model("./Models/MathDetector.ts")
    mathargs, *mathobjs = RM.initialize()

    st.markdown("""
        <div class="hero">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom: 0.3rem;">
                <div style="background:linear-gradient(135deg,#1dd3b0,#17a2f3); width:44px; height:44px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-weight:800; color:#0b0d12;">∑</div>
                <div>
                    <div class="hero-title">Mathematical Formula Detector</div>
                    <div class="hero-subtitle">Detect, extract, and render formulas from images or PDFs in a fast, export-friendly workflow.</div>
                </div>
            </div>
            <div class="chip-row">
                <span class="chip">YOLOv5 Detection</span>
                <span class="chip">Transformer OCR</span>
                <span class="chip">Smart LaTeX Refinement</span>
                <span class="chip">PDF & Image Workflow</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="empty-state">
            Choose <b>Image</b> or <b>PDF</b> from the sidebar, upload your file, then run detection to start extraction.
        </div>
        """,
        unsafe_allow_html=True,
    )

