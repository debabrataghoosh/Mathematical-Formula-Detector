
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
    """Normalize common non-standard macros to KaTeX-safe equivalents.

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
    st.set_page_config(page_title="Math Formula Detection", page_icon="➗", layout="wide")
    download_models()

    st.markdown(
        """
        <style>
        /* Modern look: bold type, soft card edges, subtle glow */
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap');
        html, body, [class*="css"]  { font-family: 'Space Grotesk', sans-serif; }
        .main { background: radial-gradient(circle at 20% 20%, rgba(0,181,173,0.12), transparent 30%),
                          radial-gradient(circle at 80% 0%, rgba(255,140,66,0.14), transparent 32%),
                          #0f1116;
                 color: #e6e8ef; }
        .stSidebar { background: #0b0d12; }
        .stSidebar, .st-bb, .st-at, .st-bc { color: #e6e8ef; }
        .css-1d391kg, .css-1lcbmhc { color: #e6e8ef; }
        .stButton>button { border-radius: 12px; border: 1px solid #1dd3b0; color: #0b0d12;
                           background: linear-gradient(135deg, #1dd3b0 0%, #17a2f3 100%);
                           box-shadow: 0 10px 30px rgba(23,162,243,0.25); font-weight: 600; }
        .stButton>button:hover { box-shadow: 0 12px 34px rgba(29,211,176,0.35); transform: translateY(-1px); }
        .stDownloadButton>button { border-radius: 12px; background: #151a23; border: 1px solid #283344;
                                   color: #e6e8ef; box-shadow: 0 6px 20px rgba(0,0,0,0.25); }
        .stDownloadButton>button:hover { border-color: #1dd3b0; color: #1dd3b0; }
        .block-container { padding-top: 1.8rem; padding-bottom: 2rem; }
        .metric-card { background: #151a23; border: 1px solid #1f2a39; border-radius: 14px;
                       padding: 1rem 1.2rem; box-shadow: 0 12px 45px rgba(0,0,0,0.35); }
        .stExpander { background: #151a23; border: 1px solid #1f2a39; border-radius: 12px; }
        .stExpander > div > div { padding: 0.75rem 1rem; }
        .stAlert { border-radius: 12px; border: 1px solid #1dd3b0; background: rgba(29,211,176,0.08); }
        .stMarkdown h1, h2, h3 { color: #e6e8ef; }
        .latex { color: #e6e8ef; }
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
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="background:linear-gradient(135deg,#1dd3b0,#17a2f3); width:44px; height:44px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-weight:800; color:#0b0d12;">∑</div>
            <div>
                <div style="font-size:28px; font-weight:700; color:#e6e8ef;">Mathematical Formula Detector</div>
                <div style="color:#94a3b8;">Detect • Extract • Render LaTeX from images and PDFs</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    inf_style = st.sidebar.selectbox("Inference Type",('Image', 'PDF'))
    if inf_style == 'Image':

        uploaded_file = st.sidebar.file_uploader("Upload Image", type=['png','jpeg', 'jpg'])

    #     res = st.sidebar.radio("Final Result",("Detection","Detection And Recogntion"))
        if uploaded_file is not None:
            if st.sidebar.button('Clear uploaded file or image!'):
                st.warning("attempt to clear uploaded_file")
                uploaded_file.seek(0)
            with st.spinner(text='In progress'):
                # Read once and render from decoded array to avoid stale file refs
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)
                preview_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
                st.sidebar.image(preview_image)
                
                # Store image in session for later use
                st.session_state.opencv_image = opencv_image

                if st.button('Launch the Detection!'):
                    results_boxes = MD.predict_formulas(opencv_image, math_model)
                    st.session_state.results_boxes = results_boxes
                    st.session_state.detection_done = True
                
                # Show detection result if detection was done
                if st.session_state.detection_done and st.session_state.results_boxes is not None:
                    results_boxes = st.session_state.results_boxes
                    images_rectangles = st.session_state.opencv_image.copy()
                    draw_rectangles(images_rectangles, results_boxes)
                    st.image(images_rectangles)
                    
                    # Add extraction option
                    if len(results_boxes) > 0:
                        st.success(f"✓ Found {len(results_boxes)} formulas!")
                        
                        # Create two columns for better layout
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("🚀 Extract Formulas to File"):
                                with st.spinner("Extracting and recognizing formulas..."):
                                    # Create output directory with timestamp
                                    from datetime import datetime
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    output_dir = f"extracted_output_{timestamp}"
                                    os.makedirs(output_dir, exist_ok=True)
                                    
                                    # Extract formula crops
                                    st.session_state.extracted_crops = FE.extract_formula_crops(st.session_state.opencv_image, st.session_state.results_boxes)
                                    
                                    # Recognize formulas
                                    st.session_state.extracted_formulas = FE.recognize_formulas(st.session_state.extracted_crops, mathargs, mathobjs)
                                    # Optionally refine LaTeX using Gemini for weak outputs
                                    try:
                                        st.session_state.extracted_formulas = FE.refine_formulas_latex_with_gemini(
                                            st.session_state.extracted_formulas,
                                            st.session_state.extracted_crops,
                                            max_calls=8
                                        )
                                    except Exception:
                                        pass
                                    # Enrich with AI descriptions if available
                                    st.session_state.extracted_formulas = FE.enrich_formulas_with_descriptions(st.session_state.extracted_formulas)
                                    
                                    # Save all files to output directory
                                    formulas = st.session_state.extracted_formulas
                                    extracted_crops = st.session_state.extracted_crops
                                    
                                    # Save consolidated PDF Report
                                    pdf_path = os.path.join(output_dir, 'formulas_report.pdf')
                                    FE.save_pdf_report(formulas, extracted_crops=extracted_crops, output_path=pdf_path, original_image=st.session_state.opencv_image)
                                    
                                    # Save Annotated Image
                                    annotated_path = os.path.join(output_dir, 'annotated_image.png')
                                    FE.save_annotated_image(st.session_state.opencv_image, formulas, annotated_path)
                                    
                                    # Save individual formula images
                                    formula_dir = os.path.join(output_dir, 'formula_images')
                                    os.makedirs(formula_dir, exist_ok=True)
                                    for idx, crop_data in enumerate(extracted_crops):
                                        img_path = os.path.join(formula_dir, f'formula_{idx+1:04d}.png')
                                        cv2.imwrite(img_path, crop_data['image'])
                                    
                                    # Create ZIP package
                                    zip_path = os.path.join(output_dir, 'extracted_formulas.zip')
                                    with zipfile.ZipFile(zip_path, 'w') as zip_file:
                                        for root, dirs, files in os.walk(output_dir):
                                            for file in files:
                                                if not file.endswith('.zip'):
                                                    file_path = os.path.join(root, file)
                                                    arcname = os.path.relpath(file_path, output_dir)
                                                    zip_file.write(file_path, arcname)
                                    
                                    st.session_state.extraction_done = True
                                    st.session_state.output_dir = output_dir
                                    
                                    st.success(f"✓ Successfully extracted {len(formulas)} formulas!")
                                    st.info(f"📁 All files saved to: **{output_dir}**")
                        
                        with col2:
                            if st.button("👁️ View Extracted Formulas"):
                                with st.spinner("Extracting and recognizing formulas..."):
                                    # Extract formula crops
                                    st.session_state.extracted_crops = FE.extract_formula_crops(st.session_state.opencv_image, st.session_state.results_boxes)
                                    
                                    # Recognize formulas
                                    st.session_state.extracted_formulas = FE.recognize_formulas(st.session_state.extracted_crops, mathargs, mathobjs)
                                    # Optionally refine LaTeX using Gemini for weak outputs
                                    try:
                                        st.session_state.extracted_formulas = FE.refine_formulas_latex_with_gemini(
                                            st.session_state.extracted_formulas,
                                            st.session_state.extracted_crops,
                                            max_calls=8
                                        )
                                    except Exception:
                                        pass
                                    # Optional: enrich with AI descriptions
                                    st.session_state.extracted_formulas = FE.enrich_formulas_with_descriptions(st.session_state.extracted_formulas)
                                    st.session_state.extraction_done = 'view'
                        
                        # Display extraction results if extraction was done
                        if st.session_state.extraction_done == True:
                            if st.session_state.extracted_formulas is not None and st.session_state.output_dir is not None:
                                formulas = st.session_state.extracted_formulas
                                output_dir = st.session_state.output_dir
                                
                                st.success(f"✓ Successfully extracted {len(formulas)} formulas!")
                                st.info(f"📁 All files saved to: **{output_dir}**")
                                
                                # Export options in columns
                                st.subheader("📥 Download Extracted Formulas")
                                
                                dl_col1, dl_col2 = st.columns(2)
                                
                                # PDF Report Download
                                with dl_col1:
                                    pdf_file = os.path.join(output_dir, 'formulas_report.pdf')
                                    if os.path.exists(pdf_file):
                                        with open(pdf_file, 'rb') as f:
                                            st.download_button(
                                                label="📄 PDF",
                                                data=f.read(),
                                                file_name="formulas_report.pdf",
                                                mime="application/pdf"
                                            )
                                
                                # ZIP Download
                                with dl_col2:
                                    zip_file = os.path.join(output_dir, 'extracted_formulas.zip')
                                    if os.path.exists(zip_file):
                                        with open(zip_file, 'rb') as f:
                                            st.download_button(
                                                label="📦 ZIP",
                                                data=f.read(),
                                                file_name="extracted_formulas.zip",
                                                mime="application/zip"
                                            )
                                
                                st.success("✓ All files are ready for download!")
                        
                        # Display view results if view was requested
                        if st.session_state.extraction_done == 'view':
                            if st.session_state.extracted_formulas is not None:
                                formulas = st.session_state.extracted_formulas
                                
                                st.subheader("🔍 Extracted Formulas Details")
                                
                                for formula in formulas:
                                    with st.expander(f"📐 Formula #{formula['id']} (Confidence: {formula['confidence']:.4f})", expanded=False):
                                        exp_col1, exp_col2 = st.columns(2)
                                        
                                        with exp_col1:
                                            st.write("**Formula Image:**")
                                            coords = formula['coordinates']
                                            crop_img = st.session_state.opencv_image[coords[1]:coords[3], coords[0]:coords[2]]
                                            st.image(crop_img, use_column_width=True)
                                            # Show AI description directly under the image if available
                                            if 'description' in formula and formula['description']:
                                                st.write("**About this formula:**")
                                                st.write(formula['description'])
                                        
                                        with exp_col2:
                                            st.write("**LaTeX Formula:**")
                                            st.code(formula['latex'], language='latex')
                                            st.write("**Rendered:**")
                                            render_latex_block(formula['latex'])
                                            st.write(f"**Bounding Box:** {formula['coordinates']}")
                    else:
                        st.warning("No formulas detected in the image.")



    #                 col1, col2, col3 = st.columns(3)
    #                 col1.header("Image")
    #                 col2.header("Latext")
    #                 col3.header("Formula")
    #                 if res == "Detection And Recogntion":
    #                     for each_box in results_boxes:
    #                         each_box = list(map(int,each_box))
    #                         crop_box = opencv_image[each_box[1]:each_box[3],each_box[0]:each_box[2],:]
    #                         crop_img = Image.fromarray(np.uint8(crop_box))
    #                         pred = RM.call_model(mathargs, *mathobjs, img=crop_img)
    #                         col1, col2, col3 = st.columns(3)
    #                         with col1:
    #                             st.image(crop_box)
    #                         with col2:
    #                             st.write(pred, width=5)
    #                         with col3:
    #                             st.markdown("$$"+pred+"$$")
    elif inf_style == 'PDF':
        imagem_referencia = st.sidebar.file_uploader("Choose an image", type=["pdf"])
        if st.sidebar.button('Clear uploaded file or image!'):
            st.write("attempt to clear uploaded_file")
            imagem_referencia.seek(0)
    #     res = st.sidebar.radio("Final Result",("Detection","Detection And Recogntion"))

        if imagem_referencia is not None:
            if imagem_referencia.type == "application/pdf":
                # Cache PDF pages to avoid repeated conversions
                if st.session_state.pdf_file_name != imagem_referencia.name:
                    pdf_bytes = imagem_referencia.read()
                    st.session_state.pdf_pages = pdf2image.convert_from_bytes(pdf_bytes)
                    st.session_state.pdf_file_name = imagem_referencia.name
                    # Reset state for new PDF
                    st.session_state.detection_done = False
                    st.session_state.extraction_done = False
                    st.session_state.results_boxes = None
                    st.session_state.extracted_formulas = None
                    st.session_state.extracted_crops = None
                    st.session_state.output_dir = None

                if st.session_state.pdf_pages:
                    page_idx = st.sidebar.number_input("Page Number", min_value=1, max_value=len(st.session_state.pdf_pages), value=1, step=1)
                    page_image = st.session_state.pdf_pages[int(page_idx) - 1]
                    # Convert PIL RGB page to OpenCV BGR for detector/recognizer
                    opencv_image = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
                    st.session_state.opencv_image = opencv_image
                    st.sidebar.image(page_image, caption=f"PDF Page {page_idx}")

                    # Reset detection/extraction state when page changes
                    if st.session_state.pdf_active_page != page_idx:
                        st.session_state.detection_done = False
                        st.session_state.extraction_done = False
                        st.session_state.results_boxes = None
                        st.session_state.extracted_formulas = None
                        st.session_state.extracted_crops = None
                        st.session_state.output_dir = None
                        st.session_state.pdf_active_page = page_idx

                    if st.button('Launch the Detection!', key='pdf_detect'):
                        results_boxes = MD.predict_formulas(opencv_image, math_model)
                        st.session_state.results_boxes = results_boxes
                        st.session_state.detection_done = True

                    if st.session_state.detection_done and st.session_state.results_boxes is not None:
                        results_boxes = st.session_state.results_boxes
                        images_rectangles = opencv_image.copy()
                        draw_rectangles(images_rectangles, results_boxes)
                        st.image(images_rectangles, caption=f"Detections on Page {page_idx}")

                        if len(results_boxes) > 0:
                            st.success(f"✓ Found {len(results_boxes)} formulas on page {page_idx}!")

                            col1, col2 = st.columns(2)

                            with col1:
                                if st.button("🚀 Extract Formulas to File (PDF)", key='pdf_extract_files'):
                                    with st.spinner("Extracting and recognizing formulas from PDF page..."):
                                        from datetime import datetime
                                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        output_dir = f"extracted_output_pdf_p{page_idx}_{timestamp}"
                                        os.makedirs(output_dir, exist_ok=True)

                                        st.session_state.extracted_crops = FE.extract_formula_crops(st.session_state.opencv_image, st.session_state.results_boxes)
                                        st.session_state.extracted_formulas = FE.recognize_formulas(st.session_state.extracted_crops, mathargs, mathobjs)
                                        # Optionally refine LaTeX using Gemini for weak outputs
                                        try:
                                            st.session_state.extracted_formulas = FE.refine_formulas_latex_with_gemini(
                                                st.session_state.extracted_formulas,
                                                st.session_state.extracted_crops,
                                                max_calls=8
                                            )
                                        except Exception:
                                            pass
                                        # Enrich with AI descriptions if available
                                        st.session_state.extracted_formulas = FE.enrich_formulas_with_descriptions(st.session_state.extracted_formulas)

                                        formulas = st.session_state.extracted_formulas
                                        extracted_crops = st.session_state.extracted_crops

                                        pdf_path = os.path.join(output_dir, 'formulas_report.pdf')
                                        FE.save_pdf_report(formulas, extracted_crops=extracted_crops, output_path=pdf_path, original_image=st.session_state.opencv_image)

                                        annotated_path = os.path.join(output_dir, 'annotated_image.png')
                                        FE.save_annotated_image(st.session_state.opencv_image, formulas, annotated_path)

                                        formula_dir = os.path.join(output_dir, 'formula_images')
                                        os.makedirs(formula_dir, exist_ok=True)
                                        for idx, crop_data in enumerate(extracted_crops):
                                            img_path = os.path.join(formula_dir, f'formula_{idx+1:04d}.png')
                                            cv2.imwrite(img_path, crop_data['image'])

                                        zip_path = os.path.join(output_dir, 'extracted_formulas.zip')
                                        with zipfile.ZipFile(zip_path, 'w') as zip_file:
                                            for root, dirs, files in os.walk(output_dir):
                                                for file in files:
                                                    if not file.endswith('.zip'):
                                                        file_path = os.path.join(root, file)
                                                        arcname = os.path.relpath(file_path, output_dir)
                                                        zip_file.write(file_path, arcname)

                                        st.session_state.extraction_done = True
                                        st.session_state.output_dir = output_dir

                                        st.success(f"✓ Successfully extracted {len(formulas)} formulas!")
                                        st.info(f"📁 All files saved to: **{output_dir}**")

                            with col2:
                                if st.button("👁️ View Extracted Formulas (PDF)", key='pdf_view_extract'):
                                    with st.spinner("Extracting and recognizing formulas from PDF page..."):
                                        st.session_state.extracted_crops = FE.extract_formula_crops(st.session_state.opencv_image, st.session_state.results_boxes)
                                        st.session_state.extracted_formulas = FE.recognize_formulas(st.session_state.extracted_crops, mathargs, mathobjs)
                                        # Optionally refine LaTeX using Gemini for weak outputs
                                        try:
                                            st.session_state.extracted_formulas = FE.refine_formulas_latex_with_gemini(
                                                st.session_state.extracted_formulas,
                                                st.session_state.extracted_crops,
                                                max_calls=8
                                            )
                                        except Exception:
                                            pass
                                        st.session_state.extracted_formulas = FE.enrich_formulas_with_descriptions(st.session_state.extracted_formulas)
                                        st.session_state.extraction_done = 'view'

                            if st.session_state.extraction_done == True:
                                if st.session_state.extracted_formulas is not None and st.session_state.output_dir is not None:
                                    formulas = st.session_state.extracted_formulas
                                    output_dir = st.session_state.output_dir

                                    st.success(f"✓ Successfully extracted {len(formulas)} formulas!")
                                    st.info(f"📁 All files saved to: **{output_dir}**")

                                    st.subheader("📥 Download Extracted Formulas")

                                    dl_col1, dl_col2 = st.columns(2)

                                    with dl_col1:
                                        pdf_file = os.path.join(output_dir, 'formulas_report.pdf')
                                        if os.path.exists(pdf_file):
                                            with open(pdf_file, 'rb') as f:
                                                st.download_button(
                                                    label="📄 PDF",
                                                    data=f.read(),
                                                    file_name="formulas_report.pdf",
                                                    mime="application/pdf"
                                                )

                                    with dl_col2:
                                        zip_file = os.path.join(output_dir, 'extracted_formulas.zip')
                                        if os.path.exists(zip_file):
                                            with open(zip_file, 'rb') as f:
                                                st.download_button(
                                                    label="📦 ZIP",
                                                    data=f.read(),
                                                    file_name="extracted_formulas.zip",
                                                    mime="application/zip"
                                                )

                                    st.success("✓ All files are ready for download!")

                            if st.session_state.extraction_done == 'view':
                                if st.session_state.extracted_formulas is not None:
                                    formulas = st.session_state.extracted_formulas

                                    st.subheader("🔍 Extracted Formulas Details")

                                    for formula in formulas:
                                        with st.expander(f"📐 Formula #{formula['id']} (Confidence: {formula['confidence']:.4f})", expanded=False):
                                            exp_col1, exp_col2 = st.columns(2)

                                            with exp_col1:
                                                st.write("**Formula Image:**")
                                                coords = formula['coordinates']
                                                crop_img = st.session_state.opencv_image[coords[1]:coords[3], coords[0]:coords[2]]
                                                st.image(crop_img, use_column_width=True)
                                                if 'description' in formula and formula['description']:
                                                    st.write("**About this formula:**")
                                                    st.write(formula['description'])

                                            with exp_col2:
                                                st.write("**LaTeX Formula:**")
                                                st.code(formula['latex'], language='latex')
                                                st.write("**Rendered:**")
                                                render_latex_block(formula['latex'])
                                                st.write(f"**Bounding Box:** {formula['coordinates']}")
                        else:
                            st.warning(f"No formulas detected on page {page_idx}.")


