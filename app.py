
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

if __name__ == '__main__':
    download_models()
    
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
    
    math_model = MD.initialize_model("./Models/MathDetector.ts")
    mathargs, *mathobjs = RM.initialize()

    st.title('Mathematical Formula Detector!')

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
                                        
                                        with exp_col2:
                                            st.write("**LaTeX Formula:**")
                                            st.code(formula['latex'], language='latex')
                                            st.write("**Rendered:**")
                                            st.markdown(f"$${formula['latex']}$$")
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
                images = pdf2image.convert_from_bytes(imagem_referencia.read())
                page_idx = st.sidebar.number_input("Page Number", min_value=1, max_value=len(images), value=1, step=1)
                opencv_image = np.array(images[int(page_idx)-1])
                results_boxes = MD.predict_formulas(opencv_image,math_model)
                images_rectangles = np.array(images[int(page_idx)-1])
                draw_rectangles(images_rectangles,results_boxes)
                st.image(images_rectangles)
    #             col1, col2, col3 = st.columns(3)
    #             col1.header("Image")
    #             col2.header("Latext")
    #             col3.header("Formula")
    #             if res == "Detection And Recogntion":
    #                 for each_box in results_boxes:
    #                     each_box = list(map(int,each_box))
    #                     crop_box = opencv_image[each_box[1]:each_box[3],each_box[0]:each_box[2],:]
    #                     crop_img = Image.fromarray(np.uint8(crop_box))
    #                     pred = RM.call_model(mathargs, *mathobjs, img=crop_img)
    #                     col1, col2, col3 = st.columns(3)
    #                     with col1:
    #                         st.image(crop_box)
    #                     with col2:
    #                         st.markdown(pred)
    #                     with col3:
    #                         st.markdown("$$"+pred+"$$")


