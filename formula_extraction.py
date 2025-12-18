"""
Formula Extraction Module
Extracts detected math formulas from images and saves them with their LaTeX representations
"""

import os
import json
import csv
import re
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
import io
from fpdf import FPDF
import os
from typing import List, Dict, Optional


def _get_gemini_api_key() -> Optional[str]:
    """Resolve Gemini API key from env or a simple .env file.
    Supports two formats:
    - Standard: GEMINI_API_KEY=...
    - Raw: first line is the key value
    """
    key = os.getenv('GEMINI_API_KEY')
    if key:
        return key.strip()
    env_path = os.path.join(os.getcwd(), '.env')
    try:
        if os.path.exists(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if '=' in content:
                    for line in content.splitlines():
                        if line.startswith('GEMINI_API_KEY='):
                            return line.split('=', 1)[1].strip()
                elif content:
                    return content  # treat as raw key
    except Exception:
        pass
    return None


def _build_gemini_client():
    """Lazily create a Gemini client if API key is available. Returns (model, enabled_bool)."""
    api_key = _get_gemini_api_key()
    if not api_key:
        return None, False
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model, True
    except Exception:
        return None, False


def describe_formula_with_gemini(latex: str) -> Optional[str]:
    """Use Gemini to generate a short, reader-friendly description of a LaTeX formula.
    Returns text or None on failure."""
    if not latex or not isinstance(latex, str):
        return None
    model, ok = _build_gemini_client()
    if not ok or model is None:
        return None
    prompt = (
        "You will be given a LaTeX math formula. Identify the formula's common name "
        "(e.g., 'Cauchy's Integral Formula', 'Gauss Divergence Theorem', 'Standard Deviation'), "
        "then provide a 2-3 sentence plain-English explanation of what it represents and typical use-cases. "
        "If the name is unclear, provide the closest general category (e.g., 'definite integral', 'vector calculus identity'). "
        "Keep it concise and helpful for a non-expert reader.\n\nLaTeX:\n" + latex
    )
    try:
        resp = model.generate_content(prompt, request_options={"timeout": 20})
        text = getattr(resp, 'text', None)
        if text:
            return text.strip()
    except Exception:
        return None
    return None


def enrich_formulas_with_descriptions(formulas: List[Dict]) -> List[Dict]:
    """Add a 'description' field to each recognized formula using Gemini if available."""
    if not formulas:
        return formulas
    for f in formulas:
        try:
            desc = describe_formula_with_gemini(f.get('latex', ''))
            if not desc:
                desc = _basic_formula_description(f.get('latex', ''))
            if desc:
                f['description'] = desc
        except Exception:
            pass
    return formulas


def _sanitize_latex_output(text: Optional[str]) -> Optional[str]:
    """Clean Gemini output to a bare LaTeX math expression.
    Removes code fences, leading/trailing $ or $$, and labels.
    """
    if not text:
        return text
    t = str(text).strip()
    # Remove triple fences
    t = re.sub(r"^```+\s*", "", t)
    t = re.sub(r"\s*```+$", "", t)
    # Remove language labels like 'latex' or 'LaTeX:' prefixes
    t = re.sub(r"^(latex|LaTeX)\s*:\s*", "", t)
    # Strip math delimiters
    t = t.strip()
    t = t.strip("$")
    t = t.strip()
    return t


def _is_latex_suspicious(expr: Optional[str]) -> bool:
    """Heuristic check: is the LaTeX likely invalid or too weak?
    - Empty or very short
    - Unbalanced braces
    - No LaTeX macros at all
    """
    if not expr or not isinstance(expr, str):
        return True
    s = expr.strip()
    if len(s) < 3:
        return True
    # Brace balance check
    bal = 0
    for ch in s:
        if ch == '{':
            bal += 1
        elif ch == '}':
            bal -= 1
            if bal < 0:
                return True
    if bal != 0:
        return True
    # At least one macro
    if not re.search(r"\\[A-Za-z]+", s):
        return True
    return False


def generate_latex_with_gemini_from_crop(crop_bgr: np.ndarray) -> Optional[str]:
    """Ask Gemini to produce clean LaTeX from a formula image crop."""
    model, ok = _build_gemini_client()
    if not ok or model is None:
        return None
    try:
        # Convert BGR np array to PIL Image
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        prompt = (
            "Return only the LaTeX math expression for this formula image. "
            "Do not include any explanations, words, or code fences. "
            "Prefer standard macros (\\frac, \\sum, \\int, \\nabla, \\partial, \\sqrt). "
            "Avoid surrounding $ or $$; output the bare LaTeX expression."
        )
        resp = model.generate_content([prompt, pil_img], request_options={"timeout": 25})
        text = getattr(resp, 'text', None)
        return _sanitize_latex_output(text)
    except Exception:
        return None


def refine_formulas_latex_with_gemini(formulas: List[Dict], extracted_crops: List[Dict], max_calls: int = 8) -> List[Dict]:
    """Replace suspicious LaTeX with Gemini-generated LaTeX from the crop image.
    Limits calls via `max_calls` to control cost.
    """
    if not formulas or not extracted_crops:
        return formulas
    calls = 0
    for f, crop in zip(formulas, extracted_crops):
        if calls >= max_calls:
            break
        curr = f.get('latex', '')
        if _is_latex_suspicious(curr):
            new_latex = generate_latex_with_gemini_from_crop(crop.get('image'))
            if new_latex and isinstance(new_latex, str) and len(new_latex.strip()) > 0:
                f['latex'] = new_latex.strip()
                calls += 1
    return formulas


def _basic_formula_description(latex: str) -> Optional[str]:
    """Heuristic fallback description when Gemini is unavailable.
    Provides a short reader-friendly summary based on LaTeX cues.
    """
    if not isinstance(latex, str) or not latex.strip():
        return None
    s = latex.replace("\\ ", " ").lower()

    # Named/common formulas first
    if "1" in s and "2\\pi" in s and "\\oint" in s:
        return (
            "**Cauchy's Integral Formula** – One of the fundamental theorems in complex analysis. "
            "This formula expresses the value of an analytic (holomorphic) function at any point inside a closed contour "
            "in terms of a contour integral around that path. It is instrumental in evaluating complex integrals, "
            "deriving power series representations (Taylor and Laurent series), and proving the residue theorem. "
            "Applications span theoretical physics (quantum field theory), engineering (signal processing), "
            "and pure mathematics (analytic number theory)."
        )
    if "\\nabla\\cdot" in s or "\\operatorname{div}" in s:
        return (
            "**Divergence Theorem (Gauss's Theorem)** – A cornerstone result in vector calculus connecting "
            "the flux of a vector field through a closed surface to the volume integral of the field's divergence. "
            "Mathematically: ∫∫_S F·n dS = ∫∫∫_V (∇·F) dV. This theorem is essential in electromagnetism "
            "(Gauss's law for electric fields), fluid dynamics (continuity equation), and heat transfer. "
            "It provides insight into how sources and sinks of a field behave within a volume."
        )
    if "\\sigma" in s and ("^2" in s or "\\sqrt" in s):
        return (
            "**Standard Deviation / Variance Formula** – A fundamental statistical measure quantifying "
            "the dispersion or spread of data points relative to their mean. Variance (σ²) is the average of "
            "squared deviations from the mean, while standard deviation (σ) is its square root. "
            "These metrics are critical in probability theory, hypothesis testing, quality control, "
            "finance (risk assessment), and machine learning (understanding data distributions and regularization)."
        )

    # General categories
    if "\\sum" in s:
        return (
            "**Series / Summation Expression** – Represents the sum of terms in a sequence, often indexed over integers. "
            "Summations are ubiquitous in discrete mathematics, combinatorics, number theory, and algorithm analysis. "
            "Common examples include arithmetic/geometric series, power series, and Fourier series. "
            "They model cumulative effects, total counts, and approximate continuous integrals in numerical methods."
        )
    if "\\int" in s and "_" in s:
        return (
            "**Definite Integral** – Calculates the accumulation of a quantity (area under a curve, total displacement, work done) "
            "over a specified interval [a, b]. The fundamental theorem of calculus links this to antiderivatives. "
            "Definite integrals are central to physics (computing work, energy, flux), engineering (signal processing), "
            "economics (consumer surplus), and probability (expected values via continuous distributions)."
        )
    if "\\oint" in s:
        return (
            "**Contour Integral** – An integral taken over a closed curve (contour) in the complex plane. "
            "This concept is vital in complex analysis, enabling evaluation via the residue theorem and Cauchy's theorem. "
            "Contour integrals simplify otherwise difficult real integrals and appear in quantum mechanics, "
            "string theory, and advanced electromagnetic field calculations."
        )
    if "\\partial" in s or "\\frac{\\partial" in s:
        return (
            "**Partial Derivative** – Measures the rate of change of a multivariable function with respect to one variable, "
            "holding others constant. Notation: ∂f/∂x. Partial derivatives are the foundation of gradient descent "
            "(machine learning optimization), thermodynamics (Maxwell relations), fluid dynamics (Navier-Stokes equations), "
            "and economics (marginal analysis). They generalize the single-variable derivative concept."
        )
    if "\\nabla" in s:
        return (
            "**Vector Calculus Operator (Nabla, ∇)** – The del operator applied to scalar or vector fields. "
            "When applied to a scalar, it yields the gradient (∇f), pointing in the direction of steepest ascent. "
            "For vector fields, ∇· gives divergence (source/sink density) and ∇× gives curl (rotational tendency). "
            "Essential in electromagnetism (Maxwell's equations), fluid mechanics, and optimization algorithms."
        )
    if "\\lim" in s:
        return (
            "**Limit Expression** – Describes the behavior of a function as its input approaches a specific value or infinity. "
            "Limits form the rigorous foundation of calculus (defining derivatives and integrals), "
            "analyze function continuity, and characterize asymptotic behavior in algorithm complexity analysis. "
            "They are indispensable in real analysis, numerical methods, and understanding convergence of sequences/series."
        )
    if "\\mathbb{e}" in s or "e^{" in s:
        return (
            "**Exponential Expression** – Functions involving the natural exponential base e ≈ 2.718. "
            "Exponential growth/decay models population dynamics, radioactive decay, compound interest, and neural activations. "
            "In differential equations, e^x is its own derivative, making it the solution to many fundamental ODEs. "
            "Appears throughout probability (exponential distribution), complex analysis (Euler's formula: e^(iθ) = cos θ + i sin θ), "
            "and information theory (entropy)."
        )
    if "\\log" in s:
        return (
            "**Logarithmic Expression** – The inverse of the exponential function, answering 'to what power must the base be raised?'. "
            "Logarithms compress large ranges (decibel scales, Richter scale), appear in complexity analysis (O(log n) algorithms), "
            "information theory (bits of information), and are central to entropy, pH calculations, and solving exponential equations."
        )
    if "\\frac" in s:
        return (
            "**Rational / Fractional Expression** – Represents the ratio of two quantities (numerator/denominator). "
            "Fractions model rates (speed = distance/time), probabilities, proportions, and normalized values. "
            "In calculus, they appear in derivative rules (quotient rule) and rational function integration. "
            "Common in physics (ratios of forces, densities), economics (marginal rates), and engineering (transfer functions)."
        )

    # Fallback generic
    return (
        "**General Mathematical Expression** – This formula represents a symbolic relationship between mathematical quantities. "
        "Without more specific LaTeX cues, it likely involves algebraic manipulation, calculus operations, or analytical techniques. "
        "Mathematical expressions encode laws of nature, logical relationships, optimization constraints, and computational algorithms. "
        "They serve as the universal language for modeling phenomena across science, engineering, economics, and technology."
    )


def _chunk_text_for_pdf(text: str, chunk_size: int = 80) -> str:
    """Insert spaces every `chunk_size` characters to allow fpdf2 to wrap long tokens.
    Avoids FPDFException when content has no spaces (e.g., long LaTeX strings)."""
    if not isinstance(text, str):
        text = str(text)
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return " ".join(chunks)




def extract_formula_crops(image, bboxes):
    """
    Extract individual formula regions from the image based on bounding boxes
    
    Parameters:
        image: opencv image (numpy array)
        bboxes: list of bounding boxes in format [x1, y1, x2, y2, conf, cls]
    
    Returns:
        list of extracted formula images
    """
    crops = []
    for bbox in bboxes:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        crop = image[y1:y2, x1:x2]
        crops.append({
            'image': crop,
            'bbox': bbox,
            'coordinates': (x1, y1, x2, y2)
        })
    return crops


def recognize_formulas(extracted_crops, model_args, model_objs):
    """
    Recognize LaTeX formulas from extracted crop images
    
    Parameters:
        extracted_crops: list of extracted crop dictionaries
        model_args: recognition model arguments
        model_objs: recognition model objects (model, tokenizer)
    
    Returns:
        list of recognized formulas with their crops
    """
    import Recog_MathForm as RM
    
    formulas = []
    for idx, crop_data in enumerate(extracted_crops):
        crop_img = Image.fromarray(np.uint8(crop_data['image']))
        try:
            latex_pred = RM.call_model(model_args, *model_objs, img=crop_img)

            # Fallback: pix2tex if installed
            if not isinstance(latex_pred, str) or latex_pred.strip() in {"", "ERROR", "[Unrecognized]"}:
                try:
                    from pix2tex.cli import LatexOCR
                    pix_model = LatexOCR()
                    latex_pred = pix_model(crop_img)
                except Exception:
                    pass

            formulas.append({
                'id': idx + 1,
                'bbox': crop_data['bbox'],
                'coordinates': crop_data['coordinates'],
                'latex': latex_pred,
                'confidence': crop_data['bbox'][4]
            })
        except Exception as e:
            print(f"Error recognizing formula {idx + 1}: {e}")
            formulas.append({
                'id': idx + 1,
                'bbox': crop_data['bbox'],
                'coordinates': crop_data['coordinates'],
                'latex': '[Unrecognized]',
                'confidence': crop_data['bbox'][4]
            })
    
    return formulas


def save_formulas_to_json(formulas, output_path='extracted_formulas.json'):
    """
    Save extracted formulas to JSON file
    
    Parameters:
        formulas: list of recognized formula dictionaries
        output_path: path to save JSON file
    """
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'total_formulas': len(formulas),
        'formulas': []
    }
    
    for formula in formulas:
        output_data['formulas'].append({
            'id': formula['id'],
            'coordinates': formula['coordinates'],
            'bbox': formula['bbox'],
            'latex': formula['latex'],
            'confidence': formula['confidence']
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return output_path


def save_formulas_to_csv(formulas, output_path='extracted_formulas.csv'):
    """
    Save extracted formulas to CSV file
    
    Parameters:
        formulas: list of recognized formula dictionaries
        output_path: path to save CSV file
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'X1', 'Y1', 'X2', 'Y2', 'LaTeX', 'Confidence'])
        
        for formula in formulas:
            coords = formula['coordinates']
            writer.writerow([
                formula['id'],
                coords[0],
                coords[1],
                coords[2],
                coords[3],
                formula['latex'],
                f"{formula['confidence']:.4f}"
            ])
    
    return output_path


def save_formula_images(extracted_crops, output_dir='extracted_formulas'):
    """
    Save individual formula images to directory
    
    Parameters:
        extracted_crops: list of extracted crop dictionaries
        output_dir: directory to save formula images
    
    Returns:
        list of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for idx, crop_data in enumerate(extracted_crops):
        filename = os.path.join(output_dir, f'formula_{idx + 1:04d}.png')
        cv2.imwrite(filename, crop_data['image'])
        saved_paths.append(filename)
    
    return saved_paths


def save_annotated_image(image, formulas, output_path='annotated_image.png'):
    """
    Save image with bounding boxes and LaTeX annotations
    
    Parameters:
        image: original opencv image
        formulas: list of recognized formula dictionaries
        output_path: path to save annotated image
    """
    annotated = image.copy()
    
    for formula in formulas:
        coords = formula['coordinates']
        x1, y1, x2, y2 = coords
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add LaTeX text above box
        text = f"ID: {formula['id']} | Conf: {formula['confidence']:.2f}"
        cv2.putText(annotated, text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add LaTeX formula as text
        latex_text = formula['latex'][:50]  # Truncate long formulas
        cv2.putText(annotated, latex_text, (x1, y2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    cv2.imwrite(output_path, annotated)
    return output_path


def save_html_report(formulas, image_path=None, output_path='formulas_report.html'):
    """
    Create an HTML report with extracted formulas
    
    Parameters:
        formulas: list of recognized formula dictionaries
        image_path: path to annotated image (optional)
        output_path: path to save HTML file
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Math Formula Extraction Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .header { background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px; }
            .formula-card { 
                border: 1px solid #ddd; 
                padding: 15px; 
                margin: 10px 0; 
                border-radius: 5px;
                background-color: #f9f9f9;
            }
            .latex { 
                background-color: #f0f0f0; 
                padding: 10px; 
                font-family: monospace; 
                border-left: 3px solid #4CAF50;
                margin: 10px 0;
            }
            .coordinates { color: #666; font-size: 0.9em; }
            .confidence { color: #4CAF50; font-weight: bold; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📐 Math Formula Extraction Report</h1>
            <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            <p>Total Formulas: <strong>""" + str(len(formulas)) + """</strong></p>
        </div>
    """
    
    if image_path and os.path.exists(image_path):
        html_content += f'<img src="{image_path}" style="max-width: 100%; border: 1px solid #ddd; margin: 20px 0;">'
    
    html_content += "<h2>Formulas Summary</h2><table><tr><th>ID</th><th>Coordinates (X1,Y1,X2,Y2)</th><th>LaTeX</th><th>Confidence</th></tr>"
    
    for formula in formulas:
        coords = formula['coordinates']
        coords_str = f"({coords[0]}, {coords[1]}, {coords[2]}, {coords[3]})"
        html_content += f"""
        <tr>
            <td>{formula['id']}</td>
            <td class="coordinates">{coords_str}</td>
            <td class="latex">{formula['latex']}</td>
            <td class="confidence">{formula['confidence']:.4f}</td>
        </tr>
        """
    
    html_content += "</table><h2>Detailed View</h2>"
    
    for formula in formulas:
        html_content += f"""
        <div class="formula-card">
            <h3>Formula #{formula['id']}</h3>
            <p><strong>Coordinates:</strong> {formula['coordinates']}</p>
            <p><strong>Confidence:</strong> <span class="confidence">{formula['confidence']:.4f}</span></p>
            <p><strong>LaTeX:</strong></p>
            <div class="latex">{formula['latex']}</div>
            <p><strong>Rendered (if LaTeX valid):</strong></p>
            <div class="latex">\\({formula['latex']}\\)</div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path


def save_pdf_report(formulas, extracted_crops=None, output_path='formulas_report.pdf', original_image=None):
    """
    Create a single-page PDF that matches the detected page view with all boxes visible.

    Parameters:
        formulas: list of recognized formula dictionaries
        extracted_crops: list of extracted crop dictionaries (unused here but kept for API compatibility)
        output_path: path to save PDF file
        original_image: numpy image (BGR) of the page to embed with boxes
    """
    pdf = FPDF(format='A4', orientation='P')
    pdf.set_auto_page_break(auto=False, margin=5)
    pdf.add_page()

    # If original image is provided, draw boxes and embed as a single page
    if original_image is not None:
        annotated = original_image.copy()
        # Draw red boxes like the UI view
        for f in formulas:
            x1, y1, x2, y2 = map(int, f['coordinates'])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Convert to RGB PIL image for FPDF
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(annotated_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)

        # Fit image to page width while keeping aspect ratio
        page_w = pdf.w - 10  # margin already set to 5 each side
        page_h = pdf.h - 10
        img_w, img_h = pil_img.size
        scale = min(page_w / img_w, page_h / img_h)
        render_w = img_w * scale
        render_h = img_h * scale

        # Center the image
        x = (pdf.w - render_w) / 2
        y = (pdf.h - render_h) / 2
        pdf.image(buf, x=x, y=y, w=render_w, h=render_h)
    else:
        # Fallback: simple table if no image passed
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Mathematical Formula Extraction Report', ln=True)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Total Formulas: {len(formulas)}", ln=True)
        pdf.ln(4)

        pdf.set_font('Arial', 'B', 9)
        pdf.cell(10, 7, 'ID', 1)
        pdf.cell(40, 7, 'Coords', 1)
        pdf.cell(120, 7, 'LaTeX', 1)
        pdf.cell(20, 7, 'Conf', 1, ln=True)
        pdf.set_font('Arial', '', 8)
        for f in formulas:
            coords = f['coordinates']
            coords_str = f"({coords[0]}, {coords[1]}, {coords[2]}, {coords[3]})"
            latex_summary = (f['latex'][:60] + '...') if len(f['latex']) > 60 else f['latex']
            pdf.cell(10, 6, str(f['id']), 1)
            pdf.cell(40, 6, coords_str, 1)
            pdf.cell(120, 6, latex_summary, 1)
            pdf.cell(20, 6, f"{f['confidence']:.3f}", 1, ln=True)

    pdf.output(output_path)
    return output_path
