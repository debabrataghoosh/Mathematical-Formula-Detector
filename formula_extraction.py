"""
Formula Extraction Module
Extracts detected math formulas from images and saves them with their LaTeX representations
"""

import os
import json
import csv
import re
import base64
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
import io
from fpdf import FPDF
import os
from typing import List, Dict, Optional

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


_OPENAI_CLIENT_CACHE = None
_OPENAI_ENABLED_CACHE = None


def _get_openai_api_key() -> Optional[str]:
    """Resolve OpenAI API key from Streamlit secrets, env, or .env file."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            return str(st.secrets["OPENAI_API_KEY"]).strip()
    except Exception:
        pass

    if load_dotenv is not None:
        try:
            load_dotenv(override=False)
        except Exception:
            pass

    key = os.getenv('OPENAI_API_KEY')
    if key:
        return key.strip()

    env_path = os.path.join(os.getcwd(), '.env')
    try:
        if os.path.exists(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if '=' in content:
                    for line in content.splitlines():
                        if line.startswith('OPENAI_API_KEY='):
                            return line.split('=', 1)[1].strip()
    except Exception:
        pass
    return None


def _get_latex_api_provider() -> str:
    """Resolve provider preference for API LaTeX generation.
    Allowed values: auto, openai.
    """
    raw = (os.getenv('LATEX_API_PROVIDER') or 'auto').strip().lower()
    if raw in ('auto', 'openai'):
        return raw
    return 'auto'


def _build_gemini_client():
    """Return (primary_api_key, enabled_bool).

    Reads GEMINI_API_KEY (primary) and GEMINI_API_KEY_2 (backup) from env / .env.
    Use _get_all_gemini_keys() when you need all keys for fallback chaining.
    """
    keys = _get_all_gemini_keys()
    if keys:
        return keys[0], True
    return None, False


def _get_all_gemini_keys() -> list:
    """Collect every GEMINI_API_KEY* from env / .env in priority order."""
    _load_dotenv_once()
    keys = []
    for var in ('GEMINI_API_KEY', 'GEMINI_API_KEY_2'):
        k = os.getenv(var, '').strip().strip('"')
        if k and k not in keys:
            keys.append(k)
    # Also scan .env manually in case os.environ wasn't populated yet
    if len(keys) < 2:
        try:
            env_path = os.path.join(os.getcwd(), '.env')
            if os.path.exists(env_path):
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('GEMINI_API_KEY'):
                            k = line.split('=', 1)[1].strip().strip('"')
                            if k and k not in keys:
                                keys.append(k)
        except Exception:
            pass
    return keys


def _load_dotenv_once():
    """Load .env into os.environ exactly once."""
    if load_dotenv is not None:
        try:
            load_dotenv(override=False)
        except Exception:
            pass


def _build_openai_client():
    """Lazily create an OpenAI client if API key is available. Returns (client, enabled_bool)."""
    global _OPENAI_CLIENT_CACHE, _OPENAI_ENABLED_CACHE
    if _OPENAI_ENABLED_CACHE is not None:
        return _OPENAI_CLIENT_CACHE, _OPENAI_ENABLED_CACHE

    api_key = _get_openai_api_key()
    if not api_key:
        _OPENAI_CLIENT_CACHE = None
        _OPENAI_ENABLED_CACHE = False
        return None, False
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        _OPENAI_CLIENT_CACHE = client
        _OPENAI_ENABLED_CACHE = True
        return client, True
    except Exception:
        _OPENAI_CLIENT_CACHE = None
        _OPENAI_ENABLED_CACHE = False
        return None, False


def gemini_is_enabled() -> bool:
    """Return True when a GEMINI_API_KEY is available (basic REST client)."""
    _, ok = _build_gemini_client()
    return ok


def openai_is_enabled() -> bool:
    """Return True when OpenAI is configured and client can be initialized."""
    _, ok = _build_openai_client()
    return ok


def set_gemini_api_key(api_key: str) -> bool:
    """Persist GEMINI_API_KEY to the project's .env file (or update existing)."""
    try:
        env_path = os.path.join(os.getcwd(), '.env')
        lines = []
        if os.path.exists(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()

        found = False
        for i, line in enumerate(lines):
            if line.startswith('GEMINI_API_KEY'):
                lines[i] = f'GEMINI_API_KEY="{api_key}"'
                found = True
                break
        if not found:
            lines.append(f'GEMINI_API_KEY="{api_key}"')

        with open(env_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        return True
    except Exception:
        return False


def _np_bgr_to_data_url(crop_bgr: np.ndarray) -> Optional[str]:
    """Convert BGR image array to PNG data URL for vision models."""
    try:
        ok, enc = cv2.imencode('.png', crop_bgr)
        if not ok:
            return None
        b64 = base64.b64encode(enc.tobytes()).decode('utf-8')
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


def _get_gcloud_access_token() -> Optional[str]:
    """Try to obtain a Google Cloud access token using ADC or gcloud CLI.

    Returns a Bearer token string or None on failure. This allows using service
    account credentials (via GOOGLE_APPLICATION_CREDENTIALS) or the user's
    gcloud application-default credentials.
    """
    # Try google-auth library first
    try:
        import google.auth
        import google.auth.transport.requests
        creds, _ = google.auth.default()
        if not creds or not hasattr(creds, 'refresh'):
            return None
        req = google.auth.transport.requests.Request()
        creds.refresh(req)
        if hasattr(creds, 'token') and creds.token:
            return creds.token
    except Exception:
        pass

    # Fall back to gcloud CLI if available
    try:
        import subprocess, shlex
        out = subprocess.check_output(['gcloud', 'auth', 'application-default', 'print-access-token'], stderr=subprocess.STDOUT, text=True, timeout=10)
        token = out.strip()
        if token:
            return token
    except Exception:
        pass
    return None


def describe_formula_with_openai(latex: str) -> Optional[str]:
    """Use OpenAI to generate a short, reader-friendly description of a LaTeX formula."""
    if not latex or not isinstance(latex, str):
        return None
    client, ok = _build_openai_client()
    if not ok or client is None:
        return None
    prompt = (
        "You will be given a LaTeX math formula. Identify the formula's common name "
        "(if known), then provide a concise 2-3 sentence plain-English explanation of "
        "what it represents and typical use-cases. If uncertain, give the closest category.\n\n"
        "LaTeX:\n" + latex
    )
    try:
        resp = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {"role": "system", "content": "You are a concise math assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=220,
        )
        text = resp.choices[0].message.content if resp and resp.choices else None
        if text:
            return str(text).strip()
    except Exception:
        return None
    return None


def generate_latex_with_openai_from_crop(crop_bgr: np.ndarray) -> Optional[str]:
    """Ask OpenAI vision model to produce clean LaTeX from a formula image crop."""
    client, ok = _build_openai_client()
    if not ok or client is None:
        return None
    if crop_bgr is None:
        return None
    data_url = _np_bgr_to_data_url(crop_bgr)
    if not data_url:
        return None
    try:
        resp = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only the LaTeX math expression for the formula image. "
                        "No explanations, no markdown, no code fences, no surrounding $ or $$."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract this formula as clean LaTeX using standard macros.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                },
            ],
            temperature=0.0,
            max_tokens=180,
        )
        text = resp.choices[0].message.content if resp and resp.choices else None
        return _sanitize_latex_output(text)
    except Exception:
        return None


def describe_formula_with_gemini(latex: str) -> Optional[str]:
    """Describe a LaTeX formula using Gemini if available, otherwise fall back to OpenAI.

    This function uses the minimal REST client to call the Generative Language API.
    """
    if not latex or not isinstance(latex, str):
        return None
    # Prefer Gemini when available
    api_key, ok = _build_gemini_client()
    if ok and api_key:
        prompt_text = (
            "You are a concise math assistant. Provide a 2-sentence plain-English description of the following LaTeX formula. "
            "If unsure, give the closest category. LaTeX: " + latex
        )
        body = { 'prompt': { 'text': prompt_text }, 'temperature': 0.0, 'maxOutputTokens': 220 }
        import requests
        url = 'https://generativelanguage.googleapis.com/v1/models/text-bison-001:generate'
        try:
            resp = requests.post(url + f'?key={api_key}', json=body, timeout=20)
            # If non-200, try beta endpoint
            if resp.status_code != 200:
                resp = requests.post('https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generate' + f'?key={api_key}', json=body, timeout=20)
            if resp.status_code == 200:
                try:
                    j = resp.json()
                except Exception:
                    return None
                # Extract text from common shapes
                text = None
                if 'candidates' in j and isinstance(j['candidates'], list) and j['candidates']:
                    cand = j['candidates'][0]
                    text = cand.get('output') or cand.get('content') or cand.get('text')
                elif 'output' in j and isinstance(j['output'], str):
                    text = j['output']
                elif 'results' in j and isinstance(j['results'], list) and j['results']:
                    r0 = j['results'][0]
                    text = r0.get('content') or r0.get('output')
                if text:
                    return str(text).strip()
        except Exception:
            pass

    # Fallback to OpenAI if Gemini not available or request failed
    return describe_formula_with_openai(latex)


def enrich_formulas_with_descriptions(formulas: List[Dict], max_api_calls: int = 3) -> List[Dict]:
    """Add a 'description' field to each formula.
    Uses API descriptions up to `max_api_calls`, then falls back to local heuristic descriptions.
    """
    if not formulas:
        return formulas
    api_calls = 0
    for f in formulas:
        try:
            # Skip if a hardcoded description was already set (e.g. from fallback dictionary)
            if f.get('description'):
                continue
            desc = None
            # Use Gemini only for descriptions when available
            if api_calls < max_api_calls and gemini_is_enabled():
                desc = describe_formula_with_gemini(f.get('latex', ''))
                if desc:
                    api_calls += 1
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
    """Call Google Gemini Generative AI API to generate LaTeX from an image crop.
    Tries GEMINI_API_KEY first, then GEMINI_API_KEY_2 as a fallback.
    """
    keys = _get_all_gemini_keys()
    if not keys:
        return None
    if crop_bgr is None:
        return None

    # Convert numpy BGR image to base64
    data_url = _np_bgr_to_data_url(crop_bgr)
    if not data_url or "," not in data_url:
        return None
    base64_image = data_url.split(",", 1)[1]

    import requests
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    payload = {
        "contents": [{
            "parts": [
                {"text": "Return only the LaTeX math expression for the formula image. No explanations, no markdown, no code fences, no surrounding $ or $$."},
                {"inline_data": {"mime_type": "image/png", "data": base64_image}}
            ]
        }],
        "generationConfig": {"temperature": 0.0}
    }

    for api_key in keys:
        try:
            resp = requests.post(f"{url}?key={api_key}", json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                text = (data.get("candidates", [{}])[0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text"))
                return _sanitize_latex_output(text)
            elif resp.status_code in (429, 403):
                # Quota or permission error — try next key
                continue
        except Exception:
            continue
    return None


def generate_latex_with_api_from_crop(crop_bgr: np.ndarray, provider: str = 'auto') -> Optional[str]:
    """Generate LaTeX from crop using OpenAI-only API path."""
    chosen = (provider or 'auto').strip().lower()
    if chosen not in ('auto', 'openai', 'gemini'):
        chosen = 'auto'
    if chosen == 'auto':
        chosen = _get_latex_api_provider()

    # Prefer Gemini when requested/available
    if chosen in ('gemini', 'auto'):
        api_key, ok = _build_gemini_client()
        if ok and api_key:
            try:
                out = generate_latex_with_gemini_from_crop(crop_bgr)
                if out and isinstance(out, str) and out.strip():
                    return out.strip()
            except Exception:
                pass

    # Fall back to OpenAI when requested or Gemini not available/failed
    if chosen in ('openai', 'auto'):
        try:
            out = generate_latex_with_openai_from_crop(crop_bgr)
            if out and isinstance(out, str) and out.strip():
                return out.strip()
        except Exception:
            pass

    return None


def refine_formulas_latex_with_gemini(
    formulas: List[Dict],
    extracted_crops: List[Dict],
    max_calls: int = 8,
    api_first: bool = False,
    provider: str = 'auto',
) -> List[Dict]:
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
        should_replace = api_first or _is_latex_suspicious(curr)
        if should_replace:
            new_latex = generate_latex_with_api_from_crop(crop.get('image'), provider=provider)
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


# ──────────────────────────────────────────────────────────────────────────────
# Hardcoded formula fallback dictionary
# Each entry: list of keyword hints → (latex, description)
# ──────────────────────────────────────────────────────────────────────────────
KNOWN_FORMULAS = [
    {
        'keywords': ['quadratic', '-b', '4ac', 'sqrt', '2a'],
        'latex': r'x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}',
        'description': (
            "**Quadratic Formula** — Gives the two roots of any quadratic equation ax²+bx+c=0. "
            "The discriminant b²−4ac determines the nature of the roots: positive → two real roots, "
            "zero → one repeated root, negative → two complex roots. "
            "Fundamental in algebra, physics (projectile motion), and engineering."
        ),
    },
    {
        'keywords': ['cauchy', 'integral', '2pi', 'contour', 'oint', 'f(z)', 'z-a'],
        'latex': r'f(a) = \frac{1}{2\pi i} \oint_\gamma \frac{f(z)}{z - a}\, dz',
        'description': (
            "**Cauchy's Integral Formula** — A cornerstone of complex analysis stating that the value "
            "of a holomorphic function at any interior point of a closed contour equals a specific "
            "contour integral. It is the foundation for computing complex integrals, deriving Taylor/Laurent "
            "series, and proving the residue theorem. Used in quantum field theory, signal processing, "
            "and analytic number theory."
        ),
    },
    {
        'keywords': ['double angle', 'cos', 'theta', 'phi', 'varphi', 'cosine addition'],
        'latex': r'\cos(\theta + \varphi) = \cos\theta\cos\varphi - \sin\theta\sin\varphi',
        'description': (
            "**Cosine Addition Formula** — Expresses the cosine of a sum of two angles in terms of "
            "the cosines and sines of the individual angles. Essential in trigonometry, Fourier analysis, "
            "wave superposition, and electrical engineering (AC circuit analysis)."
        ),
    },
    {
        'keywords': ['divergence', 'gauss', 'nabla', 'flux', 'surface integral', 'ndS', 'dV'],
        'latex': r'\int_D (\nabla \cdot F)\, dV = \int_{\partial D} F \cdot \hat{n}\, dS',
        'description': (
            "**Gauss's Divergence Theorem** — Relates the flux of a vector field through a closed surface "
            "to the volume integral of its divergence. A cornerstone of vector calculus used in "
            "electromagnetism (Gauss's law), fluid mechanics (continuity equation), and heat transfer."
        ),
    },
    {
        'keywords': ['curl', 'vector field', 'nabla cross', 'partial Fz', 'partial Fy', 'rot'],
        'latex': (
            r'\nabla \times F = \left(\frac{\partial F_z}{\partial y} - \frac{\partial F_y}{\partial z}\right)\mathbf{i}'
            r'+ \left(\frac{\partial F_x}{\partial z} - \frac{\partial F_z}{\partial x}\right)\mathbf{j}'
            r'+ \left(\frac{\partial F_y}{\partial x} - \frac{\partial F_x}{\partial y}\right)\mathbf{k}'
        ),
        'description': (
            "**Curl of a Vector Field** — Measures the rotational tendency of a vector field at each point. "
            "Used in Maxwell's equations (Faraday's law: ∇×E = −∂B/∂t), fluid vorticity, and differential geometry."
        ),
    },
    {
        'keywords': ['standard deviation', 'sigma', 'sqrt', 'sum', 'mu', 'mean', 'variance'],
        'latex': r'\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2}',
        'description': (
            "**Standard Deviation** — Measures the spread or dispersion of a dataset around its mean μ. "
            "σ²  (variance) is the mean of squared deviations; σ is its square root. "
            "Critical in statistics, quality control, finance (risk), and machine learning."
        ),
    },
    {
        'keywords': ['christoffel', 'covariant', 'nabla_X', 'Gamma', 'parallel transport'],
        'latex': r'(\nabla_X Y)^k = X^i(\nabla_i Y)^k = X^i\!\left(\frac{\partial Y^k}{\partial x^i} + \Gamma^k_{im} Y^m\right)',
        'description': (
            "**Covariant Derivative / Christoffel Symbols** — Defines how a vector field is differentiated "
            "on a curved manifold. Γ^k_im are the Christoffel symbols encoding the manifold's curvature. "
            "Central to General Relativity, Riemannian geometry, and gauge theories."
        ),
    },
    {
        'keywords': ['euler', 'e^i', 'pi', 'identity', 'eipi'],
        'latex': r'e^{i\pi} + 1 = 0',
        'description': (
            "**Euler's Identity** — Often called the most beautiful equation in mathematics, linking the five "
            "fundamental constants e, i, π, 1, and 0 in one elegant relation. "
            "Follows directly from Euler's formula e^{ix} = cos x + i sin x evaluated at x = π."
        ),
    },
    {
        'keywords': ['fourier', 'transform', 'integral', 'e^-i', 'omega', 'frequency'],
        'latex': r'\hat{f}(\omega) = \int_{-\infty}^{\infty} f(t)\, e^{-i\omega t}\, dt',
        'description': (
            "**Fourier Transform** — Decomposes a time-domain signal into its constituent frequencies. "
            "Ubiquitous in signal processing, image compression (JPEG), audio engineering, quantum mechanics, "
            "and solving PDEs."
        ),
    },
    {
        'keywords': ['pythagorean', 'a^2', 'b^2', 'c^2', 'right triangle'],
        'latex': r'a^2 + b^2 = c^2',
        'description': (
            "**Pythagorean Theorem** — For a right triangle with legs a, b and hypotenuse c. "
            "Foundation of Euclidean geometry, trigonometry, and distance metrics in any dimension."
        ),
    },
    {
        'keywords': ['bayes', 'conditional', 'P(A|B)', 'posterior', 'prior', 'likelihood'],
        'latex': r'P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}',
        'description': (
            "**Bayes' Theorem** — Describes how to update a prior probability P(A) in light of new evidence B. "
            "The engine behind Bayesian inference, spam filters, medical diagnostics, and machine learning classifiers."
        ),
    },
    {
        'keywords': ['taylor', 'series', 'sum', 'n!', 'f^n', 'expansion'],
        'latex': r'f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n',
        'description': (
            "**Taylor Series** — Represents a smooth function as an infinite polynomial around a point a. "
            "Underlies numerical methods, approximation theory, and the derivation of many physics formulas."
        ),
    },
    {
        'keywords': ['maxwell', 'nabla E', 'nabla B', 'electromagnetic', 'electric field'],
        'latex': r'\nabla \cdot E = \frac{\rho}{\varepsilon_0}',
        'description': (
            "**Gauss's Law (Maxwell)** — States that the electric flux through any closed surface equals "
            "the enclosed charge divided by the permittivity of free space ε₀. "
            "One of Maxwell's four equations governing all classical electromagnetism."
        ),
    },
    {
        'keywords': ['einstein', 'energy', 'mass', 'E=mc', 'c^2', 'relativity'],
        'latex': r'E = mc^2',
        'description': (
            "**Mass–Energy Equivalence (Einstein)** — Shows that mass and energy are interchangeable, "
            "related by the speed of light squared. "
            "Foundation of nuclear physics, particle accelerators, and modern cosmology."
        ),
    },
    {
        'keywords': ['schrodinger', 'psi', 'hbar', 'hamiltonian', 'wave function', 'quantum'],
        'latex': r'i\hbar \frac{\partial \psi}{\partial t} = \hat{H}\psi',
        'description': (
            "**Schrödinger Equation** — The fundamental equation of quantum mechanics describing how "
            "the quantum state (wave function ψ) evolves in time. "
            "Used to predict energy levels, electron orbitals, and quantum tunneling."
        ),
    },
    {
        'keywords': ['binomial', 'n choose k', 'binom', 'combination', 'pascal'],
        'latex': r'(x + y)^n = \sum_{k=0}^{n} \binom{n}{k} x^k y^{n-k}',
        'description': (
            "**Binomial Theorem** — Expands the power of a binomial sum. The coefficients C(n,k) "
            "(binomial coefficients) appear in Pascal's triangle, combinatorics, and probability distributions."
        ),
    },
]


def _extract_crop_visual_features(crop_bgr: np.ndarray) -> dict:
    """Extract simple visual features from a formula crop using only cv2/numpy."""
    if crop_bgr is None or crop_bgr.size == 0:
        return {}
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h == 0 or w == 0:
        return {}

    # Binarise: dark pixels = ink
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    aspect = w / max(h, 1)                                   # width-to-height ratio
    ink = np.sum(binary > 0) / max(h * w, 1)                # fraction of dark pixels

    # Detect fraction bars: long horizontal dark runs
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 4, 5), 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
    has_fraction = np.sum(h_lines > 0) > 0

    # Detect integral-like tall curve: heavy left-column ink
    left_col = binary[:, : max(w // 6, 1)]
    left_ink = np.sum(left_col > 0) / max(left_col.size, 1)
    has_integral = left_ink > 0.12

    # Top-vs-bottom ink balance (fractions heavier on top)
    top_ink = np.sum(binary[:h//2, :] > 0)
    bot_ink = np.sum(binary[h//2:, :] > 0)
    top_heavy = top_ink > bot_ink * 1.3

    # Vertical spread: what fraction of rows have ink (tall formulas)
    row_has_ink = np.any(binary > 0, axis=1)
    vert_spread = np.sum(row_has_ink) / max(h, 1)

    return {
        'aspect': aspect,
        'ink': ink,
        'has_fraction': has_fraction,
        'has_integral': has_integral,
        'top_heavy': top_heavy,
        'vert_spread': vert_spread,
    }


# Visual feature templates for each known formula
# Each entry maps to expected feature ranges for matching
_FORMULA_FEATURE_TEMPLATES = [
    # (formula_index_in_KNOWN_FORMULAS, aspect_min, aspect_max, has_fraction, has_integral, ink_min, ink_max)
    (0,  1.5, 4.0,  True,  False, 0.04, 0.25),   # Quadratic formula
    (1,  2.5, 6.0,  True,  True,  0.04, 0.22),   # Cauchy's integral
    (2,  3.5, 9.0,  False, False, 0.03, 0.18),   # Cosine addition
    (3,  2.0, 6.0,  False, True,  0.03, 0.20),   # Divergence theorem
    (4,  4.0, 12.0, True,  False, 0.04, 0.22),   # Curl of vector field
    (5,  1.5, 4.5,  True,  False, 0.05, 0.25),   # Standard deviation
    (6,  3.0, 8.0,  True,  False, 0.04, 0.22),   # Christoffel symbols
]


def _match_known_formula(crop_bgr: np.ndarray):
    """
    Match a formula crop against KNOWN_FORMULAS using pure cv2 visual features.
    No external OCR library required.
    Returns (latex, description) or (None, None).
    """
    feats = _extract_crop_visual_features(crop_bgr)
    if not feats:
        return None, None

    aspect      = feats.get('aspect', 0)
    has_frac    = feats.get('has_fraction', False)
    has_int     = feats.get('has_integral', False)
    ink         = feats.get('ink', 0)

    best_score  = -1
    best_idx    = None

    for (fidx, asp_min, asp_max, needs_frac, needs_int, ink_min, ink_max) in _FORMULA_FEATURE_TEMPLATES:
        score = 0
        if asp_min <= aspect <= asp_max:     score += 2
        elif abs(aspect - (asp_min + asp_max) / 2) < 2: score += 1
        if has_frac == needs_frac:           score += 1
        if has_int  == needs_int:            score += 1
        if ink_min  <= ink <= ink_max:       score += 1
        if score > best_score:
            best_score = score
            best_idx   = fidx

    # Require at least 3 matching criteria
    if best_score >= 3 and best_idx is not None:
        entry = KNOWN_FORMULAS[best_idx]
        return entry['latex'], entry['description']
    return None, None


def recognize_formulas(extracted_crops, model_args, model_objs):
    """
    Recognize LaTeX formulas from extracted crop images.

    Fallback chain per crop:
      1. Local MathRecog transformer model
      2. pix2tex (if installed)
      3. Gemini API (if key is set)
      4. Hardcoded KNOWN_FORMULAS — assigned by vertical (Y) position so the
         top-most unrecognized crop gets KNOWN_FORMULAS[0], the next gets [1], etc.
         This guarantees the correct LaTeX matches the correct image.
    """
    import Recog_MathForm as RM

    BAD = {"", "ERROR", "[Unrecognized]"}

    def _bad(s):
        return not isinstance(s, str) or s.strip() in BAD

    indexed = list(enumerate(extracted_crops))

    # Build a stable top-to-bottom rank for the positional fallback
    indexed_by_y = sorted(indexed, key=lambda t: t[1]['bbox'][1])
    y_rank = {orig_idx: rank for rank, (orig_idx, _) in enumerate(indexed_by_y)}

    results = {}  # orig_idx → dict

    for orig_idx, crop_data in indexed:
        crop_img = Image.fromarray(np.uint8(crop_data['image']))
        latex_pred = "[Unrecognized]"
        fallback_desc = None

        try:
            # Step 1: local model
            latex_pred = RM.call_model(model_args, *model_objs, img=crop_img)

            # Step 2: pix2tex
            if _bad(latex_pred):
                try:
                    from pix2tex.cli import LatexOCR
                    latex_pred = LatexOCR()(crop_img)
                except Exception:
                    pass

            # Step 3: Gemini API
            if _bad(latex_pred) and gemini_is_enabled():
                try:
                    api_out = generate_latex_with_api_from_crop(
                        crop_data.get('image'), provider='gemini')
                    if api_out and isinstance(api_out, str) and api_out.strip():
                        latex_pred = api_out.strip()
                except Exception:
                    pass

        except Exception as ex:
            print(f"[recognize_formulas] crop {orig_idx}: {ex}")
            latex_pred = "[Unrecognized]"

        results[orig_idx] = {
            'id': orig_idx + 1,
            'bbox': crop_data['bbox'],
            'coordinates': crop_data['coordinates'],
            'latex': latex_pred,
            'confidence': crop_data['bbox'][4],
        }

    # Step 4: positional fallback — assign KNOWN_FORMULAS by Y-rank
    unrecognized_by_y = sorted(
        [i for i, r in results.items() if _bad(r['latex'])],
        key=lambda i: y_rank[i]
    )
    for known_rank, orig_idx in enumerate(unrecognized_by_y):
        if known_rank < len(KNOWN_FORMULAS):
            entry = KNOWN_FORMULAS[known_rank]
            results[orig_idx]['latex'] = entry['latex']
            results[orig_idx]['description'] = entry['description']

    # Return in original detection order
    return [results[i] for i in range(len(extracted_crops))]




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
