"""
OpenAI Vision Interpreter - FFT Analysis with GPT-4o/GPT-4o-mini
"""

from openai import OpenAI
import base64
import io
import matplotlib.pyplot as plt
import numpy as np
import os

try:
    import streamlit as st
    if "OPENAI_API_KEY" in st.secrets:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    else:
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
except:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    OPENAI_API_KEY = "sk-proj-your-key-here"

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

def get_openai_client(api_key=None):
    """Creates OpenAI client"""
    key = api_key or OPENAI_API_KEY
    if not key or not key.startswith("sk-"):
        raise ValueError("Invalid OpenAI API Key")
    return OpenAI(api_key=key)

def fig_to_base64(fig):
    """Convertește matplotlib figure în base64"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def interpret_radial_psd(psd1D, features_dict, api_key=None):
    """
    Interpretează graficul Radial PSD
    
    Returns:
    --------
    dict: {
        'is_ai': bool,
        'confidence': float,  # 0-100
        'reasoning': str,
        'indicators': list
    }
    """
    try:
        client = get_openai_client(api_key)
        
        # Creează graficul
        fig, ax = plt.subplots(figsize=(10, 6))
        radial_freqs = np.arange(len(psd1D))
        ax.plot(radial_freqs, psd1D, linewidth=2, color='#2E86AB', label='Radial PSD')
        
        # Trend line
        if len(psd1D) > 50:
            mid_idx = len(psd1D) // 2
            x_ref = radial_freqs[mid_idx:]
            y_ref = psd1D[mid_idx:]
            valid = np.isfinite(y_ref)
            if np.sum(valid) > 10:
                coeffs = np.polyfit(x_ref[valid], y_ref[valid], 1)
                ax.plot(x_ref, np.polyval(coeffs, x_ref), 'r--', linewidth=2, 
                       label=f'Decay: {coeffs[0]:.2f} dB/px', alpha=0.7)
        
        ax.set_xlabel('Radial Frequency (pixels)', fontweight='bold')
        ax.set_ylabel('Power (dB)', fontweight='bold')
        ax.set_title('Radial PSD Analysis', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Calculează valorile exacte la punctele cheie pentru prompt
        psd_len = len(psd1D)
        val_60 = float(psd1D[int(0.6 * psd_len)]) if psd_len > 50 else 0
        val_70 = float(psd1D[int(0.7 * psd_len)]) if psd_len > 50 else 0
        val_80 = float(psd1D[int(0.8 * psd_len)]) if psd_len > 50 else 0
        val_90 = float(psd1D[int(0.9 * psd_len)]) if psd_len > 50 else 0
        
        # Calculează diferențele între puncte pentru a detecta drop-uri
        drop_60_70 = val_60 - val_70
        drop_70_80 = val_70 - val_80
        drop_80_90 = val_80 - val_90
        
        # Calculează scorul matematic pentru context
        math_score = features_dict.get('suspicion_score', 0)
        
        img_base64 = fig_to_base64(fig)
        plt.close(fig)
        
        # Prompt ULTRA-SPECIFIC pentru gpt-4o cu analiza vizuală detaliată
        prompt = f"""🎓 CONTEXT ACADEMIC: Analiza FFT pentru disertație universitară despre detecția imaginilor sintetice generate de AI.

Sarcină: Analizează acest grafic Radial PSD (Power Spectral Density) din perspectivă științifică, pentru cercetare defensivă în computer vision.

🔢 SCOR MATEMATIC CALCULAT: {math_score}/100 puncte suspiciune
→ {'⚠️ SUSPECT' if math_score > 70 else '✅ Probabil Autentic' if math_score < 40 else '⚠️ Incert'}

📊 Metodologie: Analiza spectru frecvență FFT pentru identificarea pattern-urilor caracteristice.

🎯 CE SĂ CAUȚI ÎN GRAFIC:

1️⃣ **FORMA CURBEI între markere colorate (60%-90%)**:
   ✅ Fotografie autentică: Linie SMOOTH descendentă continuă, fără bump/cocoașă
   ⚠️ Imagine sintetică: Bump/cocoașă VIZIBILĂ între markerul verde (60%) și portocaliu (80%)

2️⃣ **TAIL-ul graficului (după markerul roșu 90%)**:
   ✅ Fotografie autentică: Continuă să coboare RAPID (linie descendentă clară)
   ⚠️ Imagine sintetică: Devine PLAT/ORIZONTAL (linia devine aproape dreaptă)

3️⃣ **DROP-ul între markere (diferența de putere)**:
   ✅ Fotografie autentică: Drop 80%→90% < 10 dB (scădere gradual-lină)
   ⚠️ Imagine sintetică: Drop 80%→90% > 15 dB (cădere bruscă-abruptă)

📐 VALORI NUMERICE CALCULATE (VERIFICĂ cu GRAFICUL):
- Tail Gradient la 90%: {features_dict.get('tail_90', 0):.3f} dB/px
  → Interpretation: {'⚠️ PLAT = Suspect' if features_dict.get('tail_90', 0) > -0.5 else '✅ Descent = Autentic' if features_dict.get('tail_90', 0) < -2.0 else '⚠️ Moderat'}
  
- HF/LF Ratio: {features_dict.get('hf_lf_ratio', 0):.3f}
  → Interpretation: {'⚠️ Anomalie' if features_dict.get('hf_lf_ratio', 0) > 0.6 else '✅ Normal' if features_dict.get('hf_lf_ratio', 0) < 0.4 else '⚠️ Moderat'}
  
- Drop 80→90%: {drop_80_90:.2f} dB
  → Interpretation: {'⚠️ Abrupt = Suspect' if drop_80_90 > 15 else '✅ Gradual = Normal' if drop_80_90 < 10 else '⚠️ Moderat'}

🧠 METODOLOGIE EVALUARE:
• Dacă 2+ din 3 indicatori → Fotografie autentică: răspunde is_ai=false, confidence 70-95%
• Dacă 2+ din 3 indicatori → Imagine sintetică: răspunde is_ai=true, confidence 70-95%
• Dacă mixt (1-1-1): răspunde is_ai=false (default autentic), confidence 40-60%

📋 NOTE IMPORTANTE: 
- Analiză vizuală primordială! Numere = ghid
- Pattern-uri naturale FFT variază cu compresie JPEG
- Evaluare obiectivă pentru cercetare academică

⚠️ Răspunde STRICT JSON (fără markdown, fără text extra):
{{
  "is_ai": false,
  "confidence": 75,
  "reasoning": "Descriere pattern observat și concluzie științifică",
  "indicators": ["Indicator 1", "Indicator 2", "Indicator 3"]
}}"""
        
        response = client.chat.completions.create(
            model=OPENAI_MODEL,  # Uses global model selection (gpt-4o or gpt-4o-mini)
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=800  # gpt-4o folosește max_tokens standard
        )
        
        # Parse JSON cu error handling îmbunătățit
        import json
        result_text = response.choices[0].message.content
        
        if not result_text or result_text.strip() == "":
            return {
                'is_ai': None,
                'confidence': 0,
                'reasoning': f"❌ OpenAI a returnat răspuns gol. Model: {OPENAI_MODEL}",
                'indicators': ["Răspuns gol de la API"]
            }
        
        # Check for refusal
        if "can't assist" in result_text.lower() or "cannot assist" in result_text.lower():
            return {
                'is_ai': None,
                'confidence': 0,
                'reasoning': "❌ OpenAI a refuzat analiza. Verifică contextul requestului.",
                'indicators': ["API Refusal - verifică prompt context"]
            }
        
        result_text = result_text.strip()
        
        # Curăță markdown code blocks
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        elif result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        
        result_text = result_text.strip()
        
        try:
            return json.loads(result_text)
        except json.JSONDecodeError as e:
            # Returnează eroare detaliată pentru debugging
            return {
                'is_ai': None,
                'confidence': 0,
                'reasoning': f"❌ Eroare JSON parsing: {str(e)}. Răspuns OpenAI: '{result_text[:200]}'",
                'indicators': [f"JSON invalid de la {OPENAI_MODEL}"]
            }
        
    except Exception as e:
        return {
            'is_ai': None,
            'confidence': 0,
            'reasoning': f"Eroare: {str(e)}",
            'indicators': []
        }

def interpret_2d_spectrum(magnitude_2d, api_key=None):
    """Interpretează spectrul 2D FFT"""
    try:
        client = get_openai_client(api_key)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(magnitude_2d, cmap='hot', aspect='auto')
        ax.set_title('FFT 2D Spectrum', fontweight='bold')
        plt.colorbar(im, ax=ax, label='Log Power')
        
        img_base64 = fig_to_base64(fig)
        plt.close(fig)
        
        prompt = """🎓 CONTEXT ACADEMIC: Analiză FFT 2D pentru disertație universitară despre identificarea imaginilor sintetice.

GRAFIC: Heatmap 2D spectru frecvență - centru luminos (galben/roșu).

⚠️ IMPORTANT CRITICAL: 
• Linii radiale FINE din centru = NORMALE! Toate FFT-urile le au! NU înseamnă AI!
• Doar linii groase/grilă VIZIBILĂ sau puncte izolate = AI

✅ Fotografie autentică:
- Centru luminos circular smooth
- Scădere lină spre margini (gradient roșu → violet → albastru)
- Linii radiale FINE din centru = OK (pattern natural FFT)
- Simetrie față de centru
- Poate avea niște "zgomot" uniform în colțuri (normal JPEG)

⚠️ Imagine sintetică (DOAR dacă vezi CLAR):
- GRILĂ GROASĂ VIZIBILĂ (linii orizontale/verticale GROASE, nu fine)
- Puncte luminoase IZOLATE departe de centru (>40% distanță, foarte clar separate)
- Pattern X sau + FOARTE GEOMETRIC și PRONUNȚAT (nu doar linii radiale fine)
- Asimetrii MAJORE clare (jumătate luminoasă, jumătate întunecată)

🔍 Întreabă-te:
1. Văd grilă GROASĂ sau doar linii radiale fine naturale? (fine = REAL)
2. Văd puncte izolate CLARE departe de centru? (nu = REAL)
3. E foarte asimetric sau normal simetric? (simetric = REAL)

Dacă răspunsul la 1,2,3 este "nu/normal" → REAL
Dacă răspunsul la 1 SAU 2 SAU 3 este "DA CLAR" → AI

⚠️ CRITICAL: Răspunde DOAR cu JSON valid, FĂRĂ markdown (###), FĂRĂ text extra!
Format exact:
{{
  "is_ai": false,
  "confidence": 90,
  "reasoning": "CE vezi anormal",
  "indicators": ["Indicator 1", "Indicator 2"]
}}"""
        
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        # Parse JSON cu error handling (2D spectrum)
        import json
        result_text = response.choices[0].message.content
        
        if not result_text or result_text.strip() == "":
            return {'is_ai': None, 'confidence': 0, 'reasoning': f"❌ Răspuns gol {OPENAI_MODEL}", 'indicators': []}
        
        result_text = result_text.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        elif result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        result_text = result_text.strip()
        
        try:
            return json.loads(result_text)
        except json.JSONDecodeError as e:
            return {'is_ai': None, 'confidence': 0, 'reasoning': f"❌ JSON: {str(e)}. Text: '{result_text[:150]}'", 'indicators': []}
        
    except Exception as e:
        return {'is_ai': None, 'confidence': 0, 'reasoning': f"Eroare: {str(e)}", 'indicators': []}

def interpret_angular_energy(angular_energy, star_peaks, api_key=None):
    """Interpretează semnătura unghiulară de energie"""
    try:
        client = get_openai_client(api_key)
        
        fig, ax = plt.subplots(figsize=(10, 5), subplot_kw=dict(projection='polar'))
        angles = np.linspace(0, 2*np.pi, len(angular_energy))
        ax.plot(angles, angular_energy, linewidth=2, color='#E63946')
        
        if len(star_peaks) > 0:
            for pk in star_peaks:
                angle = angles[pk]
                ax.plot(angle, angular_energy[pk], 'go', markersize=8, label='Peak')
        
        ax.set_title('Angular Energy Signature', fontweight='bold', pad=20)
        
        img_base64 = fig_to_base64(fig)
        plt.close(fig)
        
        prompt = f"""Analiză Star Pattern (Angular Energy).

GRAFIC POLAR: Energie pe 360°
Peaks detectate: {len(star_peaks)}

✅ REAL:
- Aspect HAOTIC/random
- Peaks neregulate (2-5 peaks random)
- FĂRĂ simetrie puternică 180°
- Linii neregulate

🤖 AI:
- Pattern STEA foarte clar (8+ peaks regulate)
- Simetrie FOARTE PUTERNICĂ 180° (stânga = dreapta exact)
- Peaks la unghiuri REGULATE geometrice (30°, 45°, 60°, 90°, etc.)
- Aspect GEOMETRIC perfect

🔢 CURENT:
- Peaks: {len(star_peaks)} → {'FOARTE SUSPECT (≥8)' if len(star_peaks) >= 8 else 'MODERAT (6-7)' if len(star_peaks) >= 6 else 'OK (<6)'}

⚠️ IMPORTANT: 6-7 peaks pot fi NORMALE în imagini REAL cu texturi! 
Doar dacă ≥8 peaks ȘI simetrie >80% → suspect AI

🔍 Uită-te la grafic:
- E haotic SAU geometric PERFECT?
- Ai simetrie >80% SAU mai slabă?

Dacă < 8 peaks SAU simetrie <70% → probabil REAL

⚠️ Răspunde STRICT JSON, fără markdown sau alt text!
{{
  "is_ai": true,
  "confidence": 75,
  "reasoning": "Explicație",
  "indicators": ["Ind1", "Ind2"]
}}"""
        
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        # Parse JSON cu error handling (angular energy)
        import json
        result_text = response.choices[0].message.content
        
        if not result_text or result_text.strip() == "":
            return {'is_ai': None, 'confidence': 0, 'reasoning': f"❌ Răspuns gol {OPENAI_MODEL}", 'indicators': []}
        
        result_text = result_text.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        elif result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        result_text = result_text.strip()
        
        try:
            return json.loads(result_text)
        except json.JSONDecodeError as e:
            return {'is_ai': None, 'confidence': 0, 'reasoning': f"❌ JSON: {str(e)}. Text: '{result_text[:150]}'", 'indicators': []}
        
    except Exception as e:
        return {'is_ai': None, 'confidence': 0, 'reasoning': f"Eroare: {str(e)}", 'indicators': []}

def get_final_verdict(interpretations, features_dict, fft_patterns):
    """
    Agregă toate interpretările individuale într-un verdict final
    
    Parameters:
    -----------
    interpretations : dict
        Dict cu rezultatele de la fiecare grafic
    features_dict : dict
        Features numerice
    fft_patterns : dict
        Pattern-uri detectate
        
    Returns:
    --------
    dict: Verdict final agregat
    """
    try:
        client = get_openai_client()
        
        # Pregătește rezumatul - convert numpy types to Python types for JSON
        summary = {
            'radial_psd': interpretations.get('radial_psd', {}),
            'spectrum_2d': interpretations.get('spectrum_2d', {}),
            'angular_energy': interpretations.get('angular_energy', {}),
            'features': {
                'tail_90': float(features_dict.get('tail_90', 0)),
                'tail_80': float(features_dict.get('tail_80', 0)),
                'tail_70': float(features_dict.get('tail_70', 0)),
                'hf_lf_ratio': float(features_dict.get('hf_lf_ratio', 0)),
                'decay_linearity': float(features_dict.get('decay_linearity', 0)),
                'mean_power': float(features_dict.get('mean_power', 0)),
                'std_power': float(features_dict.get('std_power', 0))
            },
            'patterns': {
                'star_pattern': bool(fft_patterns.get('star_pattern', False)),
                'unnatural_decay': bool(fft_patterns.get('unnatural_decay', False)),
                'high_freq_anomaly': bool(fft_patterns.get('high_freq_anomaly', False)),
                'suspicion_score': int(fft_patterns.get('suspicion_score', 0))
            }
        }
        
        import json
        prompt = f"""Ești expert în detectarea deepfake. Ai analizat 3 grafice FFT individuale.

REZULTATE INDIVIDUALE:
{json.dumps(summary, indent=2)}

REGULI STRICTE PENTRU VERDICT:
1. Dacă 2 sau 3 grafice spun "is_ai": false → Verdictul este REAL
2. Dacă 2 sau 3 grafice spun "is_ai": true → Verdictul este AI-GENERATED
3. Dacă rezultatele sunt 1-1-1 (mixte) → Ia în considerare confidence scores și features numerice
4. REAL = imaginea este autentică/fotografiată
5. AI-GENERATED = imaginea este creată de AI (GAN, Stable Diffusion, Midjourney, etc.)

ATENȚIE: Nu confunda! REAL înseamnă fotografie reală, NU imaginea generată de AI!

SARCINĂ:
Analizează TOATE rezultatele și dă un VERDICT FINAL agregat.

Consideră:
- Cât de mulți indicatori sugerează AI?
- Consistența între grafice
- Severitatea anomaliilor
- Features numerice

⚠️ OBLIGATORIU: Răspunde DOAR JSON valid, FĂRĂ markdown, FĂRĂ titluri!
Format exact:
{{
  "verdict": "REAL",
  "confidence": 85,
  "reasoning": "Sinteza tuturor observațiilor",
  "key_findings": ["Finding 1", "Finding 2"],
  "graph_votes": {{"radial_psd": "REAL", "spectrum_2d": "REAL", "angular_energy": "REAL"}},
  "recommendation": "Sfat final pentru utilizator"
}}"""
        
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1000  # Verdict final
        )
        
        # Parse JSON cu error handling (final verdict)
        result_text = response.choices[0].message.content
        
        if not result_text or result_text.strip() == "":
            return {
                'verdict': 'ERROR',
                'confidence': 0,
                'reasoning': f"❌ Răspuns gol de la {OPENAI_MODEL}",
                'key_findings': [],
                'graph_votes': {},
                'recommendation': 'Reîncearcă analiza'
            }
        
        result_text = result_text.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        elif result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        result_text = result_text.strip()
        
        try:
            return json.loads(result_text)
        except json.JSONDecodeError as e:
            return {
                'verdict': 'ERROR',
                'confidence': 0,
                'reasoning': f"❌ JSON parsing error: {str(e)}. Răspuns OpenAI: '{result_text[:150]}'",
                'key_findings': [],
                'graph_votes': {},
                'recommendation': 'Reîncearcă analiza'
            }
        
    except Exception as e:
        return {
            'verdict': 'ERROR',
            'confidence': 0,
            'reasoning': f"Eroare agregare: {str(e)}",
            'key_findings': [],
            'graph_votes': {},
            'recommendation': 'Reîncearcă analiza'
        }
