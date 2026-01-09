"""
Google Gemini API Integration pentru interpretarea automată a graficelor FFT
Folosește noul pachet google.genai (ianuarie 2026)
"""

from google import genai
from google.genai import types
import base64
import os
import json
import matplotlib.pyplot as plt
import io
import numpy as np

# Configurare API key (setează-l în variabila de mediu sau direct aici)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAEpZBCLxVCgtyhPx5vNfEaeX55dzdw354")  # Adaugă cheia ta aici sau în .env

def configure_gemini(api_key=None):
    """Configurează Gemini API cu noul SDK"""
    key_to_use = api_key or GEMINI_API_KEY
    if not key_to_use:
        raise ValueError("❌ API Key lipsește! Setează GEMINI_API_KEY în environment sau ca parametru.")
    
    # Configurare client cu noul SDK
    client = genai.Client(api_key=key_to_use)
    return client

def create_analysis_package(psd1D, fft_patterns, features_dict, magnitude_2d=None):
    """
    Creează un pachet de date pentru interpretare Gemini
    Nu trimite imaginea originală, doar graficele și metricele numerice
    
    Parameters:
    -----------
    psd1D : array
        Profilul radial PSD 1D
    fft_patterns : dict
        Dictionary cu pattern-uri detectate
    features_dict : dict
        Features extrase din FFT
    magnitude_2d : array, optional
        Spectrul 2D pentru vizualizare
        
    Returns:
    --------
    dict : Pachet cu datele pentru analiză
    """
    package = {
        # 1. Datele numerice brute
        "psd_1d": psd1D.tolist() if isinstance(psd1D, np.ndarray) else psd1D,
        
        # 2. Statistici calculate
        "statistics": {
            "psd_mean": float(np.mean(psd1D)),
            "psd_std": float(np.std(psd1D)),
            "psd_max": float(np.max(psd1D)),
            "psd_min": float(np.min(psd1D)),
            "decay_rate": float((psd1D[len(psd1D)//2] - psd1D[-1]) / (len(psd1D)//2)) if len(psd1D) > 10 else 0,
        },
        
        # 3. Pattern-uri detectate
        "patterns": {
            "star_pattern": bool(fft_patterns.get('star_pattern', False)),
            "periodic_spikes": bool(fft_patterns.get('periodic_spikes', False)),
            "unnatural_decay": bool(fft_patterns.get('unnatural_decay', False)),
            "suspicion_score": float(fft_patterns.get('suspicion_score', 0)),
            "symmetry_ratio": float(fft_patterns.get('symmetry_ratio', 0)),
        },
        
        # 4. Features relevante
        "features": {
            "tail_90": float(features_dict.get('tail_90', 0)),
            "tail_80": float(features_dict.get('tail_80', 0)),
            "hf_lf_ratio": float(features_dict.get('hf_lf_ratio', 0)),
            "ela_std": float(features_dict.get('ela_std', 0)),
        }
    }
    
    return package

def generate_psd_plot_base64(psd1D):
    """
    Generează graficul PSD ca imagine base64 pentru Gemini Vision
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    radial_freqs = np.arange(len(psd1D))
    
    ax.plot(radial_freqs, psd1D, linewidth=2, color='#2E86AB', label='Radial PSD')
    
    # Zone markers
    n = len(psd1D)
    for pct, color, label in [(0.25, 'green', '25%'), (0.50, 'blue', '50%'), 
                               (0.75, 'orange', '75%'), (0.90, 'red', '90%')]:
        idx = int(pct * n)
        ax.axvline(x=idx, color=color, linestyle=':', alpha=0.5, label=label)
    
    # Trend line
    mid_idx = n // 2
    x_ref = radial_freqs[mid_idx:]
    y_ref = psd1D[mid_idx:]
    coeffs = np.polyfit(x_ref, y_ref, 1)
    ax.plot(x_ref, np.polyval(coeffs, x_ref), 'r--', linewidth=2, alpha=0.7,
            label=f'Decay: {coeffs[0]:.2f} dB/px')
    
    ax.set_xlabel('Radial Frequency (pixels)', fontweight='bold')
    ax.set_ylabel('Power (dB)', fontweight='bold')
    ax.set_title('Radial PSD Analysis', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left')
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64

def interpret_with_gemini(psd1D, fft_patterns, features_dict, magnitude_2d=None, 
                         api_key=None, use_vision=True):
    """
    Interpretează analiza FFT folosind Google Gemini API
    
    Parameters:
    -----------
    psd1D : array
        Profilul radial PSD
    fft_patterns : dict
        Pattern-uri detectate
    features_dict : dict
        Features FFT
    magnitude_2d : array, optional
        Spectrul 2D
    api_key : str, optional
        Gemini API key
    use_vision : bool
        Dacă True, trimite și graficul ca imagine (Gemini Vision)
        
    Returns:
    --------
    dict : Interpretarea AI cu confidence și explicații
    """
    try:
        # Configurare
        configure_gemini(api_key)
        
        # Creează pachetul de analiză
        analysis_package = create_analysis_package(psd1D, fft_patterns, features_dict, magnitude_2d)
        
        # Prompt pentru Gemini
        prompt = f"""
Ești un expert în detectarea deepfake-urilor folosind analiza spectrală FFT (Fast Fourier Transform).

**Datele tale de analizat:**

1. **PSD Radial 1D** (profil radial obținut prin media azimutală):
   - {len(psd1D)} puncte de frecvență
   - Mean: {analysis_package['statistics']['psd_mean']:.2f} dB
   - Std: {analysis_package['statistics']['psd_std']:.2f} dB
   - Decay rate: {analysis_package['statistics']['decay_rate']:.4f} dB/pixel

2. **Pattern-uri detectate:**
   - Star Pattern: {'DA ⭐' if analysis_package['patterns']['star_pattern'] else 'NU'}
   - Periodic Spikes: {'DA 📊' if analysis_package['patterns']['periodic_spikes'] else 'NU'}
   - Unnatural Decay: {'DA 📉' if analysis_package['patterns']['unnatural_decay'] else 'NU'}
   - Suspicion Score: {analysis_package['patterns']['suspicion_score']:.1f}/100
   - Symmetry Ratio: {analysis_package['patterns']['symmetry_ratio']:.2f}

3. **Features numerice:**
   - Tail gradient (90%): {analysis_package['features']['tail_90']:.2f} dB/dec
   - Tail gradient (80%): {analysis_package['features']['tail_80']:.2f} dB/dec
   - HF/LF Ratio: {analysis_package['features']['hf_lf_ratio']:.4f}
   - ELA Std: {analysis_package['features']['ela_std']:.2f}

**Sarcina ta:**

Analizează aceste date și oferă o interpretare EXPERTĂ:

1. **Verdict**: Este această imagine REAL sau AI-GENERATED (GAN/Diffusion)?
2. **Confidence**: 0-100% (cât de sigur ești)
3. **Reasoning**: Explică CE pattern-uri specific indică AI generation
4. **Key Indicators**: Lista cu top 3 indicatori decisivi
5. **Natural Signals**: Ce semne arată că ar putea fi REAL (dacă există)

**Context tehnic:**
- Imaginile REALE au decay smooth ~1/f^α (alpha ≈ 2)
- AI (GAN/Diffusion) lasă "amprente" de up-sampling: vârfuri la frecvențe medii/înalte, drop abrupt >90%
- Star pattern = semn de resampling (rotații frecvente în antrenare)
- Periodic spikes = artefacte de grila (transpose convolutions)

**Format răspuns (JSON):**
{{
  "verdict": "REAL" sau "AI-GENERATED",
  "confidence": 85,
  "reasoning": "Explicație detaliată...",
  "key_indicators": ["Indicator 1", "Indicator 2", "Indicator 3"],
  "natural_signals": ["Semn natural 1", ...] sau [],
  "recommendation": "Sugestie pentru utilizator"
}}

Răspunde DOAR cu JSON valid, fără text suplimentar.
"""
        
        # Configurare client
        client = configure_gemini(api_key)
        
        # Folosește modelul gemini-2.5-flash (cel mai nou, rapid și gratuit)
        model_name = "gemini-2.5-flash"
        
        if use_vision:
            # Generează graficul ca imagine
            plot_base64 = generate_psd_plot_base64(psd1D)
            
            # Creează conținut cu imagine
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_text(prompt),
                    types.Part.from_bytes(
                        data=base64.b64decode(plot_base64),
                        mime_type="image/png"
                    )
                ]
            )
        else:
            # Text-only mode
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
        
        # Parse răspuns
        result_text = response.text.strip()
        
        # Încearcă să extragă JSON
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        
        result = json.loads(result_text.strip())
        
        return {
            "success": True,
            "interpretation": result,
            "raw_response": response.text
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "interpretation": None
        }

# Test function
if __name__ == "__main__":
    # Test cu date dummy
    psd_test = np.linspace(50, 10, 256) + np.random.normal(0, 2, 256)
    patterns_test = {
        'star_pattern': True,
        'periodic_spikes': False,
        'unnatural_decay': True,
        'suspicion_score': 65,
        'symmetry_ratio': 0.78
    }
    features_test = {
        'tail_90': -35.5,
        'tail_80': -28.3,
        'hf_lf_ratio': 0.0012,
        'ela_std': 8.5
    }
    
    # Notă: Trebuie să ai GEMINI_API_KEY setat
    # result = interpret_with_gemini(psd_test, patterns_test, features_test)
    # print(json.dumps(result, indent=2))
    print("✅ Module Gemini Interpreter ready. Set GEMINI_API_KEY to use.")
