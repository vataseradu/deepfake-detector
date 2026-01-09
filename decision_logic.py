"""
Decision Logic Module - Deepfake Detection Verdict System
CALIBRATED VERSION based on CIFAKE Dataset Stats (January 2026)

Research Findings:
- REAL Mean HF Ratio: -1.80 (Natural falloff)
- FAKE Mean HF Ratio: -1.55 (High-frequency artifacts detected)
- Threshold: -1.67 (Midpoint)
"""

from typing import Dict, Tuple
import numpy as np
from sklearn.linear_model import LinearRegression

def analyze_fft_score(psd1D: np.ndarray) -> Tuple[float, Dict]:
    debug_metrics = {
        "log_hf_ratio": 0.0,
        "slope": 0.0
    }

    try:
        if psd1D is None or len(psd1D) < 20:
            return 50.0, debug_metrics
        
        psd1D = psd1D[5:] 
        valid_indices = psd1D > 0
        if np.sum(valid_indices) < 15:
            return 50.0, debug_metrics
        psd1D = psd1D[valid_indices]
        n_freq = len(psd1D)
        
        # ==== METRIC 1: HF Ratio ====
        cutoff = int(0.7 * n_freq)
        low_p = np.mean(psd1D[:cutoff])
        high_p = np.mean(psd1D[cutoff:])
        
        if low_p > 0 and high_p > 0:
            ratio = high_p / low_p
            log_ratio = np.log10(ratio)
        else:
            log_ratio = -6.0
            
        debug_metrics["log_hf_ratio"] = round(log_ratio, 4)

        # === CALIBRARE PE DATELE TALE ===
        # Real: -1.80 | Fake: -1.55
        # Midpoint: -1.67
        # Logică inversată față de High-Res: Aici FAKE-ul are valoare MAI MARE.
        
        # Dacă e mai mic de -1.8 (Real Mean), scorul tinde spre 0.
        # Dacă e mai mare de -1.55 (Fake Mean), scorul tinde spre 100.
        
        # Mapping: 
        # -1.90 -> 0% AI
        # -1.50 -> 100% AI
        
        ratio_score = np.clip(((log_ratio - (-1.90)) / 0.40) * 100, 0, 100)
        
        # ==== METRIC 2: Slope ====
        x = np.log10(np.arange(1, len(psd1D) + 1)).reshape(-1, 1)
        y = np.log10(psd1D)
        model = LinearRegression().fit(x, y)
        slope = model.coef_[0]
        debug_metrics["slope"] = round(slope, 4)
        
        # Slope-ul nu a arătat diferențe majore în datele tale, îl ignorăm parțial
        slope_score = 0 

        final_score = ratio_score # Ne bazăm 100% pe Ratio calibrat
        
        return float(np.clip(final_score, 0, 100)), debug_metrics
    
    except Exception as e:
        print(f"Error: {e}")
        return 50.0, debug_metrics

def get_final_verdict(fft_score: float, ela_stats: Dict[str, float]) -> Dict[str, str]:
    """
    Decizie calibrată pe mediile: Real ELA=1.02, Fake ELA=1.31
    """
    ela_std = ela_stats.get('std_intensity', 0.0)
    
    # === PRAGURI CALIBRATE ===
    # FFT Threshold la mijlocul intervalului statistic
    FFT_THRESHOLD = 50.0 
    
    # ELA Threshold: Real=1.0, Fake=1.3 -> Midpoint = 1.15
    # Punem 1.2 pentru siguranță
    ELA_THRESHOLD = 1.2
    
    # Logică Simplificată pentru CIFAKE
    is_ai_fft = fft_score > FFT_THRESHOLD
    is_ai_ela = ela_std > ELA_THRESHOLD
    
    if is_ai_fft and is_ai_ela:
        return {
            'label': '🤖 AI GENERATED',
            'color': '#FF4B4B',
            'confidence': f'{fft_score:.1f}%',
            'explanation': 'High-frequency artifacts & ELA noise detected.',
            'technical_details': f'Ratio: High (>{-1.67}) | ELA: High (>{ELA_THRESHOLD})'
        }
    elif is_ai_fft:
        return {
            'label': '⚠️ SUSPICIOUS (FFT)',
            'color': '#FFD700',
            'confidence': f'{fft_score:.1f}%',
            'explanation': 'Spectral anomalies detected, but compression is uniform.',
            'technical_details': f'Ratio: High (>{-1.67}) | ELA: Low'
        }
    elif is_ai_ela:
        return {
            'label': '⚠️ SUSPICIOUS (ELA)',
            'color': '#FFA500',
            'confidence': 'Medium',
            'explanation': 'Spectrum looks natural, but compression noise is high.',
            'technical_details': f'ELA: {ela_std:.2f} (High)'
        }
    else:
        return {
            'label': '✅ AUTHENTIC',
            'color': '#28A745',
            'confidence': f'{100-fft_score:.1f}%',
            'explanation': 'Natural frequency profile and uniform compression.',
            'technical_details': f'Ratio: Low (<{-1.67}) | ELA: Low'
        }

def get_risk_assessment(verdict):
    if 'AI' in verdict['label']: return "HIGH RISK: Synthetic artifacts detected."
    if 'SUSPICIOUS' in verdict['label']: return "MODERATE RISK: Inconclusive result."
    return "LOW RISK: Consistent with real image stats."