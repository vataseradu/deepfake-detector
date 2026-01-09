"""
Deepfake Detector - Interfață Simplificată cu Gemini AI
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import cv2

from frequency import azimuthalAverage
from scipy.fft import fft2, fftshift
from scipy import ndimage

try:
    from gemini_graph_interpreter import (
        interpret_radial_psd, 
        interpret_2d_spectrum, 
        interpret_angular_energy,
        get_final_verdict,
        OPENAI_API_KEY,
        OPENAI_MODEL
    )
    OPENAI_AVAILABLE = True
    api_key_loaded = bool(OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-"))
except ImportError:
    OPENAI_AVAILABLE = False
    OPENAI_API_KEY = None
    OPENAI_MODEL = "N/A"
    api_key_loaded = False

st.set_page_config(
    page_title="Deepfake Detector - AI Analysis",
    layout="wide"
)

st.title("Deepfake Detector")
st.markdown("**Tema de cercetare - VATASE Radu-Petrut**")

uploaded_file = st.file_uploader("Upload imagine", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(img)
    
    with st.sidebar:
        st.image(img, caption="Imagine Uploaded", width='stretch')
        st.markdown(f"**Dimensiune:** {img.size[0]} x {img.size[1]}")
        
        if OPENAI_AVAILABLE and api_key_loaded:
            model_display = OPENAI_MODEL if OPENAI_MODEL else "gpt-4o-mini"
            st.success(f"OpenAI {model_display} Activ")
            st.caption(f"API Key: {OPENAI_API_KEY[:15]}...")
            
            if model_display == "gpt-4o":
                st.info("**Model Premium**: Precizie maximă, ~$0.025/imagine")
            else:
                st.info("**Model Economic**: ~$0.003/imagine")
        elif OPENAI_AVAILABLE:
            st.warning("API Key OpenAI lipsește!")
            st.info("Pune-l în gemini_graph_interpreter.py linia 13")
    
    if st.button("Analizează", type="primary"):
        with st.spinner("Analizez imaginea..."):
            gray = np.mean(img_array, axis=2).astype(np.float64)
            
            
            window = np.hanning(gray.shape[0])[:, None] * np.hanning(gray.shape[1])[None, :]
            gray_windowed = gray * window
            fft_result = fft2(gray_windowed)
            fft_shifted = fftshift(fft_result)
            magnitude_2d = np.log10(np.abs(fft_shifted) + 1)
            
            # Compute radial PSD using azimuthal average
            psd_2d = np.abs(fft_shifted) ** 2
            radial_profile = azimuthalAverage(psd_2d)
            psd1D = 10 * np.log10(radial_profile + 1e-10)
            
            # Comprehensive features dict (from ML training)
            psd_len = len(psd1D)
            
            # Calculate drops between key points
            val_60 = psd1D[int(0.6 * psd_len)] if psd_len > 50 else 0
            val_70 = psd1D[int(0.7 * psd_len)] if psd_len > 50 else 0
            val_80 = psd1D[int(0.8 * psd_len)] if psd_len > 50 else 0
            val_90 = psd1D[int(0.9 * psd_len)] if psd_len > 50 else 0
            
            drop_60_80 = val_60 - val_80  # Total drop in middle range
            drop_80_90 = val_80 - val_90  # Critical tail drop
            
            features_dict = {
                # Tail gradients (key indicators)
                'tail_90': np.gradient(psd1D)[int(0.9*psd_len):].mean() if psd_len > 50 else 0,
                'tail_80': np.gradient(psd1D)[int(0.8*psd_len):].mean() if psd_len > 50 else 0,
                'tail_70': np.gradient(psd1D)[int(0.7*psd_len):].mean() if psd_len > 50 else 0,
                
                # Power distribution
                'mean_power': np.mean(psd1D),
                'std_power': np.std(psd1D),
                'power_range': np.max(psd1D) - np.min(psd1D),
                
                # High/Low frequency ratio (adjusted calculation)
                'hf_lf_ratio': (np.mean(psd1D[int(0.7*psd_len):]) / 
                               (np.mean(psd1D[:int(0.4*psd_len)]) + 1e-10)) if psd_len > 50 else 0,
                
                # Decay smoothness
                'decay_linearity': np.corrcoef(np.arange(psd_len//2, psd_len), 
                                               psd1D[psd_len//2:])[0,1] if psd_len > 50 else 0,
                
                # Drops between points
                'drop_60_80': drop_60_80,
                'drop_80_90': drop_80_90
            }
            
            # 🎯 MATHEMATICAL SCORING (fără API) - REVISED pentru mai puține false positives
            math_score_ai = 0  # Start neutral
            
            # Rule 1: Drop Analysis (mai important decât gradient absolut)
            # Imagini REALE: drop consistent 8-15 dB per 20%
            # Imagini AI: drop >20 dB (abrupt) sau <5 dB (prea flat)
            if drop_80_90 < 3:  # Tail FOARTE flat = suspect
                math_score_ai += 35
            elif drop_80_90 < 6:  # Tail destul de flat
                math_score_ai += 15
            elif drop_80_90 > 18:  # Drop prea abrupt
                math_score_ai += 20
            elif 8 <= drop_80_90 <= 15:  # Drop natural normal
                math_score_ai -= 25  # BONUS pentru REAL
            
            # Rule 2: HF/LF Ratio (mai permisiv)
            # JPEG compression poate crește HF, deci threshold mai mare
            hf_lf = features_dict['hf_lf_ratio']
            if hf_lf > 0.8:  # FOARTE mare = AI clar
                math_score_ai += 25
            elif hf_lf > 0.6:  # Mare dar poate fi JPEG
                math_score_ai += 10
            elif hf_lf < 0.3:  # Curat = REAL
                math_score_ai -= 20
            
            # Rule 3: Decay Linearity (important pentru consistență)
            linearity = abs(features_dict['decay_linearity'])
            if linearity < 0.5:  # Foarte non-linear = AI
                math_score_ai += 20
            elif linearity > 0.85:  # Foarte linear = natural REAL
                math_score_ai -= 15
            
            # Rule 4: Std Power (variație)
            if features_dict['std_power'] > 25:  # Variație FOARTE mare
                math_score_ai += 10
            elif features_dict['std_power'] < 10:  # Variație mică = consistent
                math_score_ai -= 10
            
            # Rule 5: Drop consistency (60-80 vs 80-90)
            # Dacă ambele drop-uri sunt consistente = REAL
            if 8 <= drop_60_80 <= 20 and 8 <= drop_80_90 <= 15:
                math_score_ai -= 20  # BONUS mare pentru consistență
            
            # Normalize to 0-100 range (adjusted baseline)
            math_score_ai = max(0, min(100, math_score_ai + 45))  # Baseline la 45 (mai neutru)
            math_verdict = "AI-GENERATED" if math_score_ai > 60 else "REAL"
            
            fft_patterns = {
                'math_score_ai': math_score_ai,
                'math_verdict': math_verdict,
                'unnatural_decay': features_dict['tail_90'] > -1.5,
                'high_freq_anomaly': features_dict['hf_lf_ratio'] > 0.35,
                'non_linear': abs(features_dict['decay_linearity']) < 0.7
            }
            
            st.markdown("---")
            st.markdown("## Analiză Completă: Matematică + AI")
            
            st.markdown("### Scoruri de Detecție")
            score_col1, score_col2, score_col3 = st.columns(3)
            
            with score_col1:
                st.markdown("#### Scor Matematic")
                st.markdown("*(Fără API - doar calcule)*")
                if math_score_ai > 70:
                    st.error(f"**AI: {math_score_ai:.0f}%**")
                    st.caption("SUSPECT AI")
                elif math_score_ai > 50:
                    st.warning(f"**AI: {math_score_ai:.0f}%**")
                    st.caption("INCERT")
                else:
                    st.success(f"**REAL: {100-math_score_ai:.0f}%**")
                    st.caption("PROBABIL REAL")
                
                st.markdown(f"**Verdict:** {math_verdict}")
                
                with st.expander("Debug - Valori Features"):
                    st.markdown(f"""
                    **Drops:**
                    - 60→80%: {drop_60_80:.2f} dB
                    - 80→90%: {drop_80_90:.2f} dB
                    
                    **Ratios:**
                    - HF/LF: {features_dict['hf_lf_ratio']:.3f}
                    - Linearity: {abs(features_dict['decay_linearity']):.3f}
                    
                    **Power:**
                    - Std Dev: {features_dict['std_power']:.2f} dB
                    
                    **Tail Gradients:**
                    - 90%: {features_dict['tail_90']:.3f} dB/px
                    - 80%: {features_dict['tail_80']:.3f} dB/px
                    """)
            
            with score_col2:
                st.markdown("#### Scor OpenAI")
                st.markdown("*(Va fi calculat mai jos)*")
                st.info("Se calculează...")
            
            with score_col3:
                st.markdown("#### Verdict Combinat")
                st.markdown("*(Final: Math 40% + AI 60%)*")
                st.info("Se calculează...")
            
            st.markdown("---")
            
            st.markdown("### Metrici Numerice Complete (Antrenare ML + FFT Analysis)")
            

            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                st.metric("Dimensiune PSD", f"{len(psd1D)} puncte")
                st.metric("Mean Power", f"{features_dict['mean_power']:.2f} dB")
            with metric_col2:
                st.metric("Std Dev", f"{features_dict['std_power']:.2f} dB")
                st.metric("Power Range", f"{features_dict['power_range']:.2f} dB")
            with metric_col3:
                st.metric("HF/LF Ratio", f"{features_dict['hf_lf_ratio']:.3f}")
                delta_hf = "Suspect" if features_dict['hf_lf_ratio'] > 0.35 else "Normal"
                st.caption(delta_hf)
            with metric_col4:
                st.metric("Decay Linearity", f"{features_dict['decay_linearity']:.3f}")
                delta_decay = "Non-linear" if abs(features_dict['decay_linearity']) < 0.7 else "Linear"
                st.caption(delta_decay)
            
            st.markdown("**Tail Gradients (Indicatori Cheie):**")
            tail_col1, tail_col2, tail_col3, tail_col4 = st.columns(4)
            with tail_col1:
                st.metric("Tail 70%", f"{features_dict['tail_70']:.3f} dB/px")
            with tail_col2:
                st.metric("Tail 80%", f"{features_dict['tail_80']:.3f} dB/px")
            with tail_col3:
                st.metric("Tail 90%", f"{features_dict['tail_90']:.3f} dB/px")
                delta_90 = "Flat" if features_dict['tail_90'] > -1.0 else "Natural"
                st.caption(delta_90)
            with tail_col4:
                suspicion = fft_patterns.get('suspicion_score', 0)
                st.metric("Suspicion Score", f"{suspicion}/100")
                if suspicion > 70:
                    st.caption("ALERT: High AI probability")
                elif suspicion > 40:
                    st.caption("WARNING: Moderate suspicion")
                else:
                    st.caption("CLEAR: Low suspicion")
            
            st.markdown("---")
            
            interpretations = {}
            
            st.markdown("### 1. FFT Radial PSD (Spectru de Frecvență)")
            
            if psd1D is not None:
                fig1, ax1 = plt.subplots(figsize=(20, 11))
                radial_freqs = np.arange(len(psd1D))
                
                ax1.plot(radial_freqs, psd1D, linewidth=4, color='#2E86AB', alpha=0.95, label='PSD Curve', zorder=3)
                
                markers = [0.6, 0.7, 0.8, 0.9]
                marker_colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545']
                marker_labels = ['60%', '70%', '80%', '90%']
                
                for pct, color, label in zip(markers, marker_colors, marker_labels):
                    idx = int(pct * len(psd1D))
                    value = psd1D[idx]
                    
                    ax1.axvline(x=idx, color=color, linestyle='--', alpha=0.7, linewidth=2.5, zorder=2)
                    
                    ax1.text(idx, ax1.get_ylim()[1]*0.97, label, 
                            color=color, fontweight='bold', ha='center', fontsize=14,
                            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=color, linewidth=2, alpha=0.9))
                    
                    ax1.text(idx, value + 2, f'{value:.1f} dB', 
                            color='black', fontweight='bold', ha='center', fontsize=12,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7, edgecolor='black', linewidth=1.5))
                
                if len(psd1D) > 50:
                    x_start = int(0.3 * len(psd1D))
                    x_end = len(psd1D)
                    x_range = np.arange(x_start, x_end)
                    # Ideal exponential decay: -2.5 dB/px slope
                    ideal_start = psd1D[x_start]
                    ideal_curve = ideal_start + (-2.5) * (x_range - x_start) / (x_end - x_start) * 10
                    ax1.plot(x_range, ideal_curve, 'g--', linewidth=2, alpha=0.4, 
                            label='Ideal REAL decay (~-2.5 dB/px)', zorder=1)
                
                # Trend line on tail (80-100%)
                if len(psd1D) > 50:
                    tail_start = int(0.8 * len(psd1D))
                    x_tail = radial_freqs[tail_start:]
                    y_tail = psd1D[tail_start:]
                    valid = np.isfinite(y_tail)
                    if np.sum(valid) > 10:
                        coeffs = np.polyfit(x_tail[valid], y_tail[valid], 1)
                        ax1.plot(x_tail, np.polyval(coeffs, x_tail), 'r--', 
                                linewidth=3, alpha=0.8, label=f'Tail Decay: {coeffs[0]:.3f} dB/px')
                
                # Zoom to relevant region (skip first 5% - DC component)
                start_idx = int(0.05 * len(psd1D))
                ax1.set_xlim(start_idx, len(psd1D))
                
                # ADAUGĂ TEXT BOX MARE cu valorile cheie pentru modelul AI
                idx_60 = int(0.6 * len(psd1D))
                idx_70 = int(0.7 * len(psd1D))
                idx_80 = int(0.8 * len(psd1D))
                idx_90 = int(0.9 * len(psd1D))
                
                drop_60_70 = psd1D[idx_60] - psd1D[idx_70]
                drop_70_80 = psd1D[idx_70] - psd1D[idx_80]
                drop_80_90 = psd1D[idx_80] - psd1D[idx_90]
                
                info_text = f"""VALORI CHEIE (pentru AI):
Tail Gradient 90%: {features_dict.get('tail_90', 0):.3f} dB/px
HF/LF Ratio: {features_dict.get('hf_lf_ratio', 0):.3f}

Drop 60→70%: {drop_60_70:.2f} dB
Drop 70→80%: {drop_70_80:.2f} dB  
Drop 80→90%: {drop_80_90:.2f} dB

✅ REAL: Drop 80→90% < 10 dB
🤖 AI: Drop 80→90% > 15 dB"""
                
                # Text box MARE în colțul din dreapta sus
                ax1.text(0.98, 0.97, info_text, 
                        transform=ax1.transAxes, fontsize=13, fontweight='bold',
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=1.0', facecolor='yellow', alpha=0.9, edgecolor='black', linewidth=3))
                
                ax1.set_xlabel('Radial Frequency (pixels)', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Power (dB)', fontsize=14, fontweight='bold')
                ax1.set_title('FFT Radial PSD - Zoom pe zona critică (60-90% frecvențe)', 
                             fontsize=15, fontweight='bold')
                ax1.grid(True, alpha=0.4, linestyle=':')
                ax1.legend(loc='upper right', fontsize=10)
                
                st.pyplot(fig1)
                plt.close(fig1)
                
                if OPENAI_AVAILABLE and api_key_loaded:
                    with st.spinner("GPT-4o-mini analizează FFT..."):
                        try:
                            result = interpret_radial_psd(psd1D, features_dict, api_key=OPENAI_API_KEY)
                            interpretations['radial_psd'] = result
                            
                            ai_confidence = result.get('confidence', 50)
                            ai_is_ai = result.get('is_ai', None)
                            if ai_is_ai is True:
                                ai_score = ai_confidence
                            elif ai_is_ai is False:
                                ai_score = 100 - ai_confidence
                            else:
                                ai_score = 50
                            
                            combined_score = (math_score_ai * 0.4) + (ai_score * 0.6)
                            combined_verdict = "AI-GENERATED" if combined_score > 55 else "REAL"
                            
                            st.markdown("---")
                            st.markdown("#### Comparație Scoruri (FFT Radial PSD)")
                            comp_col1, comp_col2, comp_col3 = st.columns(3)
                            
                            with comp_col1:
                                st.metric("Scor Matematic", f"{math_score_ai:.0f}% AI")
                                st.caption(f"Verdict: {math_verdict}")
                            
                            with comp_col2:
                                st.metric("Scor OpenAI", f"{ai_score:.0f}% AI")
                                st.caption(f"Confidence: {ai_confidence}%")
                            
                            with comp_col3:
                                if combined_score > 70:
                                    st.error(f"**COMBINAT: {combined_score:.0f}% AI**")
                                    st.caption("VERDICT: AI-GENERATED")
                                elif combined_score > 45:
                                    st.warning(f"**COMBINAT: {combined_score:.0f}% AI**")
                                    st.caption("VERDICT: INCERT")
                                else:
                                    st.success(f"**COMBINAT: {100-combined_score:.0f}% REAL**")
                                    st.caption("VERDICT: REAL")
                            
                            st.markdown("---")
                            
                            st.markdown(f"**Raționament OpenAI:** {result.get('reasoning', 'N/A')}")
                            
                            if result.get('indicators'):
                                with st.expander("Indicatori Detectați"):
                                    for ind in result['indicators']:
                                        st.markdown(f"- {ind}")
                            
                            with st.expander("Detalii Complete FFT + Features ML (Pentru Disertație)"):
                                complete_data = {
                                    # PSD Basic Stats
                                    'psd_analysis': {
                                        'points': len(psd1D),
                                        'mean_power_db': float(features_dict['mean_power']),
                                        'std_dev_db': float(features_dict['std_power']),
                                        'power_range_db': float(features_dict['power_range'])
                                    },
                                    # Tail Gradients (critical for AI detection)
                                    'tail_gradients': {
                                        'tail_70_pct': float(features_dict['tail_70']),
                                        'tail_80_pct': float(features_dict['tail_80']),
                                        'tail_90_pct': float(features_dict['tail_90']),
                                        'interpretation': 'Tail > -1.0 = suspicious (AI likely)'
                                    },
                                    # Frequency Analysis
                                    'frequency_features': {
                                        'hf_lf_ratio': float(features_dict['hf_lf_ratio']),
                                        'decay_linearity': float(features_dict['decay_linearity']),
                                        'interpretation': 'HF/LF > 0.3 or linearity < 0.8 = AI artifacts'
                                    },
                                    # Pattern Detection
                                    'detected_patterns': {
                                        'unnatural_decay': fft_patterns.get('unnatural_decay', False),
                                        'high_freq_anomaly': fft_patterns.get('high_freq_anomaly', False),
                                        'non_linear_decay': fft_patterns.get('non_linear', False),
                                        'suspicion_score': fft_patterns.get('suspicion_score', 0)
                                    },
                                    # GPT-4o-mini Analysis
                                    'ai_interpretation': {
                                        'verdict': 'AI-GENERATED' if result.get('is_ai') else 'REAL',
                                        'confidence': result.get('confidence', 0),
                                        'reasoning': result.get('reasoning', 'N/A')
                                    }
                                }
                                st.json(complete_data)
                        except Exception as e:
                            st.error(f"Eroare interpretare: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                elif not GEMINI_API_KEY:
                    st.warning("API Key lipsă în gemini_graph_interpreter.py")
                else:
                    st.warning("Gemini not available - pip install google-genai")
            
            st.markdown("---")
            
            st.markdown("### 2. Spectru 2D FFT (Pattern-uri Spațiale)")
            
            if magnitude_2d is not None:
                fig2, ax2 = plt.subplots(figsize=(14, 14))
                im = ax2.imshow(magnitude_2d, cmap='hot', aspect='auto')
                ax2.set_title('FFT 2D Spectrum - Full View', fontsize=15, fontweight='bold')
                plt.colorbar(im, ax=ax2, label='Log Power', fraction=0.046, pad=0.04)
                
                ax2.grid(True, alpha=0.2, color='cyan', linestyle=':', linewidth=0.5)
                
                st.pyplot(fig2)
                plt.close(fig2)
                
                if OPENAI_AVAILABLE and api_key_loaded:
                    with st.spinner("GPT-4o-mini analizează spectrul 2D..."):
                        try:
                            result = interpret_2d_spectrum(magnitude_2d, api_key=OPENAI_API_KEY)
                            interpretations['spectrum_2d'] = result
                            
                            if result.get('is_ai') is True:
                                st.error(f"**GPT-4o-mini: AI-GENERATED** (Confidence: {result.get('confidence', 0):.0f}%)")
                            elif result.get('is_ai') is False:
                                st.success(f"**GPT-4o-mini: REAL** (Confidence: {result.get('confidence', 0):.0f}%)")
                            else:
                                st.warning("**Gemini: INCERT**")
                            
                            st.markdown(f"**Raționament:** {result.get('reasoning', 'N/A')}")
                            
                            if result.get('indicators'):
                                with st.expander("Indicatori Detectați"):
                                    for ind in result['indicators']:
                                        st.markdown(f"- {ind}")
                        except Exception as e:
                            st.error(f"Eroare interpretare: {str(e)}")
            
            st.markdown("---")
            
            # =====================================
            # GRAFIC 2.5: NOISE RESIDUAL FFT (CRITICAL pentru GAN/Diffusion detection)
            # =====================================
            st.markdown("### 🔬 2.5. Noise Residual FFT Analysis (Detectare Artefacte GAN)")
            st.markdown("*Elimină conținutul natural și arată doar artefactele rețelelor neurale*")
            
            try:
                # 1. Obține componenta de zgomot (Noise Residual)
                # Convertim gray la uint8 pentru cv2.medianBlur
                gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
                
                # Aplicăm median blur pentru a elimina detaliile fine naturale
                denoised = cv2.medianBlur(gray_uint8, 3)
                
                # Reziduul = imaginea originală - versiunea blur-ată
                # Acest reziduu conține doar textura fină și artefactele
                noise_residual = gray_uint8.astype(np.float32) - denoised.astype(np.float32)
                
                # 2. Aplică FFT pe reziduu
                f_noise = np.fft.fft2(noise_residual)
                fshift_noise = np.fft.fftshift(f_noise)
                
                # 3. Calculează magnitudinea (cu logaritm pentru că semnalul e slab)
                magnitude_noise = 20 * np.log(np.abs(fshift_noise) + 1e-9)
                
                # 4. Afișare comparativă: Reziduu + Spectrul său
                fig_noise, (ax_res, ax_spec) = plt.subplots(1, 2, figsize=(20, 9))
                
                # Subplot 1: Reziduul de zgomot
                ax_res.imshow(noise_residual, cmap='gray')
                ax_res.set_title('Reziduu de Zgomot (High Pass Filter)\n(Artefacte vizibile dacă e AI)', 
                               fontsize=14, fontweight='bold')
                ax_res.axis('off')
                
                # Subplot 2: Spectrul FFT al reziduului
                im_spec = ax_spec.imshow(magnitude_noise, cmap='hot', aspect='auto')
                ax_spec.set_title('Spectrul Reziduului (FFT)\n⚠️ Puncte/Linii geometrice = AI | Uniform = REAL', 
                                fontsize=14, fontweight='bold')
                ax_spec.grid(True, alpha=0.2, color='cyan', linestyle=':', linewidth=0.5)
                plt.colorbar(im_spec, ax=ax_spec, label='Log Power', fraction=0.046)
                
                # Text explicativ
                fig_noise.text(0.5, 0.02, 
                             '💡 Imaginile REAL au spectru uniform/cețos. Imaginile AI (GAN/Diffusion) au PUNCTE sau LINII geometrice clare.',
                             ha='center', fontsize=12, fontweight='bold', 
                             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
                
                plt.tight_layout(rect=[0, 0.04, 1, 1])
                st.pyplot(fig_noise)
                plt.close(fig_noise)
                
                # Analiză automată: Detectează puncte strălucitoare în spectrul reziduului
                # Elimină centrul (DC component)
                h_noise, w_noise = magnitude_noise.shape
                cy_noise, cx_noise = h_noise // 2, w_noise // 2
                mask_center = np.ones_like(magnitude_noise, dtype=bool)
                mask_center[cy_noise-10:cy_noise+10, cx_noise-10:cx_noise+10] = False
                
                # Calculează statistici pe zona non-centrală
                outer_values = magnitude_noise[mask_center]
                mean_outer = np.mean(outer_values)
                std_outer = np.std(outer_values)
                max_outer = np.max(outer_values)
                
                # Threshold pentru "puncte strălucitoare" = mean + 3*std
                threshold_bright = mean_outer + 3 * std_outer
                bright_pixels = np.sum(magnitude_noise[mask_center] > threshold_bright)
                bright_ratio = bright_pixels / np.sum(mask_center)
                
                # Scoring: Multe puncte strălucitoare = AI
                noise_residual_score = 0
                if bright_ratio > 0.02:  # >2% puncte strălucitoare = suspect
                    noise_residual_score = 70 + min(30, bright_ratio * 1000)
                elif bright_ratio > 0.01:
                    noise_residual_score = 50 + bright_ratio * 2000
                else:
                    noise_residual_score = max(0, 20 + bright_ratio * 3000)
                
                noise_residual_score = min(100, noise_residual_score)
                noise_verdict = "AI-GENERATED" if noise_residual_score > 60 else "REAL"
                
                # Display rezultate
                st.markdown("#### 📊 Analiză Automată Noise Residual")
                ncol1, ncol2, ncol3, ncol4 = st.columns(4)
                
                with ncol1:
                    st.metric("Puncte Strălucitoare", f"{bright_pixels}")
                    st.caption(f"Ratio: {bright_ratio*100:.2f}%")
                
                with ncol2:
                    st.metric("Mean Power (outer)", f"{mean_outer:.1f} dB")
                    st.caption(f"Std: {std_outer:.1f}")
                
                with ncol3:
                    st.metric("Max Peak (outer)", f"{max_outer:.1f} dB")
                    delta_peak = "🔴 Suspect" if max_outer > mean_outer + 5*std_outer else "✓ Normal"
                    st.caption(delta_peak)
                
                with ncol4:
                    if noise_residual_score > 60:
                        st.error(f"**🚨 Scor: {noise_residual_score:.0f}% AI**")
                    else:
                        st.success(f"**✅ Scor: {100-noise_residual_score:.0f}% REAL**")
                    st.caption(f"Verdict: {noise_verdict}")
                
                st.info("💡 **Interpretare:** Dacă vezi puncte sau linii geometrice clare în spectrul reziduului (dreapta), e probabil AI. Imaginile REAL au spectru uniform.")
                
            except Exception as e:
                st.error(f"Eroare Noise Residual Analysis: {str(e)}")
            
            st.markdown("---")
            
            # =====================================
            # GRAFIC 3: ANGULAR ENERGY
            # =====================================
            st.markdown("### ⭐ 3. Semnătura Unghiulară (Star Pattern)")
            
            try:
                # Simple angular energy calculation
                h, w = magnitude_2d.shape
                cy, cx = h//2, w//2
                
                # Compute angular distribution (0-360 degrees)
                y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
                theta = np.arctan2(y, x)
                ang_bins = np.linspace(-np.pi, np.pi, 180)
                ang_energy = np.array([magnitude_2d[(theta >= ang_bins[i]) & (theta < ang_bins[i+1])].sum() 
                                       for i in range(len(ang_bins)-1)])
                
                # Find peaks (star pattern indicators)
                from scipy.signal import find_peaks
                star_peaks_idx, _ = find_peaks(ang_energy, height=ang_energy.mean()*1.2, distance=10)
                
                # Calculate 180° symmetry (star pattern indicator)
                half_len = len(ang_energy) // 2
                if half_len > 0:
                    symmetry_corr = np.corrcoef(ang_energy[:half_len], ang_energy[half_len:2*half_len])[0,1]
                    star_sym = max(0, symmetry_corr)  # 0-1 range
                else:
                    star_sym = 0
                
                # Mathematical star pattern score (fără API) - AJUSTAT pentru mai puține false positives
                star_math_score = 0
                
                # Peak count scoring - MAI PUȚIN STRICT (10+ = very suspect, 6-8 = moderat)
                if len(star_peaks_idx) >= 10:  # FOARTE multe peaks = AI
                    star_math_score += 50
                elif len(star_peaks_idx) >= 8:  # Multe peaks = suspect
                    star_math_score += 30
                elif len(star_peaks_idx) >= 6:  # Moderat (poate fi REAL sau AI)
                    star_math_score += 10
                elif len(star_peaks_idx) <= 3:  # Puține peaks = REAL
                    star_math_score -= 25
                
                # Symmetry scoring - MAI PUȚIN STRICT (0.8+ = suspect, 0.7 = moderat)
                if star_sym > 0.85:  # Simetrie FOARTE mare = AI
                    star_math_score += 50
                elif star_sym > 0.75:  # Simetrie mare = suspect
                    star_math_score += 30
                elif star_sym > 0.6:  # Simetrie moderată = puțin suspect
                    star_math_score += 10
                elif star_sym < 0.3:  # Asimetrie = REAL
                    star_math_score -= 30
                
                star_math_score = max(0, min(100, star_math_score + 35))  # Normalize (start mai jos)
                star_math_verdict = "AI-GENERATED" if star_math_score > 65 else "REAL"  # Prag mai înalt
                
                # Update fft_patterns with star pattern info - prag mai strict (8+ peaks ȘI symmetry >0.75)
                fft_patterns['star_pattern'] = len(star_peaks_idx) >= 8 and star_sym > 0.75
                fft_patterns['star_peaks'] = len(star_peaks_idx)
                fft_patterns['star_symmetry'] = star_sym
                fft_patterns['star_math_score'] = star_math_score
                fft_patterns['star_math_verdict'] = star_math_verdict
                
                # Define angles array for polar plot
                angles = np.linspace(-np.pi, np.pi, len(ang_energy))
                
                # Plot with annotations
                fig3, ax3 = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))
                ax3.plot(angles, ang_energy, linewidth=3, color='#E63946', alpha=0.9, label='Energy Distribution')
                
                # Mark peaks with labels
                if len(star_peaks_idx) > 0:
                    for i, pk in enumerate(star_peaks_idx):
                        angle_deg = np.degrees(angles[pk])
                        ax3.plot(angles[pk], ang_energy[pk], 'go', markersize=12, marker='*', 
                                markeredgecolor='darkgreen', markeredgewidth=2.5, zorder=5)
                        # Label each peak
                        ax3.text(angles[pk], ang_energy[pk]*1.1, f'P{i+1}\n{angle_deg:.0f}°', 
                                ha='center', fontsize=9, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                
                # Add threshold circle for reference
                mean_energy = np.mean(ang_energy)
                ax3.axhline(y=mean_energy, color='gray', linestyle=':', alpha=0.5, linewidth=2, 
                           label=f'Mean Energy: {mean_energy:.2f}')
                
                pattern_status = "⚠️ STAR PATTERN DETECTED" if fft_patterns['star_pattern'] else "✓ Normal Distribution"
                ax3.set_title(f'Angular Energy - {pattern_status}\n(Peaks: {len(star_peaks_idx)}, Symmetry: {star_sym*100:.1f}%, Math Score: {star_math_score:.0f}% AI)', 
                             fontsize=13, fontweight='bold', pad=20)
                ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
                
                st.pyplot(fig3)
                plt.close(fig3)
                
                # OpenAI interpretation
                if OPENAI_AVAILABLE and api_key_loaded:
                    with st.spinner("🤖 GPT-4o-mini analizează pattern-ul unghiular..."):
                        try:
                            result = interpret_angular_energy(ang_energy, star_peaks_idx, api_key=OPENAI_API_KEY)
                            interpretations['angular_energy'] = result
                            
                            if result.get('is_ai') is True:
                                st.error(f"**🤖 GPT-4o-mini: AI-GENERATED** (Confidence: {result.get('confidence', 0):.0f}%)")
                            elif result.get('is_ai') is False:
                                st.success(f"**✅ GPT-4o-mini: REAL** (Confidence: {result.get('confidence', 0):.0f}%)")
                            else:
                                st.warning("**⚠️ GPT-4o-mini: INCERT**")
                            
                            st.markdown(f"**Raționament:** {result.get('reasoning', 'N/A')}")
                            
                            if result.get('indicators'):
                                with st.expander("📋 Indicatori Detectați"):
                                    for ind in result['indicators']:
                                        st.markdown(f"- {ind}")
                        except Exception as e:
                            st.error(f"Eroare interpretare: {str(e)}")
            except Exception as e:
                st.error(f"Eroare calcul angular energy: {str(e)}")
            
            st.markdown("---")
            
            # =====================================
            # VERDICT FINAL OpenAI
            # =====================================
            if OPENAI_AVAILABLE and api_key_loaded and interpretations:
                st.markdown("## 🎯 Verdict Final GPT-4o-mini")
                
                with st.spinner("🤖 GPT-4o-mini agregă toate analizele..."):
                    try:
                        final = get_final_verdict(interpretations, features_dict, fft_patterns)
                        
                        # Display verdict
                        if final.get('verdict') == 'AI-GENERATED':
                            st.error(f"# 🤖 VERDICT: **{final['verdict']}**")
                        elif final.get('verdict') == 'REAL':
                            st.success(f"# ✅ VERDICT: **{final['verdict']}**")
                        else:
                            st.warning(f"# ⚠️ VERDICT: **{final.get('verdict', 'UNKNOWN')}**")
                        
                        # Confidence
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.metric("Confidence", f"{final.get('confidence', 0):.0f}%")
                        with col2:
                            st.progress(final.get('confidence', 0) / 100)
                        
                        # Reasoning
                        st.markdown("### 💭 Raționament Complet:")
                        st.write(final.get('reasoning', 'N/A'))
                        
                        # Key findings
                        if final.get('key_findings'):
                            st.markdown("### 🔑 Concluzii Cheie:")
                            for finding in final['key_findings']:
                                st.markdown(f"- {finding}")
                        
                        # Graph votes
                        if final.get('graph_votes'):
                            st.markdown("### 📊 Voturi per Grafic:")
                            vote_col1, vote_col2, vote_col3 = st.columns(3)
                            votes = final['graph_votes']
                            
                            with vote_col1:
                                vote = votes.get('radial_psd', 'N/A')
                                if vote == 'AI':
                                    st.error(f"**FFT PSD:** {vote}")
                                else:
                                    st.success(f"**FFT PSD:** {vote}")
                            
                            with vote_col2:
                                vote = votes.get('spectrum_2d', 'N/A')
                                if vote == 'AI':
                                    st.error(f"**Spectrum 2D:** {vote}")
                                else:
                                    st.success(f"**Spectrum 2D:** {vote}")
                            
                            with vote_col3:
                                vote = votes.get('angular_energy', 'N/A')
                                if vote == 'AI':
                                    st.error(f"**Angular:** {vote}")
                                else:
                                    st.success(f"**Angular:** {vote}")
                        
                        # Recommendation
                        if final.get('recommendation'):
                            st.info(f"**💡 Recomandare:** {final['recommendation']}")
                        
                    except Exception as e:
                        st.error(f"Eroare verdict final: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

else:
    st.info("👆 Încarcă o imagine pentru a începe analiza")
    
    if not OPENAI_AVAILABLE:
        st.warning("""
        ⚠️ **OpenAI nu este disponibil**
        
        Pentru interpretare automată, instalează:
        ```bash
        pip install openai
        ```
        """)
    elif not api_key_loaded:
        st.error("""
        🔑 **API Key OpenAI necesar!**
        
        Pune-l în `gemini_graph_interpreter.py` linia 13:
        ```python
        OPENAI_API_KEY = "sk-YOUR-ACTUAL-KEY-HERE"
        ```
        """)
