"""
Deepfake Detector - Academic Research Project
VATASE Radu-Petrut
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
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
            
            psd_2d = np.abs(fft_shifted) ** 2
            radial_profile = azimuthalAverage(psd_2d)
            psd1D = 10 * np.log10(radial_profile + 1e-10)
            
            psd_len = len(psd1D)
            
            val_60 = psd1D[int(0.6 * psd_len)] if psd_len > 50 else 0
            val_70 = psd1D[int(0.7 * psd_len)] if psd_len > 50 else 0
            val_80 = psd1D[int(0.8 * psd_len)] if psd_len > 50 else 0
            val_90 = psd1D[int(0.9 * psd_len)] if psd_len > 50 else 0
            
            drop_60_80 = val_60 - val_80
            drop_80_90 = val_80 - val_90
            
            features_dict = {
                'tail_90': np.gradient(psd1D)[int(0.9*psd_len):].mean() if psd_len > 50 else 0,
                'tail_80': np.gradient(psd1D)[int(0.8*psd_len):].mean() if psd_len > 50 else 0,
                'tail_70': np.gradient(psd1D)[int(0.7*psd_len):].mean() if psd_len > 50 else 0,
                'mean_power': np.mean(psd1D),
                'std_power': np.std(psd1D),
                'power_range': np.max(psd1D) - np.min(psd1D),
                'hf_lf_ratio': (np.mean(psd1D[int(0.7*psd_len):]) / 
                               (np.mean(psd1D[:int(0.4*psd_len)]) + 1e-10)) if psd_len > 50 else 0,
                'decay_linearity': np.corrcoef(np.arange(psd_len//2, psd_len), 
                                               psd1D[psd_len//2:])[0,1] if psd_len > 50 else 0,
                'drop_60_80': drop_60_80,
                'drop_80_90': drop_80_90
            }
            
            math_score_ai = 0
            
            if drop_80_90 < 3:
                math_score_ai += 35
            elif drop_80_90 < 6:
                math_score_ai += 15
            elif drop_80_90 > 18:
                math_score_ai += 20
            elif 8 <= drop_80_90 <= 15:
                math_score_ai -= 25
            
            hf_lf = features_dict['hf_lf_ratio']
            if hf_lf > 0.8:
                math_score_ai += 25
            elif hf_lf > 0.6:
                math_score_ai += 10
            elif hf_lf < 0.3:
                math_score_ai -= 20
            
            linearity = abs(features_dict['decay_linearity'])
            if linearity < 0.5:
                math_score_ai += 20
            elif linearity > 0.85:
                math_score_ai -= 15
            
            if features_dict['std_power'] > 25:
                math_score_ai += 10
            elif features_dict['std_power'] < 10:
                math_score_ai -= 10
            
            if 8 <= drop_60_80 <= 20 and 8 <= drop_80_90 <= 15:
                math_score_ai -= 20
            
            math_score_ai = max(0, min(100, math_score_ai + 45))
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
                st.markdown("*(Final: Math 60% + AI 40%)*")
                st.info("Se calculează...")
            
            st.markdown("---")
            
            st.markdown("### Metrici Numerice Complete")
            
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
            
            st.markdown("**Tail Gradients:**")
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
            
            st.markdown("### 1. FFT Radial PSD")
            
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
                
                start_idx = int(0.05 * len(psd1D))
                ax1.set_xlim(start_idx, len(psd1D))
                
                ax1.set_xlabel('Radial Frequency (pixels)', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Power (dB)', fontsize=14, fontweight='bold')
                ax1.set_title('FFT Radial PSD - Frequency Analysis', fontsize=15, fontweight='bold')
                ax1.grid(True, alpha=0.4, linestyle=':')
                ax1.legend(loc='upper right', fontsize=10)
                
                st.pyplot(fig1)
                plt.close(fig1)
                
                if OPENAI_AVAILABLE and api_key_loaded:
                    with st.spinner("Analyzing FFT with OpenAI..."):
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
                            
                            combined_score = (math_score_ai * 0.6) + (ai_score * 0.4)
                            combined_verdict = "AI-GENERATED" if combined_score > 50 else "REAL"
                            
                            st.markdown("---")
                            st.markdown("#### Comparație Scoruri")
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
                            
                            with st.expander("Detalii Complete (Pentru Disertație)"):
                                complete_data = {
                                    'psd_analysis': {
                                        'points': len(psd1D),
                                        'mean_power_db': float(features_dict['mean_power']),
                                        'std_dev_db': float(features_dict['std_power']),
                                        'power_range_db': float(features_dict['power_range'])
                                    },
                                    'tail_gradients': {
                                        'tail_70_pct': float(features_dict['tail_70']),
                                        'tail_80_pct': float(features_dict['tail_80']),
                                        'tail_90_pct': float(features_dict['tail_90']),
                                        'interpretation': 'Tail > -1.0 = suspicious (AI likely)'
                                    },
                                    'frequency_features': {
                                        'hf_lf_ratio': float(features_dict['hf_lf_ratio']),
                                        'decay_linearity': float(features_dict['decay_linearity']),
                                        'interpretation': 'HF/LF > 0.3 or linearity < 0.8 = AI artifacts'
                                    },
                                    'detected_patterns': {
                                        'unnatural_decay': fft_patterns.get('unnatural_decay', False),
                                        'high_freq_anomaly': fft_patterns.get('high_freq_anomaly', False),
                                        'non_linear_decay': fft_patterns.get('non_linear', False),
                                        'suspicion_score': fft_patterns.get('suspicion_score', 0)
                                    },
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
                else:
                    st.warning("API Key lipsă sau Gemini not available")
            
            st.markdown("---")
            
            st.markdown("### 2. Spectru 2D FFT")
            
            if magnitude_2d is not None:
                fig2, ax2 = plt.subplots(figsize=(14, 14))
                im = ax2.imshow(magnitude_2d, cmap='hot', aspect='auto')
                ax2.set_title('FFT 2D Spectrum', fontsize=15, fontweight='bold')
                plt.colorbar(im, ax=ax2, label='Log Power', fraction=0.046, pad=0.04)
                ax2.grid(True, alpha=0.2, color='cyan', linestyle=':', linewidth=0.5)
                
                st.pyplot(fig2)
                plt.close(fig2)
                
                if OPENAI_AVAILABLE and api_key_loaded:
                    with st.spinner("Analyzing 2D spectrum..."):
                        try:
                            result = interpret_2d_spectrum(magnitude_2d, api_key=OPENAI_API_KEY)
                            interpretations['spectrum_2d'] = result
                            
                            if result.get('is_ai') is True:
                                st.error(f"**OpenAI: AI-GENERATED** (Confidence: {result.get('confidence', 0):.0f}%)")
                            elif result.get('is_ai') is False:
                                st.success(f"**OpenAI: REAL** (Confidence: {result.get('confidence', 0):.0f}%)")
                            else:
                                st.warning("**OpenAI: UNCERTAIN**")
                            
                            st.markdown(f"**Raționament:** {result.get('reasoning', 'N/A')}")
                            
                            if result.get('indicators'):
                                with st.expander("Indicatori Detectați"):
                                    for ind in result['indicators']:
                                        st.markdown(f"- {ind}")
                        except Exception as e:
                            st.error(f"Eroare interpretare: {str(e)}")
            
            st.markdown("---")
            
            if OPENAI_AVAILABLE and api_key_loaded and interpretations:
                st.markdown("## Verdict Final OpenAI")
                
                with st.spinner("OpenAI agregă toate analizele..."):
                    try:
                        final = get_final_verdict(interpretations, features_dict, fft_patterns)
                        
                        if final.get('verdict') == 'AI-GENERATED':
                            st.error(f"# VERDICT: **{final['verdict']}**")
                        elif final.get('verdict') == 'REAL':
                            st.success(f"# VERDICT: **{final['verdict']}**")
                        else:
                            st.warning(f"# VERDICT: **{final.get('verdict', 'UNKNOWN')}**")
                        
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.metric("Confidence", f"{final.get('confidence', 0):.0f}%")
                        with col2:
                            st.progress(final.get('confidence', 0) / 100)
                        
                        st.markdown("### Raționament Complet:")
                        st.write(final.get('reasoning', 'N/A'))
                        
                        if final.get('key_findings'):
                            st.markdown("### Concluzii Cheie:")
                            for finding in final['key_findings']:
                                st.markdown(f"- {finding}")
                        
                        if final.get('graph_votes'):
                            st.markdown("### Voturi per Grafic:")
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
                        
                        if final.get('recommendation'):
                            st.info(f"**Recomandare:** {final['recommendation']}")
                        
                    except Exception as e:
                        st.error(f"Eroare verdict final: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

else:
    st.info("Încarcă o imagine pentru a începe analiza")
    
    if not OPENAI_AVAILABLE:
        st.warning("""
        **OpenAI nu este disponibil**
        
        Pentru interpretare automată, instalează:
        ```bash
        pip install openai
        ```
        """)
    elif not api_key_loaded:
        st.error("""
        **API Key OpenAI necesar!**
        
        Pune-l în `gemini_graph_interpreter.py` linia 13:
        ```python
        OPENAI_API_KEY = "sk-YOUR-ACTUAL-KEY-HERE"
        ```
        """)
