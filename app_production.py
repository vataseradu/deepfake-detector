"""
Deepfake Detector - Academic Research Project
VATASE Radu-Petrut
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import pickle

from frequency import azimuthalAverage
from scipy.fft import fft2, fftshift
from scipy import ndimage

# Import TensorFlow/Keras for CNN model
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    import tensorflow as tf
    from tensorflow import keras
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False
    tf = None
    keras = None

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

# Load CNN Model once at startup
@st.cache_resource
def load_cnn_model():
    """Load the trained Xception CNN model"""
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'modele', 'deepfake_phase1_best.keras')
        if not os.path.exists(model_path):
            return None, f"Model not found: {model_path}"
        
        # Recreate model architecture manually (avoid loading issues)
        from tensorflow.keras.applications import Xception
        from tensorflow.keras import layers, models
        
        # Build same architecture as training
        base_model = Xception(
            include_top=False,
            weights=None,  # Will load from saved weights
            input_shape=(256, 256, 3)  # MUST match training config!
        )
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ], name='xception_deepfake_detector')
        
        # Now load the weights
        model.load_weights(model_path)
        
        # Compile for inference
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model, None
        
    except Exception as e:
        return None, str(e)

cnn_model, cnn_error = load_cnn_model() if CNN_AVAILABLE else (None, "TensorFlow not available")

def predict_cnn(img_pil, model):
    """
    Preprocess image and predict using CNN
    
    Args:
        img_pil: PIL Image
        model: Loaded Keras model
    
    Returns:
        dict: {
            'probability_fake': float,
            'probability_real': float,
            'prediction': str ('REAL' or 'FAKE'),
            'confidence': float
        }
    """
    try:
        # Preprocess image (MUST match training exactly!)
        img_resized = img_pil.resize((256, 256))  # 256x256 as per training config
        img_array = keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        # Normalize to [0, 1] (same as training: rescale=1./255)
        img_array = img_array / 255.0
        
        # Predict
        pred = model.predict(img_array, verbose=0)[0][0]
        
        # flow_from_directory with class_mode='binary' assigns alphabetically:
        # fake=0 (class 0), real=1 (class 1)
        # Model outputs: 0 = FAKE, 1 = REAL
        prob_real = float(pred)
        prob_fake = 1.0 - prob_real
        
        prediction = "REAL" if prob_real > 0.5 else "FAKE"
        confidence = max(prob_fake, prob_real) * 100
        
        return {
            'probability_fake': prob_fake,
            'probability_real': prob_real,
            'prediction': prediction,
            'confidence': confidence
        }
    except Exception as e:
        return {
            'error': str(e),
            'probability_fake': 0.5,
            'probability_real': 0.5,
            'prediction': 'ERROR',
            'confidence': 0
        }

st.title("Deepfake Detector")
st.markdown("**Tema de cercetare - VATASE Radu-Petrut**")
st.caption("TCSI - Teoria codarii si stocarii informatiei")
st.info("Sistem hibrid: FFT + Random Forest + CNN Xception (74.67% accuracy, AUC 0.8273)")

uploaded_file = st.file_uploader("Upload imagine", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(img)
    
    with st.sidebar:
        st.image(img, caption="Imagine Uploaded", width='stretch')
        st.markdown(f"**Dimensiune:** {img.size[0]} x {img.size[1]}")
        
        # Display model status
        st.markdown("---")
        st.markdown("### Module disponibile:")
        
        # CNN Status
        if CNN_AVAILABLE and cnn_model is not None:
            st.success("✅ CNN Xception (Faza 1)")
            st.caption("deepfake_phase1_best.keras")
        elif CNN_AVAILABLE:
            st.error("❌ CNN Model Error")
            st.caption(cnn_error)
        else:
            st.warning("⚠️ TensorFlow indisponibil")
        
        # Random Forest Status
        if os.path.exists('face_rf_simple.pkl'):
            st.success("✅ Random Forest")
        else:
            st.warning("⚠️ Random Forest lipseste")
        
        if OPENAI_AVAILABLE and api_key_loaded:
            model_display = OPENAI_MODEL if OPENAI_MODEL else "gpt-4o-mini"
            st.success(f"✅ OpenAI {model_display}")
            st.caption(f"API Key: {OPENAI_API_KEY[:15]}...")
        elif OPENAI_AVAILABLE:
            st.warning("⚠️ API Key OpenAI lipsește")
            st.info("Pune-l în gemini_graph_interpreter.py linia 13")
        else:
            st.warning("⚠️ OpenAI indisponibil")
    
    if st.button("Analizează", type="primary"):
        with st.spinner("Analizez imaginea..."):
            # ===== CNN PREDICTION FIRST (DECISIVE COMPONENT) =====
            cnn_result = None
            if CNN_AVAILABLE and cnn_model is not None:
                with st.spinner("🔬 CNN Xception analysis..."):
                    cnn_result = predict_cnn(img, cnn_model)
            
            # ===== FFT ANALYSIS =====
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
            
            # Try to load Random Forest model trained on FACE dataset
            try:
                import pickle
                import os
                model_path = os.path.join(os.path.dirname(__file__), 'face_rf_simple.pkl')
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        rf_model = pickle.load(f)
                    
                    # Extract 5 features used in training
                    X_features = np.array([[
                        features_dict['tail_70'],
                        features_dict['tail_80'],
                        features_dict['tail_90'],
                        features_dict['hf_lf_ratio'],
                        features_dict['std_power']
                    ]])
                    
                    # Get probability prediction
                    proba = rf_model.predict_proba(X_features)[0]
                    # proba[0] = P(REAL), proba[1] = P(FAKE)
                    math_score_ai = proba[1] * 100
                    
                else:
                    # Fallback: neutral score if model not found
                    math_score_ai = 50
            except Exception as e:
                # Fallback heuristics (INVERTED for FACE dataset)
                math_score_ai = 50
                
                # For FACE: LOWER hf_lf suggests AI (reversed from old logic)
                hf_lf = features_dict['hf_lf_ratio']
                if hf_lf < 0.60:
                    math_score_ai += 20
                elif hf_lf < 0.65:
                    math_score_ai += 10
                elif hf_lf > 0.75:
                    math_score_ai -= 15
                
                # Tail flatness
                if features_dict['tail_90'] > -0.005:
                    math_score_ai += 15
                elif features_dict['tail_90'] < -0.01:
                    math_score_ai -= 10
                
                math_score_ai = max(0, min(100, math_score_ai))
            math_verdict = "AI-GENERATED" if math_score_ai > 60 else "REAL"
            
            fft_patterns = {
                'math_score_ai': math_score_ai,
                'math_verdict': math_verdict,
                'unnatural_decay': features_dict['tail_90'] > -1.5,
                'high_freq_anomaly': features_dict['hf_lf_ratio'] > 0.35,
                'non_linear': abs(features_dict['decay_linearity']) < 0.7
            }
            
            st.markdown("---")
            st.markdown("## Analiza completa:")
            
            # ===== PRIMARY VERDICT: CNN =====
            st.markdown("### VERDICT PRINCIPAL - CNN Xception")
            st.caption("*Componentă decisivă: 74.67% accuracy, AUC 0.8273 (Epoca 3, 100.000 imagini)*")
            
            if cnn_result and 'error' not in cnn_result:
                cnn_col1, cnn_col2, cnn_col3 = st.columns(3)
                
                with cnn_col1:
                    st.metric("Predicție CNN", cnn_result['prediction'])
                    if cnn_result['prediction'] == 'FAKE':
                        st.error(f"🚨 **AI-GENERATED**")
                    else:
                        st.success(f"✅ **REAL IMAGE**")
                
                with cnn_col2:
                    st.metric("Probabilitate FAKE", f"{cnn_result['probability_fake']*100:.1f}%")
                    st.metric("Probabilitate REAL", f"{cnn_result['probability_real']*100:.1f}%")
                
                with cnn_col3:
                    st.metric("Confidence", f"{cnn_result['confidence']:.1f}%")
                    if cnn_result['confidence'] > 80:
                        st.caption("Certitudine Foarte Mare")
                    elif cnn_result['confidence'] > 60:
                        st.caption("Certitudine Mare")
                    else:
                        st.caption("Certitudine Moderată")
                
                # Visual bar
                st.markdown("**Distribuție probabilități:**")
                prob_fake_pct = cnn_result['probability_fake'] * 100
                prob_real_pct = cnn_result['probability_real'] * 100
                
                col_bar1, col_bar2 = st.columns([prob_fake_pct/100, prob_real_pct/100])
                with col_bar1:
                    st.markdown(f"<div style='background-color: #FF4B4B; padding: 10px; text-align: center; color: white; font-weight: bold;'>FAKE: {prob_fake_pct:.1f}%</div>", unsafe_allow_html=True)
                with col_bar2:
                    st.markdown(f"<div style='background-color: #00CC66; padding: 10px; text-align: center; color: white; font-weight: bold;'>REAL: {prob_real_pct:.1f}%</div>", unsafe_allow_html=True)
            
            elif cnn_result and 'error' in cnn_result:
                st.error(f"Eroare CNN: {cnn_result['error']}")
            else:
                st.warning("⚠️ CNN indisponibil - vezi sidebar pentru detalii")
            
            st.markdown("---")
            st.markdown("### 📊 Analiză Comparativă (FFT + Random Forest)")
            st.caption("*Metode complementare pentru validare educațională*")
            
            st.markdown("### VERDICT PRINCIPAL")
            
            verdict_col1, verdict_col2 = st.columns(2)
            
            with verdict_col1:
                st.markdown("#### Scor Matematic")
                st.caption("*Bazat pe 5 features FFT antrenate pe 2041 imagini*")
                if math_score_ai > 70:
                    st.error(f"**AI: {math_score_ai:.0f}%**")
                    st.caption("🚨 SUSPECT AI")
                elif math_score_ai > 50:
                    st.warning(f"**AI: {math_score_ai:.0f}%**")
                    st.caption("⚠️ INCERT")
                else:
                    st.success(f"**REAL: {100-math_score_ai:.0f}%**")
                    st.caption("✅ PROBABIL REAL")
                
                st.markdown(f"**Verdict:** {math_verdict}")
                
                with st.expander("Debug - Valori features"):
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
            
            st.markdown("---")
            
            st.markdown("### Metrici numerice complete")
            
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
            tail_col1, tail_col2, tail_col3 = st.columns(3)
            with tail_col1:
                st.metric("Tail 70%", f"{features_dict['tail_70']:.3f} dB/px")
            with tail_col2:
                st.metric("Tail 80%", f"{features_dict['tail_80']:.3f} dB/px")
            with tail_col3:
                st.metric("Tail 90%", f"{features_dict['tail_90']:.3f} dB/px")
                delta_90 = "Flat" if features_dict['tail_90'] > -1.0 else "Natural"
                st.caption(delta_90)
            
            st.markdown("---")
            
            st.markdown("## 🎯 VERDICT PRINCIPAL")
            st.caption("*Bazat pe Random Forest antrenat pe 2041 imagini (960 AI + 1081 reale)*")
            
            verdict_col1, verdict_col2 = st.columns(2)
            
            with verdict_col1:
                st.markdown("### 🔢 Analiza matematica FFT")
                st.caption("5 features: tail_70, tail_80, tail_90, hf_lf_ratio, std_power")
                
                if math_score_ai > 70:
                    st.error(f"### {math_score_ai:.0f}% SUSPICIUNE AI")
                    st.markdown(f"**Verdict:** {math_verdict}")
                elif math_score_ai > 45:
                    st.warning(f"### {math_score_ai:.0f}% SUSPICIUNE AI")
                    st.markdown(f"**Verdict:** INCERT")
                else:
                    st.success(f"### {100-math_score_ai:.0f}% REAL")
                    st.markdown(f"**Verdict:** {math_verdict}")
            
            with verdict_col2:
                if OPENAI_AVAILABLE and api_key_loaded:
                    st.markdown("### 🤖 Interpretare GPT-4o")
                    st.caption("*Analizeaza graficele FFT + analize suplimentare*")
                    gpt_placeholder = st.empty()
                    gpt_placeholder.info("⏳ Rezultatul va aparea dupa ce toate graficele sunt generate")
                else:
                    st.markdown("#### Interpretare AI (Optional)")
                    st.warning("⚠️ OpenAI indisponibil - verifica API Key")
                    gpt_placeholder = None
            
            st.markdown("---")
            
            interpretations = {}
            
            st.markdown("### 1. FFT Radial PSD")
            
            if psd1D is not None:
                with st.expander("Afiseaza grafic FFT Radial PSD", expanded=False):
                    # Grafic CURAT fara label-uri pentru a nu induce AI in eroare
                    fig1, ax1 = plt.subplots(figsize=(12, 6))
                    radial_freqs = np.arange(len(psd1D))
                    
                    # Plot simplu fara adnotari
                    ax1.plot(radial_freqs, psd1D, linewidth=2, color='#2E86AB', alpha=0.95)
                    
                    start_idx = int(0.05 * len(psd1D))
                    ax1.set_xlim(start_idx, len(psd1D))
                    
                    ax1.set_xlabel('Radial Frequency', fontsize=12)
                    ax1.set_ylabel('Power (dB)', fontsize=12)
                    ax1.set_title('FFT Radial PSD', fontsize=13)
                    ax1.grid(True, alpha=0.3, linestyle=':')
                    
                    st.pyplot(fig1)
                    plt.close(fig1)
                    
                    # Numerele ca text pentru prompt AI
                    psd_len = len(psd1D)
                    val_60 = psd1D[int(0.6 * psd_len)] if psd_len > 50 else 0
                    val_70 = psd1D[int(0.7 * psd_len)] if psd_len > 50 else 0
                    val_80 = psd1D[int(0.8 * psd_len)] if psd_len > 50 else 0
                    val_90 = psd1D[int(0.9 * psd_len)] if psd_len > 50 else 0
                    
                    st.caption(f"📊 Valori PSD: 60%={val_60:.1f}dB, 70%={val_70:.1f}dB, 80%={val_80:.1f}dB, 90%={val_90:.1f}dB")
                
                if OPENAI_AVAILABLE and api_key_loaded:
                    with st.spinner("Analyzing FFT with OpenAI..."):
                        try:
                            # Trimite valorile numerice ca text
                            psd_text_values = {
                                'val_60': float(val_60),
                                'val_70': float(val_70),
                                'val_80': float(val_80),
                                'val_90': float(val_90)
                            }
                            result = interpret_radial_psd(psd1D, features_dict, psd_text_values=psd_text_values, api_key=OPENAI_API_KEY)
                            interpretations['radial_psd'] = result
                            
                            # Fix: Foloseste correct confidence
                            ai_confidence = result.get('confidence', 50)
                            ai_is_ai = result.get('is_ai', None)
                            
                            # ai_confidence = cat de sigur e AI
                            # Daca is_ai=True -> ai_score = confidence (% ca e AI)
                            # Daca is_ai=False -> ai_score = 100-confidence (% ca e REAL inversat)
                            if ai_is_ai is True:
                                ai_score = ai_confidence  # Ex: 75% confidence ca e AI -> 75% AI score
                            elif ai_is_ai is False:
                                ai_score = 100 - ai_confidence  # Ex: 80% confidence ca e REAL -> 20% AI score
                            else:
                                ai_score = 50
                            
                            combined_score = (math_score_ai * 0.6) + (ai_score * 0.4)
                            combined_verdict = "AI-GENERATED" if combined_score > 50 else "REAL"
                            
                            st.markdown("---")
                            st.markdown("#### 🎯 Comparatie scoruri")
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
                                with st.expander("🔍 Indicatori detectati"):
                                    for ind in result['indicators']:
                                        st.markdown(f"- {ind}")
                            
                            with st.expander("📋 Detalii complete (pentru disertatie)"):
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
            
            st.markdown("#### 2. Spectru 2D FFT")
            
            if magnitude_2d is not None:
                with st.expander("Afiseaza grafic Spectru 2D FFT", expanded=False):
                    fig2, ax2 = plt.subplots(figsize=(10, 10))
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
                st.markdown("---")
                st.markdown("## Verdict OpenAI GPT-4o")
                st.caption("*Sinteza analizei FFT + grafice suplimentare (color, gradient, noise)*")
                
                with st.spinner("⏳ GPT-4o agrega toate datele..."):
                    try:
                        final = get_final_verdict(interpretations, features_dict, fft_patterns)
                        
                        # Actualizam placeholder-ul din verdict_col2
                        if 'gpt_placeholder' in locals() and gpt_placeholder is not None:
                            with gpt_placeholder.container():
                                if final.get('verdict') == 'AI-GENERATED':
                                    st.error(f"**{final['verdict']}**")
                                elif final.get('verdict') == 'REAL':
                                    st.success(f"**{final['verdict']}**")
                                else:
                                    st.warning(f"**{final.get('verdict', 'UNKNOWN')}**")
                                st.caption(f"Confidence: {final.get('confidence', 0):.0f}%")
                        
                        vcol1, vcol2, vcol3 = st.columns([1, 2, 2])
                        
                        with vcol1:
                            if final.get('verdict') == 'AI-GENERATED':
                                st.error(f"### {final['verdict']}")
                            elif final.get('verdict') == 'REAL':
                                st.success(f"### {final['verdict']}")
                            else:
                                st.warning(f"### {final.get('verdict', 'UNKNOWN')}")
                            
                            st.metric("Confidence", f"{final.get('confidence', 0):.0f}%")
                        
                        with vcol2:
                            if final.get('key_findings'):
                                st.markdown("**Concluzii cheie:**")
                                for finding in final['key_findings'][:3]:
                                    st.markdown(f"• {finding}")
                        
                        with vcol3:
                            if final.get('graph_votes'):
                                st.markdown("**Voturi grafice:**")
                                votes = final['graph_votes']
                                st.caption(f"📊 PSD Radial: {votes.get('radial_psd', 'N/A')}")
                                st.caption(f"🎨 Spectrum 2D: {votes.get('spectrum_2d', 'N/A')}")
                        
                        with st.expander("📋 Rationament complet OpenAI"):
                            st.write(final.get('reasoning', 'N/A'))
                        

                    except Exception as e:
                        st.error(f"Eroare verdict final: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            st.markdown("---")
            st.markdown("## 📊 Analize suplimentare")
            st.caption("*Metode complementare: color, gradient, noise, metadata*")
            
            # 3. Color Histogram
            with st.expander("🌈 Color Histogram - distributie canale RGB"):
                fig3, ax3 = plt.subplots(figsize=(12, 6))
                colors = ('r', 'g', 'b')
                for i, color in enumerate(colors):
                    histogram, bin_edges = np.histogram(img_array[:, :, i], bins=256, range=(0, 256))
                    ax3.plot(bin_edges[0:-1], histogram, color=color, alpha=0.7, linewidth=2, label=f'{color.upper()} channel')
                
                ax3.set_xlabel('Pixel Value', fontsize=12)
                ax3.set_ylabel('Frequency', fontsize=12)
                ax3.set_title('RGB Color Distribution', fontsize=13)
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                st.pyplot(fig3)
                
                # Analiza simpla
                r_std = np.std(img_array[:, :, 0])
                g_std = np.std(img_array[:, :, 1])
                b_std = np.std(img_array[:, :, 2])
                st.caption(f"📌 Std Dev: R={r_std:.2f}, G={g_std:.2f}, B={b_std:.2f}")
                if abs(r_std - g_std) < 5 and abs(g_std - b_std) < 5:
                    st.info("✅ Distribuție naturală - canale echilibrate")
                else:
                    st.warning("⚠️ Distribuție neuniformă - posibilă procesare artificială")
                
                # Trimite la OpenAI
                if OPENAI_AVAILABLE and api_key_loaded:
                    with st.spinner("🤖 OpenAI analizeaza Color Histogram..."):
                        try:
                            from gemini_graph_interpreter import interpret_color_histogram
                            color_result = interpret_color_histogram(fig3, r_std, g_std, b_std, api_key=OPENAI_API_KEY)
                            interpretations['color_histogram'] = color_result
                            
                            st.markdown(f"**AI: {color_result.get('reasoning', 'N/A')}**")
                        except Exception as e:
                            st.warning(f"Eroare OpenAI: {str(e)}")
                
                plt.close(fig3)
            
            # 4. Gradient Magnitude Map
            with st.expander("📐 Gradient Magnitude - harta de detalii"):
                gray_img = np.mean(img_array, axis=2).astype(np.float32)
                
                # Sobel gradients
                grad_x = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
                grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                
                fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))
                
                ax4a.imshow(grad_magnitude, cmap='hot')
                ax4a.set_title('Gradient Magnitude', fontsize=13)
                ax4a.axis('off')
                
                # Histogram of gradients
                ax4b.hist(grad_magnitude.flatten(), bins=100, color='coral', alpha=0.7)
                ax4b.set_xlabel('Gradient Magnitude', fontsize=12)
                ax4b.set_ylabel('Frequency', fontsize=12)
                ax4b.set_title('Gradient Distribution', fontsize=13)
                ax4b.grid(True, alpha=0.3)
                
                st.pyplot(fig4)
                
                mean_grad = np.mean(grad_magnitude)
                std_grad = np.std(grad_magnitude)
                st.caption(f"📌 Mean Gradient: {mean_grad:.2f}, Std: {std_grad:.2f}")
                if std_grad < 15:
                    st.warning("⚠️ Gradient foarte uniform - posibil AI smoothing")
                else:
                    st.info("✅ Gradient variat - texturi naturale")
                
                # Trimite la OpenAI
                if OPENAI_AVAILABLE and api_key_loaded:
                    with st.spinner("🤖 OpenAI analizeaza Gradient Magnitude..."):
                        try:
                            from gemini_graph_interpreter import interpret_gradient_magnitude
                            gradient_result = interpret_gradient_magnitude(fig4, mean_grad, std_grad, api_key=OPENAI_API_KEY)
                            interpretations['gradient_magnitude'] = gradient_result
                            
                            st.markdown(f"**AI: {gradient_result.get('reasoning', 'N/A')}**")
                        except Exception as e:
                            st.warning(f"Eroare OpenAI: {str(e)}")
                
                plt.close(fig4)
            
            # 5. Noise Pattern Analysis
            with st.expander("🔍 Noise Pattern - analiza zgomotului"):
                # High-pass filter pentru noise
                gray_img = np.mean(img_array, axis=2).astype(np.float32)
                blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
                noise = gray_img - blurred
                
                fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 6))
                
                im5a = ax5a.imshow(noise, cmap='gray', vmin=-30, vmax=30)
                ax5a.set_title('Noise Pattern', fontsize=13)
                ax5a.axis('off')
                plt.colorbar(im5a, ax=ax5a, fraction=0.046, pad=0.04)
                
                # Noise histogram
                ax5b.hist(noise.flatten(), bins=100, color='steelblue', alpha=0.7)
                ax5b.set_xlabel('Noise Value', fontsize=12)
                ax5b.set_ylabel('Frequency', fontsize=12)
                ax5b.set_title('Noise Distribution', fontsize=13)
                ax5b.grid(True, alpha=0.3)
                
                st.pyplot(fig5)
                
                noise_std = np.std(noise)
                st.caption(f"📌 Noise Std Dev: {noise_std:.2f}")
                if noise_std < 5:
                    st.warning("⚠️ Zgomot foarte mic - posibilă prelucrare AI (denoising)")
                elif noise_std > 20:
                    st.warning("⚠️ Zgomot foarte mare - posibil artifact compresie")
                else:
                    st.info("✅ Zgomot natural - nivel acceptabil")
                
                # Trimite la OpenAI
                if OPENAI_AVAILABLE and api_key_loaded:
                    with st.spinner("🤖 OpenAI analizeaza Noise Pattern..."):
                        try:
                            from gemini_graph_interpreter import interpret_noise_pattern
                            noise_result = interpret_noise_pattern(fig5, noise_std, api_key=OPENAI_API_KEY)
                            interpretations['noise_pattern'] = noise_result
                            
                            st.markdown(f"**AI: {noise_result.get('reasoning', 'N/A')}**")
                        except Exception as e:
                            st.warning(f"Eroare OpenAI: {str(e)}")
                
                plt.close(fig5)
            
            # 6. EXIF Metadata
            st.markdown("---")
            st.markdown("## 📷 EXIF Metadata")
            
            try:
                from PIL.ExifTags import TAGS
                
                exif_data = img.getexif()
                
                if exif_data:
                    st.success("✅ Metadata EXIF găsită")
                    
                    exif_dict = {}
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        exif_dict[tag] = str(value)
                    
                    with st.expander("📋 Detalii EXIF Complete"):
                        for key, value in exif_dict.items():
                            st.text(f"{key}: {value}")
                    
                    # Key indicators
                    meta_col1, meta_col2, meta_col3 = st.columns(3)
                    
                    with meta_col1:
                        if 'Make' in exif_dict or 'Model' in exif_dict:
                            st.metric("Camera", exif_dict.get('Model', exif_dict.get('Make', 'N/A')))
                        else:
                            st.warning("⚠️ Lipsă info camera")
                            st.caption("AI images rar au metadata camera")
                    
                    with meta_col2:
                        if 'Software' in exif_dict:
                            software = exif_dict['Software']
                            st.metric("Software", software[:30])
                            if any(ai_tool in software.lower() for ai_tool in ['stable', 'midjourney', 'dalle', 'photoshop', 'generate']):
                                st.error("🚨 AI generation tool detectat!")
                        else:
                            st.info("Software: N/A")
                    
                    with meta_col3:
                        if 'DateTime' in exif_dict:
                            st.metric("Date", exif_dict['DateTime'][:10])
                        else:
                            st.info("Date: N/A")
                    
                    # AI indicators
                    ai_indicators = []
                    if 'Make' not in exif_dict and 'Model' not in exif_dict:
                        ai_indicators.append("❌ Lipsă metadata camera (suspect)")
                    if 'Software' in exif_dict:
                        if any(tool in exif_dict['Software'].lower() for tool in ['ai', 'generate', 'stable', 'midjourney']):
                            ai_indicators.append("🚨 Software AI detectat")
                    if not exif_data or len(exif_data) < 5:
                        ai_indicators.append("⚠️ EXIF minimal (posibil sters sau generat)")
                    
                    if ai_indicators:
                        st.warning("**Indicatori Suspicioși:**")
                        for indicator in ai_indicators:
                            st.caption(indicator)
                    else:
                        st.success("✅ Metadata completa - nicio alerta")
                
                else:
                    st.error("❌ Nicio metadata EXIF")
                    st.caption("⚠️ Imaginile AI generate rar contin EXIF data")
                    st.caption("⚠️ Sau metadata a fost stearsa intenționat")
            
            except Exception as e:
                st.warning(f"Nu s-a putut citi EXIF: {str(e)}")

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
