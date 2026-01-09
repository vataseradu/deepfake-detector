"""
Streamlit Dashboard for Deepfake Detection & Digital Forensics
===============================================================
Master's Thesis: Multimodal Architecture for Deepfake Detection
and Digital Integrity Verification

Modules:
- Error Level Analysis (ELA): Detects local manipulation
- Frequency Analysis (FFT): Detects AI-generation artifacts
- Decision Logic: calibrated metric analysis

Author: Master's Thesis Project
Date: January 2026
"""

import streamlit as st
from typing import Optional
from PIL import Image
import numpy as np

# Import analysis modules
from ela import perform_ela
from frequency import plot_spectrum
from decision_logic import analyze_fft_score, get_final_verdict, get_risk_assessment


# Page Configuration
st.set_page_config(
    page_title="Deepfake Detection & Digital Forensics System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main Title
st.title("🔬 Digital Forensic Analysis: Deepfake & Manipulation Detection")
st.markdown("""
**Master's Thesis Project** | *Multimodal Architecture for AI-Generated Image Detection* This system combines multiple forensic techniques to detect image manipulation and AI-generated content.
""")
st.markdown("---")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Analysis Settings")
    
    st.subheader("ELA Parameters")
    ela_quality = st.slider(
        "JPEG Quality for Re-compression",
        min_value=50,
        max_value=100,
        value=90,
        help="Lower quality = more sensitive to manipulation, but may show false positives"
    )
    
    st.markdown("---")
    st.subheader("About")
    st.info("""
    **Detection Methods:**
    - **ELA**: Analyzes JPEG compression artifacts
    - **FFT**: Examines frequency domain patterns
    
    **Use Cases:**
    - Detect Photoshop edits
    - Identify AI-generated images
    - Verify image authenticity
    """)

# Main Content Area
st.markdown("## 📁 Upload Image for Analysis")
uploaded_file = st.file_uploader(
    "Select an image file (JPG, PNG, JPEG)",
    type=["jpg", "png", "jpeg"],
    help="Upload a suspicious image for forensic analysis"
)

if uploaded_file is not None:
    try:
        # Load image
        original_image: Image.Image = Image.open(uploaded_file).convert('RGB')
        
        # Display original image info
        st.success(f"✅ Image loaded: {uploaded_file.name} | Size: {original_image.size[0]}x{original_image.size[1]}px")
        
        # Run analyses immediately to get verdict
        with st.spinner("🔍 Running forensic analysis..."):
            # 1. Perform ELA analysis
            ela_result, ela_stats = perform_ela(original_image, quality=ela_quality)
            
            # 2. Perform FFT analysis (Get Plot + Raw Data)
            fig_fft, psd1D = plot_spectrum(original_image)
            
            # 3. Analyze FFT Score & Get Debug Metrics (CRITICAL FOR CALIBRATION)
            fft_score, fft_debug = analyze_fft_score(psd1D)
            
            # 4. Get final verdict based on fused scores
            verdict = get_final_verdict(fft_score, ela_stats)
            risk_assessment = get_risk_assessment(verdict)
        
        # ============== VERDICT SECTION (TOP) ==============
        st.markdown("---")
        st.markdown("## 🎯 Detection Verdict")
        
        # Create verdict display with colored background
        verdict_color = verdict['color']
        verdict_label = verdict['label']
        verdict_confidence = verdict['confidence']
        verdict_explanation = verdict['explanation']
        verdict_technical = verdict['technical_details']
        
        # Custom CSS for verdict box
        st.markdown(f"""
        <div style="
            background-color: {verdict_color}15;
            border-left: 5px solid {verdict_color};
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        ">
            <h2 style="color: {verdict_color}; margin: 0 0 10px 0;">{verdict_label}</h2>
            <p style="font-size: 18px; margin: 10px 0;"><strong>Confidence:</strong> {verdict_confidence}</p>
            <p style="font-size: 16px; margin: 10px 0;">{verdict_explanation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display technical metrics in columns
        st.markdown("#### 📊 Technical Metrics")
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("FFT AI Score", f"{fft_score:.1f}%", 
                     help="Higher = More likely AI-generated")
        with metric_cols[1]:
            st.metric("ELA Mean", f"{ela_stats['mean_intensity']:.2f}",
                     help="Average compression error level")
        with metric_cols[2]:
            st.metric("ELA Std Dev", f"{ela_stats['std_intensity']:.2f}",
                     help="Variance indicator - high = manipulation")
        with metric_cols[3]:
            st.metric("ELA Max", f"{ela_stats['max_intensity']:.2f}",
                     help="Peak compression error")
        
        # Risk assessment
        st.info(risk_assessment)
        
        st.markdown(f"""
        <details>
        <summary><strong>🔧 Technical Details</strong></summary>
        <p style="font-family: monospace; font-size: 14px; margin-top: 10px;">
        {verdict_technical}
        </p>
        </details>
        """, unsafe_allow_html=True)

        # ============== CALIBRATION PANEL (NEW!) ==============
        st.markdown("---")
        st.error("🛠️ CALIBRATION DATA (Use these values for your Thesis Table)")
        
        cal_col1, cal_col2, cal_col3 = st.columns(3)
        with cal_col1:
            st.metric("Log HF Ratio", f"{fft_debug.get('log_hf_ratio', 'N/A')}", 
                     help="Valoarea critică! Real ≈ -2.0, AI ≈ -5.0")
        with cal_col2:
            st.metric("Spectral Slope", f"{fft_debug.get('slope', 'N/A')}",
                     help="Real ≈ -2.0, AI < -3.0")
        with cal_col3:
            st.metric("ELA Std Dev", f"{ela_stats['std_intensity']:.2f}",
                     help="Peste 45 indică manipulare/Photoshop")
        
        # ============== DETAILED ANALYSIS TABS ==============
        st.markdown("---")
        st.markdown("## 🔬 Detailed Forensic Analysis")
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["🔍 Error Level Analysis (ELA)", "📊 Frequency Analysis (FFT)", "📷 Original Image"])
        
        # ============== TAB 1: ELA ==============
        with tab1:
            st.markdown("### Module 1: Error Level Analysis")
            st.markdown("""
            **Purpose:** Detect local image manipulation (copy-paste, face swap, inpainting)  
            **Method:** Analyzes inconsistencies in JPEG compression error levels
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Original Image")
                st.image(original_image, caption="Input Image", width="stretch")
            
            with col2:
                st.markdown("#### ELA Result")
                st.image(ela_result, caption=f"ELA Output (Quality={ela_quality})", width="stretch")
            
            # Interpretation guide
            st.info("""
            **🔍 How to Interpret ELA Results:**
            - **Uniform brightness (low variance):** Likely authentic or fully AI-generated image
            - **Localized bright spots (high variance):** Potential manipulation detected
            - **Very bright regions:** Strong indicator of copy-paste or face swap
            - **Edge-aligned brightness:** May indicate inpainting/object removal
            
            ⚠️ **Note:** High-contrast edges and saturated colors can produce false positives
            """)
        
        # ============== TAB 2: FFT ==============
        with tab2:
            st.markdown("### Module 2: Frequency Domain Analysis")
            st.markdown("""
            **Purpose:** Detect AI-generated images (Stable Diffusion, DALL-E, Midjourney)  
            **Method:** Analyzes frequency spectrum deviations from natural 1/f power law
            """)
            
            # Display the FFT plot
            st.pyplot(fig_fft)
            
            # Display FFT score
            st.markdown("#### 📈 FFT Analysis Score")
            score_col1, score_col2 = st.columns(2)
            
            with score_col1:
                # Visual gauge
                score_color = "#FF4B4B" if fft_score > 60 else "#FFD700" if fft_score > 40 else "#28A745"
                st.markdown(f"""
                <div style="text-align: center;">
                    <h1 style="color: {score_color}; font-size: 72px; margin: 0;">{fft_score:.1f}%</h1>
                    <p style="font-size: 18px; color: #666;">AI Generation Probability</p>
                </div>
                """, unsafe_allow_html=True)
            
            with score_col2:
                st.markdown("""
                **Score Interpretation:**
                - **0-40%**: Natural high-freq content - Likely authentic
                - **40-60%**: Borderline - Possible heavy processing
                - **60-100%**: Depleted high frequencies - Likely AI-generated
                """)
            
            # Interpretation guide
            st.info("""
            **🔍 How to Interpret FFT Results:**
            
            **Left Plot (2D Frequency Spectrum):**
            - Center = Low frequencies (overall shape/color)
            - Edges = High frequencies (fine details/textures)
            - Bright spots = Strong frequency components
            
            **Right Plot (1D Power Spectral Density):**
            - **Real images:** Follow red dashed line (1/f² power law) - smooth decline
            - **AI images:** Show deviations (abrupt drops at high frequencies)
            
            **Red dashed line = Natural reference (1/f² decay)** Significant deviations suggest AI generation or heavy post-processing.
            """)
        
        # ============== TAB 3: ORIGINAL ==============
        with tab3:
            st.markdown("### Original Image")
            st.image(original_image, caption=f"{uploaded_file.name}", width="stretch")
            
            st.markdown("#### Image Information")
            info_cols = st.columns(3)
            with info_cols[0]:
                st.metric("Width", f"{original_image.size[0]}px")
            with info_cols[1]:
                st.metric("Height", f"{original_image.size[1]}px")
            with info_cols[2]:
                st.metric("Mode", original_image.mode)
    
    except Exception as e:
        st.error(f"❌ Error loading image: {str(e)}")
        st.warning("Please ensure the uploaded file is a valid image format.")

else:
    # Welcome screen when no image is uploaded
    st.info("👆 **Upload an image above to begin forensic analysis**")
    
    st.markdown("### 🎯 System Capabilities")
    
    cap_col1, cap_col2 = st.columns(2)
    
    with cap_col1:
        st.markdown("""
        **Error Level Analysis (ELA)**
        - Detects Photoshop manipulation
        - Identifies copy-paste artifacts
        - Reveals inpainting/face swaps
        - Works on JPEG images
        """)
    
    with cap_col2:
        st.markdown("""
        **Frequency Analysis (FFT)**
        - Detects AI-generated images
        - Identifies diffusion model artifacts
        - Analyzes spectral anomalies
        - Works on all image formats
        """)
    
    st.markdown("---")
    st.markdown("""
    ### 📚 Academic References
    - **ELA Method:** Krawetz, N. (2007) "A Picture's Worth..."
    - **FFT Analysis:** Frank et al. (2020) "Leveraging Frequency Analysis for Deep Fake Image Recognition"
    - **Power Law Theory:** Durall et al. (2020) "Watch Your Up-Convolution: CNN Based Generative Deep Neural Networks"
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Master's Thesis Project</strong> | Multimodal Architecture for Deepfake Detection | January 2026</p>
</div>
""", unsafe_allow_html=True)