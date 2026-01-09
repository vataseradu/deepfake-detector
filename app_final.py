"""
SISTEM FINAL INTEGRAT - STREAMLIT APPLICATION
==============================================
Aplicație completă pentru detectarea imaginilor AI-generate
Integrează toate metodele dezvoltate în disertație

Metode:
1. FFT Spectral Analysis (Log Ratio, Tail Gradients)
2. Error Level Analysis (ELA)
3. Wavelet Transform (Daubechies)
4. Local Binary Patterns (LBP)
5. Gradient Analysis
6. Color Statistics
7. Machine Learning (Random Forest optimizat)
"""

import streamlit as st
from PIL import Image, ImageChops
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import tempfile
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import pywt
from skimage.feature import local_binary_pattern, peak_local_max
from scipy.stats import entropy, skew, kurtosis
from scipy.signal import windows, welch, find_peaks
from scipy.ndimage import gaussian_filter1d
import piexif
import cv2

# Optional: Gemini API pentru interpretare automată
try:
    from gemini_interpreter import interpret_with_gemini
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ Gemini interpreter not available. Install google-genai to enable AI interpretation.")

# Page Config
st.set_page_config(
    page_title="AI Image Detector - Final System",
    page_icon="🔍",
    layout="wide"
)

# CSS
st.markdown("""
<style>
.big-font {font-size:24px !important; font-weight:bold; color:#1f77b4;}
.verdict-real {font-size:32px; font-weight:bold; color:#2ecc71; text-align:center; padding:20px; background:#d5f4e6; border-radius:10px;}
.verdict-fake {font-size:32px; font-weight:bold; color:#e74c3c; text-align:center; padding:20px; background:#fadbd8; border-radius:10px;}
.metric-box {background:#f0f2f6; padding:15px; border-radius:8px; border-left:4px solid #1f77b4; margin:10px 0;}
.feature-bar {background:#e8eaed; height:20px; border-radius:4px; margin:5px 0;}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="big-font">🔍 AI IMAGE DETECTOR - SISTEM FINAL INTEGRAT</p>', unsafe_allow_html=True)
st.markdown("**Disertație** | *Multi-Method Forensic Analysis pentru Detectarea Imaginilor AI-Generate*")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Setări Analiză")
    
    st.subheader("📊 Metode Active")
    st.info("""
    **8 Tehnici Implementate:**
    ✓ Metadata EXIF Analysis
    ✓ Welch PSD (robust FFT)
    ✓ Error Level Analysis (ELA)
    ✓ Wavelet Transform
    ✓ Local Binary Patterns
    ✓ Gradient Analysis
    ✓ Color Statistics
    ✓ Machine Learning (RF)
    """)
    
    show_details = st.checkbox("Arată Detalii Features", value=False)
    show_visualization = st.checkbox("Arată Vizualizări", value=True)
    
    st.markdown("---")
    st.subheader("ℹ️ Despre")
    st.write(f"**Acuratețe Sistem:** ~58.5%")
    st.write(f"**Dataset Antrenare:** 2041 imagini")
    st.write(f"**Features:** 27 metrici numerice")
    st.write(f"**FFT:** Hanning Window + INTER_AREA")

# ============================================================================
# IMPROVED FFT FORENSICS FUNCTIONS (Code Review Fixes)
# ============================================================================

def radial_whitening_fast(spectrum, exclude_dc_radius=5, r_min_frac=0.05):
    """Vectorized radial whitening (Fix #2 - 10-100× faster)"""
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(np.int32)
    max_r = r.max()
    r_min = int(r_min_frac * max_r)
    
    # Vectorized radial mean
    radial_mean = np.zeros(max_r + 1)
    radial_count = np.bincount(r.ravel(), minlength=max_r + 1)
    radial_sum = np.bincount(r.ravel(), weights=spectrum.ravel(), minlength=max_r + 1)
    valid = radial_count > 0
    radial_mean[valid] = radial_sum[valid] / radial_count[valid]
    
    # Exclude DC and low frequencies
    whitened = spectrum.copy()
    mask_whiten = (r >= max(exclude_dc_radius, r_min))
    whitened[mask_whiten] -= radial_mean[r[mask_whiten]]
    
    # Robust normalization
    med = np.median(whitened[mask_whiten])
    mad = np.median(np.abs(whitened[mask_whiten] - med)) + 1e-8
    z = (whitened - med) / (1.4826 * mad)
    return z

def azimuthalAverage(image, center=None):
    """
    Calculează media radială (azimutală) a spectrului de putere 2D.
    Transformă spectrul 2D într-un profil 1D radial - CORECT MATEMATIC.
    
    Parameters:
    -----------
    image : 2D array
        Spectrul de putere 2D (|FFT|^2)
    center : tuple, optional
        Centrul pentru calculul radial. Dacă None, folosește centrul imaginii.
        
    Returns:
    --------
    radial_profile : 1D array
        Profilul radial al puterii (Power vs. Frecvență radială)
    """
    y, x = np.indices(image.shape)
    if center is None:
        center = np.array([(y.max()-y.min())/2.0, (x.max()-x.min())/2.0])
    
    # Calculează distanța radială de la centru pentru fiecare pixel
    r = np.hypot(x - center[1], y - center[0])
    
    # Sortează razele și valorile corespunzătoare
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]
    
    # Obține partea întreagă a razelor pentru binning (grupare pe inele)
    r_int = r_sorted.astype(int)
    
    # Calculează media pentru fiecare inel (rază) - MEDIA AZIMUTALĂ
    tbin = np.bincount(r_int, i_sorted)
    nr = np.bincount(r_int)
    
    # Evită împărțirea la zero
    radial_profile = np.zeros_like(tbin, dtype=float)
    mask = nr > 0
    radial_profile[mask] = tbin[mask] / nr[mask]
    
    return radial_profile

def detect_symmetric_spikes(z_spectrum, z_thresh=6.0, r_min=15):
    """Detect spikes with symmetry verification (Fix #3)"""
    h, w = z_spectrum.shape
    cy, cx = h // 2, w // 2
    
    # Find peaks
    coords = peak_local_max(z_spectrum, threshold_abs=z_thresh, min_distance=10, 
                           num_peaks=200, exclude_border=10)
    if len(coords) == 0:
        return 0, 0, 0.0
    
    # Filter by radius
    r_peaks = np.sqrt((coords[:, 0] - cy)**2 + (coords[:, 1] - cx)**2)
    valid = r_peaks >= r_min
    coords = coords[valid]
    if len(coords) == 0:
        return 0, 0, 0.0
    
    # Check symmetry
    symmetric_pairs = 0
    used = set()
    for i, (y1, x1) in enumerate(coords):
        if i in used:
            continue
        y2_exp = 2 * cy - y1
        x2_exp = 2 * cx - x1
        for j, (y2, x2) in enumerate(coords):
            if j <= i or j in used:
                continue
            dist = np.sqrt((y2 - y2_exp)**2 + (x2 - x2_exp)**2)
            if dist <= 5:
                symmetric_pairs += 1
                used.add(i)
                used.add(j)
                break
    
    values = z_spectrum[coords[:, 0], coords[:, 1]]
    return len(coords), symmetric_pairs, float(np.mean(values)) if len(values) > 0 else 0.0

def angular_energy_signature(z_spectrum, r_min_frac=0.1, r_max_frac=0.45):
    """Compute angular energy distribution"""
    h, w = z_spectrum.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    dx, dy = x - cx, y - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = (np.arctan2(dy, dx) + np.pi)
    rmax = r.max()
    mask = (r > r_min_frac * rmax) & (r < r_max_frac * rmax)
    
    n_bins = 360
    theta_bins = (theta[mask] * n_bins / (2*np.pi)).astype(int) % n_bins
    energy = np.maximum(z_spectrum[mask], 0)
    
    ang_energy = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for t, e in zip(theta_bins, energy):
        ang_energy[t] += e
        counts[t] += 1
    ang_energy /= np.maximum(counts, 1)
    return ang_energy

def star_score_with_symmetry(angular_energy):
    """Compute star score with 180° symmetry (Fix #4)"""
    smoothed = gaussian_filter1d(angular_energy, sigma=3, mode='wrap')
    norm = (smoothed - np.mean(smoothed)) / (np.std(smoothed) + 1e-8)
    prominence = 1.5 * np.std(norm)
    peaks, props = find_peaks(norm, prominence=prominence, distance=5)
    
    # ✅ Require at least 6 peaks (excludes 4-fold grids like fences)
    if len(peaks) < 6:
        return 0.0, 0.0, 0
    
    # Check 180° symmetry
    n_bins = len(angular_energy)
    symmetry_errors = []
    for pk in peaks:
        opposite_bin = (pk + n_bins // 2) % n_bins
        distances = np.abs(peaks - opposite_bin)
        distances = np.minimum(distances, n_bins - distances)
        symmetry_errors.append(distances.min())
    
    avg_error = np.mean(symmetry_errors) if symmetry_errors else n_bins
    symmetry_score = 1.0 - (avg_error / (n_bins / 8))
    symmetry_score = np.clip(symmetry_score, 0, 1)
    
    base_score = len(peaks) + np.sum(props["prominences"])
    star_score = base_score * (1.0 + symmetry_score)
    
    return float(star_score), float(symmetry_score), peaks
    
    # Check 180° symmetry
    n_bins = len(angular_energy)
    symmetry_errors = []
    for pk in peaks:
        opposite_bin = (pk + n_bins // 2) % n_bins
        distances = np.abs(peaks - opposite_bin)
        distances = np.minimum(distances, n_bins - distances)
        symmetry_errors.append(distances.min())
    
    avg_error = np.mean(symmetry_errors) if symmetry_errors else n_bins
    symmetry_score = 1.0 - (avg_error / (n_bins / 8))
    symmetry_score = np.clip(symmetry_score, 0, 1)
    
    base_score = len(peaks) + np.sum(props["prominences"])
    star_score = base_score * (1.0 + symmetry_score)
    
    return float(star_score), float(symmetry_score), len(peaks)

# FFT Pattern Analysis Function
def analyze_fft_patterns(psd1D, magnitude_2d):
    """Detectează pattern-uri suspecte în spectrul FFT (Improved with Code Review Fixes)"""
    patterns = {
        'star_pattern': False,
        'periodic_spikes': False,
        'unnatural_decay': False,
        'suspicion_score': 0,
        'interpretations': [],
        # NEW: Advanced forensics metrics
        'num_spikes': 0,
        'symmetric_pairs': 0,
        'symmetry_ratio': 0.0,
        'star_score': 0.0,
        'star_symmetry': 0.0,
        'star_num_peaks': 0
    }
    
    if psd1D is None or len(psd1D) < 50:
        return patterns
    
    try:
        # ✅ IMPROVED ANALYSIS using Code Review Fixes
        
        # Apply radial whitening to 2D spectrum (if available)
        if magnitude_2d is not None:
            try:
                # Whiten spectrum (Fix #2)
                whitened = radial_whitening_fast(magnitude_2d, exclude_dc_radius=5, r_min_frac=0.05)
                
                # Detect symmetric spikes (Fix #3)
                num_spikes, sym_pairs, spike_strength = detect_symmetric_spikes(whitened, z_thresh=6.0, r_min=15)
                
                # Angular energy and star score (Fix #4)
                ang_energy = angular_energy_signature(whitened, r_min_frac=0.1, r_max_frac=0.45)
                star_sc, star_sym, star_peaks = star_score_with_symmetry(ang_energy)
                
                # Store metrics
                patterns['num_spikes'] = num_spikes
                patterns['symmetric_pairs'] = sym_pairs
                patterns['symmetry_ratio'] = sym_pairs / max(num_spikes, 1)
                patterns['star_score'] = star_sc
                patterns['star_symmetry'] = star_sym
                patterns['star_num_peaks'] = len(star_peaks) if hasattr(star_peaks, '__len__') else star_peaks
                
                # Decision logic with VERY STRICT thresholds
                # Star pattern detection (high symmetry + many peaks = AI)
                # Require VERY strong evidence to avoid false positives
                num_star_peaks = len(star_peaks) if hasattr(star_peaks, '__len__') else star_peaks
                if star_sym > 0.8 and num_star_peaks >= 10:  # VERY strict (was 0.7 and 8)
                    patterns['star_pattern'] = True
                    patterns['suspicion_score'] += 45
                    patterns['interpretations'].append(
                        f"⚠️ STAR PATTERN EXTREM: {num_star_peaks} vârfuri cu simetrie 180° = {star_sym:.1%} (CERT AI!)"
                    )
                elif star_sym > 0.7 and num_star_peaks >= 8:
                    patterns['suspicion_score'] += 20
                    patterns['interpretations'].append(
                        f"⚠️ Star pattern puternic: {num_star_peaks} vârfuri, simetrie {star_sym:.1%} - suspect AI"
                    )
                elif star_sym > 0.5 and num_star_peaks >= 6:
                    patterns['suspicion_score'] += 8
                    patterns['interpretations'].append(
                        f"ℹ️ Star pattern moderat: {num_star_peaks} vârfuri, simetrie {star_sym:.1%} - artefact minor"
                    )
                
                # Resampling detection (symmetric spike pairs) - VERY STRICT
                # Only AI creates VERY symmetric patterns
                if patterns['symmetry_ratio'] > 0.75 and num_spikes > 15:  # VERY strict (was 0.5 and 10)
                    patterns['periodic_spikes'] = True
                    patterns['suspicion_score'] += 40
                    patterns['interpretations'].append(
                        f"⚠️ RESAMPLING EXTREM: {sym_pairs} perechi din {num_spikes} spike-uri ({patterns['symmetry_ratio']:.1%}) - CERT AI"
                    )
                elif patterns['symmetry_ratio'] > 0.6 and num_spikes > 12:
                    patterns['suspicion_score'] += 15
                    patterns['interpretations'].append(
                        f"⚠️ Spike-uri foarte simetrice: {sym_pairs}/{num_spikes} = {patterns['symmetry_ratio']:.1%} - posibil AI"
                    )
                elif patterns['symmetry_ratio'] > 0.45 and num_spikes > 8:
                    patterns['suspicion_score'] += 5
                    patterns['interpretations'].append(
                        f"ℹ️ Simetrie moderată: {sym_pairs}/{num_spikes} = {patterns['symmetry_ratio']:.1%}"
                    )
                
                # Combined high-confidence AI detection
                if patterns['star_pattern'] and patterns['periodic_spikes']:
                    patterns['interpretations'].append(
                        "🚨 DUBLU INDICATOR: Star pattern + Resampling = foarte probabil AI-generated!"
                    )
                
            except Exception as e:
                # Fallback to original method if advanced analysis fails
                pass
        
        # 1. FALLBACK: Original Star-like Pattern Detection (if advanced fails)
        # AI models create periodic structures in frequency domain
        # Check for symmetrical bright spots
        if magnitude_2d is not None and not patterns['star_pattern']:
            center_y, center_x = np.array(magnitude_2d.shape) // 2
            radius = min(center_y, center_x) // 3
            
            # Sample points in circular pattern
            angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
            intensities = []
            for angle in angles:
                y = int(center_y + radius * np.sin(angle))
                x = int(center_x + radius * np.cos(angle))
                if 0 <= y < magnitude_2d.shape[0] and 0 <= x < magnitude_2d.shape[1]:
                    intensities.append(magnitude_2d[y, x])
            
            if len(intensities) >= 8:
                # Check for periodic pattern (star-like) - VERY STRICT thresholds
                std_intensities = np.std(intensities)
                mean_intensities = np.mean(intensities)
                if std_intensities > 0 and mean_intensities > 0:
                    coefficient_of_variation = std_intensities / mean_intensities
                    # VERY STRICT thresholds - only strong AI artifacts trigger
                    # Real images should NOT trigger this (CV < 1.2)
                    if coefficient_of_variation > 1.5:  # VERY stricter (was 0.8)
                        patterns['star_pattern'] = True
                        patterns['suspicion_score'] += 35
                        patterns['interpretations'].append("⚠️ Pattern stea EXTREM după Hanning - FOARTE PROBABIL AI")
                    elif coefficient_of_variation > 1.0:  # Stricter (was 0.5)
                        patterns['suspicion_score'] += 15
                        patterns['interpretations'].append("⚠️ Pattern simetric puternic - suspect AI")
                    elif coefficient_of_variation > 0.7:
                        patterns['suspicion_score'] += 5
                        patterns['interpretations'].append("ℹ️ Pattern simetric slab - posibil natural")
        
        # 2. FALLBACK: Periodic Spikes Detection (1D PSD) - VERY STRICT
        # AI creates VERY regular patterns, real photos are chaotic
        if not patterns['periodic_spikes']:
            peaks, _ = find_peaks(psd1D, prominence=np.max(psd1D)*0.15)  # Higher prominence
            if len(peaks) > 10:  # Need MANY peaks (was 5)
                peak_distances = np.diff(peaks)
                if len(peak_distances) > 8:  # Need consistent spacing
                    distance_std = np.std(peak_distances)
                    distance_mean = np.mean(peak_distances)
                    # VERY strict: CV must be < 0.15 (was 0.3)
                    if distance_mean > 0 and distance_std / distance_mean < 0.15:
                        patterns['periodic_spikes'] = True
                        patterns['suspicion_score'] += 30
                        patterns['interpretations'].append("⚠️ Spike-uri EXTREM de periodice - FOARTE PROBABIL AI")
                    elif distance_std / distance_mean < 0.25:
                        patterns['suspicion_score'] += 10
                        patterns['interpretations'].append("ℹ️ Spike-uri oarecum regulate - posibil minor artefact")
        
        # 3. Unnatural Decay Detection
        # AI images often have too-perfect exponential decay
        n = len(psd1D)
        mid_point = n // 2
        
        # Check if decay is TOO smooth (R² very close to 1)
        x_decay = np.log10(np.arange(mid_point, n) + 1).reshape(-1, 1)
        y_decay = np.log10(psd1D[mid_point:])
        
        from sklearn.linear_model import LinearRegression
        model_decay = LinearRegression().fit(x_decay, y_decay)
        r_squared = model_decay.score(x_decay, y_decay)
        
        if r_squared > 0.995:  # EXTREMELY perfect fit - only AI does this
            patterns['unnatural_decay'] = True
            patterns['suspicion_score'] += 25
            patterns['interpretations'].append("⚠️ Decay PERFECT matematic (R²={:.3f}) - FOARTE PROBABIL AI".format(r_squared))
        elif r_squared > 0.98:  # Very suspicious
            patterns['suspicion_score'] += 12
            patterns['interpretations'].append("⚠️ Decay foarte uniform (R²={:.3f}) - suspect AI".format(r_squared))
        elif r_squared > 0.95:  # Moderately suspicious
            patterns['suspicion_score'] += 5
            patterns['interpretations'].append("ℹ️ Decay uniform (R²={:.3f}) - posibil compresie sau AI".format(r_squared))
        elif r_squared < 0.90:  # Natural variation
            patterns['interpretations'].append("✅ Decay natural cu variație (R²={:.3f}) - caracteristic poze reale".format(r_squared))
        
        # 4. High-frequency drop-off check - STRICTER thresholds
        tail_90 = int(0.9 * n)
        high_freq_mean = np.mean(psd1D[tail_90:])
        mid_freq_mean = np.mean(psd1D[mid_point:tail_90])
        
        if mid_freq_mean > 0:
            drop_ratio = high_freq_mean / mid_freq_mean
            if drop_ratio < 0.001:  # EXTREMELY dramatic drop - only AI
                patterns['suspicion_score'] += 20
                patterns['interpretations'].append("⚠️ Drop EXTREM în HF (ratio={:.4f}) - AI pierde complet detaliile fine".format(drop_ratio))
            elif drop_ratio < 0.005:  # Very dramatic drop
                patterns['suspicion_score'] += 10
                patterns['interpretations'].append("⚠️ Drop sever în HF (ratio={:.4f}) - posibil AI".format(drop_ratio))
            elif drop_ratio < 0.015:  # Moderate drop
                patterns['suspicion_score'] += 3
                patterns['interpretations'].append("ℹ️ Drop moderat în HF (ratio={:.4f}) - compresie normală".format(drop_ratio))
            else:
                patterns['interpretations'].append("✅ Frecvențe înalte păstrate (ratio={:.4f}) - natural".format(drop_ratio))
        
    except Exception as e:
        patterns['interpretations'].append(f"ℹ️ Eroare în analiza pattern-urilor: {e}")
    
    return patterns

# Metadata Extraction Function
def extract_metadata(img_path):
    """Extrage metadata EXIF din imagine"""
    metadata_info = {
        'has_exif': False,
        'datetime': None,
        'make': None,
        'model': None,
        'software': None,
        'gps': False,
        'completeness_score': 0.0
    }
    
    try:
        exif_dict = piexif.load(img_path)
        score = 0
        
        # Check DateTime
        if piexif.ExifIFD.DateTimeOriginal in exif_dict.get('Exif', {}):
            metadata_info['datetime'] = exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal].decode('utf-8')
            score += 20
        elif piexif.ImageIFD.DateTime in exif_dict.get('0th', {}):
            metadata_info['datetime'] = exif_dict['0th'][piexif.ImageIFD.DateTime].decode('utf-8')
            score += 15
        
        # Check Make/Model
        if piexif.ImageIFD.Make in exif_dict.get('0th', {}):
            metadata_info['make'] = exif_dict['0th'][piexif.ImageIFD.Make].decode('utf-8')
            score += 25
        if piexif.ImageIFD.Model in exif_dict.get('0th', {}):
            metadata_info['model'] = exif_dict['0th'][piexif.ImageIFD.Model].decode('utf-8')
            score += 25
        
        # Check Software
        if piexif.ImageIFD.Software in exif_dict.get('0th', {}):
            metadata_info['software'] = exif_dict['0th'][piexif.ImageIFD.Software].decode('utf-8')
            score += 15
        
        # Check GPS
        if exif_dict.get('GPS'):
            metadata_info['gps'] = True
            score += 20
        
        metadata_info['completeness_score'] = score
        metadata_info['has_exif'] = score > 0
        
    except Exception:
        pass
    
    return metadata_info

# Feature Extraction Function
def extract_all_features(img_pil, img_path=None):
    """Extract toate features din imagine"""
    try:
        im = img_pil.convert('RGB')
        features = {}
        
        # Metadata analysis (daca avem path)
        metadata_info = None
        if img_path:
            metadata_info = extract_metadata(img_path)
        
        # 1. ELA
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = tmp.name
            im.save(tmp_path, 'JPEG', quality=90)
            resaved = Image.open(tmp_path)
            ela_im = ImageChops.difference(im, resaved)
            ela_array = np.array(ela_im)
            gray_ela = np.mean(ela_array, axis=2)
            features['ela_std'] = float(np.std(gray_ela))
            features['ela_mean'] = float(np.mean(gray_ela))
            features['ela_max'] = float(np.max(gray_ela))
            resaved.close()
            try:
                os.unlink(tmp_path)
            except:
                pass
        except Exception as e:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            features['ela_std'] = 0.0
            features['ela_mean'] = 0.0
            features['ela_max'] = 0.0
        
        # 2. FFT/PSD - CORECT MATEMATIC cu Media Azimutală (Azimuthal Average)
        # Aceasta este metoda CORECTĂ pentru detectarea deepfake prin analiza de frecvență
        img_gray_orig = np.array(im.convert('L')).astype(np.float64)
        
        # Resize to standard size for consistency (512x512 optimal for FFT)
        max_size = 512
        h, w = img_gray_orig.shape
        if h != max_size or w != max_size:
            # Use INTER_AREA for downsampling (best for anti-aliasing)
            img_gray = cv2.resize(img_gray_orig, (max_size, max_size), interpolation=cv2.INTER_AREA)
        else:
            img_gray = img_gray_orig
        
        # Normalize to [0, 1]
        img_normalized = img_gray / 255.0
        
        # ✅ JPEG mitigation (light blur to suppress 8×8 block artifacts)
        # This doesn't destroy resampling artifacts (which are global)
        img_normalized = cv2.GaussianBlur(img_normalized, (3, 3), 0.7)
        # === METHOD 2: 2D FFT with CORRECT Azimuthal Average for Radial PSD ===
        # This is the MATHEMATICALLY CORRECT approach for deepfake detection
        
        # Apply 2D Hanning window
        h_win, w_win = img_normalized.shape
        window_2d = np.outer(np.hanning(h_win), np.hanning(w_win))
        img_windowed = img_normalized * window_2d
        
        # 2D FFT
        f_2d = np.fft.fft2(img_windowed)
        fshift_2d = np.fft.fftshift(f_2d)  # Shift zero-frequency to center
        magnitude_2d = np.abs(fshift_2d)
        
        # Calculate Power Spectral Density 2D: |F(u,v)|^2
        psd_2d = magnitude_2d ** 2
        
        # === CORRECT RADIAL PROFILE using Azimuthal Average ===
        # This transforms the 2D spectrum into 1D radial profile
        radial_profile = azimuthalAverage(psd_2d, center=None)
        
        # Skip DC (index 0) and first few bins (low freq dominated)
        skip_radial = max(3, len(radial_profile) // 100)
        radial_profile_trimmed = radial_profile[skip_radial:]
        
        # Convert to dB scale (standard for PSD visualization)
        # 10*log10 for power (not 20*log10 which is for amplitude)
        psd_radial_db = 10 * np.log10(radial_profile_trimmed + 1e-10)
        
        # Filter out invalid values
        valid_mask = np.isfinite(psd_radial_db) & (psd_radial_db > -100)
        psd_radial_db = psd_radial_db[valid_mask]
        
        # This is the CORRECT 1D PSD for analysis and visualization
        psd1D = psd_radial_db  # Use azimuthal average as primary PSD
        
        # === 2D Magnitude for visualization (clipped for better display) ===
        magnitude_2d_db = 20 * np.log10(magnitude_2d + 1e-10)
        magnitude_clipped = np.clip(magnitude_2d_db,
                                   np.percentile(magnitude_2d_db, 1),
                                   np.percentile(magnitude_2d_db, 99.5))
        magnitude_2d_for_pattern = magnitude_clipped
        
        # === NUMERICAL FEATURES FROM RADIAL PSD ===
        
        if len(psd1D) >= 50:
            n_freq = len(psd1D)
            
            # === NUMERICAL FEATURES FOR AI DECISION (not just visualization) ===
            
            # Feature 1: Overall Slope (Power Law Decay)
            # Real images follow ~1/f law, AI images deviate
            freqs_log = np.log10(np.arange(1, n_freq + 1))
            psd_for_slope = psd1D.copy()
            slope_model = LinearRegression().fit(freqs_log.reshape(-1, 1), psd_for_slope)
            overall_slope = float(slope_model.coef_[0])
            slope_r2 = float(slope_model.score(freqs_log.reshape(-1, 1), psd_for_slope))
            
            # Feature 2: Energy Ratio (Low vs High Frequencies)
            cutoff = int(0.6 * n_freq)
            low_freq_energy = np.sum(10 ** (psd1D[:cutoff] / 10))  # Convert back from dB
            high_freq_energy = np.sum(10 ** (psd1D[cutoff:] / 10))
            energy_ratio = high_freq_energy / (low_freq_energy + 1e-8)
            
            # Traditional log ratio for backward compatibility
            low_p = np.mean(psd1D[:cutoff])
            high_p = np.mean(psd1D[cutoff:])
            features['log_hf_ratio'] = float(high_p - low_p)  # dB difference
            
            # Store new numerical features
            features['fft_slope'] = overall_slope  # Should be negative for natural images
            features['fft_slope_r2'] = slope_r2  # Fit quality
            features['fft_energy_ratio'] = float(np.log10(energy_ratio + 1e-8))
            # Feature 3: Tail Gradients (High-frequency decay rates)
            # These are NUMERICAL features, not just for visualization
            for zone, pct in [('70', 0.7), ('80', 0.8), ('90', 0.9)]:
                start_idx = int(pct * n_freq)
                if len(psd1D[start_idx:]) > 5:
                    x_tail = np.log10(np.arange(start_idx, n_freq) + 1).reshape(-1, 1)
                    y_tail = psd1D[start_idx:]  # Already in dB
                    tail_model = LinearRegression().fit(x_tail, y_tail)
                    features[f'tail_{zone}'] = float(tail_model.coef_[0])  # Slope in dB/decade
                else:
                    features[f'tail_{zone}'] = 0.0
            
            # Feature 4: Spike Detection (Periodic Artifacts)
            # Look for sudden jumps in PSD that indicate AI grid patterns
            psd_diff = np.diff(psd1D)
            spike_threshold = np.std(psd_diff) * 3  # 3-sigma rule
            num_spikes = np.sum(np.abs(psd_diff) > spike_threshold)
            features['fft_num_spikes'] = int(num_spikes)
            
        else:
            # Not enough frequency bins - set defaults
            features.update({
                'log_hf_ratio': 0.0,
                'fft_slope': 0.0,
                'fft_slope_r2': 0.0,
                'fft_energy_ratio': 0.0,
                'tail_70': 0.0,
                'tail_80': 0.0,
                'tail_90': 0.0,
                'fft_num_spikes': 0
            })
        
        # === END FFT NUMERICAL FEATURES ===
        # These features (slope, energy_ratio, tail gradients, spikes) are used by ML model
        # The visualizations (graphs) are ONLY for human interpretation
        # AI decision is based on the numerical array psd1D, not the JPEG graph
        
        # 3. Wavelet
        coeffs = pywt.dwt2(img_gray, 'db4')
        cA, (cH, cV, cD) = coeffs
        features['wavelet_cH_std'] = float(np.std(cH))
        features['wavelet_cV_std'] = float(np.std(cV))
        features['wavelet_cD_std'] = float(np.std(cD))
        features['wavelet_energy'] = float(np.sum(cH**2))
        
        # 4. LBP
        lbp = local_binary_pattern(img_gray, P=24, R=3, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)
        features['lbp_entropy'] = float(entropy(hist + 1e-10))
        features['lbp_std'] = float(np.std(lbp))
        
        # 5. Gradient
        gx = np.gradient(img_gray, axis=1)
        gy = np.gradient(img_gray, axis=0)
        magnitude_grad = np.sqrt(gx**2 + gy**2)
        features['gradient_mean'] = float(np.mean(magnitude_grad))
        features['gradient_std'] = float(np.std(magnitude_grad))
        features['gradient_max'] = float(np.max(magnitude_grad))
        features['gradient_skew'] = float(skew(magnitude_grad.ravel()))
        
        # 6. Color
        r, g, b = im.split()
        for channel, name in [(r, 'red'), (g, 'green'), (b, 'blue')]:
            arr = np.array(channel)
            features[f'{name}_mean'] = float(np.mean(arr))
            features[f'{name}_std'] = float(np.std(arr))
        
        return features, psd1D if 'psd1D' in locals() else None, metadata_info, magnitude_2d_for_pattern if 'magnitude_2d_for_pattern' in locals() else None
    except Exception as e:
        st.error(f"Eroare extracție features: {e}")
        return None, None, None, None

# Load Model
@st.cache_resource
def load_model():
    """Încarcă modelul antrenat"""
    try:
        with open('final_model.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['scaler'], data['feature_names'], data
    except:
        st.warning("⚠️ Model final_model.pkl nu a fost găsit. Folosesc predicție bazată pe reguli.")
        return None, None, None, {}

model, scaler, feature_names, model_data = load_model()

# Main Upload
st.markdown("## 📤 Încarcă Imagine pentru Analiză")
uploaded_file = st.file_uploader(
    "Selectează o imagine (JPG, PNG, JPEG)",
    type=['jpg', 'jpeg', 'png'],
    help="Încarcă o imagine pentru a detecta dacă este generată de AI"
)

if uploaded_file is not None:
    # Load image
    original_image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(original_image, caption=f"Imagine Încărcată: {uploaded_file.name}", width='stretch')
        st.write(f"**Dimensiuni:** {original_image.size[0]} x {original_image.size[1]} px")
    
    with col2:
        with st.spinner("🔍 Analiză în curs... (extragere features + metadata)"):
            # Save temp file for metadata extraction
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                temp_path = tmp.name
            original_image.save(temp_path, 'JPEG', quality=95)
            
            # Extract features
            features_dict, psd1D, metadata_info, magnitude_2d = extract_all_features(original_image, temp_path)
            
            # Analyze FFT patterns (Hany Farid approach)
            fft_patterns = analyze_fft_patterns(psd1D, magnitude_2d)
            
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
            
            if features_dict is None:
                st.error("❌ Eroare la procesarea imaginii")
                st.stop()
            
            # Metadata adjustment
            metadata_boost = 0
            metadata_override = False
            phone_photo_pattern = False
            skip_ml_model = False
            
            # Detect phone photo pattern (stronger detection)
            ela_very_low = features_dict['ela_std'] < 2.5  # Increased threshold
            ela_mean_low = features_dict['ela_mean'] < 1.5
            gradient_normal = 3.0 < features_dict['gradient_mean'] < 10.0  # Wider range
            wavelet_high = features_dict['wavelet_energy'] > 4000000  # Lower threshold
            color_variance_normal = 40 < features_dict['red_std'] < 80  # Color diversity check
            
            # Phone photos: very low ELA (aggressive compression) + normal gradients + high wavelet energy
            if ela_very_low and ela_mean_low and (gradient_normal or wavelet_high):
                phone_photo_pattern = True
            
            # ============================================
            # FFT FORENSIC SCORING SYSTEM
            # All FFT indicators contribute weighted percentages
            # ============================================
            fft_forensic_score = 0  # 0 = REAL, 100 = AI
            fft_contributions = {}  # Track each component's contribution
            
            # 1. Radial Whitening Spike Analysis (25% weight)
            spike_score = 0
            if 'suspicion_score' in fft_patterns:
                base_suspicion = fft_patterns['suspicion_score']
                if base_suspicion >= 50:
                    spike_score = 25  # High AI probability
                elif base_suspicion >= 30:
                    spike_score = 15  # Moderate AI probability
                elif base_suspicion >= 10:
                    spike_score = 5   # Low AI probability
                fft_contributions['Whitened Spikes'] = spike_score
            
            # 2. 2D Spectrum Star Pattern (20% weight)
            star_2d_score = 0
            if fft_patterns.get('star_pattern', False):
                star_2d_score = 20  # Strong AI indicator
                fft_contributions['2D Star Pattern'] = star_2d_score
            
            # 3. Angular Energy Star Peaks (20% weight)
            # Compute if magnitude_2d is available
            angular_star_score = 0
            if magnitude_2d is not None:
                try:
                    whitened = radial_whitening_fast(magnitude_2d, exclude_dc_radius=5, r_min_frac=0.05)
                    ang_energy = angular_energy_signature(whitened, r_min_frac=0.1, r_max_frac=0.45)
                    star_sc, star_sym, star_peaks_idx = star_score_with_symmetry(ang_energy)
                    num_peaks = len(star_peaks_idx) if hasattr(star_peaks_idx, '__len__') else (star_peaks_idx if isinstance(star_peaks_idx, int) else 0)
                    
                    # High star symmetry + many peaks = strong AI
                    if star_sym > 0.7 and num_peaks >= 8:
                        angular_star_score = 20  # Very high AI probability
                    elif star_sym > 0.5 and num_peaks >= 5:
                        angular_star_score = 12  # High AI probability
                    elif star_sym > 0.3 or num_peaks >= 3:
                        angular_star_score = 5   # Moderate AI probability
                    
                    fft_contributions['Angular Star'] = angular_star_score
                except:
                    pass
            
            # 4. Spike Symmetry Ratio (15% weight)
            symmetry_score = 0
            if 'symmetry_ratio' in fft_patterns:
                sym_ratio = fft_patterns['symmetry_ratio']
                if sym_ratio > 0.7:
                    symmetry_score = 15  # Very high symmetry = AI
                elif sym_ratio > 0.5:
                    symmetry_score = 10  # High symmetry = likely AI
                elif sym_ratio > 0.3:
                    symmetry_score = 5   # Moderate symmetry
                fft_contributions['Spike Symmetry'] = symmetry_score
            
            # 5. Periodic Spikes Detection (10% weight)
            periodic_score = 0
            if fft_patterns.get('periodic_spikes', False):
                periodic_score = 10  # Regular periodic patterns = AI
                fft_contributions['Periodic Spikes'] = periodic_score
            
            # 6. Unnatural Decay Pattern (10% weight)
            decay_score = 0
            if fft_patterns.get('unnatural_decay', False):
                decay_score = 10  # Sharp frequency cutoff = AI
                fft_contributions['Unnatural Decay'] = decay_score
            
            # Calculate total FFT forensic score (0-100)
            fft_forensic_score = spike_score + star_2d_score + angular_star_score + symmetry_score + periodic_score + decay_score
            fft_forensic_score = min(100, fft_forensic_score)  # Cap at 100
            
            # Convert to penalty/bonus - VERY STRICT THRESHOLDS
            # Low scores should have ZERO or NEGATIVE impact
            fft_suspicion_penalty = 0
            if fft_forensic_score >= 70:
                fft_suspicion_penalty = 45  # Extremely high AI evidence
            elif fft_forensic_score >= 50:
                fft_suspicion_penalty = 40  # Very high AI evidence
            elif fft_forensic_score >= 35:
                fft_suspicion_penalty = 30  # High AI evidence
            elif fft_forensic_score >= 25:
                fft_suspicion_penalty = 20  # Moderate-high AI evidence
            elif fft_forensic_score >= 18:
                fft_suspicion_penalty = 10  # Moderate AI evidence
            elif fft_forensic_score >= 12:
                fft_suspicion_penalty = 0   # Low - NEUTRAL (no impact)
            elif fft_forensic_score <= 8:
                fft_suspicion_penalty = -20  # Very clean FFT → STRONG boost REAL
            
            # CRITICAL: If strong evidence of real photo (metadata + phone pattern), skip ML entirely
            
            if metadata_info and metadata_info['has_exif']:
                score = metadata_info['completeness_score']
                
                if phone_photo_pattern and score >= 40:
                    skip_ml_model = True
                    prob_real = max(70.0, 90.0 - fft_suspicion_penalty)
                    prob_fake = 100.0 - prob_real
                elif score >= 70:
                    metadata_boost = 40
                    metadata_override = True
                elif score >= 50:
                    metadata_boost = 35
                    metadata_override = True
                elif score >= 30:
                    metadata_boost = 25
                    metadata_override = True
                else:
                    metadata_boost = 15
            elif phone_photo_pattern:
                # Phone pattern without metadata - moderate boost
                metadata_boost = 20
            
            # Predict (only if not skipping ML)
            if not skip_ml_model and model is not None and scaler is not None:
                # Prepare features
                feature_vector = np.array([[features_dict[name] for name in feature_names]])
                feature_vector_scaled = scaler.transform(feature_vector)
                
                # Apply feature selection if model has selector
                selector = model_data.get('selector', None)
                if selector is not None:
                    feature_vector_scaled = selector.transform(feature_vector_scaled)
                
                # Prediction
                prediction = model.predict(feature_vector_scaled)[0]
                probability = model.predict_proba(feature_vector_scaled)[0]
                
                # Base probabilities from ML model (0=REAL, 1=FAKE)
                prob_real = probability[0] * 100
                prob_fake = probability[1] * 100
                
                # FORENSIC RULE-BASED OVERRIDES (strongest evidence)
                # These rules override ML when forensic evidence is clear
                
                # Rule 1: Very low ELA + high wavelet energy = likely REAL phone photo
                if features_dict['ela_std'] < 2.0 and features_dict['wavelet_energy'] > 5000000:
                    if metadata_info and metadata_info['has_exif']:
                        # Strong evidence of real phone photo
                        prob_real = max(prob_real, 80.0)
                        prob_fake = 100 - prob_real
                
                # Rule 2: High ELA + low tail decay = likely AI
                if features_dict['ela_std'] > 6.0 and abs(features_dict['tail_80']) < 3.5:
                    # High noise but sharp frequency cutoff = AI signature
                    prob_fake = max(prob_fake, 70.0)
                    prob_real = 100 - prob_fake
                
                # Rule 3: Very uniform ELA (< 1.5) without metadata = suspicious
                if features_dict['ela_std'] < 1.5 and not (metadata_info and metadata_info['has_exif']):
                    prob_fake = max(prob_fake, 75.0)
                    prob_real = 100 - prob_fake
                
                # CORRECT ORDER OF ADJUSTMENTS:
                # Step 1: FFT forensics (BIDIRECTIONAL: increases AI prob OR decreases it)
                if fft_suspicion_penalty != 0:
                    if fft_suspicion_penalty > 0:
                        # High FFT score → increase AI probability
                        prob_fake = min(95, prob_fake + fft_suspicion_penalty)
                        prob_real = 100 - prob_fake
                    else:
                        # Low FFT score → increase REAL probability (bonus)
                        prob_real = min(95, prob_real + abs(fft_suspicion_penalty))
                        prob_fake = 100 - prob_real
                
                # Step 2: Metadata boost (increases confidence in REAL if good metadata)
                if metadata_boost > 0:
                    prob_real = min(98, prob_real + metadata_boost)
                    prob_fake = 100 - prob_real
                
                # Step 3: Phone photo pattern override (strongest evidence)
                if phone_photo_pattern:
                    if metadata_info and metadata_info['has_exif'] and metadata_info['completeness_score'] >= 40:
                        prob_real = max(85, prob_real)
                        prob_fake = 100 - prob_real
                        metadata_override = True
                    elif metadata_info and metadata_info['has_exif']:
                        prob_real = max(70, prob_real)
                        prob_fake = 100 - prob_real
                    else:
                        prob_real = min(prob_real + 10, 60)
                        prob_fake = 100 - prob_real
                
                # Step 4: Final metadata override for very strong evidence
                elif metadata_override and prob_fake > 50:
                    prob_real = max(75, prob_real)
                    prob_fake = 100 - prob_real
                
                # Final verdict
                is_fake = prob_fake > prob_real
                confidence = prob_fake if is_fake else prob_real
            
            elif not skip_ml_model:
                # Fallback if ML model not loaded
                # Fallback: Rule-based with metadata priority
                ela_score = features_dict['ela_std']
                tail_score = abs(features_dict['tail_80'])
                
                # Check metadata first
                if metadata_info and metadata_info['completeness_score'] >= 70:
                    # Strong metadata = likely REAL
                    is_fake = False
                    prob_real = min(85, 60 + metadata_info['completeness_score'] / 4)
                    prob_fake = 100 - prob_real
                else:
                    # Use forensic features
                    is_fake = ela_score < 5.0 or tail_score > 5.0
                    prob_fake = min(90, max(10, 50 + (5.0 - ela_score) * 5 + (tail_score - 3) * 10))
                    
                    # Apply metadata boost if available
                    if metadata_info and metadata_info['has_exif']:
                        prob_fake = max(10, prob_fake - metadata_info['completeness_score'] / 3)
                    
                    prob_real = 100 - prob_fake
                
                confidence = prob_fake if is_fake else prob_real
            
            # Final verdict (applies to all paths)
            if skip_ml_model:
                is_fake = False  # Already set as REAL
                confidence = prob_real
            else:
                is_fake = prob_fake > prob_real
                confidence = prob_fake if is_fake else prob_real
        
        # Display Verdict
        st.markdown("### 🎯 REZULTAT ANALIZĂ")
        
        if is_fake:
            st.markdown(f'<div class="verdict-fake">🚨 IMAGINE AI-GENERATĂ<br>Confidence: {confidence:.1f}%</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="verdict-real">✅ IMAGINE REALĂ<br>Confidence: {confidence:.1f}%</div>', unsafe_allow_html=True)
        
        # Metadata Info
        if metadata_info and metadata_info['has_exif']:
            st.markdown("### 🔐 Metadata EXIF Detectată")
            
            score = metadata_info['completeness_score']
            
            # Check for phone photo pattern (same logic as above)
            ela_very_low = features_dict['ela_std'] < 2.5
            ela_mean_low = features_dict['ela_mean'] < 1.5
            gradient_normal = 3.0 < features_dict['gradient_mean'] < 10.0
            wavelet_high = features_dict['wavelet_energy'] > 4000000
            phone_pattern = ela_very_low and ela_mean_low and (gradient_normal or wavelet_high)
            
            if phone_pattern and score >= 40:
                st.success(f"📱 **Poză cu Telefon Detectată!** Metadata: {score}/100")
                st.info(f"✅ ELA foarte mic ({features_dict['ela_std']:.2f}) este **NORMAL** pentru telefoane moderne cu compresie agresivă → Clasificare forțată ca REAL")
            elif phone_pattern:
                st.warning(f"📱 **Pattern Telefon Detectat** dar metadata incompletă ({score}/100) - suspiciune moderată")
            elif score >= 70:
                st.success(f"✅ Metadata Completă: {score}/100 - INDICATOR PUTERNIC DE IMAGINE REALĂ")
            elif score >= 50:
                st.success(f"✅ Metadata Parțială: {score}/100")
            else:
                st.info(f"ℹ️ Metadata Minimă: {score}/100")
            
            meta_col1, meta_col2 = st.columns(2)
            with meta_col1:
                if metadata_info['make']:
                    st.write(f"📱 **Device:** {metadata_info['make']}")
                if metadata_info['model']:
                    st.write(f"📷 **Model:** {metadata_info['model']}")
            with meta_col2:
                if metadata_info['datetime']:
                    st.write(f"🕐 **Data:** {metadata_info['datetime']}")
                if metadata_info['gps']:
                    st.write(f"🌍 **GPS:** Da")
                if metadata_info['software']:
                    st.write(f"💻 **Software:** {metadata_info['software']}")
        else:
            st.markdown("### ⚠️ Metadata EXIF")
            st.error("❌ Nicio metadata EXIF găsită")
        
        # FFT Forensic Score Breakdown
        if 'fft_forensic_score' in locals() and fft_forensic_score > 0:
            st.markdown("### 🔬 FFT Forensic Analysis")
            st.markdown(f"**Total FFT AI Score: {fft_forensic_score:.1f}/100**")
            
            if 'fft_contributions' in locals() and fft_contributions:
                # Create visual breakdown
                cols_fft = st.columns(len(fft_contributions))
                for idx, (name, score) in enumerate(fft_contributions.items()):
                    with cols_fft[idx]:
                        # Color based on score
                        if score >= 15:
                            st.markdown(f"**🔴 {name}**")
                        elif score >= 8:
                            st.markdown(f"**🟠 {name}**")
                        else:
                            st.markdown(f"**🟡 {name}**")
                        st.metric("", f"+{score:.0f}%")
                
                # Overall interpretation with STRICT thresholds
                st.progress(fft_forensic_score / 100)
                if fft_forensic_score >= 70:
                    st.error("🔥 **FFT Analysis: EXTREME AI Indicators** - Certitudine foarte ridicată (+45% AI)")
                elif fft_forensic_score >= 50:
                    st.error("⚠️ **FFT Analysis: VERY STRONG AI Indicators** - Evidentă puternică (+40% AI)")
                elif fft_forensic_score >= 35:
                    st.error("🔴 **FFT Analysis: STRONG AI Indicators** - Pattern-uri clare AI (+30% AI)")
                elif fft_forensic_score >= 25:
                    st.warning("⚠️ **FFT Analysis: MODERATE-HIGH AI** - Multiple anomalii (+20% AI)")
                elif fft_forensic_score >= 18:
                    st.warning("🟠 **FFT Analysis: MODERATE AI** - Câteva anomalii (+10% AI)")
                elif fft_forensic_score >= 12:
                    st.info("🟡 **FFT Analysis: LOW/NEUTRAL** - Anomalii minore (0% impact)")
                elif fft_forensic_score <= 8:
                    st.success("✅ **FFT Analysis: VERY CLEAN** - Spectru natural perfect (-20% AI, +20% REAL)")
                else:
                    st.success("✅ **FFT Analysis: CLEAN** - Spectru aproape natural")
                
                # Detailed breakdown
                with st.expander("📋 Detailed FFT Contributions"):
                    st.markdown("**How each FFT indicator contributes to AI detection:**")
                    for name, score in fft_contributions.items():
                        st.write(f"• **{name}**: +{score:.1f}% AI probability")
                    st.markdown(f"\n**Total FFT Score**: {fft_forensic_score:.1f}/100")
                    st.markdown(f"**Direct Impact**: +{fft_suspicion_penalty}% AI probability")
                    st.markdown(f"**Impact Level**: {'VERY HIGH' if fft_suspicion_penalty >= 30 else 'HIGH' if fft_suspicion_penalty >= 20 else 'MODERATE' if fft_suspicion_penalty >= 10 else 'LOW'}")
        
        # Probability bars
        st.markdown("### 📊 Probabilități Finale")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("🟢 REAL", f"{prob_real:.1f}%")
            st.progress(prob_real / 100)
        with col_b:
            st.metric("🔴 AI-GENERATE", f"{prob_fake:.1f}%")
            st.progress(prob_fake / 100)
    
    # ============================================
    # ANALIZA GRAFICĂ CU GEMINI AI
    # ============================================
    st.markdown("---")
    st.markdown("## 📊 Analiză Grafică cu AI")
    
    # Get API key for Gemini
    api_key_gemini = st.text_input("🔑 Gemini API Key (opțional pentru interpretare AI)", 
                                    type="password", 
                                    help="Obține de la: https://aistudio.google.com/apikey",
                                    key="gemini_api_key_input")
    
    # Storage for all interpretations
    gemini_interpretations = {}
    
    # ============================================
    # GRAFIC 1: FFT RADIAL PSD
    # ============================================
    st.markdown("### 📈 1. FFT Radial Power Spectral Density")
    
    if psd1D is not None and fft_patterns:
        # ============================================
        # SECTION 1: IMPROVED METRICS DISPLAY
        # ============================================
        st.markdown("#### 🔬 Advanced Forensics Metrics")
        
        # Display new metrics in organized columns
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.markdown("**🎯 Spike Detection**")
            st.metric("Total Spikes", fft_patterns.get('num_spikes', 0))
            st.metric("Symmetric Pairs", fft_patterns.get('symmetric_pairs', 0))
            sym_ratio = fft_patterns.get('symmetry_ratio', 0.0) * 100
            st.metric("Symmetry Ratio", f"{sym_ratio:.1f}%", 
                     delta="High" if sym_ratio > 50 else "Normal",
                     delta_color="inverse" if sym_ratio > 50 else "off")
        
        with metric_col2:
            st.markdown("**⭐ Star Pattern**")
            st.metric("Star Score", f"{fft_patterns.get('star_score', 0.0):.2f}")
            star_sym = fft_patterns.get('star_symmetry', 0.0) * 100
            st.metric("180° Symmetry", f"{star_sym:.1f}%",
                         delta="High" if star_sym > 70 else "Normal",
                         delta_color="inverse" if star_sym > 70 else "off")
            st.metric("Angular Peaks", fft_patterns.get('star_num_peaks', 0))
            
            with metric_col3:
                st.markdown("**📊 Overall Suspicion**")
                suspicion = fft_patterns['suspicion_score']
                st.metric("Suspicion Score", f"{suspicion}/100")
                
                # Classification based on metrics
                if fft_patterns.get('star_symmetry', 0) > 0.7 and fft_patterns.get('star_num_peaks', 0) >= 8:
                    verdict = "🚨 HIGH AI Risk"
                    verdict_color = "🔴"
                elif fft_patterns.get('symmetry_ratio', 0) > 0.5 and fft_patterns.get('num_spikes', 0) > 10:
                    verdict = "⚠️ Resampling Detected"
                    verdict_color = "🟠"
                elif suspicion >= 30:
                    verdict = "⚠️ Suspicious"
                    verdict_color = "🟡"
                else:
                    verdict = "✅ Normal"
                    verdict_color = "🟢"
                
                st.markdown(f"### {verdict_color} {verdict}")
            
            st.markdown("---")
            
            # ============================================
            # SECTION 2: DETAILED INTERPRETATION
            # ============================================
            st.markdown("#### 💡 Interpretări Detaliate")
            
            # Create expandable sections for each type of analysis
            with st.expander("🔍 Spike Analysis (Fix #3: Symmetry Checking)", expanded=True):
                num_spikes = fft_patterns.get('num_spikes', 0)
                sym_pairs = fft_patterns.get('symmetric_pairs', 0)
                sym_ratio = fft_patterns.get('symmetry_ratio', 0.0)
                
                if num_spikes == 0:
                    st.info("✅ Nu s-au detectat spike-uri spectrale - imagine netedă, fără artefacte periodice")
                else:
                    st.write(f"**Detectate:** {num_spikes} spike-uri în spectrul FFT")
                    st.write(f"**Perechi simetrice:** {sym_pairs} ({sym_ratio:.1%} din total)")
                    
                    if sym_ratio > 0.5:
                        st.error(f"""
                        🚨 **RESAMPLING SUSPECT DETECTAT**
                        - {sym_ratio:.1%} din spike-uri sunt simetrice față de centru
                        - Acest pattern indică redimensionare/interpolare (semn tipic AI)
                        - Imaginile reale nu prezintă astfel de simetrii perfecte
                        """)
                    elif sym_ratio > 0.3:
                        st.warning(f"""
                        ⚠️ **Artefacte de interpolare moderate**
                        - {sym_ratio:.1%} simetrie - posibil resize/crop
                        - Monitorizați împreună cu alte metrici
                        """)
                    else:
                        st.success("✅ Spike-urile nu sunt simetrice - pattern natural")
            
            with st.expander("⭐ Star Pattern Analysis (Fix #4: 180° Symmetry)", expanded=True):
                star_score = fft_patterns.get('star_score', 0.0)
                star_sym = fft_patterns.get('star_symmetry', 0.0)
                star_peaks = fft_patterns.get('star_num_peaks', 0)
                
                st.write(f"**Star Score:** {star_score:.2f}")
                st.write(f"**Simetrie 180°:** {star_sym:.1%}")
                st.write(f"**Peak-uri angulare:** {star_peaks}")
                
                if star_sym > 0.7 and star_peaks >= 8:
                    st.error(f"""
                    🚨 **STAR PATTERN PUTERNIC - INDICATOR AI**
                    - {star_peaks} peak-uri cu simetrie rotațională de {star_sym:.1%}
                    - Pattern-ul "stea" este caracteristic procesării AI (resampling GAN)
                    - Imaginile reale nu generează astfel de pattern-uri radiale perfecte
                    - **Concluzie: FOARTE PROBABIL AI-GENERATED**
                    """)
                elif star_sym > 0.5 and star_peaks >= 6:
                    st.warning(f"""
                    ⚠️ **Star pattern moderat detectat**
                    - {star_peaks} peak-uri, simetrie {star_sym:.1%}
                    - Posibil artefact de generare sau procesare intensă
                    """)
                elif star_peaks >= 4:
                    st.info(f"""
                    ℹ️ **Pattern angular detectat**
                    - {star_peaks} peak-uri, dar simetrie scăzută ({star_sym:.1%})
                    - Poate fi grid natural (gard, fereastră) - NU AI
                    - Fix #4 a prevenit falsa pozitiv!
                    """)
                else:
                    st.success("✅ Fără pattern stea - spectru natural")
            
            with st.expander("🔬 Technical Details (All 5 Fixes Applied)", expanded=False):
                st.markdown("""
                **Îmbunătățiri Aplicate:**
                
                ✅ **Fix #1: Windowing in PSD**
                - Fereastră Hann 2D aplicată pentru eliminarea spectral leakage
                - Previne spike-uri false de la marginile imaginii
                
                ✅ **Fix #2: Vectorized Whitening**
                - Radial whitening optimizat (10-100× mai rapid)
                - Exclusion DC component și low frequencies
                
                ✅ **Fix #3: Spike Symmetry**
                - Verificare perechi simetrice Hermitian
                - Detecție resampling prin simetrie față de centru
                
                ✅ **Fix #4: Star 180° Symmetry**
                - Verificare simetrie rotațională 180°
                - Distinge AI resampling de grid-uri naturale (garduri)
                
                ✅ **Fix #5: JPEG Mitigation**
                - Gaussian blur σ=0.7 pentru suprimarea blocurilor 8×8
                - Păstrează artefactele globale de resampling
                
                **Parametri Folosiți:**
                - Z-threshold: 6.0σ (99.9997% confidence)
                - Min radius: 15px (exclude DC)
                - Min peaks (star): 6 (evită false positive de la garduri 4-fold)
                - Symmetry tolerance: 5px
                """)
            
            # ============================================
            # SECTION 3: LEGACY INTERPRETATION
            # ============================================
            if fft_patterns['interpretations']:
                st.markdown("#### 📋 Interpretări Suplimentare")
                for interp in fft_patterns['interpretations']:
                    if "🚨" in interp:
                        st.error(interp)
                    elif "⚠️" in interp:
                        st.warning(interp)
                    elif "✅" in interp:
                        st.success(interp)
                    else:
                        st.info(interp)
            
            # Suspicion Score
            st.markdown("---")
            col_score1, col_score2 = st.columns([2, 1])
            with col_score1:
                st.markdown("#### 🎯 Suspicion Score FFT (Fallback Detection)")
                suspicion = fft_patterns['suspicion_score']
                if suspicion >= 60:
                    st.error(f"🔥 Scor EXTREM DE RIDICAT: {suspicion}/100 - CERT AI")
                elif suspicion >= 40:
                    st.error(f"❌ Scor Foarte Ridicat: {suspicion}/100 - FOARTE PROBABIL AI")
                elif suspicion >= 25:
                    st.error(f"🔴 Scor Ridicat: {suspicion}/100 - SUSPECT AI")
                elif suspicion >= 15:
                    st.warning(f"⚠️ Scor Moderat: {suspicion}/100 - Necesită verificare")
                elif suspicion >= 8:
                    st.info(f"ℹ️ Scor Minim: {suspicion}/100 - Puține anomalii")
                else:
                    st.success(f"✅ Scor Foarte Scăzut: {suspicion}/100 - NORMAL/REAL")
                st.progress(suspicion / 100)
            
            with col_score2:
                st.markdown("#### 🚨 Pattern-uri Detectate")
                if fft_patterns['star_pattern']:
                    st.write("⭐ Pattern Stea")
                if fft_patterns['periodic_spikes']:
                    st.write("📊 Spike-uri Periodice")
                if fft_patterns['unnatural_decay']:
                    st.write("📉 Decay Artificial")
                if not (fft_patterns['star_pattern'] or fft_patterns['periodic_spikes'] or fft_patterns['unnatural_decay']):
                    st.write("✅ Fără pattern-uri")
            
        # Plot FFT Radial PSD
        if psd1D is not None:
            fig_psd, ax = plt.subplots(figsize=(12, 6))
            
            # Create radial frequency array for plotting (in pixels from center)
            radial_freqs = np.arange(len(psd1D))  # Radial distance in pixels
            
            # Plot PSD directly in dB (already in dB format from extract_all_features)
            # Use simple plot with linear scales for both axes for better readability
            ax.plot(radial_freqs, psd1D, linewidth=2, color='#2E86AB', 
                   label='Radial PSD (dB)', alpha=0.9)
            
            # Add subtle frequency zone markers (radial distance zones)
            if len(psd1D) > 50:
                n = len(psd1D)
                zones = [
                    (0.25, 'green', 'Low (25%)'),
                    (0.50, 'blue', 'Mid (50%)'),
                    (0.75, 'orange', 'High (75%)'),
                    (0.90, 'red', 'Tail (90%)')
                ]
                
                for pct, color, label in zones:
                    idx = int(pct * n)
                    if idx < len(radial_freqs):
                        freq_val = radial_freqs[idx]
                        ax.axvline(x=freq_val, color=color, linestyle=':', 
                                  alpha=0.5, linewidth=1.5, label=label)
            
            # Calculate and show decay trend line for visualization
            if len(psd1D) > 50:
                # Show expected natural decay (reference line)
                mid_idx = len(psd1D) // 2
                end_idx = len(psd1D)
                
                # Simple linear fit for reference
                x_ref = radial_freqs[mid_idx:end_idx]
                y_ref = psd1D[mid_idx:end_idx]
                valid_mask = np.isfinite(y_ref)
                
                if np.sum(valid_mask) > 10:
                    x_ref_valid = x_ref[valid_mask]
                    y_ref_valid = y_ref[valid_mask]
                    
                    # Fit polynomial (degree 1)
                    coeffs = np.polyfit(x_ref_valid, y_ref_valid, 1)
                    trend_line = np.polyval(coeffs, x_ref)
                    
                    # Plot trend line
                    ax.plot(x_ref, trend_line, 
                           color='red', linestyle='--', linewidth=2, 
                           alpha=0.7, label=f'Decay trend ({coeffs[0]:.2f} dB/pixel)')
            
            # Mark detected anomalies with minimal text boxes
            anomaly_y_pos = 0.95
            if 'fft_star_pattern' in features_dict and features_dict['fft_star_pattern'] > 0.5:
                ax.text(0.98, anomaly_y_pos, "⭐ Star Pattern", transform=ax.transAxes,
                       fontsize=11, fontweight='bold', color='white',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.8),
                       verticalalignment='top', horizontalalignment='right')
                anomaly_y_pos -= 0.08
            
            if 'fft_periodic_spikes' in features_dict and features_dict['fft_periodic_spikes'] > 0.5:
                ax.text(0.98, anomaly_y_pos, "📊 Periodic Spikes", transform=ax.transAxes,
                       fontsize=11, fontweight='bold', color='white',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='darkorange', alpha=0.8),
                       verticalalignment='top', horizontalalignment='right')
                anomaly_y_pos -= 0.08
            
            if 'fft_unnatural_decay' in features_dict and features_dict['fft_unnatural_decay'] > 0.5:
                ax.text(0.98, anomaly_y_pos, "📉 Unnatural Decay", transform=ax.transAxes,
                       fontsize=11, fontweight='bold', color='white',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='purple', alpha=0.8),
                       verticalalignment='top', horizontalalignment='right')
                anomaly_y_pos -= 0.08
            
            # Show high-frequency drop if significant
            if 'tail_90' in features_dict:
                tail_val = features_dict['tail_90']
                if abs(tail_val) > 20:  # Steep drop = suspicious
                    ax.text(0.98, anomaly_y_pos, f"⚠️ HF Drop: {tail_val:.1f} dB/dec", 
                           transform=ax.transAxes,
                           fontsize=11, fontweight='bold', color='white',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='crimson', alpha=0.8),
                           verticalalignment='top', horizontalalignment='right')
            
            ax.set_xlabel('Radial Frequency (pixels from center)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Power Spectral Density (dB)', fontsize=13, fontweight='bold')
            ax.set_title('FFT Radial PSD - Azimuthal Average (Mathematically Correct)', fontsize=15, fontweight='bold')
            ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
            
            # Add legend for frequency zones
            ax.legend(loc='lower left', fontsize=10, framealpha=0.95, 
                     shadow=True, fancybox=True)
            
            plt.tight_layout()
            st.pyplot(fig_psd)
            plt.close()
            
            st.markdown("""
            **🔍 Cum să interpretezi graficul (Media Azimutală - Corect Matematic):**
            
            - **Axa X (Radial Frequency)**: Distanța radială de la centru (pixels) - frecvența spațială
            - **Axa Y (Power)**: Power Spectral Density în dB (10*log₁₀ pentru putere)
            
            **✅ Metoda Azimutală (Correct for 2D deepfake detection):**
            1. **2D FFT** - Transformată Fourier 2D pe toată imaginea
            2. **PSD 2D** - Calculează |F(u,v)|² (putere, nu amplitudine)
            3. **Azimuthal Average** - Face media radială (toate unghiurile la aceeași rază)
            4. **Radial Profile** - Rezultă profil 1D: putere vs. frecvență radială
            
            **De ce această metodă:**
            - 🎯 **Corectă matematic**: Respectă natura 2D a imaginilor
            - 🔬 **Detectează artefacte GAN**: Up-sampling lasă "amprente" radiale
            - 📊 **Natural law**: Imagini reale urmează 1/f^α decay smooth
            
            **Semne caracteristice:**
            - ✅ **Imagini REALE**: Curbă exponențială descendentă lină, fără cocoașe
            - ⚠️ **Imagini AI (GAN/Diffusion)**: 
              - **Vârfuri** (bumps) la frecvențe medii/înalte
              - **Drop abrupt** la final (>90%)
              - **Cocoașă ridicată** în high-freq tail (semnul resampling-ului)
            """)
            
            # Plot 2: 2D Spectrum (if available)
            if magnitude_2d is not None:
                st.markdown("#### 🌟 Spectrum 2D (Star Pattern Detection)")
                fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Original 2D spectrum
                im1 = ax1.imshow(magnitude_2d, cmap='hot', aspect='auto')
                ax1.set_title('FFT 2D Spectrum (Log Scale)', fontweight='bold')
                ax1.set_xlabel('Frequency X')
                ax1.set_ylabel('Frequency Y')
                plt.colorbar(im1, ax=ax1, label='Log Power')
                
                # Zoomed center (star pattern area)
                center_y, center_x = np.array(magnitude_2d.shape) // 2
                crop_size = min(center_y, center_x) // 2
                cropped = magnitude_2d[center_y-crop_size:center_y+crop_size, 
                                      center_x-crop_size:center_x+crop_size]
                im2 = ax2.imshow(cropped, cmap='hot', aspect='auto')
                ax2.set_title('Zoom Central (Star Pattern Zone)', fontweight='bold')
                ax2.set_xlabel('Frequency X (zoomed)')
                ax2.set_ylabel('Frequency Y (zoomed)')
                plt.colorbar(im2, ax=ax2, label='Log Power')
                
                if fft_patterns['star_pattern']:
                    ax2.text(crop_size, crop_size*1.8, '⚠️ STAR PATTERN DETECTED', 
                            color='yellow', fontsize=12, fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                            ha='center')
                
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
            
            # ============================================
            # COMPREHENSIVE FORENSICS VISUALIZATIONS
            # ============================================
            st.markdown("---")
            st.markdown("### 🔬 Advanced Forensics Analysis")
            
            if magnitude_2d is not None:
                try:
                    # Compute all advanced metrics
                    whitened = radial_whitening_fast(magnitude_2d, exclude_dc_radius=5, r_min_frac=0.05)
                    num_spikes, sym_pairs, spike_strength = detect_symmetric_spikes(whitened, z_thresh=6.0, r_min=15)
                    ang_energy = angular_energy_signature(whitened, r_min_frac=0.1, r_max_frac=0.45)
                    star_sc, star_sym, star_peaks_idx = star_score_with_symmetry(ang_energy)
                        
                        # Get spike coordinates
                        coords = peak_local_max(whitened, threshold_abs=6.0, min_distance=10, 
                                               num_peaks=200, exclude_border=10)
                        h, w = whitened.shape
                        cy, cx = h // 2, w // 2
                        if len(coords) > 0:
                            r_peaks = np.sqrt((coords[:, 0] - cy)**2 + (coords[:, 1] - cx)**2)
                            valid = r_peaks >= 15
                            coords = coords[valid]
                        
                        # VISUALIZATION 1: Whitened Spectrum with Spike Pairs
                        st.markdown("#### 📊 Whitened Spectrum (Fix #2 + #3)")
                        fig_w, ax_w = plt.subplots(figsize=(12, 8))
                        im_w = ax_w.imshow(whitened, cmap='seismic', vmin=-10, vmax=10, aspect='auto')
                        if len(coords) > 0:
                            ax_w.scatter(coords[:, 1], coords[:, 0], c='yellow', s=50, marker='x', 
                                       linewidths=2, label=f'{num_spikes} spikes', zorder=3)
                            # Draw symmetric pairs
                            if sym_pairs > 0:
                                used = set()
                                for i, (y1, x1) in enumerate(coords):
                                    if i in used:
                                        continue
                                    y2_exp, x2_exp = 2*cy - y1, 2*cx - x1
                                    for j, (y2, x2) in enumerate(coords):
                                        if j <= i or j in used:
                                            continue
                                        if np.sqrt((y2-y2_exp)**2 + (x2-x2_exp)**2) <= 5:
                                            ax_w.plot([x1, x2], [y1, y2], 'lime', lw=1.5, alpha=0.6, zorder=2)
                                            used.add(i)
                                            used.add(j)
                                            break
                                ax_w.plot([], [], 'lime', lw=2, label=f'{sym_pairs} sym pairs')
                        ax_w.scatter(cx, cy, c='red', s=200, marker='+', lw=3, label='Center', zorder=4)
                        ax_w.set_title(f'Radial Whitening | Spikes: {num_spikes} | Symmetric: {sym_pairs} ({fft_patterns.get("symmetry_ratio",0)*100:.1f}%)', fontweight='bold')
                        ax_w.legend(loc='upper right')
                        plt.colorbar(im_w, ax=ax_w, label='Z-score (σ)')
                        plt.tight_layout()
                        st.pyplot(fig_w)
                        plt.close()
                        
                        # VISUALIZATION 2: Angular Energy
                        st.markdown("#### ⭐ Angular Energy Signature (Fix #4)")
                        fig_a, (ax_a1, ax_a2) = plt.subplots(1, 2, figsize=(16, 6))
                        angles_deg = np.arange(len(ang_energy))
                        ax_a1.plot(angles_deg, ang_energy, lw=2, color='#2E86AB', alpha=0.7)
                        # Handle star_peaks_idx safely (can be int or array)
                        num_star_peaks = len(star_peaks_idx) if hasattr(star_peaks_idx, '__len__') else (star_peaks_idx if isinstance(star_peaks_idx, int) else 0)
                        star_peaks_array = star_peaks_idx if hasattr(star_peaks_idx, '__len__') else []
                        if len(star_peaks_array) > 0:
                            ax_a1.scatter(star_peaks_array, ang_energy[star_peaks_array], c='red', s=100, marker='*', 
                                        edgecolors='black', lw=1.5, label=f'{num_star_peaks} peaks', zorder=3)
                            for pk in star_peaks_array:
                                opposite = (pk + len(ang_energy)//2) % len(ang_energy)
                                ax_a1.axvline(pk, color='red', ls='--', alpha=0.3, lw=1)
                                ax_a1.axvline(opposite, color='blue', ls='--', alpha=0.3, lw=1)
                        ax_a1.set_xlabel('Angle (degrees)', fontweight='bold')
                        ax_a1.set_ylabel('Energy', fontweight='bold')
                        ax_a1.set_title(f'Angular Energy | Peaks: {num_star_peaks} | Symmetry: {star_sym:.1%}', fontweight='bold')
                        ax_a1.grid(True, alpha=0.3)
                        ax_a1.legend()
                        # Polar
                        theta = np.linspace(0, 2*np.pi, len(ang_energy))
                        ax_a2 = plt.subplot(122, projection='polar')
                        ax_a2.plot(theta, ang_energy, lw=2, color='#2E86AB')
                        ax_a2.fill(theta, ang_energy, alpha=0.3, color='#2E86AB')
                        if len(star_peaks_array) > 0:
                            theta_peaks = np.array(star_peaks_array) * 2*np.pi / len(ang_energy)
                            ax_a2.scatter(theta_peaks, ang_energy[star_peaks_array], c='red', s=150, marker='*', 
                                        edgecolors='black', lw=2, zorder=3)
                        ax_a2.set_title(f'Star Pattern (Polar) | Score: {star_sc:.2f}', fontweight='bold', pad=20)
                        plt.tight_layout()
                        st.pyplot(fig_a)
                        plt.close()
                        
                        # VISUALIZATION 3: Symmetry Analysis
                        if num_spikes > 0 and len(coords) > 1:
                            st.markdown("#### 🔄 Spike Symmetry Matrix (Fix #3)")
                            fig_s, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(16, 6))
                            distances = np.zeros((len(coords), len(coords)))
                            for i in range(len(coords)):
                                for j in range(len(coords)):
                                    y2_exp = 2*cy - coords[j][0]
                                    x2_exp = 2*cx - coords[j][1]
                                    distances[i,j] = np.sqrt((coords[i][0]-y2_exp)**2 + (coords[i][1]-x2_exp)**2)
                            im_s = ax_s1.imshow(distances, cmap='RdYlGn_r', aspect='auto', vmax=20)
                            ax_s1.set_title('Symmetry Distance Matrix', fontweight='bold')
                            ax_s1.set_xlabel('Spike Index')
                            ax_s1.set_ylabel('Spike Index')
                            plt.colorbar(im_s, ax=ax_s1, label='Distance (px)')
                            min_dist = np.min(distances + np.eye(len(coords))*1000, axis=1)
                            ax_s2.hist(min_dist, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
                            ax_s2.axvline(5, color='red', ls='--', lw=2, label='Threshold (5px)')
                            ax_s2.set_xlabel('Distance to Symmetric Pair (px)', fontweight='bold')
                            ax_s2.set_ylabel('Frequency', fontweight='bold')
                            ax_s2.set_title(f'Distribution | {sym_pairs} pairs < 5px', fontweight='bold')
                            ax_s2.legend()
                            ax_s2.grid(True, alpha=0.3, axis='y')
                            plt.tight_layout()
                            st.pyplot(fig_s)
                            plt.close()
                        
                        # Summary card
                        st.markdown("---")
                        st.markdown("### 📋 FFT Forensics Summary")
                        
                        # Show FFT Forensic Score prominently
                        if 'fft_forensic_score' in locals():
                            st.markdown(f"### 🎯 FFT AI Detection Score: **{fft_forensic_score:.1f}/100**")
                            score_col1, score_col2 = st.columns([3, 1])
                            with score_col1:
                                st.progress(fft_forensic_score / 100)
                            with score_col2:
                                if fft_forensic_score >= 60:
                                    st.error("🔴 HIGH AI")
                                elif fft_forensic_score >= 40:
                                    st.warning("🟠 MODERATE")
                                elif fft_forensic_score >= 20:
                                    st.info("🟡 LOW AI")
                                else:
                                    st.success("🟢 NORMAL")
                        
                        st.markdown("---")
                        
                        s1, s2, s3 = st.columns(3)
                        with s1:
                            st.markdown("**Radial Whitening:**")
                            st.metric("Spikes", num_spikes)
                            st.metric("Symmetric Pairs", sym_pairs)
                            st.metric("Symmetry Ratio", f"{fft_patterns.get('symmetry_ratio',0):.1%}")
                        with s2:
                            st.markdown("**Angular Energy:**")
                            st.metric("Star Score", f"{star_sc:.2f}")
                            st.metric("180° Symmetry", f"{star_sym:.1%}")
                            st.metric("Angular Peaks", num_star_peaks)
                        with s3:
                            st.markdown("**Overall Assessment:**")
                            if star_sym > 0.7 and num_star_peaks >= 8:
                                st.error("🔴 HIGH AI RISK\n\nMultiple star patterns with high symmetry")
                            elif fft_patterns.get('symmetry_ratio',0) > 0.5:
                                st.warning("🟠 RESAMPLING\n\nSymmetric spike pairs detected")
                            elif fft_patterns.get('suspicion_score',0) >= 30:
                                st.warning("🟡 SUSPICIOUS\n\nSome anomalies found")
                            else:
                                st.success("🟢 NORMAL\n\nNo significant AI patterns")
                            
                            # Show contribution breakdown
                            if 'fft_contributions' in locals() and fft_contributions:
                                with st.expander("📊 Score Breakdown"):
                                    for name, score in fft_contributions.items():
                                        st.write(f"• {name}: **+{score:.0f}%**")
                    except Exception as e:
                        st.error(f"⚠️ Advanced viz failed: {e}")
                else:
                    st.info("ℹ️ 2D spectrum not available for advanced visualizations")
            else:
                st.info("Activează 'Arată Vizualizări' din sidebar pentru grafice detaliate.")
        else:
            st.warning("Nu s-au putut genera analize FFT pentru această imagine.")
    
    with tab4:
        st.markdown("### 🔐 Analiză Metadata EXIF")
        
        if metadata_info and metadata_info['has_exif']:
            st.success("✅ Metadata EXIF prezentă - indicator puternic de imagine REALĂ")
            
            st.markdown("**Detalii Complete:**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📱 Device Info")
                if metadata_info['make']:
                    st.write(f"• **Make:** `{metadata_info['make']}`")
                else:
                    st.write("• **Make:** ❌ N/A")
                
                if metadata_info['model']:
                    st.write(f"• **Model:** `{metadata_info['model']}`")
                else:
                    st.write("• **Model:** ❌ N/A")
            
            with col2:
                st.markdown("#### 🕐 Temporal Info")
                if metadata_info['datetime']:
                    st.write(f"• **DateTime:** `{metadata_info['datetime']}`")
                else:
                    st.write("• **DateTime:** ❌ N/A")
                
                if metadata_info['gps']:
                    st.write("• **GPS Data:** ✅ Prezent")
                else:
                    st.write("• **GPS Data:** ❌ Absent")
            
            st.markdown("#### 💻 Software Info")
            if metadata_info['software']:
                st.write(f"• **Software:** `{metadata_info['software']}`")
            else:
                st.write("• **Software:** ❌ N/A")
            
            st.markdown("---")
            st.markdown("#### 📊 Completeness Score")
            st.progress(metadata_info['completeness_score'] / 100)
            st.write(f"**{metadata_info['completeness_score']}/100 puncte**")
            
            st.info("""
            **Interpretare:**
            - **80-100**: Metadata foarte completă (cameră/telefon)
            - **50-79**: Metadata parțială (editare software)
            - **20-49**: Metadata minimă (suspiciune)
            - **0-19**: Lipsa metadata (probabil AI-generate)
            """)
        else:
            st.error("❌ NICIO METADATA EXIF GĂSITĂ")
            
            st.warning("""
            **Semne de alarmă:**
            
            Imaginile reale făcute cu camera/telefon conțin ÎNTOTDEAUNA metadata:
            - Data și ora capturii
            - Model dispozitiv (ex: iPhone 15, Samsung Galaxy S24)
            - Setări cameră (ISO, aperture, focal length)
            - Uneori locație GPS
            
            Imaginile AI-generate **NU au** această informație sau au metadata suspectă/editată.
            
            ⚠️ **Lipsa completă a metadata = indicator puternic de imagine sintetică!**
            """)

    with tab5:
        st.markdown("### 📚 Interpretare Rezultate")
        
        # Gemini AI Interpretation
        if GEMINI_AVAILABLE:
            st.markdown("#### 🤖 Interpretare Automată cu Google Gemini AI")
            
            with st.expander("💡 Obține interpretare AI a graficelor FFT"):
                st.info("""
                Gemini va analiza **doar graficele și datele numerice** din analiza FFT:
                - Profilul radial PSD 1D
                - Statistici și metrici calculate
                - Pattern-uri detectate
                - Features numerice
                
                **NU** se trimite imaginea originală către API!
                """)
                
                api_key_input = st.text_input("Gemini API Key", type="password", 
                                              help="Obține de la: https://makersuite.google.com/app/apikey")
                
                use_vision = st.checkbox("Folosește Gemini Vision (trimite și graficul PSD ca imagine)", value=True)
                
                if st.button("🚀 Analizează cu Gemini AI", type="primary"):
                    if not api_key_input:
                        st.error("❌ Te rog introdu API Key-ul Gemini!")
                    else:
                        with st.spinner("🔄 Gemini analizează datele..."):
                            result = interpret_with_gemini(
                                psd1D=psd1D,
                                fft_patterns=fft_patterns,
                                features_dict=features_dict,
                                magnitude_2d=magnitude_2d if 'magnitude_2d' in locals() else None,
                                api_key=api_key_input,
                                use_vision=use_vision
                            )
                            
                            if result['success']:
                                interp = result['interpretation']
                                
                                # Verdict
                                if interp['verdict'] == "AI-GENERATED":
                                    st.error(f"### 🤖 Verdict Gemini: **{interp['verdict']}**")
                                else:
                                    st.success(f"### ✅ Verdict Gemini: **{interp['verdict']}**")
                                
                                # Confidence
                                col1, col2 = st.columns([1, 3])
                                with col1:
                                    st.metric("Confidence", f"{interp['confidence']}%")
                                with col2:
                                    st.progress(interp['confidence'] / 100)
                                
                                # Reasoning
                                st.markdown("#### 💭 Raționament AI:")
                                st.write(interp['reasoning'])
                                
                                # Key Indicators
                                st.markdown("#### 🔑 Indicatori Cheie:")
                                for idx, indicator in enumerate(interp['key_indicators'], 1):
                                    st.markdown(f"{idx}. {indicator}")
                                
                                # Natural Signals (if any)
                                if interp.get('natural_signals'):
                                    st.markdown("#### 🌿 Semne Naturale Detectate:")
                                    for signal in interp['natural_signals']:
                                        st.markdown(f"- {signal}")
                                
                                # Recommendation
                                st.markdown("#### 💡 Recomandare:")
                                st.info(interp['recommendation'])
                                
                                # Raw response (expandable)
                                with st.expander("📄 Răspuns brut Gemini"):
                                    st.code(result['raw_response'])
                            else:
                                st.error(f"❌ Eroare la interpretare Gemini: {result['error']}")
                                st.info("Verifică API Key-ul și conexiunea la internet.")
            
            st.markdown("---")
        else:
            st.warning("""
            ⚠️ **Gemini AI Interpreter nu este disponibil**
            
            Pentru a activa interpretarea automată cu AI:
            ```bash
            pip install google-generativeai
            ```
            Apoi repornește aplicația.
            """)
            st.markdown("---")
        
        # Ghid Manual
        st.markdown("### 📖 Ghid Manual de Interpretare (Hany Farid Approach)")
        
        st.markdown("""
        Acest sistem folosește **8 tehnici independente** de analiză criminalistică digitală,
        inspirate din cercetările lui Hany Farid (UC Berkeley) - pionier în domeniul forensicii digitale.
        
        ---
        """)
        
        col_guide1, col_guide2 = st.columns(2)
        
        with col_guide1:
            st.markdown("#### 1️⃣ FFT Radial PSD (Azimuthal Average - Corect Matematic)")
            st.success("""
            **Ce detectează:** Pattern-uri artificiale în spectrul de frecvență 2D
            
            **Metoda Azimutală (CORECTĂ pentru deepfake):**
            1. 2D FFT pe întreaga imagine (cu Hanning window)
            2. Calculează PSD 2D: |F(u,v)|²
            3. **Media Azimutală** - face media radială pentru toate unghiurile
            4. Rezultă profil 1D: Power vs. Frecvență Radială
            
            **Detectează:** Artefacte de up-sampling GAN/Diffusion
            
            **Indicatori AI:**
            - Vârfuri (bumps) în mid/high frequency
            - Drop abrupt la >90% (pierdere HF din resampling)
            - "Cocoașă" ridicată în tail (semnul resampling-ului CNN)
            - Imaginile REALE au decay smooth ~1/f²
            """)
            
            st.markdown("#### 2️⃣ Error Level Analysis (ELA)")
            st.info("""
            **Detectează:** Zone uniform comprimate (tipic AI-generated)
            
            **Indicatori AI:** ELA foarte mic (<2.0) + uniformitate între zone
            """)
            
            st.markdown("#### 3️⃣ Wavelet Transform")
            st.info("""
            **Detectează:** Detalii multi-scale anormale
            
            **Indicatori AI:** Energie uniformă între componente (cH, cV, cD)
            """)
            
            st.markdown("#### 4️⃣ Local Binary Patterns (LBP)")
            st.info("""
            **Detectează:** Texturi micro-locale anormale
            
            **Indicatori AI:** Entropie atipică în pattern-uri locale
            """)
        
        with col_guide2:
            st.markdown("#### 5️⃣ Gradient Analysis")
            st.info("""
            **Detectează:** Tranziții anormale între pixeli
            
            **Indicatori AI:** Gradient prea uniform sau skewness atipic
            """)
            
            st.markdown("#### 6️⃣ Color Statistics")
            st.info("""
            **Detectează:** Distribuție anormală a culorilor RGB
            
            **Indicatori AI:** Saturație sau corelație atipică între canale
            """)
            
            st.markdown("#### 7️⃣ Metadata EXIF")
            st.success("""
            **Ce detectează:** Date despre captură
            
            **Cum funcționează:**
            - Extrage EXIF (Make, Model, DateTime, GPS)
            - Calculează completeness score
            
            **Indicator PUTERNIC:**
            - ✅ Metadata completă = REAL (cameră/telefon)
            - ❌ Lipsă metadata = AI (generate fără EXIF)
            
            **Acest criteriu OVERRIDE modelul ML!**
            """)
            
            st.markdown("#### 8️⃣ Machine Learning Ensemble")
            st.info("""
            **Ce detectează:** Pattern-uri complexe învățate
            
            **Cum funcționează:**
            - Random Forest (300 trees)
            - Antrenat pe 2041 imagini SOTA AI (2025-2026)
            - Agregă toate cele **27 features numerice**
            - 7 din FFT (slope, ratios, tails), 3 ELA, 4 Wavelet, 2 LBP, 4 Gradient, 6 Color, 1 spikes
            
            **Limitări:**
            - Acuratețe ~58% pe dataset SOTA
            - Domain mismatch cu poze casual
            - De aceea folosim metadata override!
            """)
        
        st.markdown("---")
        st.markdown("### 🎯 Filosofia Hany Farid")
        st.success("""
        > **"Nu există nicio tehnică criminalistică perfectă. De aceea nu te oprești după o singură verificare, ci continui."**
        > — Hany Farid, UC Berkeley
        
        Acest sistem **NU** promite 100% acuratețe. În schimb:
        
        1. ✅ **Agregă 8 metode independente** (dacă una eșuează, altele compensează)
        2. ✅ **Oferă transparență** (vezi TOATE datele, nu doar "DA/NU")
        3. ✅ **Explică deciziile** (înțelegi DE CE a ales un verdict)
        4. ✅ **Recunoaște limitările** (metadata override pentru edge cases)
        
        **Pentru disertație/proiect:** Acesta este un **Asistent de Analiză Criminalistică**, nu un "Oracol".
        Demonstrează înțelegerea profundă a mai multor tehnici complementare.
        """)

    # Download section
    st.markdown("---")
    st.markdown("### 💾 Export Rezultate")
    
    report = f"""
RAPORT ANALIZĂ IMAGINE AI-GENERATE
===================================
Fișier: {uploaded_file.name}
Data: {st.session_state.get('analysis_time', 'N/A')}

VERDICT: {'AI-GENERATE' if is_fake else 'REALĂ'}
Confidence: {confidence:.2f}%

PROBABILITĂȚI:
- Real: {prob_real:.2f}%
- AI-Generate: {prob_fake:.2f}%

TOP 5 FEATURES:
"""
    
    if model is not None and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in range(min(5, len(feature_names))):
            idx = indices[i]
            name = feature_names[idx]
            value = features_dict[name]
            report += f"\n{i+1}. {name}: {value:.6f} (importance: {importances[idx]:.6f})"
    
    st.download_button(
        label="📥 Download Raport TXT",
        data=report,
        file_name=f"raport_{uploaded_file.name.split('.')[0]}.txt",
        mime="text/plain"
    )

else:
    st.info("👆 Încarcă o imagine pentru a începe analiza.")
    
    # Example section
    st.markdown("---")
    st.markdown("### 💡 Cum funcționează?")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**1️⃣ Metadata EXIF**")
        st.write("Verifică prezența și validitatea metadata (data, device, GPS).")
    
    with col2:
        st.markdown("**2️⃣ Extracție Features**")
        st.write("Sistemul extrage 23 de metrici diferite din imagine folosind 7 tehnici forensice.")
    
    with col3:
        st.markdown("**3️⃣ Machine Learning**")
        st.write("Modelul Random Forest antrenat pe 2041 imagini face predicția finală.")
    
    with col4:
        st.markdown("**4️⃣ Verdict**")
        st.write("Rezultatul final cu confidence score și detalii complete despre analiză.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
<b>Sistem Final Integrat - Disertație</b><br>
Multi-Method Forensic Analysis | 8 Tehnici | 27 Features Numerice + Metadata<br>
Dataset: 2041 imagini High-Resolution Faces (SOTA AI 2025-2026) | Acuratețe: ~58.5%<br>
<i>📊 FFT cu Hanning Window + INTER_AREA | 🔐 Metadata EXIF Override</i><br>
<small>✅ Matematic corect: np.outer window, 20*log10 dB, azimuthal average, numerical features</small>
</div>
""", unsafe_allow_html=True)
