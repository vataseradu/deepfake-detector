"""
Frequency Analysis Module (FFT & Azimuthal Averaging)
======================================================
Detects AI-generated images by analyzing frequency domain artifacts.
Diffusion models (Stable Diffusion, DALL-E, Midjourney) often produce
unnatural frequency distributions compared to real camera images.

Mathematical Foundation:
- Natural images follow 1/f power law (pink noise) in frequency domain
- AI-generated images show abrupt drop-offs at high frequencies
- FFT converts spatial domain → frequency domain for analysis

Author: Master's Thesis - Deepfake Detection Project
Date: January 2026
"""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image


def azimuthalAverage(image: np.ndarray, center: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculates radial (azimuthal) average of 2D power spectrum.
    
    Mathematical Explanation:
    For a 2D spectrum S(kx, ky), we compute the 1D profile S(k) where:
        k = sqrt(kx² + ky²)  [radial frequency]
        S(k) = mean of all S(kx, ky) where sqrt(kx² + ky²) = k
    
    This reduces 2D data to 1D for easier visualization and analysis.
    
    Parameters:
    -----------
    image : np.ndarray
        2D array representing power spectrum magnitude
    center : np.ndarray, optional
        [x, y] coordinates of spectrum center. If None, uses image center
    
    Returns:
    --------
    np.ndarray
        1D array of radially averaged power values
        Index i corresponds to frequency radius i pixels from center
    
    References:
    -----------
    - Dzanic et al. (2020) "Fourier Spectrum Discrepancies in Deep Network Generated Images"
    """
    y, x = np.indices(image.shape)
    if center is None:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    
    # Calculate radial distance from center for each pixel
    r = np.hypot(x - center[0], y - center[1])
    
    # Sort by radius
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]
    
    # Discretize radius to integer bins
    r_int = r_sorted.astype(int)
    
    # Calculate average for each radial bin
    tbin = np.bincount(r_int, i_sorted)
    nr = np.bincount(r_int)
    radial_profile = tbin / nr
    
    return radial_profile

def plot_spectrum(image_pil: Image.Image) -> Tuple[Figure, np.ndarray]:
    """
    Generates 2D and 1D frequency spectrum plots for deepfake detection.
    
    Mathematical Process:
    1. Convert image to grayscale I(x,y)
    2. Apply 2D FFT: F(kx, ky) = FFT2D[I(x,y)]
    3. Shift zero-frequency to center: Fshift = fftshift(F)
    4. Calculate magnitude: |Fshift|
    5. Compute 1D Power Spectral Density (PSD): PSD(k) = azimuthalAverage(|Fshift|²)
    
    Interpretation for Deepfake Detection:
    - Real images: Log-log plot shows linear decline (1/f^α, α ≈ 2)
    - AI images: Show deviations from power law, especially at high frequencies
    - Abrupt drops indicate frequency band limitations in generation models
    
    Parameters:
    -----------
    image_pil : Image.Image
        Input PIL Image (RGB or grayscale)
    
    Returns:
    --------
    tuple[Figure, np.ndarray]
        - Figure: Matplotlib figure with 2 subplots:
            * Left: 2D frequency spectrum visualization (log magnitude)
            * Right: 1D radial power spectral density (log-log plot)
        - np.ndarray: Raw 1D Power Spectral Density array for quantitative analysis
    
    References:
    -----------
    - Frank et al. (2020) "Leveraging Frequency Analysis for Deep Fake Image Recognition"
    - Durall et al. (2020) "Watch Your Up-Convolution"
    """
    # Convert to grayscale
    img_gray = np.array(image_pil.convert('L'))
    
    # Step 1: Compute 2D FFT
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    
    # Step 2: Calculate magnitude spectrum (log scale for visualization)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # +1 to avoid log(0)
    
    # Step 3: Calculate 1D Power Spectral Density
    psd1D = azimuthalAverage(np.abs(fshift) ** 2)
    
    # Create figure for Streamlit
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: 2D Spectrum (Visual representation)
    im = ax1.imshow(magnitude_spectrum, cmap='inferno')
    ax1.set_title('2D Frequency Spectrum (FFT)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('kx (Horizontal Frequency)')
    ax1.set_ylabel('ky (Vertical Frequency)')
    plt.colorbar(im, ax=ax1, label='Log Magnitude')
    
    # Plot 2: 1D Radial Profile (Mathematical analysis)
    ax2.loglog(psd1D, linewidth=2, color='#2E86AB')
    ax2.set_title('1D Power Spectral Density', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Spatial Frequency (k)')
    ax2.set_ylabel('Power (|F(k)|²)')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Add reference line for natural 1/f² decay
    k = np.arange(1, len(psd1D))
    reference = psd1D[1] * (k[0] / k) ** 2
    ax2.loglog(k, reference[:len(k)], 'r--', alpha=0.5, linewidth=1.5, label='1/f² (Natural)')
    ax2.legend()
    
    plt.tight_layout()
    return fig, psd1D  # Return both figure and raw data