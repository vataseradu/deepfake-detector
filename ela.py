"""
Error Level Analysis (ELA) Module
==================================
Detects local image manipulation (Photoshop, inpainting, face swapping) by analyzing
JPEG compression artifacts. The principle: manipulated regions have different compression
error levels than the rest of the image.

Mathematical Foundation:
- JPEG compression is lossy and introduces quantization errors
- Re-saving at the same quality level should produce similar errors throughout
- Manipulated areas show higher error levels (appear brighter in ELA output)

Author: Master's Thesis - Deepfake Detection Project
Date: January 2026
"""

import os
from typing import Tuple, Dict
from PIL import Image, ImageChops, ImageEnhance
import numpy as np


def perform_ela(image: Image.Image, quality: int = 90) -> Tuple[Image.Image, dict]:
    """
    Performs Error Level Analysis on a PIL Image.
    
    Algorithm:
    1. Re-save the image at a specified JPEG quality level
    2. Calculate pixel-wise absolute difference between original and re-saved
    3. Normalize and scale the difference to enhance visibility
    4. Calculate statistical metrics for quantitative analysis
    
    Mathematical Explanation:
    For each pixel (i,j):
        ELA(i,j) = |Original(i,j) - Resaved(i,j)| * scale_factor
    where scale_factor = 255 / max(differences) for normalization
    
    Parameters:
    -----------
    image : Image.Image
        Input PIL Image in RGB mode
    quality : int, optional
        JPEG compression quality (1-100). Default: 90
        Higher values = less compression = smaller differences in unmodified areas
    
    Returns:
    --------
    tuple[Image.Image, dict]
        - Image.Image: ELA result image where bright areas indicate potential manipulation
        - dict: Statistical metrics containing:
            * 'mean_intensity': Average brightness (0-255)
            * 'std_intensity': Standard deviation (variance indicator)
            * 'max_intensity': Peak brightness (0-255)
            * 'median_intensity': Median brightness (0-255)
        
    Notes:
    ------
    - Uniform brightness suggests consistent compression (likely authentic or fully AI-generated)
    - Localized bright spots suggest manipulation (copy-paste, face swap, inpainting)
    - False positives can occur in high-contrast edges or saturated colors
    """
    temp_filename = "temp_ela.jpg"
    
    # Step 1: Re-save at specified quality
    image.save(temp_filename, 'JPEG', quality=quality)
    
    # Step 2: Load re-saved image and compute difference
    resaved_image = Image.open(temp_filename)
    ela_image = ImageChops.difference(image, resaved_image)
    
    # Step 3: Normalize and scale for visibility
    extrema = ela_image.getextrema()  # Get min/max per channel
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1  # Avoid division by zero
    
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    # Calculate statistics for decision logic
    ela_array = np.array(ela_image)
    gray_ela = np.mean(ela_array, axis=2)  # Convert to grayscale for analysis
    
    ela_stats = {
        'mean_intensity': float(np.mean(gray_ela)),
        'std_intensity': float(np.std(gray_ela)),
        'max_intensity': float(np.max(gray_ela)),
        'median_intensity': float(np.median(gray_ela))
    }
    
    # Clean up temporary file
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
        
    return ela_image, ela_stats


def ela_to_array(ela_image: Image.Image) -> np.ndarray:
    """
    Converts ELA result to NumPy array for further analysis.
    
    Parameters:
    -----------
    ela_image : Image.Image
        ELA result from perform_ela()
    
    Returns:
    --------
    np.ndarray
        3D array of shape (height, width, 3) with dtype uint8
    """
    return np.array(ela_image)


def get_ela_statistics(ela_image: Image.Image) -> dict:
    """
    Calculates statistical metrics from ELA result for automated classification.
    
    Metrics:
    --------
    - mean_intensity: Average brightness (higher = more manipulation)
    - std_intensity: Standard deviation (higher = localized manipulation)
    - max_intensity: Peak brightness (255 = strong manipulation signal)
    
    Parameters:
    -----------
    ela_image : Image.Image
        ELA result from perform_ela()
    
    Returns:
    --------
    dict
        Dictionary containing statistical metrics
    """
    ela_array = ela_to_array(ela_image)
    gray_ela = np.mean(ela_array, axis=2)  # Convert to grayscale
    
    return {
        'mean_intensity': float(np.mean(gray_ela)),
        'std_intensity': float(np.std(gray_ela)),
        'max_intensity': float(np.max(gray_ela)),
        'median_intensity': float(np.median(gray_ela))
    }
