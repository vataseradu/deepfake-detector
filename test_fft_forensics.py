"""
VALIDATION TEST SUITE FOR FFT FORENSICS
========================================
Minimal reproducible tests using synthetic patterns.
No external datasets required - all patterns generated programmatically.

Tests:
1. Resampling Pattern → Should detect symmetric spikes
2. Fence/Grid Pattern → Should NOT trigger high star score
3. Star Pattern → Should detect angular peaks with symmetry
4. Natural Image → Should show low scores
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from fft_forensics_improved import (
    preprocess_for_fft,
    fft_log_spectrum,
    radial_whitening,
    detect_spectral_spikes,
    angular_energy_signature,
    star_score_robust
)
import tempfile
import os


# =============================================================================
# SYNTHETIC PATTERN GENERATORS
# =============================================================================

def create_resampled_pattern(size=512, scale_factor=1.5):
    """
    Create synthetic image with known resampling artifacts.
    
    Process:
    1. Generate natural image with 1/f spectrum (pink noise)
    2. Downsample and upsample (introduces periodic artifacts)
    
    Expected: High number of symmetric spikes
    """
    # Original image with natural 1/f spectrum
    freq_y, freq_x = np.meshgrid(
        np.fft.fftfreq(size),
        np.fft.fftfreq(size),
        indexing='ij'
    )
    freq_radial = np.sqrt(freq_x**2 + freq_y**2) + 1e-10
    
    # Pink noise (1/f spectrum)
    spectrum = 1.0 / freq_radial
    phase = np.random.rand(size, size) * 2 * np.pi
    fft_pink = spectrum * np.exp(1j * phase)
    
    img_original = np.fft.ifft2(fft_pink).real
    img_original = (img_original - img_original.min()) / (img_original.max() - img_original.min())
    
    # Resample (introduces periodic artifacts)
    new_size = int(size / scale_factor)
    img_small = cv2.resize(img_original, (new_size, new_size), interpolation=cv2.INTER_CUBIC)
    img_resampled = cv2.resize(img_small, (size, size), interpolation=cv2.INTER_CUBIC)
    
    return img_original, img_resampled


def create_fence_pattern(size=512, spacing=20):
    """
    Create fence/grid pattern that should NOT trigger high star score.
    
    This tests FALSE POSITIVE mitigation - natural grids shouldn't be flagged as AI.
    
    Expected: Low star score (< 10), or if high, low symmetry score
    """
    img = np.ones((size, size), dtype=np.float32) * 0.7
    
    # Vertical bars
    for x in range(0, size, spacing):
        img[:, x:x+3] = 0.2
    
    # Horizontal bars
    for y in range(0, size, spacing):
        img[y:y+3, :] = 0.2
    
    return img


def create_star_pattern(size=512, num_arms=8):
    """
    Create artificial star pattern in frequency domain.
    
    This simulates AI resampling artifacts (radial spokes).
    
    Expected: High star score, high symmetry
    """
    y, x = np.indices((size, size))
    cy, cx = size // 2, size // 2
    
    theta = np.arctan2(y - cy, x - cx)
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Create arms
    img = np.zeros((size, size))
    for i in range(num_arms):
        angle = i * 2 * np.pi / num_arms
        mask = np.abs((theta - angle + np.pi) % (2*np.pi) - np.pi) < 0.1
        mask &= (r > 50) & (r < 200)
        img[mask] = 1.0
    
    return img


def create_natural_image(size=512):
    """
    Create smooth natural-looking image (Gaussian blobs).
    
    Expected: Very low scores (< 5 spikes, star_score < 5)
    """
    img = np.zeros((size, size))
    
    # Add random Gaussian blobs
    np.random.seed(42)
    for _ in range(10):
        cy = np.random.randint(0, size)
        cx = np.random.randint(0, size)
        sigma = np.random.randint(20, 80)
        amplitude = np.random.rand() * 0.5 + 0.3
        
        y, x = np.indices((size, size))
        gaussian = amplitude * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
        img += gaussian
    
    img = np.clip(img, 0, 1)
    return img


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_resampling_detection():
    """Test 1: Resampled image should have symmetric spikes"""
    print("\n" + "="*60)
    print("TEST 1: RESAMPLING DETECTION")
    print("="*60)
    
    original, resampled = create_resampled_pattern(size=512, scale_factor=1.5)
    
    # Preprocess
    img_prep = preprocess_for_fft(resampled, jpeg_mitigation=False)
    
    # Compute spectrum
    spectrum = fft_log_spectrum(img_prep)
    
    # Whiten
    whitened = radial_whitening(spectrum, exclude_dc_radius=5, r_min_frac=0.05)
    
    # Detect spikes
    coords, vals, sym_pairs = detect_spectral_spikes(
        whitened, 
        z_thresh=6.0,
        r_min=15,
        symmetry_tolerance=5
    )
    
    # Results
    num_spikes = len(coords)
    num_sym_pairs = len(sym_pairs)
    symmetry_ratio = num_sym_pairs / max(num_spikes, 1)
    
    print(f"📊 Results:")
    print(f"  Total spikes detected: {num_spikes}")
    print(f"  Symmetric pairs: {num_sym_pairs}")
    print(f"  Symmetry ratio: {symmetry_ratio:.2%}")
    print(f"  Mean spike strength: {np.mean(vals):.2f} σ" if len(vals) > 0 else "  Mean spike strength: N/A")
    
    # Pass criteria: Should have significant symmetry
    passed = symmetry_ratio > 0.3 and num_spikes > 5
    
    if passed:
        print(f"\n✅ TEST PASSED: Resampling artifacts detected correctly")
    else:
        print(f"\n⚠️  TEST UNCLEAR: Low symmetry ratio or few spikes")
        print(f"   (This may be normal for some synthetic patterns)")
    
    return {
        "test": "resampling",
        "passed": passed,
        "num_spikes": num_spikes,
        "symmetry_ratio": symmetry_ratio,
        "whitened": whitened,
        "coords": coords
    }


def test_fence_false_positive():
    """Test 2: Fence should NOT trigger high AI score"""
    print("\n" + "="*60)
    print("TEST 2: FALSE POSITIVE MITIGATION (Fence Pattern)")
    print("="*60)
    
    fence = create_fence_pattern(size=512, spacing=20)
    
    # Preprocess
    img_prep = preprocess_for_fft(fence, jpeg_mitigation=False)
    
    # Compute spectrum
    spectrum = fft_log_spectrum(img_prep)
    
    # Whiten
    whitened = radial_whitening(spectrum)
    
    # Detect spikes
    coords, vals, sym_pairs = detect_spectral_spikes(whitened, z_thresh=6.0)
    
    # Angular analysis
    ang_energy = angular_energy_signature(whitened)
    star_result = star_score_robust(ang_energy)
    
    # Results
    print(f"📊 Results:")
    print(f"  Total spikes: {len(coords)}")
    print(f"  Symmetric pairs: {len(sym_pairs)}")
    print(f"  Star score: {star_result['score']:.2f}")
    print(f"  Star symmetry: {star_result['symmetry_score']:.2f}")
    print(f"  Angular peaks: {star_result['num_peaks']}")
    
    # Pass criteria: Should NOT be flagged as high-confidence AI
    # Either low star score OR low symmetry (grid has 4-fold, not continuous radial)
    high_confidence_ai = (star_result['symmetry_score'] > 0.7 and star_result['num_peaks'] > 6)
    passed = not high_confidence_ai
    
    if passed:
        print(f"\n✅ TEST PASSED: Fence pattern not flagged as high-confidence AI")
    else:
        print(f"\n⚠️  TEST FAILED: Fence incorrectly flagged as AI")
        print(f"   May need to tune symmetry threshold or min_peaks")
    
    return {
        "test": "fence_false_positive",
        "passed": passed,
        "star_score": star_result['score'],
        "star_symmetry": star_result['symmetry_score'],
        "whitened": whitened,
        "ang_energy": ang_energy
    }


def test_star_pattern_detection():
    """Test 3: Star pattern should be detected with high symmetry"""
    print("\n" + "="*60)
    print("TEST 3: STAR PATTERN DETECTION (AI Signature)")
    print("="*60)
    
    star = create_star_pattern(size=512, num_arms=8)
    
    # Preprocess
    img_prep = preprocess_for_fft(star, jpeg_mitigation=False)
    
    # Compute spectrum
    spectrum = fft_log_spectrum(img_prep)
    
    # Whiten
    whitened = radial_whitening(spectrum)
    
    # Angular analysis
    ang_energy = angular_energy_signature(whitened)
    star_result = star_score_robust(ang_energy)
    
    # Results
    print(f"📊 Results:")
    print(f"  Star score: {star_result['score']:.2f}")
    print(f"  Star symmetry: {star_result['symmetry_score']:.2f}")
    print(f"  Angular peaks: {star_result['num_peaks']}")
    
    # Pass criteria: Should detect star with high symmetry
    passed = star_result['num_peaks'] >= 6 and star_result['symmetry_score'] > 0.5
    
    if passed:
        print(f"\n✅ TEST PASSED: Star pattern detected with high symmetry")
    else:
        print(f"\n❌ TEST FAILED: Star pattern not detected")
    
    return {
        "test": "star_pattern",
        "passed": passed,
        "star_score": star_result['score'],
        "star_symmetry": star_result['symmetry_score'],
        "ang_energy": ang_energy,
        "star_peaks": star_result['peaks']
    }


def test_natural_image():
    """Test 4: Natural smooth image should show low scores"""
    print("\n" + "="*60)
    print("TEST 4: NATURAL IMAGE (Control Test)")
    print("="*60)
    
    natural = create_natural_image(size=512)
    
    # Preprocess
    img_prep = preprocess_for_fft(natural, jpeg_mitigation=False)
    
    # Compute spectrum
    spectrum = fft_log_spectrum(img_prep)
    
    # Whiten
    whitened = radial_whitening(spectrum)
    
    # Detect spikes
    coords, vals, sym_pairs = detect_spectral_spikes(whitened, z_thresh=6.0)
    
    # Angular analysis
    ang_energy = angular_energy_signature(whitened)
    star_result = star_score_robust(ang_energy)
    
    # Results
    print(f"📊 Results:")
    print(f"  Total spikes: {len(coords)}")
    print(f"  Star score: {star_result['score']:.2f}")
    print(f"  Star symmetry: {star_result['symmetry_score']:.2f}")
    
    # Pass criteria: Should have very low scores
    passed = len(coords) < 10 and star_result['score'] < 10
    
    if passed:
        print(f"\n✅ TEST PASSED: Natural image shows low artifact scores")
    else:
        print(f"\n⚠️  TEST WARNING: Natural image shows unexpected artifacts")
    
    return {
        "test": "natural_image",
        "passed": passed,
        "num_spikes": len(coords),
        "star_score": star_result['score'],
        "whitened": whitened
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_all_tests(results):
    """Create comprehensive visualization of all test results"""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION")
    print("="*60)
    
    fig = plt.figure(figsize=(20, 12))
    
    # Test 1: Resampling
    result_resamp = results['resampling']
    ax1 = plt.subplot(3, 4, 1)
    original, resampled = create_resampled_pattern()
    ax1.imshow(resampled, cmap='gray')
    ax1.set_title(f'Test 1: Resampled Image', fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 4, 2)
    ax2.imshow(result_resamp['whitened'], cmap='seismic', vmin=-10, vmax=10)
    ax2.scatter(result_resamp['coords'][:, 1], result_resamp['coords'][:, 0], 
               c='yellow', s=20, marker='x', linewidths=2)
    ax2.set_title(f"Spikes: {result_resamp['num_spikes']}, Sym: {result_resamp['symmetry_ratio']:.1%}", 
                 fontweight='bold')
    ax2.axis('off')
    
    # Test 2: Fence
    result_fence = results['fence']
    ax3 = plt.subplot(3, 4, 5)
    fence = create_fence_pattern()
    ax3.imshow(fence, cmap='gray')
    ax3.set_title('Test 2: Fence Pattern', fontweight='bold')
    ax3.axis('off')
    
    ax4 = plt.subplot(3, 4, 6)
    ax4.imshow(result_fence['whitened'], cmap='seismic', vmin=-10, vmax=10)
    ax4.set_title(f"Star: {result_fence['star_score']:.1f}, Sym: {result_fence['star_symmetry']:.2f}", 
                 fontweight='bold')
    ax4.axis('off')
    
    ax5 = plt.subplot(3, 4, 7)
    ax5.plot(result_fence['ang_energy'], linewidth=2)
    ax5.set_title('Angular Energy', fontweight='bold')
    ax5.set_xlabel('Angle (bins)')
    ax5.grid(True, alpha=0.3)
    
    # Test 3: Star
    result_star = results['star']
    ax6 = plt.subplot(3, 4, 9)
    star = create_star_pattern()
    ax6.imshow(star, cmap='gray')
    ax6.set_title('Test 3: Star Pattern', fontweight='bold')
    ax6.axis('off')
    
    ax7 = plt.subplot(3, 4, 10)
    ax7.plot(result_star['ang_energy'], linewidth=2, color='red')
    if len(result_star['star_peaks']) > 0:
        ax7.scatter(result_star['star_peaks'], 
                   result_star['ang_energy'][result_star['star_peaks']], 
                   c='yellow', s=100, marker='*', edgecolors='black', linewidths=2)
    ax7.set_title(f"Peaks: {len(result_star['star_peaks'])}, Sym: {result_star['star_symmetry']:.2f}", 
                 fontweight='bold')
    ax7.set_xlabel('Angle (bins)')
    ax7.grid(True, alpha=0.3)
    
    # Test 4: Natural
    result_natural = results['natural']
    ax8 = plt.subplot(3, 4, 3)
    natural = create_natural_image()
    ax8.imshow(natural, cmap='gray')
    ax8.set_title('Test 4: Natural Image', fontweight='bold')
    ax8.axis('off')
    
    ax9 = plt.subplot(3, 4, 4)
    ax9.imshow(result_natural['whitened'], cmap='seismic', vmin=-10, vmax=10)
    ax9.set_title(f"Spikes: {result_natural['num_spikes']}, Star: {result_natural['star_score']:.1f}", 
                 fontweight='bold')
    ax9.axis('off')
    
    # Summary
    ax_summary = plt.subplot(3, 1, 3)
    ax_summary.axis('off')
    
    summary_text = f"""
    ═══════════════════════════════════════════════════════════════════════
    TEST SUMMARY - FFT FORENSICS VALIDATION
    ═══════════════════════════════════════════════════════════════════════
    
    Test 1: Resampling Detection          {'✅ PASSED' if result_resamp['passed'] else '❌ FAILED'}
        • Spikes: {result_resamp['num_spikes']}
        • Symmetry ratio: {result_resamp['symmetry_ratio']:.1%}
        • Expected: High symmetry (>30%) indicates resampling
    
    Test 2: False Positive (Fence)        {'✅ PASSED' if result_fence['passed'] else '⚠️  FAILED'}
        • Star score: {result_fence['star_score']:.2f}
        • Symmetry: {result_fence['star_symmetry']:.2f}
        • Expected: Natural grids should NOT be flagged as AI
    
    Test 3: Star Pattern Detection        {'✅ PASSED' if result_star['passed'] else '❌ FAILED'}
        • Angular peaks: {len(result_star['star_peaks'])}
        • Symmetry: {result_star['star_symmetry']:.2f}
        • Expected: 180° symmetric peaks (AI signature)
    
    Test 4: Natural Image Control         {'✅ PASSED' if result_natural['passed'] else '⚠️  WARNING'}
        • Spikes: {result_natural['num_spikes']}
        • Star score: {result_natural['star_score']:.2f}
        • Expected: Very low scores (< 10)
    
    ═══════════════════════════════════════════════════════════════════════
    OVERALL: {sum([r['passed'] for r in results.values()])}/4 tests passed
    ═══════════════════════════════════════════════════════════════════════
    """
    
    ax_summary.text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
                   verticalalignment='center', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('fft_forensics_validation.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved: fft_forensics_validation.png")
    
    return fig


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """Execute complete test suite"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "FFT FORENSICS TEST SUITE" + " "*19 + "║")
    print("║" + " "*12 + "Code Review Validation Tests" + " "*18 + "║")
    print("╚" + "="*58 + "╝")
    
    results = {}
    
    # Run tests
    results['resampling'] = test_resampling_detection()
    results['fence'] = test_fence_false_positive()
    results['star'] = test_star_pattern_detection()
    results['natural'] = test_natural_image()
    
    # Visualize
    visualize_all_tests(results)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    passed = sum([r['passed'] for r in results.values()])
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.0f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Implementation validated.")
    elif passed >= total - 1:
        print("\n✅ Implementation mostly correct. Minor tuning may be needed.")
    else:
        print("\n⚠️  Multiple tests failed. Review implementation.")
    
    print("\n📊 Detailed results saved to: fft_forensics_validation.png")
    print("="*60)
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
    
    # Keep plot open
    plt.show()
