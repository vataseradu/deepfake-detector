# Testing Guide - Deepfake Detection System

## 🧪 How to Test & Validate Results

### Expected Behavior (CORRECTED)

#### ✅ Real Camera Images Should Show:
- **FFT Score**: 0-40% (LOW score = authentic)
- **High-Freq Ratio**: > 0.01 (visible in debug panel)
- **ELA**: Uniform brightness (low variance)
- **Verdict**: AUTHENTIC (Green) or SUSPICIOUS (Yellow)

#### ❌ AI-Generated Images Should Show:
- **FFT Score**: 60-100% (HIGH score = AI)
- **High-Freq Ratio**: < 0.001 (visible in debug panel)
- **ELA**: Uniform brightness (AI doesn't manipulate, it creates from scratch)
- **Verdict**: AI GENERATED (Red)

#### ✏️ Photoshop/Manipulated Images Should Show:
- **FFT Score**: Variable (depends on base image)
- **ELA**: HIGH variance (bright spots where edited)
- **Verdict**: MANIPULATED (Orange)

---

## 📊 Debug Panel Usage

1. Upload test image
2. Go to **FFT Analysis** tab
3. Expand **"🔧 Detailed FFT Diagnostics"**
4. Check these metrics:

### Key Diagnostic Metrics:

**HF/LF Ratio (Most Important!):**
- `> 0.01` → Real image ✅
- `0.001 - 0.01` → Borderline ⚠️
- `< 0.001` → AI-generated ❌

**Visual Check:**
- In the debug plot, look at the blue power curve
- Real images: Curve maintains power all the way to the right
- AI images: Curve drops sharply or "dies off" on the right side

---

## 🎯 Test Cases

### Test Set 1: Real Photos
**Source**: Your own smartphone photos, DSLR images, unedited JPEGs

**Expected Results:**
- FFT Score: 10-35%
- HF/LF Ratio: 0.02-0.15
- Verdict: AUTHENTIC (Green)

**Why**: Real cameras capture sensor noise, lens artifacts, and natural high-frequency detail.

---

### Test Set 2: AI-Generated (Stable Diffusion, Midjourney, DALL-E)
**Source**: 
- Midjourney Discord exports
- Stable Diffusion web UI outputs
- DALL-E 3 downloads

**Expected Results:**
- FFT Score: 65-95%
- HF/LF Ratio: 0.0001-0.005
- Verdict: AI GENERATED (Red)

**Why**: Diffusion models denoise iteratively, suppressing high frequencies. They can't generate realistic fine-grained noise.

---

### Test Set 3: Photoshop Edits
**Source**: 
- Take a real photo
- Edit it in Photoshop (copy-paste face, remove object, etc.)
- Save as JPEG

**Expected Results:**
- FFT Score: 20-45% (depends on base image)
- ELA Std Dev: > 50 (HIGH)
- ELA shows bright spots at edited regions
- Verdict: MANIPULATED (Orange)

**Why**: ELA detects compression inconsistencies at edit boundaries.

---

### Test Set 4: Heavily Compressed Real Photos
**Source**: 
- Download images from social media (Instagram, Facebook)
- Re-save at low JPEG quality multiple times

**Expected Results:**
- FFT Score: 40-60% (BORDERLINE)
- HF/LF Ratio: 0.005-0.02
- Verdict: SUSPICIOUS (Yellow)

**Why**: Heavy compression removes high frequencies, mimicking AI characteristics.

---

## 🔍 Troubleshooting

### Problem: Real images scored as AI (False Positive)

**Possible Causes:**
1. **Heavy JPEG compression** → Try testing with RAW/PNG files
2. **Instagram/social media downloads** → Platforms heavily compress
3. **HDR or beauty mode** → Smartphone processing smooths images
4. **Very small image size** → < 500px may not have enough freq bins

**Solution:**
- Test with original, unprocessed camera files
- Check HF/LF ratio in debug panel
- If ratio > 0.01 but score is high, report as bug

---

### Problem: AI images scored as real (False Negative)

**Possible Causes:**
1. **Older AI models** (pre-2023) had different artifacts
2. **Post-processed AI images** → Someone added noise/grain
3. **Hybrid images** → AI upscaling of real photos
4. **Very high resolution AI** → Some models preserve frequencies better

**Solution:**
- Check the debug panel spectral plot
- Look for unnatural patterns (oscillations, sudden drops)
- Verify with ELA (AI images should have uniform ELA)

---

## 📈 Calibration & Threshold Tuning

Current thresholds (in `decision_logic.py`):

```python
FFT_THRESHOLD_HIGH = 60.0  # Adjust if too sensitive
FFT_THRESHOLD_LOW = 40.0   # Adjust if too lenient
ELA_STD_THRESHOLD = 45.0   # Manipulation sensitivity
```

**If you get too many false positives (real → AI):**
- Increase `FFT_THRESHOLD_HIGH` to 70
- Increase `FFT_THRESHOLD_LOW` to 50

**If you get too many false negatives (AI → real):**
- Decrease `FFT_THRESHOLD_HIGH` to 50
- Decrease `FFT_THRESHOLD_LOW` to 30

---

## 📚 Academic Validation

**Reference Papers for Testing:**

1. **Durall et al. (2020)** - "Watch Your Up-Convolution"
   - Introduced spectral analysis for GAN detection
   - Found ProGAN, StyleGAN lack high frequencies
   
2. **Frank et al. (2020)** - "Leveraging Frequency Analysis"
   - Tested on multiple datasets (LSUN, CelebA)
   - Achieved 99%+ accuracy with frequency features

3. **Dzanic et al. (2020)** - "Fourier Spectrum Discrepancies"
   - Quantified spectral differences
   - Showed AI images have ~100x lower high-freq power

**Your thesis should include:**
- Confusion matrix (TP, TN, FP, FN rates)
- ROC curve (varying FFT thresholds)
- Test on standard datasets (FFHQ real vs. StyleGAN)

---

## 🛠️ Dataset Recommendations

**For Real Images:**
- **RAISE** (Raw Images of Scenes and Events) - Uncompressed NEF files
- **FFHQ** (Flickr-Faces-HQ) - High quality real faces
- Your own smartphone photos (important!)

**For AI Images:**
- **StyleGAN2 outputs** (from official NVIDIA repo)
- **Stable Diffusion samples** (from HuggingFace)
- **Midjourney exports** (Discord community)
- **DALL-E 3 generations** (via ChatGPT/Bing)

**Balanced Test Set Example:**
- 100 real photos (various sources, cameras, subjects)
- 100 AI images (25 each: StyleGAN, SD, MJ, DALL-E)
- 50 manipulated (Photoshop edits of real photos)
- 50 compressed/degraded real photos (edge cases)

---

## ✅ Validation Checklist

Before presenting your thesis:

- [ ] Test on at least 200 images (100 real, 100 AI)
- [ ] Calculate accuracy, precision, recall, F1-score
- [ ] Document false positives/negatives with examples
- [ ] Compare to baseline methods (ResNet classifier, etc.)
- [ ] Test edge cases (compressed, small, black & white)
- [ ] Verify ELA works on Photoshop edits
- [ ] Include confusion matrix in thesis
- [ ] Show example debug panel outputs
- [ ] Explain why method works (diffusion theory)

---

## 🎓 For Your Thesis Defense

**Expected Questions:**

Q: "Why should we trust frequency analysis?"
A: Backed by multiple peer-reviewed papers (Durall, Frank, Dzanic). Diffusion models mathematically must suppress high frequencies during denoising.

Q: "What about false positives on compressed images?"
A: System flags them as SUSPICIOUS (yellow), not definitively AI. Recommend checking metadata and source.

Q: "Can AI generators defeat this by adding noise?"
A: They can, but it would require post-processing. Authentic-looking noise is hard to synthesize. Future work: detect artificial noise patterns.

Q: "Why not just use a deep learning classifier?"
A: 
1. Interpretability - We can explain WHY an image is flagged
2. Generalization - Works on new models without retraining
3. Academic rigor - Grounded in signal processing theory

---

## 📞 Support

If results don't match expectations:

1. Check debug panel HF/LF ratio
2. Verify image is not heavily compressed
3. Test with known ground truth (your own photo vs. Midjourney)
4. Adjust thresholds if needed
5. Document edge cases for thesis discussion

Good luck with your Master's defense! 🎓
