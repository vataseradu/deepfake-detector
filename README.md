# Deepfake Detector - AI Analysis System

**Academic Research Project** | January 2026  
**Author:** VATASE Radu-Petrut

## Project Overview

Advanced deepfake detection system using FFT (Fast Fourier Transform) analysis combined with OpenAI GPT-4o-mini for intelligent interpretation of frequency domain patterns.

### Live Demo

**Try the application here:** [Streamlit Cloud Link](will-be-updated-after-deployment)

## Features

- **FFT Radial PSD Analysis** - Detects unnatural frequency patterns in AI-generated images
- **2D Spectrum Visualization** - Identifies spatial artifacts from GAN/Diffusion models
- **Mathematical Scoring** - Rule-based detection without requiring API access
- **AI-Powered Interpretation** - GPT-4o-mini analyzes spectral patterns for enhanced accuracy
- **Combined Verdict System** - Merges mathematical and AI analysis (40% Math + 60% AI)

## Installation & Usage

### Quick Start (Recommended)

**Access the live application:** [Your Streamlit Cloud URL]

No installation needed - works directly in browser!

### Local Installation

Prerequisites:
- Python 3.8+
- OpenAI API Key (optional - for AI interpretation)

Setup:

1. Clone repository:
```bash
git clone https://github.com/your-username/deepfake-detector.git
cd deepfake-detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API Key (optional):
   - Open `gemini_graph_interpreter.py`
   - Add OpenAI API key on line 13

4. Run application:
```bash
streamlit run app_production.py
```

## How to Use

1. Upload an image (JPG, JPEG, or PNG)
2. Click "Analizează" button
3. View comprehensive analysis:
   - Mathematical detection score
   - FFT frequency analysis
   - 2D spectrum patterns
   - AI interpretation (if API key configured)
   - Combined final verdict

## Technical Details

### Detection Methods

**Tail Gradient Analysis**
- Analyzes power decay at 70%, 80%, 90% frequency ranges
- Real images: smooth exponential decay
- AI images: abrupt drops or unnatural flatness

**HF/LF Ratio**
- High Frequency / Low Frequency power ratio
- AI-generated images show elevated HF content

**Decay Linearity**
- Measures consistency of frequency decay
- Non-linear patterns indicate AI artifacts

**Star Pattern Detection**
- Identifies symmetric radial patterns in 2D spectrum
- Common artifact in GAN-generated images

### Technology Stack

- Streamlit - Web interface
- NumPy/SciPy - FFT computations
- Matplotlib - Visualization
- OpenCV - Image processing
- OpenAI GPT-4o-mini - AI interpretation
- PIL - Image loading

## Project Structure

```
deepfake-detector/
├── app_production.py              # Main application (production-ready)
├── frequency.py                   # FFT utility functions
├── gemini_graph_interpreter.py    # OpenAI integration
├── requirements.txt               # Dependencies
├── README.md                      # Documentation
└── .gitignore                     # Git ignore rules
```

## Academic Context

This project is part of a research thesis on deepfake detection using frequency domain analysis. The approach combines traditional signal processing with modern AI interpretation.

### Research Contributions

1. **Hybrid Detection** - Combines rule-based math with AI interpretation
2. **Frequency Domain Focus** - Exploits artifacts invisible in spatial domain
3. **Explainable Results** - Detailed metrics for academic evaluation

## Performance

- Mathematical scoring: Works offline (no API needed)
- AI interpretation: Requires OpenAI API (~$0.003-0.025 per image)
- Processing time: 2-5 seconds per image
- Best results: Images > 256x256 pixels

## Deployment

This application is deployed on Streamlit Community Cloud for easy access and demonstration.

## Contact

**VATASE Radu-Petrut**  
Academic Research Project 2026

---

**Note:** Application works without API key (mathematical analysis only). For full AI interpretation, OpenAI API key required.

## Deployment Guide (Streamlit Cloud)

### Prerequisites
1. GitHub account
2. Streamlit Cloud account (free at share.streamlit.io)

### Steps

1. **Push to GitHub:**
```bash
git init
git add app_production.py frequency.py gemini_graph_interpreter.py requirements.txt README.md .gitignore
git commit -m "Initial commit - Deepfake Detector"
git branch -M main
git remote add origin https://github.com/your-username/deepfake-detector.git
git push -u origin main
```

2. **Deploy on Streamlit Cloud:**
   - Go to share.streamlit.io
   - Click "New app"
   - Select your GitHub repository
   - Set main file: `app_production.py`
   - Click "Deploy"

3. **Configure Secrets (Optional - for AI features):**
   - In Streamlit Cloud dashboard, go to App Settings
   - Add secrets in TOML format:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```

Your app will be live at: `https://your-app-name.streamlit.app`

print(f"Mean intensity: {stats['mean_intensity']:.2f}")
```

### Frequency Analysis
```python
from PIL import Image
from frequency import plot_spectrum

image = Image.open("test_image.jpg")
fig = plot_spectrum(image)
fig.savefig("frequency_analysis.png")
```

## 🎓 Academic References

1. **Krawetz, N. (2007)** - "A Picture's Worth..." - FotoForensics blog
   - Foundation of Error Level Analysis

2. **Frank, J., et al. (2020)** - "Leveraging Frequency Analysis for Deep Fake Image Recognition"
   - IEEE Conference on Computer Vision and Pattern Recognition Workshops

3. **Durall, R., et al. (2020)** - "Watch Your Up-Convolution: CNN Based Generative Deep Neural Networks"
   - International Conference on Computer Vision

4. **Dzanic, T., et al. (2020)** - "Fourier Spectrum Discrepancies in Deep Network Generated Images"
   - NeurIPS Workshop on Machine Learning and the Physical Sciences

## 🛠️ Development Standards

### Code Quality
- **Type Hinting:** All functions use Python type hints
- **Docstrings:** Mathematical explanations for all algorithms
- **Error Handling:** Graceful image loading and processing errors

### Streamlit Best Practices (2026)
- ✅ Use `use_container_width=True` instead of deprecated `width` parameter
- ✅ Implement tabs for multi-module interfaces
- ✅ Provide sidebar configuration options

## 🔮 Future Enhancements

- [ ] C2PA/CAI metadata verification
- [ ] Deep learning classifier (CNN-based)
- [ ] Batch processing support
- [ ] Export forensic reports (PDF)
- [ ] Real-time webcam analysis
- [ ] REST API endpoint

## 📝 License

Master's Thesis Research Project - Academic Use

## 👤 Author

Master's Thesis Project | 2026

---

**Note:** This system is designed for academic research and educational purposes. Results should be interpreted by qualified forensic analysts.
