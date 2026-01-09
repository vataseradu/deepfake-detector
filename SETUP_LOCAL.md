# Setup Local - Configurare Path-uri și API Keys

## 📁 Configurare Dataset Local

Scripturile de training necesită path-uri locale către dataset. **NU commitează aceste path-uri în Git!**

### Path-uri necesare:

```python
# Exemplu pentru Windows
FAKE_PATH = r"C:\path\to\your\dataset\training_fake"
REAL_PATH = r"C:\path\to\your\dataset\training_real"

# Exemplu pentru Linux/Mac
FAKE_PATH = "/path/to/your/dataset/training_fake"
REAL_PATH = "/path/to/your/dataset/training_real"
```

### Fișiere care necesită configurare:

- `train_simple_face.py` - Liniile 19-20
- `train_enhanced_face.py` - Liniile 19-20
- `batch_test.py` - Liniile 18-19
- `test_real_images.py` - Linia 12
- `test_fake_images.py` - Linia 12
- `calibrate_face_dataset.py` - Linia 16

## 🔑 OpenAI API Key

Aplicația `app_production.py` caută API key-ul în următoarele locații (în ordine):

1. **Streamlit Secrets** (pentru deployment): 
   - Setează în Streamlit Cloud Dashboard → Secrets
   - Format: `OPENAI_API_KEY = "sk-..."`

2. **Environment Variable**:
   ```bash
   # Windows PowerShell
   $env:OPENAI_API_KEY = "sk-..."
   
   # Linux/Mac
   export OPENAI_API_KEY="sk-..."
   ```

3. **Fișier .env** (local development):
   ```
   OPENAI_API_KEY=sk-...
   ```

⚠️ **IMPORTANT**: Nu commita niciodată API keys în Git!

## 🚀 Rulare Locală

1. Clonează repository:
   ```bash
   git clone https://github.com/vataseradu/deepfake-detector.git
   cd deepfake-detector
   ```

2. Creează virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. Instalează dependențe:
   ```bash
   pip install -r requirements.txt
   ```

4. Configurează API key (vezi secțiunea de mai sus)

5. Rulează aplicația:
   ```bash
   streamlit run app_production.py
   ```

## 📊 Training Models (Optional)

Pentru a antrena modele noi, ai nevoie de dataset FACE:
- 960+ imagini AI-generated (fake)
- 1081+ imagini reale (real)

Descarcă dataset-ul și actualizează path-urile în scripturile de training.

## 🔒 Securitate

✅ **CE SĂ COMMITEZI:**
- Cod sursă (`.py`)
- Documentație (`.md`)
- Requirements (`requirements.txt`)
- Models trained (`.pkl`, `.keras`) - dacă nu sunt prea mari
- Result plots (`.png` - doar rezultate, NU imagini de test)

❌ **CE SĂ NU COMMITEZI:**
- API keys
- Path-uri personale (hardcoded)
- Imagini de test/training
- Fișiere `.env`
- Cache (`__pycache__`)
- Virtual environments (`.venv`)
