# 🤖 Ghid Integrare Google Gemini AI pentru Interpretare FFT

## 📋 Pas 1: Obține API Key

1. Mergi la **Google AI Studio**: https://makersuite.google.com/app/apikey
2. Autentifică-te cu contul Google
3. Click pe **"Get API Key"** sau **"Create API Key"**
4. Copiază cheia (format: `AIzaSy...`)

⚠️ **IMPORTANT**: Păstrează cheia secretă! Nu o partaja public.

---

## 📦 Pas 2: Instalează pachetul

În terminal, cu virtual environment activat:

```powershell
# Activează venv (dacă nu e deja activ)
.\.venv\Scripts\Activate.ps1

# Instalează Gemini SDK
pip install google-generativeai

# Verifică instalarea
python -c "import google.generativeai as genai; print('✅ Gemini installed!')"
```

---

## 🚀 Pas 3: Repornește aplicația

```powershell
# Oprește Streamlit curent
Get-Process -Name streamlit | Stop-Process -Force

# Repornește
streamlit run app_final.py
```

Acum mesajul `⚠️ Gemini interpreter not available` ar trebui să dispară!

---

## 💡 Pas 4: Folosește interpretarea AI

1. Încarcă o imagine în aplicație
2. Așteaptă analiza să se finalizeze
3. Mergi la tab-ul **"📚 Interpretare"**
4. Deschide expandable-ul **"💡 Obține interpretare AI a graficelor FFT"**
5. Introdu API Key-ul în câmpul de text
6. (Opțional) Bifează "Folosește Gemini Vision" pentru a trimite și graficul
7. Click pe **"🚀 Analizează cu Gemini AI"**

---

## 📊 Ce primești:

### 1. Verdict AI:
- ✅ **REAL** - imaginea pare autentică
- 🤖 **AI-GENERATED** - imaginea pare generată de AI

### 2. Confidence Score:
- 0-100% - cât de sigur este AI-ul de verdict
- Progress bar vizual

### 3. Raționament detaliat:
- Explicație tehnică CE pattern-uri indică AI
- De ce anume acea concluzie

### 4. Key Indicators:
- Top 3 indicatori cei mai importanți
- Exemple: "Vârf la frecvență 120px", "Drop abrupt >90%"

### 5. Natural Signals:
- Ce semne arată că ar putea fi REAL (dacă există)
- Contraargumente

### 6. Recommendation:
- Sugestii pentru utilizator
- Ce să verifice în plus

---

## 🔒 Ce date se trimit către Gemini?

### ✅ SE TRIMITE:
- **Graficul PSD** - ca imagine PNG generată în memorie
- **Date numerice**: PSD profile (array de valori)
- **Statistici**: mean, std, decay rate
- **Pattern-uri detectate**: star pattern, periodic spikes, etc.
- **Features**: tail gradients, HF/LF ratio, ELA std

### ❌ NU SE TRIMITE:
- **Imaginea originală încărcată de utilizator**
- **Metadata EXIF** (locație, cameră, etc.)
- **Orice informație personală**

---

## 💰 Costuri Google Gemini API

### Gemini 1.5 Flash (recomandat):
- **Free tier**: 15 requests/minute, 1500 requests/day
- **Paid**: $0.075 per 1M tokens input, $0.30 per 1M tokens output
- **Vision**: $0.0015 per imagine (cu graficul)

### Gemini 1.5 Pro (mai precis):
- **Free tier**: 2 requests/minute, 50 requests/day
- **Paid**: $1.25 per 1M tokens input, $5 per 1M tokens output

Pentru uz personal/teză, **free tier este suficient**!

---

## 🛠️ Troubleshooting

### Eroare: "API Key invalid"
```
❌ Verifică că ai copiat corect întreaga cheie
❌ Asigură-te că nu are spații la început/sfârșit
❌ Verifică că API Key-ul este activ în console
```

### Eroare: "Quota exceeded"
```
⏳ Ai depășit limita free tier (15 req/min sau 1500 req/day)
💡 Așteaptă câteva minute sau upgrade la plan paid
```

### Eroare: "Model not available"
```
🔧 Schimbă modelul în gemini_interpreter.py:
   - gemini-1.5-flash (implicit, mai rapid)
   - gemini-1.5-pro (mai precis, mai scump)
   - gemini-pro (text-only, fără vision)
```

### Aplicația nu detectează Gemini după instalare
```
1. Repornește complet terminalul
2. Re-activează virtual environment
3. Verifică: python -c "import google.generativeai; print('OK')"
4. Repornește Streamlit
```

---

## 🎯 Exemple de prompt-uri Gemini

Gemini primește un prompt customizat cu toate datele tale:

```
Ești un expert în detectarea deepfake-urilor folosind analiza FFT.

Datele tale:
- PSD Radial 1D: 256 puncte, Mean: 45.2 dB, Decay: -0.15 dB/pixel
- Pattern-uri: Star Pattern: DA ⭐, Periodic: NU, Decay: DA 📉
- Features: Tail -35.5 dB/dec, HF/LF: 0.0012

Context tehnic:
- Imagini REALE: decay smooth ~1/f²
- AI (GAN): vârfuri mid/high freq, drop abrupt >90%

Răspunde cu JSON:
{
  "verdict": "REAL" sau "AI-GENERATED",
  "confidence": 85,
  "reasoning": "...",
  "key_indicators": [...],
  "natural_signals": [...],
  "recommendation": "..."
}
```

---

## 📚 Resurse Utile

- **Google AI Studio**: https://makersuite.google.com/
- **Gemini API Docs**: https://ai.google.dev/docs
- **Pricing**: https://ai.google.dev/pricing
- **Cookbook**: https://github.com/google-gemini/cookbook

---

## ⚙️ Configurare Avansată

### Setează API Key ca variabilă de mediu (recomandată):

**PowerShell (temporar - sesiune curentă):**
```powershell
$env:GEMINI_API_KEY = "AIzaSy..."
```

**PowerShell (permanent - user):**
```powershell
[System.Environment]::SetEnvironmentVariable('GEMINI_API_KEY', 'AIzaSy...', 'User')
```

**Apoi în cod** (modifică gemini_interpreter.py):
```python
import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Citește automat
```

Astfel nu mai trebuie să introduci manual în aplicație!

---

## 🧪 Test Manual

Test rapid în Python:

```python
import google.generativeai as genai

# Configurare
genai.configure(api_key="AIzaSy...")

# Test simplu
model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("Hello! Confirm you're working.")
print(response.text)

# Dacă vezi răspuns → funcționează! ✅
```

---

## 📊 Comparație modele:

| Model | Speed | Accuracy | Cost | Vision | Use Case |
|-------|-------|----------|------|--------|----------|
| **gemini-1.5-flash** | ⚡⚡⚡ | ⭐⭐⭐ | 💰 | ✅ | Testing, uz personal |
| **gemini-1.5-pro** | ⚡⚡ | ⭐⭐⭐⭐⭐ | 💰💰💰 | ✅ | Production, high accuracy |
| **gemini-pro** | ⚡⚡⚡ | ⭐⭐⭐⭐ | 💰 | ❌ | Text-only, no graphs |

Pentru teză: **gemini-1.5-flash** e perfect! 🎓

---

**Status**: ✅ Gata de utilizare  
**Support**: Dacă ai probleme, verifică logs în terminal  
**Enjoy**: Happy AI-powered analysis! 🚀🤖
