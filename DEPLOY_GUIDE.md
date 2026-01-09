# Ghid Deploy Streamlit Cloud

## De ce Streamlit Cloud?

✅ **Gratis** pentru proiecte publice  
✅ **Live instant** - profesorul accesează direct din browser  
✅ **Nu necesită setup** - fără instalări, dependințe, Python  
✅ **Auto-deploy** - orice push pe GitHub => aplicația se actualizează  
✅ **Perfect pentru facultate** - demonstrație profesională

## Pași pentru Deploy

### 1. Pregătire GitHub

Creează un repository nou pe GitHub (public):
- Nume: `deepfake-detector` (sau similar)
- Descriere: "Academic research - Deepfake detection using FFT analysis"

### 2. Push Cod pe GitHub

Din terminal (PowerShell), în folderul proiectului:

```bash
# Inițializează git (dacă nu e deja)
git init

# Adaugă fișierele importante
git add app_production.py
git add frequency.py
git add gemini_graph_interpreter.py
git add requirements.txt
git add README.md
git add .gitignore

# Commit
git commit -m "Initial commit - Deepfake Detector"

# Setează branch-ul principal
git branch -M main

# Conectează la GitHub (înlocuiește cu URL-ul tău)
git remote add origin https://github.com/TauUsername/deepfake-detector.git

# Push
git push -u origin main
```

### 3. Deploy pe Streamlit Cloud

1. **Mergi la:** https://share.streamlit.io/

2. **Sign in cu GitHub** (dacă nu ai cont, creează unul - e gratuit)

3. **Click "New app"**

4. **Configurează:**
   - Repository: Alege `your-username/deepfake-detector`
   - Branch: `main`
   - Main file path: `app_production.py`
   - App URL: Alege un nume (ex: `deepfake-detector-radu`)

5. **Click "Deploy"**

**Așteaptă 2-3 minute** - Streamlit instalează dependințele și pornește aplicația

### 4. Configurare API Key (Opțional)

Pentru funcția AI (GPT-4o-mini):

1. În dashboard-ul Streamlit, click pe app-ul tău
2. Click "Settings" (hamburger menu)
3. Click "Secrets"
4. Adaugă în format TOML:

```toml
OPENAI_API_KEY = "sk-your-actual-key-here"
```

5. Click "Save"

Apoi în `gemini_graph_interpreter.py`, modifică:
```python
import os
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
```

### 5. Verificare

Aplicația ta e live la: `https://your-app-name.streamlit.app`

Test rapid:
- Upload o imagine
- Verifică că analizele funcționează
- Verifică graficele

### 6. Share cu Profesorul

Link-ul e format:
```
https://deepfake-detector-radu.streamlit.app
```

**Pune acest link în:**
- README.md pe GitHub
- Documentația lucrării
- Email către profesor

## Troubleshooting

### Eroare: "ModuleNotFoundError"
- Verifică că toate dependințele sunt în `requirements.txt`
- Redeploy app-ul

### Aplicația nu pornește
- Check logs în Streamlit Cloud dashboard
- Verifică că `app_production.py` nu are erori de sintaxă

### API Key nu funcționează
- Verifică că ai adăugat în Secrets (nu în cod direct)
- Redeploy după adăugarea secrets

## Avantaje pentru Evaluare

✅ Profesorul poate testa instant - fără instalări  
✅ Accesibil de pe orice device (laptop, telefon)  
✅ Professional presentation  
✅ Istoricul versiunilor pe GitHub  
✅ Cod + demo live = impresie maximă

## Cost

**ZERO LEI** - Streamlit Community Cloud e complet gratuit pentru proiecte publice!

API OpenAI (opțional):
- ~$0.003-0.025 per imagine
- 100 imagini ≈ $0.30-2.50
- Poți folosi fără API - funcționează scoring matematic

## Link-uri Utile

- Streamlit Cloud: https://share.streamlit.io/
- Documentație: https://docs.streamlit.io/deploy/streamlit-community-cloud
- GitHub: https://github.com/

## Next Steps

După deploy:
1. ✅ Testează aplicația live
2. ✅ Updatează README.md cu link-ul live
3. ✅ Trimite link-ul profesorului
4. ✅ Adaugă imagini de test în documentație

---

**Succes cu disertația! 🎓**
