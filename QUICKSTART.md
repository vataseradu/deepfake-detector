# Quick Start - Streamlit Cloud Deploy

## ⚡ TL;DR (5 minute setup)

1. **GitHub**: Push codul
2. **Streamlit Cloud**: Deploy la share.streamlit.io
3. **Share**: Trimite link-ul profesorului
4. **Done!** ✅

---

## 📋 Checklist Pre-Deploy

✅ Fișiere necesare:
- [x] app_production.py
- [x] frequency.py
- [x] gemini_graph_interpreter.py
- [x] requirements.txt
- [x] README.md
- [x] .gitignore

✅ API Key (opțional):
- [ ] Ai OpenAI API key pentru features AI
- [ ] Sau funcționează doar cu scoring matematic (fără API)

---

## 🚀 Deploy în 3 Pași

### Pas 1: GitHub (1 minut)

```bash
git init
git add app_production.py frequency.py gemini_graph_interpreter.py requirements.txt README.md .gitignore
git commit -m "Deploy: Deepfake Detector"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/deepfake-detector.git
git push -u origin main
```

### Pas 2: Streamlit Cloud (2 minute)

1. Du-te la: **https://share.streamlit.io/**
2. Sign in cu GitHub
3. Click **"New app"**
4. Alege repository-ul tău
5. Main file: `app_production.py`
6. Click **"Deploy"**

### Pas 3: Test (1 minut)

1. Așteaptă deployment (2-3 min)
2. Aplicația se deschide automat
3. Upload o imagine test
4. Verifică că funcționează

**Done! 🎉**

---

## 🔑 API Key Setup (Opțional)

Pentru AI interpretation (GPT-4o-mini):

1. În Streamlit dashboard → Settings → Secrets
2. Adaugă:
```toml
OPENAI_API_KEY = "sk-your-key-here"
```
3. Save + Redeploy

**Fără API**: Aplicația funcționează cu scoring matematic!

---

## 🎯 Link-ul Tău

După deploy, aplicația va fi la:
```
https://your-app-name.streamlit.app
```

**Pune acest link în:**
- README.md (actualizează secțiunea Live Demo)
- Documentația disertației
- Email către profesor

---

## 💡 Tips

**Pentru prezentare:**
- Testează cu 3-4 imagini înainte (real + fake)
- Screenshot-uri pentru documentație
- Explică că funcționează și fără API

**Cost:**
- Streamlit Cloud: GRATIS ✅
- OpenAI API (opțional): ~$0.003/imagine

**Probleme?**
- Check logs în Streamlit dashboard
- Verifică requirements.txt
- Test local cu: `streamlit run app_production.py`

---

## 📞 Support

- Streamlit docs: https://docs.streamlit.io/
- Streamlit forum: https://discuss.streamlit.io/
- GitHub issues pentru bugs

---

**Succes! 🎓**
