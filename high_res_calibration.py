"""
High-Res Research Tool for Master's Thesis
==========================================
Target: High-Resolution Face Dataset
Path: C:/Users/Vatase Radu/Downloads/datetrainingFACE

Metrics:
1. Log HF Ratio (Standard Energy)
2. Tail Gradient (Detection of AI 'Frequency Cutoff')
3. ELA Std (Compression Noise)
"""

import os
import glob
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageEnhance
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# === WORKER FUNCTION ===

def analyze_image(file_info):
    img_path, label = file_info
    
    try:
        with Image.open(img_path) as im:
            im = im.convert('RGB')
            
            # --- 1. ELA (High Res) ---
            pid = multiprocessing.current_process().pid
            temp_filename = f"temp_hr_{pid}.jpg"
            
            # La High-Res, compresia 90 e standard
            im.save(temp_filename, 'JPEG', quality=90)
            resaved = Image.open(temp_filename)
            ela_im = ImageChops.difference(im, resaved)
            
            ela_array = np.array(ela_im)
            gray_ela = np.mean(ela_array, axis=2)
            ela_std = float(np.std(gray_ela))
            
            try:
                resaved.close()
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            except:
                pass

            # --- 2. FFT (Detailed Analysis) ---
            img_gray = np.array(im.convert('L'))
            f = np.fft.fft2(img_gray)
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift) ** 2
            
            # Azimuthal Average
            y, x = np.indices(magnitude.shape)
            center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
            r = np.hypot(x - center[0], y - center[1])
            ind = np.argsort(r.flat)
            r_sorted = r.flat[ind]
            i_sorted = magnitude.flat[ind]
            r_int = r_sorted.astype(int)
            tbin = np.bincount(r_int, i_sorted)
            nr = np.bincount(r_int)
            nr[nr == 0] = 1
            psd1D = tbin / nr
            
            # Validation & Trimming
            psd1D = psd1D[5:]
            valid = psd1D > 0
            if np.sum(valid) < 50: return None # High res needs more points
            psd1D = psd1D[valid]
            n_freq = len(psd1D)
            
            # A. HF Ratio (General)
            cutoff = int(0.6 * n_freq) # 60% frequency mark
            low_p = np.mean(psd1D[:cutoff])
            high_p = np.mean(psd1D[cutoff:])
            
            log_ratio = -10.0
            if low_p > 0 and high_p > 0:
                log_ratio = np.log10(high_p / low_p)
                
            # B. MULTIPLE TAIL GRADIENTS (Enhanced Analysis)
            # Analizăm pe zone diferite pentru a găsi "sweet spot"
            
            tail_70 = 0.0  # 70-100%
            tail_80 = 0.0  # 80-100%
            tail_90 = 0.0  # 90-100%
            spectral_slope = 0.0  # Overall slope
            
            # Zone 70-100%
            start_70 = int(0.7 * n_freq)
            if len(psd1D[start_70:]) > 10:
                x_70 = np.log10(np.arange(start_70, n_freq) + 1).reshape(-1, 1)
                y_70 = np.log10(psd1D[start_70:])
                model_70 = LinearRegression().fit(x_70, y_70)
                tail_70 = float(model_70.coef_[0])
            
            # Zone 80-100%
            start_80 = int(0.8 * n_freq)
            if len(psd1D[start_80:]) > 10:
                x_80 = np.log10(np.arange(start_80, n_freq) + 1).reshape(-1, 1)
                y_80 = np.log10(psd1D[start_80:])
                model_80 = LinearRegression().fit(x_80, y_80)
                tail_80 = float(model_80.coef_[0])
            
            # Zone 90-100%
            start_90 = int(0.9 * n_freq)
            if len(psd1D[start_90:]) > 5:
                x_90 = np.log10(np.arange(start_90, n_freq) + 1).reshape(-1, 1)
                y_90 = np.log10(psd1D[start_90:])
                model_90 = LinearRegression().fit(x_90, y_90)
                tail_90 = float(model_90.coef_[0])
            
            # Overall Spectral Slope (20-100%)
            start_20 = int(0.2 * n_freq)
            if n_freq - start_20 > 20:
                x_full = np.log10(np.arange(start_20, n_freq) + 1).reshape(-1, 1)
                y_full = np.log10(psd1D[start_20:])
                model_full = LinearRegression().fit(x_full, y_full)
                spectral_slope = float(model_full.coef_[0])

            return {
                "type": label,
                "filename": os.path.basename(img_path),
                "log_hf_ratio": log_ratio,
                "tail_70": tail_70,
                "tail_80": tail_80,
                "tail_90": tail_90,
                "spectral_slope": spectral_slope,
                "ela_std": ela_std,
                "n_freq": n_freq
            }
            
    except Exception:
        return None

def main():
    # === CONFIGURARE CALE ===
    # Folosim r"" pentru raw string în Windows
    BASE_PATH = r"C:\Users\Vatase Radu\Downloads\datetrainingFACE"
    
    # Ajustează numele folderelor dacă diferă (ex: 'real' vs 'training_real')
    REAL_PATH = os.path.join(BASE_PATH, "training_real") 
    FAKE_PATH = os.path.join(BASE_PATH, "training_fake")
    
    print(f"📂 Caut imagini în: {BASE_PATH}")
    
    if not os.path.exists(REAL_PATH):
        print(f"❌ Nu găsesc: {REAL_PATH}")
        print("Verifică numele folderelor (sunt 'training_real' și 'training_fake'?)")
        return

    # Indexare
    real_files = [(f, "REAL") for f in glob.glob(os.path.join(REAL_PATH, "*.*"))]
    fake_files = [(f, "FAKE") for f in glob.glob(os.path.join(FAKE_PATH, "*.*"))]
    
    print(f"   Găsite: {len(real_files)} REALE, {len(fake_files)} FAKE.")

    # Eșantionare (Sampling) - Procesăm 1000 din fiecare pentru viteză
    SAMPLE_SIZE = 1000
    if len(real_files) > SAMPLE_SIZE:
        print(f"⚡ Limitare activă: Analizăm random {SAMPLE_SIZE} imagini per clasă.")
        random.shuffle(real_files)
        random.shuffle(fake_files)
        real_files = real_files[:SAMPLE_SIZE]
        fake_files = fake_files[:SAMPLE_SIZE]
    
    all_files = real_files + fake_files
    
    # Procesare Paralelă
    print(f"🚀 Pornire analiză High-Res pe {len(all_files)} imagini...")
    results = []
    with ProcessPoolExecutor() as executor:
        results_gen = list(tqdm(executor.map(analyze_image, all_files), total=len(all_files)))
    
    results = [r for r in results_gen if r is not None]

    if not results:
        print("❌ Eroare: Nicio imagine procesată.")
        return

    # Salvare CSV
    csv_file = "high_res_calibration.csv"
    keys = results[0].keys()
    with open(csv_file, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
    
    # Statistici Complete
    real_ratios = [x['log_hf_ratio'] for x in results if x['type'] == 'REAL']
    fake_ratios = [x['log_hf_ratio'] for x in results if x['type'] == 'FAKE']
    
    real_tail70 = [x['tail_70'] for x in results if x['type'] == 'REAL']
    fake_tail70 = [x['tail_70'] for x in results if x['type'] == 'FAKE']
    
    real_tail80 = [x['tail_80'] for x in results if x['type'] == 'REAL']
    fake_tail80 = [x['tail_80'] for x in results if x['type'] == 'FAKE']
    
    real_tail90 = [x['tail_90'] for x in results if x['type'] == 'REAL']
    fake_tail90 = [x['tail_90'] for x in results if x['type'] == 'FAKE']
    
    real_slope = [x['spectral_slope'] for x in results if x['type'] == 'REAL']
    fake_slope = [x['spectral_slope'] for x in results if x['type'] == 'FAKE']
    
    real_nfreq = [x['n_freq'] for x in results if x['type'] == 'REAL']
    fake_nfreq = [x['n_freq'] for x in results if x['type'] == 'FAKE']

    print("\n" + "="*60)
    print("🧠 STATISTICI FINALE HIGH-RES (DIAGNOSTIC COMPLET)")
    print("="*60)
    print(f"📊 Frecvențe analizate: REAL={np.mean(real_nfreq):.0f} | FAKE={np.mean(fake_nfreq):.0f}")
    print("="*60)
    print(f"Log HF Ratio:    REAL={np.mean(real_ratios):.4f}±{np.std(real_ratios):.4f} | FAKE={np.mean(fake_ratios):.4f}±{np.std(fake_ratios):.4f}")
    print(f"Spectral Slope:  REAL={np.mean(real_slope):.4f}±{np.std(real_slope):.4f} | FAKE={np.mean(fake_slope):.4f}±{np.std(fake_slope):.4f}")
    print("-" * 60)
    print("🎯 TAIL GRADIENTS (ZONA CRITICĂ):")
    print(f"Tail 70-100%:    REAL={np.mean(real_tail70):.4f}±{np.std(real_tail70):.4f} | FAKE={np.mean(fake_tail70):.4f}±{np.std(fake_tail70):.4f}")
    print(f"Tail 80-100%:    REAL={np.mean(real_tail80):.4f}±{np.std(real_tail80):.4f} | FAKE={np.mean(fake_tail80):.4f}±{np.std(fake_tail80):.4f}")
    print(f"Tail 90-100%:    REAL={np.mean(real_tail90):.4f}±{np.std(real_tail90):.4f} | FAKE={np.mean(fake_tail90):.4f}±{np.std(fake_tail90):.4f}")
    print("="*60)
    
    # Calculare Delta (Diferența)
    delta_70 = abs(np.mean(real_tail70) - np.mean(fake_tail70))
    delta_80 = abs(np.mean(real_tail80) - np.mean(fake_tail80))
    delta_90 = abs(np.mean(real_tail90) - np.mean(fake_tail90))
    
    print("\n🔍 DIFERENȚE MĂSURATE (Δ):")
    print(f"Δ Tail 70%: {delta_70:.4f}")
    print(f"Δ Tail 80%: {delta_80:.4f}")
    print(f"Δ Tail 90%: {delta_90:.4f}")
    print(f"Δ Log Ratio: {abs(np.mean(real_ratios) - np.mean(fake_ratios)):.4f}")
    print("="*60)
    
    # Plotting Complet
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('High-Res Spectral Analysis - Complete Diagnostic', fontsize=16, fontweight='bold')
    
    # Row 1: Tail Gradients
    axes[0, 0].hist(real_tail70, bins=30, alpha=0.6, label='REAL', color='green')
    axes[0, 0].hist(fake_tail70, bins=30, alpha=0.6, label='FAKE', color='red')
    axes[0, 0].set_title(f'Tail 70-100% (Δ={delta_70:.4f})')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('Gradient')
    
    axes[0, 1].hist(real_tail80, bins=30, alpha=0.6, label='REAL', color='green')
    axes[0, 1].hist(fake_tail80, bins=30, alpha=0.6, label='FAKE', color='red')
    axes[0, 1].set_title(f'Tail 80-100% (Δ={delta_80:.4f})')
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('Gradient')
    
    axes[0, 2].hist(real_tail90, bins=30, alpha=0.6, label='REAL', color='green')
    axes[0, 2].hist(fake_tail90, bins=30, alpha=0.6, label='FAKE', color='red')
    axes[0, 2].set_title(f'Tail 90-100% (Δ={delta_90:.4f})')
    axes[0, 2].legend()
    axes[0, 2].set_xlabel('Gradient')
    
    # Row 2: Other Metrics
    axes[1, 0].hist(real_ratios, bins=30, alpha=0.6, label='REAL', color='green')
    axes[1, 0].hist(fake_ratios, bins=30, alpha=0.6, label='FAKE', color='red')
    axes[1, 0].set_title('Log HF Ratio')
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('Log Ratio')
    
    axes[1, 1].hist(real_slope, bins=30, alpha=0.6, label='REAL', color='green')
    axes[1, 1].hist(fake_slope, bins=30, alpha=0.6, label='FAKE', color='red')
    axes[1, 1].set_title('Overall Spectral Slope')
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Slope')
    
    # Box plot pentru cel mai discriminativ tail
    best_tail_data = [real_tail70, fake_tail70] if delta_70 >= max(delta_80, delta_90) else ([real_tail80, fake_tail80] if delta_80 >= delta_90 else [real_tail90, fake_tail90])
    axes[1, 2].boxplot(best_tail_data, labels=['REAL', 'FAKE'])
    axes[1, 2].set_title('Best Discriminator (Box Plot)')
    axes[1, 2].set_ylabel('Gradient Value')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("high_res_plot.png", dpi=150)
    print("📈 Grafic salvat: high_res_plot.png")
    plt.show()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
