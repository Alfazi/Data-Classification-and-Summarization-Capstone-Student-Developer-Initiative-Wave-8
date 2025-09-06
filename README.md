# Capstone Project: Mental Health Text Classification with ML & Granite LLM

## 📌 Deskripsi
Proyek ini mengklasifikasikan postingan Reddit terkait kesehatan mental ke dalam lima kategori:
- Stress
- Depression
- Bipolar disorder
- Personality disorder
- Anxiety

Dua pendekatan dibandingkan:
1. **Baseline Machine Learning**: TF-IDF + Logistic Regression
2. **Granite LLM (via Replicate API)**: Model IBM Granite yang memberi label + penjelasan

## 📊 Dataset
- **Sumber**: Reddit Mental Health [Kaggle](https://www.kaggle.com/datasets/neelghoshal/reddit-mental-health-data)
- **Jumlah data**: 5.957 postingan
- **Fitur**:
  - `text`: isi postingan
  - `target`: label numerik
  - `label`: kategori teks (mapped ke 5 kelas)

## ⚙️ Alur Notebook
1. **Setup & Import**
   - Install library: `pandas`, `scikit-learn`, `matplotlib`, `wordcloud`, `replicate`
2. **Load Dataset**
   - Pemetaan label numerik → teks
3. **Preprocessing**
   - Case folding, hapus link, tanda baca, normalisasi spasi
4. **EDA**
   - Distribusi kelas (bar chart)
   - WordCloud per kategori (contoh: Depression)
5. **Baseline ML**
   - TF-IDF (max_features=5000)
   - Logistic Regression (class_weight balanced)
   - Akurasi ≈ 70%, macro F1 ≈ 0.70
6. **Integrasi Granite**
   - Prompt → JSON {label, explanation}
   - Contoh prediksi:
     ```json
     {
       "label": 1,
       "explanation": "Describes isolation and drowning feeling, common in depression."
     }
     ```
   - Hasil disimpan ke: `reddit_mental_health_granite_sample.csv`

## 🏆 Hasil
- **Logistic Regression**: lebih stabil kuantitatif (akurasi 70%)
- **Granite**: memberi *reasoning* interpretatif, meski kadang hasil tidak konsisten
- Insight: kombinasi ML + LLM bisa memberi klasifikasi cepat sekaligus transparan

## ⚙️ Teknologi
- Python (pandas, scikit-learn, matplotlib, seaborn, wordcloud)
- TF-IDF + Logistic Regression
- IBM Granite LLM (Replicate API)

## 📝 Catatan
Proyek ini hanya untuk penelitian akademik, **bukan diagnosis medis**.
