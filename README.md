# Spotify Müzik Verisi Analizi ve Makine Öğrenmesi Projesi

**Öğrenci:** Firuze Eroğlu  
**Numara:** 201613709044  
**Dönem:** 2025 Güz (F25)

## Proje Hakkında
Bu depo, Spotify verilerini kullanarak gerçekleştirilen kapsamlı bir makine öğrenmesi dönem projesini içermektedir. Proje kapsamında üç temel makine öğrenmesi görevi uygulanmıştır:

1.  **Sınıflandırma (Classification):**
    *   **Amaç:** Bir şarkının ses özelliklerine (dans edilebilirlik, enerji, ses şiddeti vb.) dayanarak "Popüler" olup olmadığını (Binary Classification) tahmin etmek.
    *   **Kullanılan Modeller:** Logistic Regression, Random Forest (Optimize Edilmiş), XGBoost.
    *   **Veri Seti:** `dataset.csv` (Spotify Tracks Dataset)

2.  **Regresyon (Regression):**
    *   **Amaç:** Şarkı ve sanatçı özelliklerini kullanarak şarkının popülerlik puanını (0-100 arası) sayısal olarak tahmin etmek.
    *   **Kullanılan Modeller:** Linear Regression, Random Forest Regressor, Gradient Boosting.
    *   **Veri Seti:** `spotify_data clean.csv` (Spotify Global Music Dataset)

3.  **Kümeleme (Clustering):**
    *   **Amaç:** 2024'ün en çok dinlenen şarkılarını, farklı platformlardaki (Spotify, TikTok, YouTube) etkileşimlerine göre gruplandırmak (segmentasyon).
    *   **Kullanılan Algoritmalar:** K-Means, MiniBatch K-Means, DBSCAN.
    *   **Veri Seti:** `Most Streamed Spotify Songs 2024.csv`

## Dosya Yapısı

*   `ML_Project_Final.ipynb`: Projenin tüm kodlarını, analizlerini, görselleştirme çıktılarını ve detaylı rapor metnini içeren Jupyter Notebook dosyası.
*   `ML_Project_Report_Final.pdf`: Notebook dosyasının rapor formatındaki çıktısı.
*   `m_l_png/`: Proje çalıştırıldığında üretilen grafikler (Confusion Matrix, SHAP analizi, Regresyon grafikleri vb.).
*   `dataset.csv`: Sınıflandırma probleminde kullanılan veri seti.
*   `spotify_data clean.csv`: Regresyon probleminde kullanılan veri seti.
*   `Most Streamed Spotify Songs 2024.csv`: Kümeleme probleminde kullanılan veri seti.

## Nasıl Çalıştırılır?

Projenin bağımlılıklarını yükledikten sonra (pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, shap, imblearn), `ML_Project_Final.ipynb` dosyasını Jupyter Notebook veya Google Colab üzerinden çalıştırabilirsiniz.
