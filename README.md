# 🎵 Spotify Data Analysis & Machine Learning Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/frzerxz/spotify-ml-analysis/blob/main/Spotify_ML_Final.ipynb)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This project is an end-to-end Machine Learning analysis based on real-world Spotify data. It covers three core ML domains: **Classification**, **Regression**, and **Clustering**. 

The goal is to predict song popularity, analyze the factors behind a "hit" song, and segment music tracks based on streaming and social media interaction metrics.

## 🚀 Key Features & Techniques
*   **End-to-End Pipeline:** Data cleaning, preprocessing, modeling, and evaluation.
*   **Advanced Preprocessing:** 
    *   Handled **imbalanced data** using `RandomUnderSampler`.
    *   StandardScaler for feature normalization.
    *   Outlier detection and removal using **IQR (Interquartile Range)**.
*   **Model Optimization:** Hyperparameter tuning using `RandomizedSearchCV`.
*   **Explainable AI (XAI):** Used **SHAP (SHapley Additive exPlanations)** values to interpret model decisions.

---

## 📊 Project Modules

### 1. Classification Task: Hit Song Prediction 🏆
*   **Goal:** Predict whether a song will be "Popular" (Popularity > 50) based on audio features like `danceability`, `energy`, `loudness`.
*   **Models Used:** Logistic Regression, Random Forest (Optimized), XGBoost.
*   **Key Insight:** Addressed severe class imbalance. While the base Accuracy was ~56%, optimization and undersampling improved the **Recall** significantly, allowing the model to detect hits accurately (~74% Accuracy).
*   **Top Features:** `Loudness`, `Acousticness`, `Energy`.

### 2. Regression Task: Popularity Score Estimation 📈
*   **Goal:** Predict the exact popularity score (0-100) of a track using metadata (Artist popularity, Album type, etc.).
*   **Models Used:** Linear Regression, Random Forest Regressor, Gradient Boosting.
*   **Key Insight:** The most dominant factor for a song's success is not its audio quality, but the **Artist's Popularity** and Fanbase.
*   **Performance:** R² Score ~ 0.24 (indicating external factors like marketing play a huge role beyond metadata).

### 3. Clustering Task: Music Segmentation 🎧
*   **Goal:** Group songs into segments based on streams, playlist counts, and social media (TikTok/YouTube) views without labels (Unsupervised).
*   **Models Used:** K-Means, MiniBatch K-Means, DBSCAN.
*   **Technique:** Used **PCA (Principal Component Analysis)** to visualize clusters in 2D.
*   **Result:** The optimal number of clusters was found to be **K=2** (via Silhouette Score), clearly separating "Mega-Hits" from "Niche/Average" songs.

---

## 🛠️ Tech Stack
*   **Language:** Python
*   **Libraries:** 
    *   `pandas`, `numpy` (Data Manipulation)
    *   `scikit-learn` (Modeling, Preprocessing)
    *   `xgboost` (Advanced Gradient Boosting)
    *   `shap` (Model Explainability)
    *   `imbalanced-learn` (Handling Imbalanced Data)
    *   `matplotlib`, `seaborn` (Visualization)

## 📂 Project Structure
```
├── Spotify_ML_Final.ipynb               # Main Jupyter Notebook containing all codes
├── ML_Project_Report.pdf                # Detailed Project Report
├── dataset.csv                          # Classification Dataset
├── spotify_data clean.csv               # Regression Dataset
├── Most_Streamed_Spotify_Songs_2024.csv # Clustering Dataset
└── README.md                            # Project Documentation
```

## ✍️ Author
**Firuze Eroğlu**  
*Computer Engineering Student*

---
*This project was developed as a term assignment for the Machine Learning course (Fall 2025).*
