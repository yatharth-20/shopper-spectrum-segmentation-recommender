# 🛍️ Shopper Spectrum: Customer Segmentation & Product Recommendations

A complete end-to-end Machine Learning capstone project that segments e-commerce customers using RFM analysis and K-Means clustering, then classifies new customers using supervised learning, and recommends products via item-based collaborative filtering.

---

## 📌 Project Overview

| Component | Technique |
|-----------|-----------|
| Customer Segmentation | RFM Analysis + K-Means Clustering (k=4) |
| Segment Classification | Decision Tree, Random Forest, Gradient Boosting |
| Product Recommendations | Item-Based Collaborative Filtering (Cosine Similarity) |
| Deployment | Interactive Streamlit Dashboard |

---

## 📂 Project Structure

```
Shopper Spectrum/
├── Sample_ML_Submission_Template.ipynb   # Full ML notebook (executed with outputs)
├── app.py                                 # Streamlit dashboard
├── .gitignore
└── README.md
```

> **Note:** `online_retail.csv` (46MB) and `shopper_spectrum_models.pkl` (123MB) are excluded from this repo due to GitHub file size limits.
> Download the dataset from [UCI Machine Learning Repository – Online Retail](https://archive.ics.uci.edu/ml/datasets/online+retail), then run the notebook to regenerate the model file.

---

## 🧠 ML Concepts Covered

- **Unsupervised Learning:** K-Means Clustering, Elbow Method, Silhouette Score
- **RFM Analysis:** Recency, Frequency, Monetary feature engineering
- **Supervised Learning:** Decision Tree, Random Forest, Gradient Boosting with GridSearchCV
- **NLP Pipeline:** Tokenization, Stemming, TF-IDF Vectorization
- **Hypothesis Testing:** T-test, Spearman Correlation, ANOVA
- **Recommendation Systems:** Cosine Similarity Item-Based Collaborative Filtering
- **Data Preprocessing:** Log transformation, StandardScaler, missing value handling

---

## 📊 Model Performance

| Model | Accuracy |
|-------|----------|
| Decision Tree (tuned) | ~95% |
| Random Forest (tuned) | ~97% |
| Gradient Boosting (tuned) | ~97% |

---

## 🚀 Running the App

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy streamlit plotly
```

### 2. Download the dataset
Download `Online Retail.xlsx` from [UCI Repository](https://archive.ics.uci.edu/ml/datasets/online+retail) and save as `online_retail.csv`.

### 3. Run the notebook to generate models
```bash
jupyter notebook Sample_ML_Submission_Template.ipynb
```
Run all cells — this saves `shopper_spectrum_models.pkl`.

### 4. Launch the Streamlit app
```bash
streamlit run app.py
```

---

## 📁 Dataset

- **Source:** [UCI Machine Learning Repository – Online Retail](https://archive.ics.uci.edu/ml/datasets/online+retail)
- **Records:** 541,909 transactions
- **Period:** Dec 2010 – Dec 2011
- **Geography:** UK-based online retailer, 38 countries

---

## 🎯 Customer Segments

| Segment | Description |
|---------|-------------|
| 🏆 High-Value | Recently active, frequent buyers, high spend |
| 📈 Regular | Moderately active customers |
| 🔄 Occasional | Infrequent, lower spend |
| ⚠️ At-Risk | Not purchased recently, low frequency |

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `Seaborn` · `Streamlit` · `Plotly` · `SciPy`
