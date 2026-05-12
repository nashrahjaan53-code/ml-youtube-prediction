# 🎬 YouTube Watch Predictor — ML Project

Predicting if someone will watch a YouTube video using Machine Learning.

## 📊 Dataset
- 500 synthetic YouTube video records
- Features: Thumbnail Score, Channel Subscribers, Video Length, Tags Count, Upload Hour

## 🤖 Algorithms Used
| Algorithm | Task | Score |
|-----------|------|-------|
| Simple Linear Regression | Predict Watch Time (1 feature) | R² = 0.29 |
| Multiple Linear Regression | Predict Watch Time (all features) | R² = 0.98 |
| Logistic Regression | Watched or Not (0/1) | Accuracy = 97% |
| Multi-class Logistic Regression | Low/Mid/High Engagement | Accuracy = 64% |

## 📈 Visualizations
- Feature distributions
- Correlation heatmap
- Regression line & Actual vs Predicted
- Confusion matrices
- Model comparison chart

## 🛠️ Libraries Used
- pandas, numpy, matplotlib, seaborn, scikit-learn

## ▶️ How to Run
pip install pandas numpy matplotlib seaborn scikit-learn
python youtube_ml.py
