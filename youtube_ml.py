import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, confusion_matrix, classification_report
)

import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
N = 500

df = pd.DataFrame({
    "thumbnail_score"  : np.random.randint(1, 11, N),       # 1-10 rating
    "title_length"     : np.random.randint(20, 100, N),     # chars in title
    "channel_subs"     : np.random.randint(1000, 5_000_000, N),
    "video_length_min" : np.random.randint(1, 61, N),       # video duration
    "tags_count"       : np.random.randint(1, 31, N),       # number of tags
    "upload_hour"      : np.random.randint(0, 24, N),       # time uploaded
})

# 🎯 Target 1: watch_time (in seconds) — for regression
# Logic: better thumbnail + more subs + right video length = more watch time
df["watch_time"] = (
    df["thumbnail_score"] * 40
    + df["channel_subs"] / 50000
    + df["video_length_min"] * 10
    - df["upload_hour"] * 2
    + np.random.normal(0, 30, N)   # some noise
).clip(0)

# 🎯 Target 2: watched (0 or 1) — for logistic regression
df["watched"] = (df["watch_time"] > df["watch_time"].median()).astype(int)

# 🎯 Target 3: engagement_level — for multi-class logistic
df["engagement_level"] = pd.cut(
    df["watch_time"],
    bins=3,
    labels=["Low", "Mid", "High"]
)

print(df.head())
print(df.shape)
print(df.describe())
print("\nClass counts for watched:\n", df["watched"].value_counts())
print("\nClass counts for engagement:\n", df["engagement_level"].value_counts())

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("YouTube Dataset - Feature Distributions", fontsize=16)

cols = ["thumbnail_score", "title_length", "channel_subs",
        "video_length_min", "tags_count", "watch_time"]

for i, col in enumerate(cols):
    ax = axes[i//3][i%3]
    ax.hist(df[col], bins=20, color="#ff0000", edgecolor="white", alpha=0.8)
    ax.set_title(col)

plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 6))
sns.heatmap(
    df[["thumbnail_score", "title_length", "channel_subs",
        "video_length_min", "tags_count", "watch_time"]].corr(),
    annot=True, fmt=".2f", cmap="Reds"
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 4))
df["watched"].value_counts().plot(kind="bar", color=["#ff0000", "#444444"],
                                   edgecolor="white")
plt.title("Watched vs Not Watched")
plt.xticks([0, 1], ["Not Watched (0)", "Watched (1)"], rotation=0)
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Features and Target
X = df[["thumbnail_score", "channel_subs", "video_length_min",
        "tags_count", "upload_hour"]]
y_reg   = df["watch_time"]       # for regression
y_cls   = df["watched"]          # for logistic
y_multi = df["engagement_level"] # for multi-class

# Split (80% train, 20% test)
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

X_train, X_test, y_train_cls, y_test_cls = train_test_split(
    X, y_cls, test_size=0.2, random_state=42
)

X_train, X_test, y_train_multi, y_test_multi = train_test_split(
    X, y_multi, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape)
print("Test size :", X_test.shape)
# ── Train ──────────────────────────────────────────
lr_simple = LinearRegression()
lr_simple.fit(X_train[["thumbnail_score"]], y_train_reg)

# ── Predict ────────────────────────────────────────
y_pred_simple = lr_simple.predict(X_test[["thumbnail_score"]])

# ── Metrics ────────────────────────────────────────
print("\n── Simple Linear Regression ──")
print(f"R² Score : {r2_score(y_test_reg, y_pred_simple):.4f}")
print(f"RMSE     : {np.sqrt(mean_squared_error(y_test_reg, y_pred_simple)):.2f}")

# ── Graph ──────────────────────────────────────────
plt.figure(figsize=(8, 5))

# scatter actual points
plt.scatter(X_test["thumbnail_score"], y_test_reg,
            color="#ff0000", alpha=0.6, label="Actual")

# regression line using numpy instead
x_line = np.linspace(X_test["thumbnail_score"].min(),
                     X_test["thumbnail_score"].max(), 100)
y_line = lr_simple.predict(x_line.reshape(-1, 1))

plt.plot(x_line, y_line, color="white", linewidth=2, label="Regression Line")

plt.title("Simple Linear Regression\nThumbnail Score → Watch Time")
plt.xlabel("Thumbnail Score")
plt.ylabel("Watch Time (seconds)")
plt.legend()
plt.tight_layout()
plt.show()
# ── Train ──────────────────────────────────────────
lr_multi = LinearRegression()
lr_multi.fit(X_train, y_train_reg)

# ── Predict ────────────────────────────────────────
y_pred_multi = lr_multi.predict(X_test)

# ── Metrics ────────────────────────────────────────
print("\n── Multiple Linear Regression ──")
print(f"R² Score : {r2_score(y_test_reg, y_pred_multi):.4f}")
print(f"RMSE     : {np.sqrt(mean_squared_error(y_test_reg, y_pred_multi)):.2f}")

# ── Feature Importance (coefficients) ──────────────
coef_df = pd.DataFrame({
    "Feature"     : X_train.columns,
    "Coefficient" : lr_multi.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\nFeature Coefficients:")
print(coef_df)

# ── Graph 1: Actual vs Predicted ───────────────────
plt.figure(figsize=(8, 5))
plt.scatter(y_test_reg, y_pred_multi,
            color="#ff0000", alpha=0.6, label="Predicted vs Actual")
plt.plot([y_test_reg.min(), y_test_reg.max()],
         [y_test_reg.min(), y_test_reg.max()],
         color="white", linewidth=2, label="Perfect Fit Line")
plt.title("Multiple Linear Regression\nActual vs Predicted Watch Time")
plt.xlabel("Actual Watch Time")
plt.ylabel("Predicted Watch Time")
plt.legend()
plt.tight_layout()
plt.show()

# ── Graph 2: Feature Coefficients Bar Chart ─────────
plt.figure(figsize=(8, 5))
plt.barh(coef_df["Feature"], coef_df["Coefficient"],
         color="#ff0000", edgecolor="white")
plt.title("Feature Importance\n(Coefficients from Multiple Linear Regression)")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.show()
# ── Train ──────────────────────────────────────────
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train_cls)

# ── Predict ────────────────────────────────────────
y_pred_cls = log_reg.predict(X_test)

# ── Metrics ────────────────────────────────────────
print("\n── Logistic Regression ──")
print(f"Accuracy : {accuracy_score(y_test_cls, y_pred_cls):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_cls, y_pred_cls))

# ── Graph: Confusion Matrix ─────────────────────────
cm = confusion_matrix(y_test_cls, y_pred_cls)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=["Not Watched", "Watched"],
            yticklabels=["Not Watched", "Watched"])
plt.title("Logistic Regression\nConfusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ── Train ──────────────────────────────────────────
# ── Train ──────────────────────────────────────────
multi_log = LogisticRegression(max_iter=1000)
multi_log.fit(X_train, y_train_multi)
# ── Predict ────────────────────────────────────────
y_pred_multi_cls = multi_log.predict(X_test)

# ── Metrics ────────────────────────────────────────
print("\n── Multi-class Logistic Regression ──")
print(f"Accuracy : {accuracy_score(y_test_multi, y_pred_multi_cls):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_multi, y_pred_multi_cls))

# ── Graph: Confusion Matrix ─────────────────────────
cm_multi = confusion_matrix(y_test_multi, y_pred_multi_cls,
                             labels=["Low", "Mid", "High"])

plt.figure(figsize=(7, 5))
sns.heatmap(cm_multi, annot=True, fmt="d", cmap="Reds",
            xticklabels=["Low", "Mid", "High"],
            yticklabels=["Low", "Mid", "High"])
plt.title("Multi-class Logistic Regression\nConfusion Matrix (Low / Mid / High)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
# ── Model Scores ───────────────────────────────────
models    = ["Simple Linear\nRegression", "Multiple Linear\nRegression",
             "Logistic\nRegression", "Multi-class\nLogistic Regression"]
scores    = [0.2955, 0.9839, 0.97, 0.64]
colors    = ["#555555", "#ff0000", "#ff4444", "#ff9999"]

# ── Graph ──────────────────────────────────────────
plt.figure(figsize=(10, 6))
bars = plt.bar(models, scores, color=colors, edgecolor="white", width=0.5)

# add score labels on top of each bar
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.02,
             f"{score:.2f}", ha="center", fontsize=12, color="white")

plt.title("Model Comparison — All 4 Algorithms", fontsize=15)
plt.ylabel("R² Score / Accuracy")
plt.ylim(0, 1.1)
plt.tight_layout()
plt.show()