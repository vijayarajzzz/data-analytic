import pandas as pd
import matplotlib.pyplot as plt

# 🎨 Make graphs look professional
plt.style.use("ggplot")

# Load your CSV
df = pd.read_csv(r"C:\ML_Project\waste_ai_project\models\dataset\predictions.csv")

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ---------------- 1. Waste Distribution ----------------
counts = df["predicted_label"].value_counts()

print("\n📊 Waste Distribution:")
print(counts)

counts.plot(kind='bar')
plt.title("Waste Distribution")
plt.xlabel("Waste Type")
plt.ylabel("Count")
plt.xticks(rotation=45)          # ✅ FIX: rotate labels
plt.tight_layout()               # ✅ FIX: prevent overlap
plt.show()

# ---------------- 2. Daily Trend ----------------
daily = df.groupby(df["timestamp"].dt.date).size()

print("\n📈 Daily Waste Trend:")
print(daily)

daily.plot(marker='o')           # ✅ nicer line
plt.title("Daily Waste Trend")
plt.xlabel("Date")
plt.ylabel("Total Waste")
plt.xticks(rotation=45)          # ✅ FIX
plt.tight_layout()               # ✅ FIX
plt.show()

# ---------------- 3. Category per Day (IMPROVED) ----------------

# ✅ Take top 5 categories only (avoid clutter)
top_categories = df["predicted_label"].value_counts().head(5).index

filtered = df[df["predicted_label"].isin(top_categories)]

category_day = filtered.groupby(
    [filtered["timestamp"].dt.date, "predicted_label"]
).size().unstack()

category_day.plot(figsize=(10,5), marker='o')
plt.title("Top Waste Types per Day")
plt.xlabel("Date")
plt.ylabel("Count")
plt.xticks(rotation=45)          # ✅ FIX
plt.tight_layout()               # ✅ FIX
plt.show()

# ---------------- 4. Prediction (NEW ⭐) ----------------

prediction = daily.rolling(window=3).mean()

prediction.plot(marker='o')
plt.title("Waste Forecast (Moving Average)")
plt.xlabel("Date")
plt.ylabel("Predicted Waste")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()