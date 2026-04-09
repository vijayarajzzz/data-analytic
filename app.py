import streamlit as st
import plotly.express as px
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import os
import cv2
import time
st.markdown("""
<style>
/* Smooth fade-in */
section.main > div {
    animation: fadeIn 0.6s ease-in;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}

/* Card hover effect */
[data-testid="stMetric"] {
    transition: 0.3s;
}
[data-testid="stMetric"]:hover {
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #020617);
}
</style>
""", unsafe_allow_html=True)

# ---------------- PERFORMANCE FIX ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/mobilenet_finetuned.keras")

model = load_model()


def load_data():
    if not os.path.exists("models/predictions.csv"):
        return pd.DataFrame()

    df = pd.read_csv("models/predictions.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format='mixed', errors='coerce')
    return df
# 🔥 Import Grad-CAM
from gradcam import generate_gradcam, overlay_heatmap

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Waste AI System",
    layout="wide",
    page_icon="♻️"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: #e2e8f0;
}

h1, h2, h3 {
    color: #22c55e;
    font-weight: 600;
}

.block-container {
    padding: 2rem;
}

.stButton>button {
    background: linear-gradient(90deg, #22c55e, #16a34a);
    border-radius: 12px;
    font-weight: bold;
}

.stButton>button:hover {
    transform: scale(1.05);
    transition: 0.2s;
}

section[data-testid="stSidebar"] {
    background: #020617;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Home", "Prediction", "Analytics", "Insights"],
        icons=["house", "camera", "bar-chart", "graph-up"],
        default_index=0
    )

# ---------------- LOAD MODEL ----------------


class_labels = [
    'battery','biological','cardboard','clothes',
    'glass','metal','paper','plastic','shoes','trash'
]

# ---------------- SAVE FUNCTION ----------------
def save_prediction(label, confidence):
    csv_file = "models/predictions.csv"   # ALWAYS SAME FOLDER AS app.py

    file_exists = os.path.exists(csv_file)

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["timestamp", "true_label", "predicted_label"])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            "user_input",
            label
        ])
# ---------------- HOME ----------------
if selected == "Home":
    st.title("♻️ Intelligent Waste Segregation Analytics Using Machine Learning")
    st.caption("AI-powered waste classification & analytics platform")

    df = load_data()
    
    col1, col2, col3 = st.columns(3)

    if not df.empty:
        total = len(df)
        most_common = df["predicted_label"].mode()[0]
        categories = df["predicted_label"].nunique()

        # 🔥 Animated counter
        placeholder = col1.empty()
        for i in range(max(0, total - 10), total + 1):
            placeholder.metric("📊 Total Predictions", i)
            time.sleep(0.05)

        col2.metric("♻️ Most Common", most_common)
        col3.metric("🧩 Categories", categories)

    else:
        col1.metric("📊 Total Predictions", "0")
        col2.metric("♻️ Most Common", "-")
        col3.metric("🧩 Categories", "0")

    st.divider()

    # ---------------- QUICK VISUAL ----------------
    if not df.empty:
        st.subheader("📈 Recent Activity")

        recent = df.tail(50)
        trend = recent.groupby(recent["timestamp"].dt.date).size()

        st.line_chart(trend)

    st.divider()

    # ---------------- FEATURES ----------------
    st.subheader("🚀 Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        - 📸 Upload waste images  
        - 🤖 AI classification  
        - 🔥 Grad-CAM visualization  
        """)

    with col2:
        st.markdown("""
        - 📊 Waste analytics  
        - 🧠 Predictive insights  
        - 📈 Trend monitoring  
        """)

    st.divider()

    # ---------------- INSIGHT PREVIEW ----------------
    if not df.empty:
        st.subheader("🧠 Quick Insight")

        daily = df.groupby(df["timestamp"].dt.date).size()

        if daily.iloc[-1] < daily.mean():
            st.success("Waste is decreasing 📉")
        else:
            st.warning("Waste is increasing 📈")

    else:
        st.info("Upload some images to see insights 🚀")

# ---------------- PREDICTION ----------------
elif selected == "Prediction":
    
    st.title("📸 Waste Prediction (Multi-Image)")

    uploaded_files = st.file_uploader(
        "Upload up to 10 images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:

        if len(uploaded_files) > 10:
            st.warning("⚠️ You can upload maximum 10 images only.")
        else:
            st.success(f"{len(uploaded_files)} images uploaded successfully!")

            cols = st.columns(2)  # grid layout

            for idx, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file)

                # preprocess
                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                with st.spinner(f"Processing image {idx+1}..."):
                    prediction = model.predict(img_array)

                pred_class = class_labels[np.argmax(prediction)]
                confidence = np.max(prediction)

                # Grad-CAM
                heatmap = generate_gradcam(model, img_array)
                output = overlay_heatmap(heatmap, np.array(image))

                # display in grid
                with cols[idx % 2]:
                    st.image(image, caption=f"Image {idx+1}", use_container_width=True)

                    st.success(f"Prediction: {pred_class}")
                    st.progress(int(confidence * 100))
                    st.write(f"Confidence: {confidence:.2f}")

                    st.image(output, caption="Grad-CAM", use_container_width=True)

                    st.divider()

                # save prediction
                save_prediction(pred_class, confidence)

# ---------------- ANALYTICS ----------------
elif selected == "Analytics":
    st.title("📊 Waste Analytics Dashboard")

    df = load_data()

    if df.empty:
        st.warning("No data available. Upload images first.")
    else:
        # ---------------- KPI CARDS ----------------
        st.subheader("📌 Key Metrics")

        daily = df.groupby(df["timestamp"].dt.date).size()

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Waste", len(df))
        col2.metric("Most Common", df["predicted_label"].mode()[0])
        col3.metric("Avg Daily Waste", int(daily.mean()))

        st.divider()

        # ---------------- FILTER ----------------
        st.subheader("🎛 Filter Data")

        categories = df["predicted_label"].unique()
        selected_categories = st.multiselect(
            "Select Waste Types",
            categories,
            default=categories[:5]
        )

        filtered = df[df["predicted_label"].isin(selected_categories)]

        # ---------------- CATEGORY TREND ----------------
        st.subheader("📈 Waste Trend by Category")

        category_day = filtered.groupby(
            [filtered["timestamp"].dt.date, "predicted_label"]
        ).size().reset_index(name="count")

        fig = px.line(
            category_day,
            x="timestamp",
            y="count",
            color="predicted_label",
            markers=True
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ---------------- TOTAL TREND ----------------
        st.subheader("📊 Total Waste Trend")

        daily = filtered.groupby(filtered["timestamp"].dt.date).size().reset_index(name="count")

        fig2 = px.area(
            daily,
            x="timestamp",
            y="count",
            title="Total Waste Over Time"
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        # ---------------- DISTRIBUTION ----------------
        st.subheader("📦 Waste Distribution")

        dist = filtered["predicted_label"].value_counts().reset_index()
        dist.columns = ["category", "count"]

        fig3 = px.pie(
            dist,
            names="category",
            values="count",
            hole=0.4
        )

        st.plotly_chart(fig3, use_container_width=True)

        st.divider()

        # ---------------- LATEST DATA ----------------
        st.subheader("📄 Latest Predictions")

        st.dataframe(df.tail(15), use_container_width=True)

        

# ---------------- INSIGHTS ----------------
elif selected == "Insights":
    st.title("🔮 Predictive Insights & Alerts")

    df = load_data()

    if df.empty:
        st.warning("No data available.")
    else:
        daily = df.groupby(df["timestamp"].dt.date).size()
        counts = df["predicted_label"].value_counts()

        trend = daily.diff().mean()
        top_waste = counts.idxmax()
        prediction_series = daily.rolling(window=3).mean()
        latest_prediction = prediction_series.iloc[-1]

        # KPI
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Waste", len(df))
        col2.metric("Top Waste", top_waste)
        col3.metric("Avg Daily", int(daily.mean()))

        st.divider()

        # Trend
        st.subheader("📈 Trend Analysis")
        st.line_chart(daily)

        if trend > 0:
            st.error("🚨 Waste is increasing rapidly!")
        elif trend < 0:
            st.success("✅ Waste is decreasing.")
        else:
            st.info("➖ Waste is stable.")

       

        # AI Insight
        st.markdown(f"""
        ### 🤖 AI Insight

        - Most common waste: **{top_waste}**
        - Average per day: **{int(daily.mean())}**
        - Trend: **{"Increasing" if trend > 0 else "Decreasing"}**

        👉 Recommendation: Improve recycling for **{top_waste}**
        """)