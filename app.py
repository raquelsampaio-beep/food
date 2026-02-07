# Streamlit Interactive Dashboard for NYC Food Orders
# To run:
# 1. pip install streamlit pandas plotly scikit-learn
# 2. streamlit run app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="NYC Food Orders Dashboard", layout="wide")

st.title("🍔 NYC Food Orders – Interactive Segmentation Dashboard")
st.markdown("Explore customer behavior, temporal patterns, and segmentation interactively.")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("food_order.csv")

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
cuisine_filter = st.sidebar.multiselect(
    "Cuisine Type",
    options=sorted(df["cuisine_type"].unique()),
    default=sorted(df["cuisine_type"].unique())
)

day_filter = st.sidebar.multiselect(
    "Day Type",
    options=df["day_of_the_week"].unique(),
    default=df["day_of_the_week"].unique()
)

filtered_df = df[
    (df["cuisine_type"].isin(cuisine_filter)) &
    (df["day_of_the_week"].isin(day_filter))
]

# KPIs
st.subheader("📊 Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Orders", len(filtered_df))
col2.metric("Avg Order Value ($)", round(filtered_df["cost_of_the_order"].mean(), 2))
col3.metric("Avg Delivery Time (min)", round(filtered_df["delivery_time"].mean(), 1))

# Orders by day
st.subheader("📅 Orders by Day Type")
fig_day = px.bar(
    filtered_df["day_of_the_week"].value_counts().reset_index(),
    x="day_of_the_week",
    y="count",
    labels={"count": "Orders", "day_of_the_week": "Day Type"}
)
st.plotly_chart(fig_day, use_container_width=True)

# Orders by cuisine
st.subheader("🍽️ Orders by Cuisine")
fig_cuisine = px.bar(
    filtered_df["cuisine_type"].value_counts().reset_index().head(10),
    x="cuisine_type",
    y="count",
    labels={"count": "Orders", "cuisine_type": "Cuisine"}
)
st.plotly_chart(fig_cuisine, use_container_width=True)

# Customer segmentation
st.subheader("🧠 Customer Segmentation")

customer_agg = filtered_df.groupby("customer_id").agg(
    total_orders=("order_id", "count"),
    avg_order_value=("cost_of_the_order", "mean"),
    weekend_orders=("day_of_the_week", lambda x: (x == "Weekend").sum())
).reset_index()

customer_agg["weekend_ratio"] = customer_agg["weekend_orders"] / customer_agg["total_orders"]

features = customer_agg[["total_orders", "avg_order_value", "weekend_ratio"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

k = st.slider("Number of clusters", min_value=2, max_value=5, value=3)
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
customer_agg["cluster"] = kmeans.fit_predict(X_scaled)

fig_cluster = px.scatter(
    customer_agg,
    x="total_orders",
    y="avg_order_value",
    color="cluster",
    size="weekend_ratio",
    labels={
        "total_orders": "Total Orders",
        "avg_order_value": "Avg Order Value ($)",
        "weekend_ratio": "Weekend Ratio"
    }
)

st.plotly_chart(fig_cluster, use_container_width=True)

st.markdown("---")
st.markdown("**Tip:** Use the filters and cluster slider to explore different customer segments interactively.")
