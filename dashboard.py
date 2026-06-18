import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Page config
st.set_page_config(page_title="NeuroRetail AI", layout="wide")
st.title("NeuroRetail™ AI Dashboard")

# ============================================
# LOAD DATA
# ============================================

@st.cache_data
def load_data():
    sales = pd.read_csv('sales_clean.csv')
    sales['date'] = pd.to_datetime(sales['date'])
    products = pd.read_csv('products_clean.csv')
    customers = pd.read_csv('customers_clustered.csv')
    return sales, products, customers

sales, products, customers = load_data()

# ============================================
# SIDEBAR
# ============================================

st.sidebar.header("Controls")

# CSV Upload
st.sidebar.subheader("📁 Upload Your Data")

with st.sidebar.expander("CSV Format Help"):
    st.markdown("""
    **Sales CSV** must have these columns:
    ```
    date,product_id,units_sold,promotion_flag
    2024-01-01,1057,4,0
    ```
    - `date`: YYYY-MM-DD format
    - `product_id`: integer
    - `units_sold`: integer
    - `promotion_flag`: 0 or 1

    **Products CSV** must have these columns:
    ```
    product_id,category,price,cost,stock_level,category_encoded
    1000,Clothing,142.65,64.98,303,0
    ```
    - `category_encoded`: integer mapping of category name
    """)

uploaded_sales = st.sidebar.file_uploader("Upload Sales CSV", type=['csv'], key='sales_upload')
uploaded_products = st.sidebar.file_uploader("Upload Products CSV", type=['csv'], key='products_upload')

if uploaded_sales is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_sales)
        uploaded_df['date'] = pd.to_datetime(uploaded_df['date'])
        st.sidebar.success(f"✅ Loaded {len(uploaded_df)} sales records")
        # Store in session state for use throughout the app
        st.session_state['uploaded_sales'] = uploaded_df
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")

if uploaded_products is not None:
    try:
        uploaded_products_df = pd.read_csv(uploaded_products)
        st.sidebar.success(f"✅ Loaded {len(uploaded_products_df)} products")
        st.session_state['uploaded_products'] = uploaded_products_df
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")

# Use uploaded data if available, otherwise use defaults
if 'uploaded_sales' in st.session_state:
    sales = st.session_state['uploaded_sales']
if 'uploaded_products' in st.session_state:
    products = st.session_state['uploaded_products']

if st.sidebar.button("🔄 Retrain Model"):
    import subprocess
    result = subprocess.run(['python', 'retrain_loop.py'], capture_output=True, text=True)
    st.sidebar.success("Retrained!")
    st.sidebar.text(result.stdout)

st.sidebar.metric("Total Products", len(products))
st.sidebar.metric("Total Customers", len(customers))
st.sidebar.metric("Sales Records", f"{len(sales):,}")

# ============================================
# ROW 1: SALES FORECAST
# ============================================

st.header("📈 Sales Forecast")

# Load model
try:
    model = joblib.load('sales_model.pkl')
    
    # Prepare recent data for prediction
    sales_sorted = sales.sort_values('date')
    
    # Merge product info
    plot_data = sales_sorted.merge(
        products[['product_id', 'price', 'cost', 'stock_level', 'category_encoded']],
        on='product_id'
    )
    
    plot_data['month'] = plot_data['date'].dt.month
    plot_data['day_of_week'] = plot_data['date'].dt.dayofweek
    plot_data['is_weekend'] = (plot_data['day_of_week'] >= 5).astype(int)
    plot_data['profit_margin'] = (plot_data['price'] - plot_data['cost']) / plot_data['price']
    plot_data['sales_yesterday'] = plot_data.groupby('product_id')['units_sold'].shift(1).fillna(0)
    plot_data['rolling_7d_avg'] = plot_data.groupby('product_id')['units_sold'].rolling(7, min_periods=1).mean().reset_index(0, drop=True)
    
    feature_cols = [
        'month', 'day_of_week', 'is_weekend', 'promotion_flag',
        'price', 'stock_level', 'category_encoded',
        'profit_margin', 'sales_yesterday', 'rolling_7d_avg'
    ]
    
    plot_data['predicted'] = model.predict(plot_data[feature_cols])
    plot_data['predicted'] = plot_data['predicted'].clip(lower=0)
    
    # Monthly actual vs predicted
    monthly = plot_data.groupby(plot_data['date'].dt.to_period('M')).agg(
        actual=('units_sold', 'sum'),
        predicted=('predicted', 'sum')
    ).reset_index()
    monthly['date'] = monthly['date'].astype(str)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly['date'], monthly['actual'], marker='o', label='Actual', linewidth=2)
    ax.plot(monthly['date'], monthly['predicted'], marker='s', label='Predicted', linewidth=2, linestyle='--')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Units Sold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
except FileNotFoundError:
    st.warning("Model not trained yet. Run pipeline_sales.py first.")

# ============================================
# ROW 2: PRODUCTS + CLUSTERS (side by side)
# ============================================

col1, col2 = st.columns(2)

with col1:
    st.header("🏷️ Product Performance")
    
    # Simple classification based on sales
    product_sales = sales.groupby('product_id')['units_sold'].sum().reset_index()
    product_sales = product_sales.merge(products[['product_id', 'category', 'price', 'stock_level']], on='product_id')
    
    # Label: top 25% Hot, bottom 25% Low
    product_sales['perf'] = pd.qcut(
        product_sales['units_sold'],
        q=[0, 0.25, 0.75, 1.0],
        labels=['🔴 Low', '🟡 Steady', '🟢 Hot']
    )
    
    # Show table
    display_df = product_sales[['product_id', 'category', 'price', 'units_sold', 'perf']].head(20)
    st.dataframe(display_df, use_container_width=True)
    
    # Category breakdown
    st.subheader("Hot Products by Category")
    cat_hot = product_sales[product_sales['perf'] == '🟢 Hot']['category'].value_counts()
    st.bar_chart(cat_hot)

with col2:
    st.header("👥 Customer Clusters")
    
    # PCA for 2D plot
    features = customers[['avg_spend', 'purchase_frequency']]
    features_scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(features_scaled)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for cluster_id in [0, 1, 2]:
        mask = customers['cluster'] == cluster_id
        cluster_spend = customers[mask]['avg_spend'].mean()
        
        if cluster_spend > 400:
            label = f'Premium (n={mask.sum()})'
        elif cluster_spend > 80:
            label = f'Regular (n={mask.sum()})'
        else:
            label = f'Budget (n={mask.sum()})'
        
        ax.scatter(points_2d[mask, 0], points_2d[mask, 1],
                  c=colors[cluster_id], label=label, alpha=0.6, s=15)
    
    ax.set_xlabel('Spend + Frequency Combined')
    ax.set_ylabel('Secondary Pattern')
    ax.legend()
    st.pyplot(fig)
    
    # Cluster stats
    st.subheader("Cluster Profiles")
    for c in range(3):
        cdata = customers[customers['cluster'] == c]
        st.metric(
            f"Cluster {c}",
            f"${cdata['avg_spend'].mean():.0f} avg spend",
            f"{cdata['purchase_frequency'].mean():.1f} visits/month"
        )

# ============================================
# ROW 3: FEATURE IMPORTANCE
# ============================================

st.header("🔍 What Drives Sales?")

try:
    model = joblib.load('sales_model.pkl')
    
    # Get feature importance from the model step of pipeline
    feature_names = [
        'month', 'day_of_week', 'is_weekend', 'promotion_flag',
        'price', 'stock_level', 'category',
        'profit_margin', 'yesterday_sales', '7_day_avg'
    ]
    
    importances = model.named_steps['model'].feature_importances_
    
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(imp_df['Feature'], imp_df['Importance'])
    ax.set_xlabel('Importance')
    st.pyplot(fig)
    
except FileNotFoundError:
    st.info("Train the model to see feature importance.")