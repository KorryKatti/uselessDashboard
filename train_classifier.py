import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load
sales = pd.read_csv('sales_clean.csv')
sales['date'] = pd.to_datetime(sales['date'])
products = pd.read_csv('products_clean.csv')

# ============================================
# 1. CREATE LABELS FROM SALES DATA
# ============================================

# Aggregate sales per product
product_perf = sales.groupby('product_id').agg(
    total_units=('units_sold', 'sum'),
    days_with_sales=('date', 'nunique'),
    avg_daily_units=('units_sold', 'mean')
).reset_index()

# Label: top 25% = Hot, bottom 25% = Low, middle = Steady
product_perf['perf_label'] = pd.qcut(
    product_perf['avg_daily_units'],
    q=[0, 0.25, 0.75, 1.0],
    labels=['Low', 'Steady', 'Hot']
)

print("Label distribution:")
print(product_perf['perf_label'].value_counts())

# ============================================
# 2. MERGE WITH PRODUCT INFO
# ============================================

data = product_perf.merge(products, on='product_id')

# Features: product attributes (no sales history — model predicts from product info only)
feature_cols = ['price', 'cost', 'stock_level', 'category_encoded']

X = data[feature_cols]
y = data['perf_label']

# ============================================
# 3. TRAIN/TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================
# 4. TRAIN
# ============================================

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ============================================
# 5. EVALUATE
# ============================================

y_pred = clf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=2))

# Feature importance
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature importance:")
print(importance)

# ============================================
# 6. SHOW EXAMPLE PREDICTIONS
# ============================================

sample = X_test.head(5).copy()
sample['actual'] = y_test.head(5).values
sample['predicted'] = clf.predict(X_test.head(5))
print("\nSample predictions:")
print(sample[['price', 'category_encoded', 'actual', 'predicted']])