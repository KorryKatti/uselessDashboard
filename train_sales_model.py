import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load both files
sales = pd.read_csv('sales_clean.csv')
sales['date'] = pd.to_datetime(sales['date'])

products = pd.read_csv('products_clean.csv')

# ============================================
# 1. MERGE PRODUCT INFO INTO SALES
# ============================================

sales = sales.merge(
    products[['product_id', 'price', 'cost', 'stock_level', 'category_encoded']],
    on='product_id',
    how='left'
)

# ============================================
# 2. ENGINEER FEATURES
# ============================================

# Time features
sales['month'] = sales['date'].dt.month
sales['day_of_week'] = sales['date'].dt.dayofweek
sales['is_weekend'] = (sales['day_of_week'] >= 5).astype(int)
sales['quarter'] = sales['date'].dt.quarter

# Product features
sales['profit_margin'] = (sales['price'] - sales['cost']) / sales['price']
sales['price_bucket'] = pd.cut(sales['price'], bins=[0, 50, 200, 500, 2000], labels=[0, 1, 2, 3])

# Sort by date so rolling features make sense
sales = sales.sort_values('date')

# Lag features: what happened yesterday per product
sales['sales_yesterday'] = sales.groupby('product_id')['units_sold'].shift(1)
sales['sales_last_week'] = sales.groupby('product_id')['units_sold'].shift(7)

# Rolling average: momentum over last 7 days
sales['rolling_7d_avg'] = sales.groupby('product_id')['units_sold'].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)

# Fill NaN from lag features (first days have no yesterday)
sales['sales_yesterday'] = sales['sales_yesterday'].fillna(0)
sales['sales_last_week'] = sales['sales_last_week'].fillna(0)

# price_bucket from pd.cut can produce NaN for out-of-range values
sales['price_bucket'] = sales['price_bucket'].fillna(0).astype(int)

# ============================================
# 3. DEFINE FEATURES AND TARGET
# ============================================

feature_columns = [
    'month', 'day_of_week', 'is_weekend', 'quarter',
    'promotion_flag',
    'price', 'stock_level', 'category_encoded',
    'profit_margin', 'price_bucket',
    'sales_yesterday', 'sales_last_week', 'rolling_7d_avg'
]

X = sales[feature_columns]
y = sales['units_sold']

# ============================================
# 4. TRAIN/TEST SPLIT (time-based, not random)
# ============================================

# For time series, never shuffle. Train on past, test on future.
split_date = '2024-10-01'
train_mask = sales['date'] < split_date
test_mask = sales['date'] >= split_date

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"Train: {len(X_train)} rows (Jan-Sep)")
print(f"Test: {len(X_test)} rows (Oct-Dec)")

# ============================================
# 5. TRAIN
# ============================================

model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# ============================================
# 6. EVALUATE
# ============================================

y_pred = model.predict(X_test)
y_pred = np.maximum(y_pred, 0)  # Can't sell negative units

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"Avg units in test: {y_test.mean():.2f}")

# Compare to old dumb model's MAE
print(f"\nPrevious model MAE was: 1.15 (but time-split was different, compare carefully)")

# ============================================
# 7. FEATURE IMPORTANCE
# ============================================

importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 features:")
print(importance.head(5))