import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error
import joblib

# Load
sales = pd.read_csv('sales_clean.csv')
sales['date'] = pd.to_datetime(sales['date'])
products = pd.read_csv('products_clean.csv')

# Merge
sales = sales.merge(
    products[['product_id', 'price', 'cost', 'stock_level', 'category_encoded']],
    on='product_id'
)

# Features
sales['month'] = sales['date'].dt.month
sales['day_of_week'] = sales['date'].dt.dayofweek
sales['is_weekend'] = (sales['day_of_week'] >= 5).astype(int)
sales['profit_margin'] = (sales['price'] - sales['cost']) / sales['price']

sales = sales.sort_values('date')
sales['sales_yesterday'] = sales.groupby('product_id')['units_sold'].shift(1).fillna(0)
sales['rolling_7d_avg'] = sales.groupby('product_id')['units_sold'].rolling(7, min_periods=1).mean().reset_index(0, drop=True)

# Features and target
feature_cols = [
    'month', 'day_of_week', 'is_weekend', 'promotion_flag',
    'price', 'stock_level', 'category_encoded',
    'profit_margin', 'sales_yesterday', 'rolling_7d_avg'
]

X = sales[feature_cols]
y = sales['units_sold']

# Time split
split_date = '2024-10-01'
train_mask = sales['date'] < split_date

X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

# ============================================
# THE PIPELINE (replaces manual imputer+scaler+model)
# ============================================

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    ))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
y_pred = np.maximum(y_pred, 0)
mae = mean_absolute_error(y_test, y_pred)
print(f"Pipeline MAE: {mae:.2f}")

# ============================================
# TIME SERIES CROSS-VALIDATION
# ============================================

tscv = TimeSeriesSplit(n_splits=3)
scores = cross_val_score(pipeline, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')
print(f"Cross-val MAE: {-scores.mean():.2f} (+/- {scores.std():.2f})")

# ============================================
# SAVE MODEL
# ============================================

joblib.dump(pipeline, 'sales_model.pkl')
print("Saved sales_model.pkl")