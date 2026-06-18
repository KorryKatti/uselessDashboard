import pandas as pd
import numpy as np
import joblib
from datetime import datetime

def retrain_if_new_data():
    """Check for new data, retrain if found, return True if retrained."""
    
    # Load current data
    sales = pd.read_csv('sales_clean.csv')
    sales['date'] = pd.to_datetime(sales['date'])
    
    # Check if we already processed this
    last_date = sales['date'].max()
    last_retrain_file = 'last_retrain.txt'
    
    try:
        with open(last_retrain_file, 'r') as f:
            last_retrain = pd.to_datetime(f.read().strip())
        
        if last_date <= last_retrain:
            print(f"No new data. Last sale: {last_date.date()}, Last retrain: {last_retrain.date()}")
            return False
    except FileNotFoundError:
        pass
    
    print(f"New data found. Last sale: {last_date.date()}. Retraining...")
    
    # Merge products
    products = pd.read_csv('products_clean.csv')
    sales = sales.merge(
        products[['product_id', 'price', 'cost', 'stock_level', 'category_encoded']],
        on='product_id'
    )
    
    # Rebuild features (same as pipeline_sales.py)
    sales['month'] = sales['date'].dt.month
    sales['day_of_week'] = sales['date'].dt.dayofweek
    sales['is_weekend'] = (sales['day_of_week'] >= 5).astype(int)
    sales['profit_margin'] = (sales['price'] - sales['cost']) / sales['price']
    sales = sales.sort_values('date')
    sales['sales_yesterday'] = sales.groupby('product_id')['units_sold'].shift(1).fillna(0)
    sales['rolling_7d_avg'] = sales.groupby('product_id')['units_sold'].rolling(7, min_periods=1).mean().reset_index(0, drop=True)
    
    feature_cols = [
        'month', 'day_of_week', 'is_weekend', 'promotion_flag',
        'price', 'stock_level', 'category_encoded',
        'profit_margin', 'sales_yesterday', 'rolling_7d_avg'
    ]
    
    X = sales[feature_cols]
    y = sales['units_sold']
    
    # Load old model or train fresh
    try:
        model = joblib.load('sales_model.pkl')
        model.fit(X, y)  # Retrain on all data
    except FileNotFoundError:
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import GradientBoostingRegressor
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42))
        ])
        model.fit(X, y)
    
    # Save
    joblib.dump(model, 'sales_model.pkl')
    
    with open(last_retrain_file, 'w') as f:
        f.write(str(last_date.date()))
    
    print(f"Retrained and saved. MAE on full data: {np.mean(np.abs(y - model.predict(X))):.2f}")
    return True

# Run it
if __name__ == '__main__':
    retrained = retrain_if_new_data()
    print(f"Retrained: {retrained}")