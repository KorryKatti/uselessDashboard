import numpy as np
import pandas as pd
from datetime import datetime

np.random.seed(42)

n_products = 500

categories = ['Electronics', 'Clothing', 'Groceries', 'Home & Garden', 'Sports']
category_weights = [0.25, 0.30, 0.20, 0.15, 0.10]

products = pd.DataFrame({
    'product_id': range(1000, 1000 + n_products),
    'category': np.random.choice(categories, size=n_products, p=category_weights)
})

price_ranges = {
    'Electronics': (50, 2000),
    'Clothing': (10, 200),
    'Groceries': (2, 50),
    'Home & Garden': (15, 500),
    'Sports': (20, 800)
}

products['price'] = products['category'].apply(
    lambda category: np.round(np.random.uniform(*price_ranges[category]), 2)
)

products['cost'] = np.round(
    products['price'] * np.random.uniform(0.4, 0.7, n_products),
    2
)

products['stock_level'] = np.random.randint(0, 500, n_products)

mask = np.random.random(n_products) < 0.03
products.loc[mask, 'stock_level'] = np.nan

start_date = datetime(2024, 1, 1)
dates = pd.date_range(start=start_date, periods=365, freq='D')

sales_records = []

for date in dates:
    base_transactions = 80 if date.dayofweek >= 5 else 50

    if date.month == 12:
        base_transactions = int(base_transactions * 1.5)

    n_transactions = np.random.poisson(base_transactions)

    for _ in range(n_transactions):
        product = products.sample(1).iloc[0]

        seasonal_boost = 1.0

        if product['category'] == 'Sports' and date.month in [6, 7, 8]:
            seasonal_boost = 1.5

        if product['category'] == 'Electronics' and date.month == 11:
            seasonal_boost = 2.0

        units = max(1, np.random.poisson(2 * seasonal_boost))

        promotion = np.random.random() < 0.2

        if promotion:
            units = int(units * 1.5)

        sales_records.append({
            'date': date,
            'product_id': product['product_id'],
            'units_sold': units,
            'promotion_flag': int(promotion)
        })

sales = pd.DataFrame(sales_records)
sales = sales.sort_values('date').reset_index(drop=True)

n_customers = 1000

segments = ['Budget', 'Regular', 'Premium']
segment_weights = [0.40, 0.45, 0.15]

customers = pd.DataFrame({
    'customer_id': range(5000, 5000 + n_customers),
    'segment': np.random.choice(segments, size=n_customers, p=segment_weights)
})

segment_params = {
    'Budget': {'avg_spend': (10, 50), 'frequency': (1, 5)},
    'Regular': {'avg_spend': (50, 200), 'frequency': (5, 15)},
    'Premium': {'avg_spend': (200, 1000), 'frequency': (10, 25)}
}

customers['avg_spend'] = customers['segment'].apply(
    lambda segment: np.round(
        np.random.uniform(*segment_params[segment]['avg_spend']),
        2
    )
)

customers['purchase_frequency'] = customers['segment'].apply(
    lambda segment: np.random.randint(
        *segment_params[segment]['frequency']
    )
)

preferred_categories = {
    'Budget': ['Groceries', 'Clothing'],
    'Regular': ['Clothing', 'Electronics', 'Home & Garden'],
    'Premium': ['Electronics', 'Sports', 'Home & Garden']
}

customers['preferred_categories'] = customers['segment'].apply(
    lambda segment: np.random.choice(
        preferred_categories[segment],
        size=np.random.randint(1, 3),
        replace=False
    ).tolist()
)

customers['preferred_categories'] = customers['preferred_categories'].apply(
    lambda categories: '|'.join(categories)
)

products.to_csv('products.csv', index=False)
sales.to_csv('sales.csv', index=False)
customers.to_csv('customers.csv', index=False)