import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.impute import SimpleImputer

products = pd.read_csv('products.csv')
sales = pd.read_csv('sales.csv')
customers = pd.read_csv('customers.csv')
sales['date'] = pd.to_datetime(sales['date'])

# fix missing values
imputer = SimpleImputer(strategy='median')
# median is better than mean ig
products['stock_level']=imputer.fit_transform(products[['stock_level']])

print(f"missing after fix: {products['stock_level'].isna().sum()}")

# text to umbers
# i think i do this on myt website as well
le = LabelEncoder()
products['category_encoded'] = le.fit_transform(products['category'])

# mapping
print("mapping:")
for i, name in enumerate(le.classes_):
    print(f"  {i} = {name}")
    # le classes lmao

# how many categories each customer likes
customers['num_preferences'] = customers['preferred_categories'].str.count(r'\|') + 1

# SCALE NUMBERS ( mean = 0 , std = 1 )
scaler = StandardScaler()

# scale product prices and costs
products[['price_scaled', 'cost_scaled', 'stock_scaled']] = scaler.fit_transform(
    products[['price', 'cost', 'stock_level']]
)

# Scale customer spend
customers[['spend_scaled', 'freq_scaled']] = scaler.fit_transform(
    customers[['avg_spend', 'purchase_frequency']]
)

# save clean data
products.to_csv('products_clean.csv', index=False)
sales.to_csv('sales_clean.csv', index=False)
customers.to_csv('customers_clean.csv', index=False)

print("Saved clean files.")