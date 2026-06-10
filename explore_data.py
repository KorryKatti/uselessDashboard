import pandas as pd
import numpy as np

products = pd.read_csv('products.csv')
sales = pd.read_csv('sales.csv')
customers = pd.read_csv('customers.csv')

sales['date'] = pd.to_datetime(sales['date'])

products["profit_margin"] = (
    products["price"] - products["cost"]
)/ products["price"]

sales["month"]=sales["date"].dt.month
sales["day_of_week"]=sales["date"].dt.day_of_week

monthly_sales = (
    sales.groupby("month")["units_sold"].sum()
)

promotion_effect = (
    sales.groupby("promotion_flag")["units_sold"].mean()
)

segment_spend = (
    customers.groupby("segment")["avg_spend"]
    .mean()
)

sales_with_products = sales.merge(
    products[["product_id", "category", "price"]],
    on="product_id"
)

category_sales = sales_with_products.groupby("category").agg(
    total_units=("units_sold", "sum"),
    avg_price=("price", "mean")
)

negative_price_count = (products["price"] < 0).sum()

future_date_count = (
    sales["date"] > pd.Timestamp("2024-12-31")
).sum()

high_stock_count = (
    products["stock_level"] > 400
).sum()
