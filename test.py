import pandas as pd

input_file = 'D:/Personal/Uni/MSc/Studies/1year/1point/rf/supply_chain_optimization/supply_chain_optimization/data/sales_data.csv' 
df = pd.read_csv(input_file)
    
# we clean the column names to avoid issues later
df.columns = df.columns.str.strip()

# our strategy is focus on a single item. to train a stable RL agent, we need a consistent history of one product. we select the product-store combination with the most data points available
top_product_id = df['Product ID'].value_counts().idxmax()
top_store_id = df[df['Product ID'] == top_product_id]['Store ID'].value_counts().idxmax()

print(f"Filtering for Product ID: {top_product_id} in Store ID: {top_store_id}")
