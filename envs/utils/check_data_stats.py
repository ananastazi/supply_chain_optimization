import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/train_data.csv')

print("--- DATASET STATISTICS ---")
print(f"Max Inventory in History: {df['Inventory Level'].max()}")
print(f"Max Units Sold (Demand):  {df['Units Sold'].max()}")
print(f"Average Price:            {df['Price'].mean():.2f}")
print("--------------------------")

# CALCULATIONS FOR CONFIG
max_inv = df['Inventory Level'].max()
max_demand = df['Units Sold'].max()
avg_price = df['Price'].mean()

rec_capacity = int(max_inv * 1.5)
rec_max_order = int(max_demand * 1.5)
rec_reward_scale = (max_demand * avg_price) / 10.0 # Heuristic to keep reward ~10

print(f"RECOMMENDED CONFIG:")
print(f"max_capacity:     {rec_capacity}")
print(f"max_order:        {rec_max_order}")
print(f"unit_price:       {avg_price:.2f}")
print(f"REWARD_SCALE:     {rec_reward_scale:.1f}")