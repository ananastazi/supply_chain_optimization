import pandas as pd
import numpy as np
import os

def preprocess_retail_data(input_file, output_dir='data'):
    print("--- Starting Phase 1.1: Data Engineering ---")
    
    # 1. Load Data
    # Adjust the delimiter if necessary (some Kaggle CSVs use ;)
    df = pd.read_csv(input_file)
    
    # Standardize Column Names (Remove spaces, lowercase)
    df.columns = df.columns.str.strip()
    
    # 2. Filter for a Single Product & Store
    # We must train the agent on a specific item's history first.
    # Let's pick the product with the most data points.
    top_product_id = df['Product ID'].value_counts().idxmax()
    top_store_id = df[df['Product ID'] == top_product_id]['Store ID'].value_counts().idxmax()
    
    print(f"Filtering for Product ID: {top_product_id} in Store ID: {top_store_id}")
    
    # Create the single time-series dataframe
    df_filtered = df[
        (df['Product ID'] == top_product_id) & 
        (df['Store ID'] == top_store_id)
    ].copy()
    
    # 3. Date Handling
    df_filtered['Date'] = pd.to_datetime(df_filtered['Date'])
    df_filtered = df_filtered.sort_values('Date')
    
    # 4. Feature Engineering for the Sampler
    df_filtered['Month'] = df_filtered['Date'].dt.month
    df_filtered['DayOfWeek'] = df_filtered['Date'].dt.dayofweek # 0=Mon, 6=Sun
    
    # --- LOGIC: Un-censoring Demand ---
    # Goal: If Inventory was 0, Sales is not the True Demand.
    
    # Initialize Adjusted Demand with Sales
    # Note: This dataset has 'Units Sold'
    df_filtered['Adjusted_Demand'] = df_filtered['Units Sold']
    
    # Define Stockout Condition
    # We use <= 0 just in case of data errors (negative stock)
    stockout_mask = df_filtered['Inventory Level'] <= 0
    
    print(f"Detected {stockout_mask.sum()} stockout days out of {len(df_filtered)} records.")
    
    # Strategy: Average of surrounding 3 days (Moving Average)
    # We calculate a rolling mean of the whole column, then apply it only to stockout days
    rolling_demand = df_filtered['Units Sold'].rolling(window=3, center=True, min_periods=1).mean()
    
    # Apply the logic: 
    # If stockout, use Rolling Avg. If Rolling Avg is roughly same as Sales, boost it slightly.
    # Here we use the simplified logic from your prompt for clarity:
    # "Simple: Sales * 1.5" or "Advanced: Avg of surrounding"
    
    # Let's apply a hybrid: Use Rolling Avg, but ensure it's at least 1.2x the sales cap
    estimated_demand = np.maximum(rolling_demand, df_filtered['Units Sold'] * 1.2)
    
    df_filtered.loc[stockout_mask, 'Adjusted_Demand'] = estimated_demand.loc[stockout_mask].astype(int)

    # 5. Clean Final Columns
    final_columns = ['Date', 'Month', 'DayOfWeek', 'Inventory Level', 'Units Sold', 'Adjusted_Demand', 'Price']
    if 'Price' not in df_filtered.columns:
        # Some versions of this dataset might not have price, handle gracefully
        df_filtered['Price'] = 10.0 
        
    final_df = df_filtered[final_columns].reset_index(drop=True)
    
    # --- Phase 1.2: Data Splitting ---
    split_idx = int(len(final_df) * 0.8)
    
    train_df = final_df.iloc[:split_idx]
    test_df = final_df.iloc[split_idx:]
    
    # Save Outputs
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'train_data.csv')
    test_path = os.path.join(output_dir, 'test_data.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Phase 1 Complete.")
    print(f"Training Data: {len(train_df)} rows saved to {train_path}")
    print(f"Testing Data:  {len(test_df)} rows saved to {test_path}")

if __name__ == "__main__":
    # Replace with your actual downloaded filename
    csv_filename = 'D:/Personal/Uni/MSc/Studies/1year/1point/rf/supply_chain_optimization/supply_chain_optimization/data/sales_data.csv' 
    
    # Create dummy file for demonstration if you don't have the file yet
    if not os.path.exists(csv_filename):
        print(f"Warning: {csv_filename} not found. Please download the dataset.")
    else:
        preprocess_retail_data(csv_filename)