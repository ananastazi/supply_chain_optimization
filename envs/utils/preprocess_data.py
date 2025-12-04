import pandas as pd
import numpy as np
import os

def preprocess_retail_data(input_file, output_dir='data'):
    print("Starting Phase 1.1: Data Engineering")
    
    # we load the raw dataset
    df = pd.read_csv(input_file)
    
    # we clean the column names to avoid issues later
    df.columns = df.columns.str.strip()
    
    # our strategy is focus on a single item. to train a stable RL agent, we need a consistent history of one product. we select the product-store combination with the most data points available
    top_product_id = df['Product ID'].value_counts().idxmax()
    top_store_id = df[df['Product ID'] == top_product_id]['Store ID'].value_counts().idxmax()
    
    print(f"Filtering for Product ID: {top_product_id} in Store ID: {top_store_id}")
    
    # we filter the dataset to keep only that specific product history and store
    df_filtered = df[
        (df['Product ID'] == top_product_id) & 
        (df['Store ID'] == top_store_id)
    ].copy()
    
    # we standardize dates and sort chronologically, which is crucial for time-series splitting later
    df_filtered['Date'] = pd.to_datetime(df_filtered['Date'])
    df_filtered = df_filtered.sort_values('Date')

    # we extract month and day of week because the agent needs these to learn seasonal patterns
    df_filtered['Month'] = df_filtered['Date'].dt.month
    df_filtered['DayOfWeek'] = df_filtered['Date'].dt.dayofweek 
    
    # if we have a problem of stockouts in the data. when inventory is zero, sales are zero too, but real demand was likely higher. if we train on raw sales, the agent will learn that stockouts = low demand, which is wrong.
    #so we decide to initialize adjuste demand with the actual sales
    df_filtered['Adjusted_Demand'] = df_filtered['Units Sold']
    
    #then, we detect stockout days where inventory level is zero or below
    stockout_mask = df_filtered['Inventory Level'] <= 0
    
    print(f"Detected {stockout_mask.sum()} stockout days out of {len(df_filtered)} records.")
    
    #after that, we estimate missing demand: we calculate the average sales of the neighboring days (the day before, the current day and the day after). this helps us guess what the demand was when we had no stock
    rolling_demand = df_filtered['Units Sold'].rolling(window=3, center=True, min_periods=1).mean()
    
    # now we decide which number to use for the stockout days. we take the rolling average we just calculate. but to be safe, we also check that this number is at least 20% higher than the sales we recorded that day (which is often zero). we pick the highest number between the two, so we don't underestimate the real demand
    estimated_demand = np.maximum(rolling_demand, df_filtered['Units Sold'] * 1.2)
    
    #we update the adjusted demand column only for the stockout days
    df_filtered.loc[stockout_mask, 'Adjusted_Demand'] = estimated_demand.loc[stockout_mask].astype(int)

    # then we clean up: we keep only the relevant columns for our model
    final_columns = ['Date', 'Month', 'DayOfWeek', 'Inventory Level', 'Units Sold', 'Adjusted_Demand', 'Price']
    if 'Price' not in df_filtered.columns:
        # if the dataset is missing the price column, we add it with a default value of 10
        df_filtered['Price'] = 10.0 
    # we reset the index to make the dataset clean and ready
    final_df = df_filtered[final_columns].reset_index(drop=True)
    
    # Phase: Data splitting--> we split the data into two parts based on time. we use the first 80% to train the agent and the last 20% to test it 
    split_idx = int(len(final_df) * 0.8)
    
    train_df = final_df.iloc[:split_idx]
    test_df = final_df.iloc[split_idx:]
    
    # we create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'train_data.csv')
    test_path = os.path.join(output_dir, 'test_data.csv')
    
    #we save the processed datasets to csv files
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Phase 1 Complete.")
    print(f"Training Data: {len(train_df)} rows saved to {train_path}")
    print(f"Testing Data:  {len(test_df)} rows saved to {test_path}")

if __name__ == "__main__":
   
    csv_filename = 'D:/Personal/Uni/MSc/Studies/1year/1point/rf/supply_chain_optimization/supply_chain_optimization/data/sales_data.csv' 
    
    
    if not os.path.exists(csv_filename):
        print(f"Warning: {csv_filename} not found. Please download the dataset.")
    else:
        preprocess_retail_data(csv_filename)