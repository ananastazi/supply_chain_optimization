config = {
        "train_data_path": "data/train_data.csv",  # Path to training data
        "test_data_path": "data/test_data.csv",    # Path to testing data

        # Data-Dependent Constants
        "max_capacity": 987,     # e.g., if max history was 520
        "max_order": 301,         # e.g., if max demand was 110
        "unit_price": 66.36,       # From your csv
        "reward_scale": 1333.8,   # Calculated scale factor
        
        # Random Lead Time Settings
        "min_lead_time": 3,
        "max_lead_time": 8,       # Max buffer size for NN
        
        # Reward Shaping (The "Leaner" Agent Tweak)
        "holding_cost": 1.5,      # Higher cost to discourage hoarding
        "shortage_penalty": 50.0  
}