config = {
        "train_data_path": "data/train_data.csv",  
        "test_data_path": "data/test_data.csv",
            
        # Data-Dependent Constants
        "max_capacity": 987,     
        "max_order": 301,         
        "unit_price": 66.36,      
        "reward_scale": 1333.8,   # Calculated scale factor
        
        # Random Lead Time Settings
        "min_lead_time": 3,
        "max_lead_time": 8,       # Max buffer size for NN
        
        # Reward Shaping (The "Leaner" Agent Tweak)
        "holding_cost": 1.5,      # Higher cost to discourage hoarding
        "shortage_penalty": 50.0  
}