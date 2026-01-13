config = {
        "train_data_path": "data/train_data.csv",  
        "test_data_path": "data/test_data.csv",
            
        # Data-Dependent Constants
        "max_capacity": 987,     
        "max_order": 301,   
        "unit_cost": 45.0,      
        "unit_price": 70,      
        "reward_scale": 1000.0,   # Calculated scale factor
        
        # Random Lead Time Settings
        "min_lead_time": 3,
        "max_lead_time": 8,       # Max buffer size for NN
        
        # Reward Shaping (The "Leaner" Agent Tweak)
        "holding_cost": 0.5,      # Higher cost to discourage hoarding
        "shortage_penalty": 10.0  
}