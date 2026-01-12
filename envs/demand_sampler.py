import pandas as pd
import numpy as np

class DemandSampler:
    def __init__(self, data_path='data/train_data.csv', episode_length=365, sequential=False):
        self.df = pd.read_csv(data_path)
        self.episode_length = episode_length
        self.sequential = sequential
        self.max_start_index = len(self.df) - self.episode_length
        self.step_index = 0 # Tracks the current row in sequential mode
        
        if self.max_start_index <= 0:
            self.max_start_index = 0
            self.episode_length = len(self.df)

        self.current_idx = 0
        self.demand_multiplier = 1.0

    def reset(self):
        """
        Resets the sampler. 
        If sequential=True (Testing), starts at 0 and NO augmentation.
        If sequential=False (Training), random start AND augmentation.
        """
        if self.sequential:
            # TESTING MODE: Strict history, no magic multipliers
            self.current_idx = 0
            self.demand_multiplier = 1.0 
        else:
            # TRAINING MODE: Random windows + Scenario Augmentation
            if self.max_start_index > 0:
                self.current_idx = np.random.randint(0, self.max_start_index)
            else:
                self.current_idx = 0
            
            # Scenario Augmentation:
            # 30% chance to simulate a "Boom" market (High Demand)
            if np.random.random() < 0.3:
                self.demand_multiplier = np.random.uniform(1.2, 1.5)
            else:
                self.demand_multiplier = 1.0
            
        return self._get_current_date_info()
    
    def _get_current_date_info(self):
        row = self.df.iloc[self.current_idx]
        return row['Month'], row['DayOfWeek']

    def sample(self):
        # Safety check for end of dataset
        if self.current_idx >= len(self.df):
            if self.sequential:
                self.current_idx = 0 # Loop back or handle as done
            else:
                self.current_idx = np.random.randint(0, self.max_start_index)

        # Read history
        row = self.df.iloc[self.current_idx]
        exact_demand = row['Adjusted_Demand']
        
        # Move pointer
        self.current_idx += 1
        
        # Add Noise and Scenario Multiplier
        if self.sequential:
             # In test mode, we usually want exact history or very low noise
             final_demand = exact_demand 
        else:
            # In train mode, add noise and multiplier
            noise = np.random.normal(loc=1.0, scale=0.1)
            final_demand = exact_demand * noise * self.demand_multiplier
        
        return max(0, int(final_demand)), row['Month'], row['DayOfWeek']