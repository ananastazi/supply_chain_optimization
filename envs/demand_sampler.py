import pandas as pd
import numpy as np

class DemandSampler:
    def __init__(self, data_path='data/train_data.csv', sequential=False):
        self.df = pd.read_csv(data_path)
        self.sequential = sequential
        self.step_index = 0 # Tracks the current row in sequential mode
        
        if not self.sequential:
            # Pre-compute demand groups for efficient sampling during training
            self.demand_groups = self.df.groupby(['Month', 'DayOfWeek'])['Adjusted_Demand'].apply(list).to_dict()
            self.global_mean = self.df['Adjusted_Demand'].mean()
            self.global_std = self.df['Adjusted_Demand'].std()

    def reset(self):
        """Resets the sequential reader to the start of the file."""
        self.step_index = 0

    def sample(self, month, day_of_week):
        #Returns demand
        if self.sequential:
            if self.step_index >= len(self.df):
                self.step_index = 0 
            
            exact_demand = self.df.iloc[self.step_index]['Adjusted_Demand']
            self.step_index += 1
            return int(exact_demand)

        key = (month, day_of_week)
        
        if key in self.demand_groups:
            base_demand = np.random.choice(self.demand_groups[key])
        else:
            base_demand = np.random.normal(self.global_mean, self.global_std)
        
        return max(0, int(self.add_noise(base_demand)))

    def add_noise(self, value):
        noise_factor = np.random.normal(loc=1.0, scale=0.075)
        return value * noise_factor