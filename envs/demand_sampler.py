import pandas as pd
import numpy as np

class DemandSampler:
    def __init__(self, data_path='data/train_data.csv', sequential=False):
        """
        Args:
            data_path: Path to csv.
            sequential (bool): 
                If False (Training): Randomly samples history + noise.
                If True (Testing): Reads file row-by-row deterministically.
        """
        self.df = pd.read_csv(data_path)
        self.sequential = sequential
        self.step_index = 0 # Track which row we are on for sequential mode
        
        # --- TRAINING MODE LOGIC ---
        if not self.sequential:
            # Pre-compute groups for fast sampling
            self.demand_groups = self.df.groupby(['Month', 'DayOfWeek'])['Adjusted_Demand'].apply(list).to_dict()
            self.global_mean = self.df['Adjusted_Demand'].mean()
            self.global_std = self.df['Adjusted_Demand'].std()

    def reset(self):
        """Resets the sequential reader to the start of the file."""
        self.step_index = 0

    def sample(self, month, day_of_week):
        """
        Returns demand.
        """
        # --- 1. SEQUENTIAL / TEST MODE ---
        if self.sequential:
            # Check if we ran out of data
            if self.step_index >= len(self.df):
                self.step_index = 0 # Loop back or handle as done
            
            # Get the exact historical value from the specific row
            exact_demand = self.df.iloc[self.step_index]['Adjusted_Demand']
            self.step_index += 1
            return int(exact_demand)

        # --- 2. TRAINING / RANDOM MODE ---
        key = (month, day_of_week)
        
        if key in self.demand_groups:
            base_demand = np.random.choice(self.demand_groups[key])
        else:
            base_demand = np.random.normal(self.global_mean, self.global_std)
        
        # Add noise only during training
        return max(0, int(self.add_noise(base_demand)))

    def add_noise(self, value):
        # Generate random noise (e.g. +/- 7.5%)
        noise_factor = np.random.normal(loc=1.0, scale=0.075)
        return value * noise_factor