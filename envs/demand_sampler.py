import pandas as pd
import numpy as np

class DemandSampler:
    def __init__(self, data_path='data/train_data.csv'):
        """
        Initializes the Data Sampler using the Training Data.
        """
        self.df = pd.read_csv(data_path)
        
        # Pre-compute groups to speed up the environment step
        # Dictionary structure: { (Month, DayOfWeek): [List of demand values] }
        # This prevents filtering the dataframe 100,000 times during training
        self.demand_groups = self.df.groupby(['Month', 'DayOfWeek'])['Adjusted_Demand'].apply(list).to_dict()
        
        # Calculate global stats for fallback
        self.global_mean = self.df['Adjusted_Demand'].mean()
        self.global_std = self.df['Adjusted_Demand'].std()

    def sample(self, month, day_of_week):
        """
        Phase 1.3 Method: Filters for matching days and returns random value.
        """
        key = (month, day_of_week)
        
        if key in self.demand_groups:
            # Bootstrapping: Pick one random historical value from this bucket
            historical_demands = self.demand_groups[key]
            base_demand = np.random.choice(historical_demands)
        else:
            # Fallback if we have no data for this specific Month+Day combination
            base_demand = np.random.normal(self.global_mean, self.global_std)
        
        # Apply the Noise Method
        final_demand = self.add_noise(base_demand)
        
        return max(0, int(final_demand))

    def add_noise(self, value):
        """
        Phase 1.3 Method: Adds ± 5-10% Gaussian noise.
        """
        # Generate a noise factor centered at 1.0 (e.g., 0.95 to 1.05)
        # Scale=0.075 implies ~7.5% standard deviation noise
        noise_factor = np.random.normal(loc=1.0, scale=0.075)
        
        noisy_value = value * noise_factor
        
        return noisy_value

# --- Usage Demonstration ---
if __name__ == "__main__":
    # Test the Class
    try:
        sampler = DemandSampler()
        print("Sampler initialized successfully.")
        
        print("\n--- Testing Sampling Logic ---")
        # Simulate sampling a Monday (0) in January (1)
        for i in range(5):
            d = sampler.sample(month=1, day_of_week=0)
            print(f"Sample {i+1} (Jan, Mon): Demand = {d}")
            
    except FileNotFoundError:
        print("Error: train_data.csv not found. Run preprocess_data.py first.")