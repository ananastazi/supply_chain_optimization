import gymnasium as gym
import numpy as np
from gymnasium import spaces
from envs.demand_sampler import DemandSampler  

class InventoryEnv(gym.Env):
    """
    Custom Environment for Inventory Management
    State: [Current_Stock, Pipeline_Orders..., Day_Sin, Day_Cos, Month_Sin, Month_Cos, Lead Time Signal]
    Action: Order Quantity (Normalized 0-1)
    """

    def __init__(self, config=None):
        super(InventoryEnv, self).__init__()
        
        # Configuration (Business Rules)
        self.config = config or {}
        self.max_capacity = self.config.get("max_capacity", 1000) # Max shelf space
        # Days to arrive
        self.min_lead_time = self.config.get("min_lead_time", 3) 
        self.max_lead_time = self.config.get("max_lead_time", 10) 
        self.max_order = self.config.get("max_order", 100)       # Max items per order
        
        # Costs & Prices
        self.unit_price = 70.0      # Revenue per unit sold
        self.holding_cost = 1.5     # Cost to keep 1 unit overnight
        self.shortage_penalty = 20.0 # Penalty for missing a sale (Opportunity cost + Brand damage)
        self.reward_scale = self.config.get("reward_scale", 2000.0)

        # Data Engine 
        data_path = self.config.get("data_path", 'data/train_data.csv')
        is_sequential = self.config.get("sequential", False)
        self.sampler = DemandSampler(data_path, sequential=is_sequential)

        # Action Space 
        # Continuous value between 0 and 1.
        # 0.0 = Don't order. 1.0 = Order 'max_order' units.
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Observation Space
        # Components:
        # 1. Current Stock (1 value)
        # 2. Pipeline Vector (Length = lead_time) -> "What is arriving in 1 day, 2 days..." (10 Values)
        # 3. Time Encoding (4 values) -> Sin/Cos of Day and Month
        # 4. Lead Time Signal (1 Value)
        obs_size = 1 + self.max_lead_time + 4 + 1 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self.recent_demand_history = []
        # Simulation State Variables
        self.inventory = 0
        self.pipeline = [] # Queue for orders in transit
        self.actual_lead_time = 5
        self.current_step = 0
        self.day_of_week = 0 # 0=Mon
        self.month = 1       # 1=Jan

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset the Sampler
        # It returns the actual date of the data chunk to use
        start_month, start_day = self.sampler.reset()
        
        # Sync Environment Time with Data Time
        self.month = start_month
        self.day_of_week = start_day
        self.current_step = 0
        
        # Initialization of History
        current_idx = self.sampler.current_idx
        initial_demand = self.sampler.df.iloc[current_idx]['Adjusted_Demand']
        self.recent_demand_history = [float(initial_demand)] * 5

        # Randomized Lead Time
        self.actual_lead_time = np.random.randint(self.min_lead_time, self.max_lead_time + 1)
        
        # Inventory Initialization
        self.inventory = np.random.randint(50, 150)
        
        # Clear Pipeline
        self.pipeline = [0] * self.actual_lead_time
        
        return self._get_obs(), {}

    def step(self, action):
        # Interpret Action
        # Model outputs [0.0 - 1.0], we scale it to [0 - max_order]
        order_amount_pct = np.clip(action[0], 0, 1)
        order_quantity = int(order_amount_pct * self.max_order)

        # Demand Generation
        # "What is the demand for this specific Month & Day?"
        real_demand, m, d = self.sampler.sample()

        # Update Rolling History
        self.recent_demand_history.append(real_demand)
        if len(self.recent_demand_history) > 5:
            self.recent_demand_history.pop(0)

        # Receive goods that were ordered 'lead_time' days ago
        arriving_inventory = self.pipeline.pop(0) 
        self.inventory += arriving_inventory
        
        # Add new order to the back of the pipeline
        self.pipeline.append(order_quantity)

        # Calculate Sales
        # we can't sell what we don't have
        sales = min(self.inventory, real_demand)
        missed_sales = real_demand - sales
        
        # Update Inventory
        self.inventory -= sales
        
        # Shelf space limit:
        # If we have too much, it spills over
        self.inventory = min(self.inventory, self.max_capacity)

        # Reward Calculation
        # Profit = Revenue - Cogs - Holding Cost - Shortage Penalty
        revenue = sales * self.config["unit_price"]

        # Money leaving to pay the supplier
        # We pay for what we ORDER, not what we sell (The Risk!)
        procurement_cost = order_quantity * self.config["unit_cost"]
        holding_costs = self.inventory * self.config["holding_cost"]
        penalty = missed_sales * self.config["shortage_penalty"]

        raw_reward = revenue - procurement_cost - holding_costs - penalty

        reward = raw_reward / self.reward_scale

        # Time updates
        self.current_step += 1
        self.day_of_week = (self.day_of_week + 1) % 7
        if self.current_step % 30 == 0: # Rough approximation of a month passing
            self.month = (self.month % 12) + 1

        # Termination
        # Episode ends after 1 year (365 steps)
        terminated = self.current_step >= 365
        truncated = False
        
        info = {
            "demand": real_demand,
            "sales": sales,
            "inventory": self.inventory,
            "ordered": order_quantity,
            "profit": raw_reward,
            "lead_time": self.actual_lead_time
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        """
        Helper to construct the state vector.
        Normalizes values to help the Neural Network.
        """
        # Normalize Inventory
        # We divide by log(max_capacity) to keep it near 0-1
        norm_inv = np.log1p(self.inventory) / np.log1p(self.max_capacity)
        
        # Normalize Pipeline
        norm_pipeline = [x / self.max_order for x in self.pipeline]

        # Pad with -1 or 0 to fill the vector up to max_lead_time
        # -1 helps the NN distinguish "Zero Order coming" between "Nothing exists here"
        padding_needed = self.max_lead_time - len(norm_pipeline)
        padded_pipeline = norm_pipeline + [-1.0] * padding_needed

        avg_recent = np.mean(self.recent_demand_history)
        norm_recent_demand = avg_recent / self.max_order
        
        # Cyclic Time Encoding
        # This helps NN understand that Day 6 (Sun) is close to Day 0 (Mon)
        day_sin = np.sin(2 * np.pi * self.day_of_week / 7)
        day_cos = np.cos(2 * np.pi * self.day_of_week / 7)
        month_sin = np.sin(2 * np.pi * self.month / 12)
        month_cos = np.cos(2 * np.pi * self.month / 12)

        norm_lead_time = self.actual_lead_time / self.max_lead_time

        obs = np.concatenate(([norm_inv], padded_pipeline, [day_sin, day_cos, month_sin, month_cos, norm_lead_time], [norm_recent_demand]))
        return obs.astype(np.float32)