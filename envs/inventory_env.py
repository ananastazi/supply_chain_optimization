import gymnasium as gym
import numpy as np
from gymnasium import spaces
from envs.demand_sampler import DemandSampler  

class InventoryEnv(gym.Env):
    """
    Custom Environment for Inventory Management
    State: [Current_Stock, Pipeline_Orders..., Day_Sin, Day_Cos, Month_Sin, Month_Cos]
    Action: Order Quantity (Normalized 0-1)
    """

    def __init__(self, config=None):
        super(InventoryEnv, self).__init__()
        
        # --- 1. Configuration (Business Rules) ---
        self.config = config or {}
        self.max_capacity = self.config.get("max_capacity", 200) # Max shelf space
        self.lead_time = self.config.get("lead_time", 5)         # Days to arrive
        self.max_order = self.config.get("max_order", 100)       # Max items per order
        
        # Costs & Prices (From your prompt)
        self.unit_price = 10.0      # Revenue per unit sold
        self.holding_cost = 0.5     # Cost to keep 1 unit overnight
        self.shortage_penalty = 20.0 # Penalty for missing a sale (Opportunity cost + Brand damage)
        
        # Load the Data Engine from Phase 1
        data_path = self.config.get("data_path", 'data/train_data.csv')
        is_sequential = self.config.get("sequential", False)
        self.sampler = DemandSampler(data_path, sequential=is_sequential)

        # --- 2. Action Space (Phase 2.2) ---
        # Continuous value between 0 and 1.
        # 0.0 = Don't order. 1.0 = Order 'max_order' units.
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # --- 3. Observation Space (Phase 2.1) ---
        # Components:
        # 1. Current Stock (1 value)
        # 2. Pipeline Vector (Length = lead_time) -> "What is arriving in 1 day, 2 days..."
        # 3. Time Encoding (4 values) -> Sin/Cos of Day and Month
        obs_size = 1 + self.lead_time + 4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        # Simulation State Variables
        self.inventory = 0
        self.pipeline = [] # Queue for orders in transit
        self.current_step = 0
        self.day_of_week = 0 # 0=Mon
        self.month = 1       # 1=Jan

    def reset(self, seed=None, options=None):
        """
        Resets the environment to start a new 'Year' (Episode).
        """
        super().reset(seed=seed)

        self.sampler.reset()
        
        # Start with some random stock to prevent immediate crash
        self.inventory = np.random.randint(20, 50)
        
        # Empty pipeline (nothing ordered yet)
        self.pipeline = [0] * self.lead_time
        
        # Reset Time
        self.current_step = 0
        self.day_of_week = 0
        self.month = 1
        
        return self._get_obs(), {}

    def step(self, action):
        """
        Phase 2.3: The Physics of the simulation.
        """
        # 1. Interpret Action
        # Model outputs [0.0 - 1.0], we scale it to [0 - max_order]
        order_amount_pct = np.clip(action[0], 0, 1)
        order_quantity = int(order_amount_pct * self.max_order)

        # 2. Demand Generation (Using Phase 1 Sampler)
        # We ask: "What is the demand for this specific Month & Day?"
        real_demand = self.sampler.sample(self.month, self.day_of_week)

        # 3. Pipeline Dynamics (Time Delay)
        # a. Receive goods that were ordered 'lead_time' days ago
        arriving_inventory = self.pipeline.pop(0) 
        self.inventory += arriving_inventory
        
        # b. Add NEW order to the back of the pipeline
        self.pipeline.append(order_quantity)

        # 4. Calculate Sales
        # You can't sell what you don't have
        sales = min(self.inventory, real_demand)
        missed_sales = real_demand - sales
        
        # 5. Update Inventory
        self.inventory -= sales
        
        # Cap inventory (Shelf space limit) - Optional logic
        # If we have too much, it spills over (or we just pay holding cost on all of it)
        # self.inventory = min(self.inventory, self.max_capacity)

        # 6. Reward Calculation
        # Profit = Revenue - Holding Cost - Shortage Penalty
        revenue = sales * self.unit_price
        holding_costs = self.inventory * self.holding_cost
        penalty = missed_sales * self.shortage_penalty
        
        raw_reward = revenue - holding_costs - penalty
        
        reward = raw_reward / 2000.0 

        # 7. Time updates
        self.current_step += 1
        self.day_of_week = (self.day_of_week + 1) % 7
        if self.current_step % 30 == 0: # Rough approximation of a month passing
            self.month = (self.month % 12) + 1

        # 8. Termination
        # Episode ends after 1 year (365 steps)
        terminated = self.current_step >= 365
        truncated = False
        
        info = {
            "demand": real_demand,
            "sales": sales,
            "inventory": self.inventory,
            "ordered": order_quantity,
            "profit": raw_reward
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        """
        Helper to construct the state vector.
        Normalizes values to help the Neural Network.
        """
        # Normalize Inventory
        norm_inv = self.inventory / self.max_capacity
        
        # Normalize Pipeline
        norm_pipeline = [x / self.max_order for x in self.pipeline]
        
        # Cyclic Time Encoding (The "Sin/Cos" Trick)
        # This helps NN understand that Day 6 (Sun) is close to Day 0 (Mon)
        day_sin = np.sin(2 * np.pi * self.day_of_week / 7)
        day_cos = np.cos(2 * np.pi * self.day_of_week / 7)
        month_sin = np.sin(2 * np.pi * self.month / 12)
        month_cos = np.cos(2 * np.pi * self.month / 12)

        obs = np.concatenate(([norm_inv], norm_pipeline, [day_sin, day_cos, month_sin, month_cos]))
        return obs.astype(np.float32)