import pandas as pd
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from envs.inventory_env import InventoryEnv

# --- Configuration ---
TEST_DATA_PATH = 'data/test_data.csv'
PPO_MODEL_PATH = "models/ppo/best_model.zip"
SAC_MODEL_PATH = "models/sac/best_model.zip"

def run_test(agent, env_config, agent_name="Agent"):
    """
    Runs a full sequential pass through the test data.
    """
    # Initialize Env with the config (Sequential Mode)
    env = InventoryEnv(config=env_config)
    
    # Reset (Forces sampler to start at Row 0 of test_data.csv)
    obs, _ = env.reset(seed=42) 
    terminated = False
    
    total_reward = 0
    total_sales = 0
    total_demand = 0
    stockout_days = 0
    inventory_levels = []
    steps = 0
    
    print(f"Running simulation for {agent_name}...")
    
    while not terminated:
        if agent == "Baseline":
            # --- TRADITIONAL HEURISTIC (Min-Max) ---
            # "If inventory < 20, Order up to 80."
            current_inv = env.inventory
            if current_inv < 20:
                needed = 80 - current_inv
                action = [needed / env.max_order] # Normalize
            else:
                action = [0.0]
        else:
            # --- AI AGENT ---
            # deterministic=True removes random noise from the agent's brain
            action, _ = agent.predict(obs, deterministic=True)
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Accumulate metrics (Unscale reward to get real $$$)
        total_reward += (reward * 2000.0) 
        total_sales += info['sales']
        total_demand += info['demand']
        inventory_levels.append(info['inventory'])
        
        if info['inventory'] == 0 and info['demand'] > 0:
            stockout_days += 1
        
        steps += 1
            
    # Calculate KPIs
    service_level = (total_sales / total_demand) * 100 if total_demand > 0 else 0
    avg_inv = np.mean(inventory_levels)
    
    print(f"--- {agent_name} Final Report ---")
    print(f"Steps Simulatd:  {steps}")
    print(f"Total Profit:    ${total_reward:,.2f}")
    print(f"Service Level:   {service_level:.2f}%")
    print(f"Stockout Days:   {stockout_days}")
    print(f"Avg Inventory:   {avg_inv:.1f} units")
    print("=" * 30)
    
    return {
        "Name": agent_name,
        "Profit": total_reward,
        "Service Level": service_level,
        "Stockouts": stockout_days
    }

if __name__ == "__main__":
    print("--- STARTING PHASE 5: BACKTESTING ---")
    print("Loading Models...")
    
    # Load Agents
    try:
        ppo_model = PPO.load(PPO_MODEL_PATH)
        print("PPO Loaded.")
    except:
        ppo_model = None
        print("PPO Model not found.")

    try:
        sac_model = SAC.load(SAC_MODEL_PATH)
        print("SAC Loaded.")
    except:
        sac_model = None
        print("SAC Model not found.")

    # Define Env Config for Testing
    # This triggers the 'Sequential' logic in the Env and Sampler
    test_config = {
        "data_path": TEST_DATA_PATH, 
        "sequential": True 
    }
    
    # 1. Run Baseline (Traditional Logic)
    run_test("Baseline", test_config, agent_name="Traditional Rule (Min-Max)")
    
    # 2. Run PPO
    if ppo_model:
        run_test(ppo_model, test_config, agent_name="PPO Agent")
        
    # 3. Run SAC
    if sac_model:
        run_test(sac_model, test_config, agent_name="SAC Agent")