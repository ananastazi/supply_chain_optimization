import pandas as pd
import numpy as np
import os
from stable_baselines3 import PPO, SAC

from envs.inventory_env import InventoryEnv
from envs.env_config import config

from baseline_optimizer import optimize_baseline 
from plots import generate_plots

# Configuration
AGENTS = {
    "PPO": ("models/ppo/best_model.zip", PPO),
    "SAC": ("models/sac/best_model.zip", SAC),
}
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_simulation(agent_name, agent_model, base_config, heuristic_params=None, seed=42):
    """
    Runs a single simulation for a specific agent on the TEST data.
    """
    print(f"Simulating: {agent_name} (Seed={seed})...")
    
    np.random.seed(seed)
    
    # Configure for Testing (Strict Sequential History)
    test_config = base_config.copy()
    test_config['data_path'] = base_config['test_data_path']
    test_config['sequential'] = True
    
    env = InventoryEnv(config=test_config)
    
    obs, _ = env.reset(seed=seed) 
    terminated = False
    
    history = {
        "step": [], "inventory": [], "action": [], 
        "demand": [], "sales": [], "profit": [], "cumulative_profit": []
    }
    
    cum_profit = 0
    step = 0
    
    while not terminated:
        if agent_name == "Optimized Baseline":
            s, S = heuristic_params
            current_inv = env.inventory
            if current_inv < s:
                needed = min(S - current_inv, env.max_order)
                action = [needed / env.max_order]
            else:
                action = [0.0]
        else:
            # Deterministic=True is standard for testing RL agents
            action, _ = agent_model.predict(obs, deterministic=True)
            
        # Step Environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Log Data
        real_profit = reward * base_config['reward_scale'] 
        cum_profit += real_profit
        
        history["step"].append(step)
        history["inventory"].append(info['inventory'])
        history["action"].append(action[0] * env.max_order)
        history["demand"].append(info['demand'])
        history["sales"].append(info['sales'])
        history["profit"].append(real_profit)
        history["cumulative_profit"].append(cum_profit)
        step += 1
        
    df = pd.DataFrame(history)
    df["Agent"] = agent_name
    
    metrics = {
        "Agent": agent_name,
        "Total Profit": cum_profit,
        "Service Level (%)": (df['sales'].sum() / df['demand'].sum() * 100) if df['demand'].sum() > 0 else 0,
        "Avg Inventory": df['inventory'].mean(),
        "Stockout Days": (df['inventory'] == 0).sum()
    }
    return df, metrics


if __name__ == "__main__":
    
    best_s, best_S = optimize_baseline(config, seed=42)
    
    all_dfs = []
    all_metrics = []
    
    # Run Baseline
    df_base, met_base = run_simulation(
        "Optimized Baseline", 
        None, 
        config, 
        heuristic_params=(best_s, best_S),
        seed=42
    )
    all_dfs.append(df_base)
    all_metrics.append(met_base)
    
    # Run RL Agents
    for name, (path, Algo) in AGENTS.items():
        if os.path.exists(path):
            print(f"Loading {name} from {path}")
            model = Algo.load(path)
            df, met = run_simulation(name, model, config, seed=42)
            all_dfs.append(df)
            all_metrics.append(met)
        else:
            print(f"Warning: Model for {name} not found at {path}")

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        metrics_df = pd.DataFrame(all_metrics)
        
        print("\n" + "="*40)
        print("FINAL RESULTS TABLE")
        print("="*40)
        print(metrics_df.sort_values(by="Total Profit", ascending=False).to_string(index=False))
        
        generate_plots(full_df, metrics_df, output_dir=RESULTS_DIR)
        