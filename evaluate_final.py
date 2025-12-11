import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO, SAC
import os

from envs.inventory_env import InventoryEnv
from envs.env_config import config

# Configuration

TRAIN_DATA_PATH = config['train_data_path']       
TEST_DATA_PATH = config['test_data_path']  
REWARD_SCALE = config['reward_scale']                 

AGENTS = {
    "PPO": ("models/ppo/best_model.zip", PPO),
    "SAC": ("models/sac/best_model.zip", SAC),
}

os.makedirs("results", exist_ok=True)


# Helper functions

def optimize_heuristic(base_config):
    """
    Grid Search on TRAINING data.
    """
    print("--- Optimizing (s, S) Heuristic on Training Data ---")
    
    # Updating config to use Training Data
    train_config = base_config.copy()
    train_config['data_path'] = TRAIN_DATA_PATH
    train_config['sequential'] = True
    
    env = InventoryEnv(config=train_config)
    
    best_profit = -float('inf')
    best_params = (10, 50) 
    
    # s: Reorder point
    # S: Order up to
    for s in range(0, 300, 50): 
        for S in range(s + 50, 1000, 50): 
            obs, _ = env.reset(seed=42)
            terminated = False
            total_reward = 0
            
            while not terminated:
                current_inv = env.inventory
                if current_inv < s:
                    needed = min(S - current_inv, env.max_order)
                    action = [needed / env.max_order]
                else:
                    action = [0.0]
                
                
                _, reward, terminated, _, _ = env.step(action)
                total_reward += reward
            
            if total_reward > best_profit:
                best_profit = total_reward
                best_params = (s, S)

    print(f"Optimization Complete. Best Params: s={best_params[0]}, S={best_params[1]}")
    return best_params


def run_simulation(agent_name, agent_model, base_config, heuristic_params=None):
    print(f"Simulating: {agent_name}...")
    
    # Updating config to use TEST Data
    test_config = base_config.copy()
    test_config['data_path'] = TEST_DATA_PATH
    test_config['sequential'] = True
    
    env = InventoryEnv(config=test_config)
    
    
    obs, _ = env.reset(seed=999) 
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
            action, _ = agent_model.predict(obs, deterministic=True)
            

        obs, reward, terminated, truncated, info = env.step(action)
        
    
        real_profit = reward * REWARD_SCALE 
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


# Visualization functions

def generate_plots(full_df, metrics_df):
    sns.set_theme(style="whitegrid")
    
    # Plot A: Cumulative Profit 
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=full_df, x="step", y="cumulative_profit", hue="Agent", linewidth=2)
    plt.title("Cumulative Profit Comparison")
    plt.ylabel("Profit ($)")
    plt.xlabel("Days")
    plt.tight_layout()
    plt.savefig("results/1_cumulative_profit.png")
    plt.close()

    # Plot B: Inventory Dynamics 
    mid_point = len(full_df) // len(full_df['Agent'].unique()) // 2
    subset = full_df[(full_df['step'] > mid_point) & (full_df['step'] < mid_point + 60)]
    
    plt.figure(figsize=(12, 6))
    plt.fill_between(subset['step'].unique(), subset[subset['Agent']==subset['Agent'].iloc[0]]['demand'], 
                     color='gray', alpha=0.2, label='Demand')
    sns.lineplot(data=subset, x="step", y="inventory", hue="Agent", linewidth=2)
    plt.title("Inventory Levels vs Demand (60 Day Zoom)")
    plt.ylabel("Units")
    plt.axhline(0, color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig("results/2_inventory_dynamics.png")
    plt.close()

    # Plot C: Action Distribution
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=full_df, x="action", hue="Agent", fill=True, common_norm=False, alpha=0.3)
    plt.title("Action Distribution (Order Sizes)")
    plt.xlabel("Order Quantity")
    plt.tight_layout()
    plt.savefig("results/3_action_distribution.png")
    plt.close()

    # Plot D: Efficiency Frontier
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=metrics_df, x="Avg Inventory", y="Service Level (%)", 
                    hue="Agent", style="Agent", s=200)
    
    for i, row in metrics_df.iterrows():
        plt.text(row['Avg Inventory'], row['Service Level (%)']+0.2, 
                 f"{row['Agent']}\n${row['Total Profit']:,.0f}", fontsize=9)
        
    plt.title("Efficiency Frontier (Cost vs Service)")
    plt.tight_layout()
    plt.savefig("results/4_efficiency_frontier.png")
    plt.close()


if __name__ == "__main__":
    
    # Optimizing Baseline
    best_s, best_S = optimize_heuristic(config)
    
    # Running Simulations
    all_dfs = []
    all_metrics = []
    
    # Running Baseline
    df_base, met_base = run_simulation("Optimized Baseline", None, config, heuristic_params=(best_s, best_S))
    all_dfs.append(df_base)
    all_metrics.append(met_base)
    
    # Running RL Agents
    for name, (path, Algo) in AGENTS.items():
        try:
            model = Algo.load(path)
            df, met = run_simulation(name, model, config)
            all_dfs.append(df)
            all_metrics.append(met)
        except Exception as e:
            print(f"Skipping {name}: {e}")

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        metrics_df = pd.DataFrame(all_metrics)
        
        print("\n" + "="*40)
        print("FINAL RESULTS TABLE")
        print("="*40)
        print(metrics_df.sort_values(by="Total Profit", ascending=False).to_string(index=False))
        
        generate_plots(full_df, metrics_df) 
        print("\nDone!")