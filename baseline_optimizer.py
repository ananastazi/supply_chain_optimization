import numpy as np
from envs.inventory_env import InventoryEnv

def optimize_baseline(base_config, seed=42, n_episodes=5):
    """
    Monte Carlo Grid Search to find the best (s, S) parameters.
    """
    print(f"Optimizing Baseline (Seed={seed})")
    
    # We use sequential=False so the baseline faces random lead times/demand
    train_config = base_config.copy()
    train_config['data_path'] = base_config['train_data_path']
    train_config['sequential'] = False 
    
    env = InventoryEnv(config=train_config)
    
    best_profit = -float('inf')
    best_params = (0, 0)
    
    # Grid Search Range
    s_values = range(0, 300, 50)         # Reorder Point
    S_values_offset = range(50, 600, 50) # Order Up To (S > s)
    
    for s in s_values: 
        for diff in S_values_offset: 
            S = s + diff
            
            avg_profit = 0
            
            # We want every (s, S) candidate to face the exact same scenarios.
            # So we reset the seed inside the loop for every parameter set.
            for i in range(n_episodes):
                
                # We derive a specific seed for this episode index (e.g., 42, 43, 44...)
                # This ensures Episode 1 is always the same "Weather", 
                # regardless of which (s, S) we are testing.
                run_seed = seed + i
                
                # Reset Global Numpy (Used by DemandSampler & Lead Times)
                np.random.seed(run_seed)
                
                # Reset Environment
                obs, _ = env.reset(seed=run_seed)
                
                terminated = False
                episode_reward = 0
                
                while not terminated:
                    current_inv = env.inventory
                    
                    # Heuristic Policy Logic
                    if current_inv < s:
                        needed = min(S - current_inv, env.max_order)
                        action = [needed / env.max_order]
                    else:
                        action = [0.0]
                    
                    _, reward, terminated, _, _ = env.step(action)
                    episode_reward += reward
                
                avg_profit += episode_reward

            avg_profit /= n_episodes
            
            if avg_profit > best_profit:
                best_profit = avg_profit
                best_params = (s, S)
                print(f"Scanned s={s}, S={S}: Profit={avg_profit:.0f}")

    print(f"Best Params fot the baseline: s={best_params[0]}, S={best_params[1]}")
    return best_params