import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Import your custom environment
# Ensure your folder structure allows this import
from envs.inventory_env import InventoryEnv

def train_ppo():
    print("--- Starting PPO Training ---")
    
    # 1. Setup Directories
    models_dir = "models/ppo"
    log_dir = "logs/ppo"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 2. Initialize Environment
    # We wrap it in DummyVecEnv which is standard for SB3 training
    env = InventoryEnv()
    
    # Sanity check: Ensure the environment follows Gym API
    check_env(env)
    
    # Vectorize the environment (wrapper for SB3)
    env = DummyVecEnv([lambda: InventoryEnv()])

    # 3. Define the Model (PPO)
    # Hyperparameters:
    # - MlpPolicy: Standard Dense Neural Network (Input -> Hidden -> Output)
    # - ent_coef=0.01: Adds a bonus for randomness. 
    #   If the agent is 100% sure of an action, it gets penalized slightly.
    #   This forces it to keep trying new things ("Exploration").
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir,
        ent_coef=0.01,         # Your requested exploration parameter
        learning_rate=0.0003,  # Standard default
        n_steps=2048,          # Number of steps to run per update
        batch_size=64
    )

    # 4. Train
    # 100,000 timesteps is usually enough for inventory problems to converge
    TIMESTEPS = 100000
    model.learn(total_timesteps=TIMESTEPS, tb_log_name="PPO_run")

    # 5. Save the Model
    model_path = os.path.join(models_dir, "ppo_inventory_final")
    model.save(model_path)
    
    print(f"--- PPO Training Complete. Model saved to {model_path}.zip ---")

if __name__ == "__main__":
    train_ppo()