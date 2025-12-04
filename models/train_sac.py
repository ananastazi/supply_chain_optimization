import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Import your custom environment
from envs.inventory_env import InventoryEnv

def train_sac():
    print("--- Starting SAC Training ---")
    
    # 1. Setup Directories
    models_dir = "models/sac"
    log_dir = "logs/sac"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 2. Initialize Environment
    # SAC works fine with a single environment, but we wrap it for consistency
    env = InventoryEnv()
    check_env(env)
    env = DummyVecEnv([lambda: InventoryEnv()])

    # 3. Define the Model (SAC)
    # Hyperparameters:
    # - buffer_size=100000: Memories to keep. Large buffers stabilize training.
    # - learning_starts=1000: Don't learn immediately. 
    #   Just take random actions for 1000 steps to fill the buffer first.
    #   This breaks the correlation of initial steps.
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir,
        buffer_size=100000,    # Large memory
        learning_starts=1000,  # Collect random data first
        batch_size=256,        # SAC likes larger batches than PPO
        ent_coef='auto'        # SAC automatically tunes entropy (Recommended)
    )

    # 4. Train
    # SAC is slower per step (more math) but learns faster in terms of episodes.
    TIMESTEPS = 50000 # SAC often needs fewer steps than PPO
    model.learn(total_timesteps=TIMESTEPS, tb_log_name="SAC_run")

    # 5. Save the Model
    model_path = os.path.join(models_dir, "sac_inventory_final")
    model.save(model_path)
    
    print(f"--- SAC Training Complete. Model saved to {model_path}.zip ---")

if __name__ == "__main__":
    train_sac()