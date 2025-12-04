import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from envs.inventory_env import InventoryEnv

def make_env(data_path):
    env = InventoryEnv(config={"data_path": data_path})
    env = Monitor(env)
    return env

def train_sac():
    print("--- Starting SAC Advanced Training ---")
    
    models_dir = "models/sac"
    log_dir = "logs/sac"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Training & Validation Envs
    train_env = DummyVecEnv([lambda: make_env('data/train_data.csv')])
    eval_env = DummyVecEnv([lambda: make_env('data/test_data.csv')])

    # Callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=log_dir,
        eval_freq=10000,
        n_eval_episodes=20,
        deterministic=True,
        render=False
    )

    # SAC Model
    model = SAC(
        "MlpPolicy", 
        train_env, 
        verbose=1, 
        tensorboard_log=log_dir,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        ent_coef='auto'
    )

    # Train
    TIMESTEPS = 300000
    print(f"Training for {TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TIMESTEPS, callback=eval_callback, tb_log_name="SAC_Advanced")

    print("--- SAC Training Complete ---")

if __name__ == "__main__":
    train_sac()