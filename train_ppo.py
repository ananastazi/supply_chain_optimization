import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from envs.utils.make_env import make_env

from envs.env_config import config

def train_ppo():
    print("--- Starting PPO Advanced Training ---")
    
    models_dir = "models/ppo"
    log_dir = "logs/ppo"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_env = DummyVecEnv([
        lambda: make_env(config["train_data_path"], is_sequential=False)
    ])
    
    # We want to test how it performs on the specific historical sequence of the test set.
    eval_env = DummyVecEnv([
        lambda: make_env(config["test_data_path"], is_sequential=True)
    ])

    # Callback:
    # - Evaluate every 10,000 steps
    # - Test for 20 episodes to get a stable average
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=log_dir,
        eval_freq=10000,
        n_eval_episodes=20,
        deterministic=True, # Use deterministic actions (no random exploration) during test
        render=False
    )

    # Model
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1, 
        tensorboard_log=log_dir,
        ent_coef=0.01,
        learning_rate=0.00005,
        batch_size=64
    )

    # Train
    TIMESTEPS = 500000 
    
    print(f"Training for {TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TIMESTEPS, callback=eval_callback, tb_log_name="PPO_Advanced")

    print("--- PPO Training Complete ---")

if __name__ == "__main__":
    train_ppo()