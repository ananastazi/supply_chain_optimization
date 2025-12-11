import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from envs.env_config import config
from envs.utils.make_env import make_env

def train_sac():
    print("Starting SAC Advanced Training")
    
    # 1. SETUP FOLDERS
    # we create specific folders to save our "brains" (models) and the graphs (logs)
    models_dir = "models/sac"
    log_dir = "logs/sac"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 2. TRAINING ENVIRONMENT
    # we use 'is_sequential=False'. this shuffles the days randomly
    # goal: Force the agent to learn the general rules of supply chain, not just memorize the dates
    train_env = DummyVecEnv([
        lambda: make_env(config["train_data_path"], is_sequential=False)
    ])
    
    # 3. EVALUATION ENVIRONMENT
    # we use 'is_sequential=True'. This keeps the real chronological order
    # goal: Test how the agent performs in a realistic timeline simulation
    eval_env = DummyVecEnv([
        lambda: make_env(config["test_data_path"], is_sequential=True)
    ])

    # we use a callback to test the agent on the evaluation environment every 10000 steps. this ensures we save the best version of the model, avoiding overfitting
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=log_dir,
        eval_freq=10000,
        n_eval_episodes=20,
        deterministic=True,
        render=False
    )

    # 5. DEFINING THE AGENT (SAC)
    # we chose SAC as a challenger to PPO. It's usually better at "Continuous Control" (like deciding the exact amount of inventory to buy)
    model = SAC(
        "MlpPolicy", 
        train_env, 
        verbose=1, 
        tensorboard_log=log_dir,
        buffer_size=100000, # unlike PPO, SAC remembers past experiences. It stores 100k steps here to "review" them later
        learning_starts=5000, #we increased this number significantly, because our environment is very random (demand and delivery times change a lot). if the agent starts learning immediately, it might jump to wrong conclusions based on just a few lucky or unlucky days. by waiting 5000 steps (about 13 simulated years), we force it to gather a lot of diverse experiences before it starts forming a strategy
        batch_size=256, 
        ent_coef='auto', # we let the algorithm automatically tune the entropy of the policy to balance exploration and exploitation
        gradient_steps=1,
        learning_rate=3e-4, 
    )


    # 6. TRAINING THE AGENT
    # we run for 300,000 steps. this is less than PPO because SAC is "Sample Efficient" (it learns faster from data)
    TIMESTEPS = 300000 
    print(f"Training for {TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TIMESTEPS, callback=eval_callback, tb_log_name="SAC_Advanced")

    print("SAC Training Complete")

if __name__ == "__main__":
    train_sac()