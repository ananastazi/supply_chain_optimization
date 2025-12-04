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
    print("Starting SAC Advanced Training")
    
    #we define the paths to save the models and the logs for tensorboard analysis
    models_dir = "models/sac"
    log_dir = "logs/sac"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    #environment setup: we stricltly separate the data with training environment that uses the first 80% of history to learn and then evaluation environment that uses the last 20% to validat performance on unseen
    train_env = DummyVecEnv([lambda: make_env('data/train_data.csv')])
    eval_env = DummyVecEnv([lambda: make_env('data/test_data.csv')])

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

    # SAC (Soft Actor-Critic) Model: we implemented the SAC algotithm as a challenger to PPO. Because our action space is continuous like we order specific quantities of inventory, SAC is a good fit for this kind of problem. And also because we hypothesized that SAC might be more sample-efficient than PPO.
    model = SAC(
        "MlpPolicy", 
        train_env, 
        verbose=1, 
        tensorboard_log=log_dir,
        buffer_size=100000, #this algorithm is an "off-policy" meaning it learns from past experiences stored in a  buffer. we set a large buffer to provide a rich history for the agent to learn from 
        learning_starts=1000, #we force the agent to explore randomly for the first 1000 steps. this prevents it from converging too early on a bad strategy without trying options
        batch_size=256, #we let the algorithm automatically tune the entropy of the policy to balance exploration and exploitation
        ent_coef='auto'
    )

    # Training loop: we train for 300000 steps to allow the critic network to stabilize its predicitions
    TIMESTEPS = 300000 # set to 80 000
    print(f"Training for {TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TIMESTEPS, callback=eval_callback, tb_log_name="SAC_Advanced")

    print("SAC Training Complete")

if __name__ == "__main__":
    train_sac()