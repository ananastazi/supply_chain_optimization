from stable_baselines3.common.monitor import Monitor

from envs.env_config import config
from envs.inventory_env import InventoryEnv

def make_env(data_path, is_sequential=False):
    """
    Creates an environment with specific data and sampling mode.
    """
    env_config = config.copy()
    
    env_config["data_path"] = data_path
    env_config["sequential"] = is_sequential # True for Test, False for Train
    
    env = InventoryEnv(config=env_config)
    
    env = Monitor(env)
    return env