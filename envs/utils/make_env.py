from stable_baselines3.common.monitor import Monitor

from envs.env_config import config
from envs.inventory_env import InventoryEnv

def make_env(data_path, is_sequential=False):
    """
    Creates an environment with specific data and sampling mode.
    """
    # 1. Create a shallow copy of the global config
    # This prevents overwriting the global variable for other environments
    env_config = config.copy()
    
    # 2. Inject the specific settings for THIS environment instance
    env_config["data_path"] = data_path
    env_config["sequential"] = is_sequential # True for Test, False for Train
    
    # 3. Initialize Env with the modified config
    env = InventoryEnv(config=env_config)
    
    # 4. Wrap in Monitor for logging
    env = Monitor(env)
    return env