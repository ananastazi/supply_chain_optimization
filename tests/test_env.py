from envs.inventory_env import InventoryEnv

# Initialize
env = InventoryEnv()
obs, _ = env.reset()

print("Initial State:", obs)

# Run for 10 days with random actions
for i in range(10):
    # Random action between 0 and 1
    action = env.action_space.sample()
    
    obs, reward, terminated, _, info = env.step(action)
    
    print(f"Day {i+1}: Stock={info['inventory']}, Demand={info['demand']}, Order={info['ordered']}, Reward={reward:.2f}")

print("Environment Test Passed!")