# Reinforcement learning : FrozenLake Application and Explanations
# With a performance and training recorder 

# Imports
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import gymnasium as gym
import numpy as np
import logging
import random
import matplotlib.pyplot as plt
import pickle

# Elf Configuration :
# Four Actions :
    
    # - Move left
    # - Move down
    # - Move right
    # - Move up

# Three Rewards:
    
    # Reach goal: +1
    # Reach hole: 0
    # Reach frozen: 0
    
# Episode End

#Termination:

# - Moves into a hole.
# - Reaches the goal at max(nrow) * max(ncol) - 1 (location [max(nrow)-1, max(ncol)-1]).

# Truncation (using the time_limit wrapper):

# Length of the episode is :
    # - 100 for FrozenLake4x4 
    # - 200 for FrozenLake8x8.
    
    
# Training configuration
training_period = 250          # Record video every 250 episodes
env_name = "FrozenLake-v1"  # Replace with your environment

# Set up logging for episode statistics
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Environment creation and initialization
env = gym.make(env_name,
               # name of the environment
               map_name="8x8", 
               # size of the map
               is_slippery = False, 
               # the step function doesn't always honor the action
               render_mode="rgb_array"
               # image for the recording 
               )

# Hyper-parameters
alpha = 0.9             # learning rate
gamma = 0.9             # discount factor (immediate/long-term rewards)
epsilon = 1             # exploration rate (choice: exploration 1.0, long exploration vs exploitation 0.0, greedy)
epsilon_decay = 0.0001  # speed of transition from exploration to exploitation,
                        # (balance between exploration (0.999) and exploitation (0.99))
epsilon_min = 0         # minimal exploration
episodes = 15000        # number of try (5000/20000 for serious training)
rng = np.random.default_rng()   # random number generator
# max_steps = 1000        # number of steps

rewards_per_episode = np.zeros(episodes) # rewards for each episodes

# Recording Env
env = RecordVideo(
    env,
    video_folder="elf-frozen-lake",    # Folder to save videos
    name_prefix="training",
    episode_trigger=lambda x: x % training_period == 0  # Only record every 250th episode
)

# Track statistics for every episode (lightweight)
env = RecordEpisodeStatistics(env)

print(f"Starting training for {episodes} episodes")
print(f"Videos will be recorded every {training_period} episodes")
print("Videos will be saved to: elf-frozen-lake/")

# q_table initialized with a 64 x 4 array shape 
q_table = np.zeros((env.observation_space.n, env.action_space.n))

for i in range(episodes):
    
    state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
    terminated = False      # True when fall in hole or reached goal
    truncated = False       # True when actions > 200
    
    total_reward = 0
    
    # While Loop : Repetitive action until terminated or truncated
    while(not terminated and not truncated):
        
        # Epsilon - greedy policy
        if rng.random() < epsilon:
            
            # actions : # 0 = left, 1 = down, 2 = right, 3 = up
            action = env.action_space.sample( ) # Exploration
            
        else:
            action = np.argmax(q_table[state,:]) # Exploitation 
            
        # Step is stored with the : reward, new_state, info and wether it's
        # terminated or truncated
        new_state, reward, terminated, truncated, info = env.step(action)
        
        # Retrieving old values and adding new ones to the table
        # old_value = q_table[state][action] # Actual Q value
        #next_max = np.max(q_table[new_state]) 
    
        if terminated:
            next_max = reward
        else:
            next_max = reward + gamma * np.max(q_table[new_state,:]) # Max Q value of state S'
        
        # Update table q with Bellman's formula
        # q_table[state][action] = (1 - alpha) * old_value + alpha * next_max
        
        q_table[state, action] += alpha * (next_max - q_table[state, action])
        
        # Refresh status and total number of rewards
        state = new_state
        total_reward += reward
        
    # Epsilon Reduction
    epsilon = max(epsilon - epsilon_decay, epsilon_min)

    if epsilon<=epsilon_min:
        alpha = 0.0001

    if total_reward == 1:
        rewards_per_episode[i] = 1
    
    #print(f"Episode {episode + 1}, reward = {total_reward}")
    
    # Log episode statistics (available in info after episode ends)
    if "episode" in info:
        episode_data = info["episode"]
        logging.info(f"Episode {i}: "
                    f"reward={episode_data['r']:.1f}")

        # Additional analysis for milestone episodes
        if i % 1000 == 0:
            # Look at recent performance (last 100 episodes)
            recent_rewards = list(env.return_queue)[-100:]
            if recent_rewards:
                avg_recent = sum(recent_rewards) / len(recent_rewards)
                print(f" -> Average reward over last 100 episodes: {avg_recent:.1f}")
                
env.close() # closing environment

sum_rewards = np.zeros(episodes)
for t in range(episodes):
    sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
plt.plot(sum_rewards)
plt.savefig('frozen_lake8x8.png')


