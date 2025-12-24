# Reinforcement learning: Reinforcement learning Lunar v3

# Creation of an agent that performs actions in an environment.
# Actions modify the agent's states, and the change results in a
# reward.
# The agent's goal: Maximize rewards!
# So it learns which actions to perform to obtain the most rewards.

# State: Position, Emotion, etc.
# Action: Movement (e.g., up), Passage, Question/answer, etc.
# Reward: Points, Money, etc.

# f(s) = a, or: what action to perform when in a given state.
# 
# f(state) = action, in other words.

# This function is called: Action Policy.

# There are several methods, one of which is Q-Learning.
# Q-learning: learning the value of an action in a given state.

# S ------------|-----------|
#               |  Q(S, A)  | -------> Expected final score
# A ------------|-----------|

# A Q function that takes a state and an action as input and predicts a
# final score that can be expected to be obtained.

# Learning the Q function: exploration of the environment by the agent 
# to generate data: S, A, R (state, action, reward)

# Q(S, A) table <- data updated for the purpose of predicting future
# rewards by choosing certain actions in the present.


# Initially, the table is filled with random values.

# BellMan equation (screen + ML Book): (expected total score)
    # Q(S,A) = E*[R(S') + y*max*Q(S', A')]
    
# Table updated using the equation and the reward obtained:
    # Q(S,A) = (1-n)*Q(S,A)+n*(R(S') + y*max*Q(S', A'))
    # n = learning rate
    

# “Epsilon” probability to avoid stopping exploration as soon as the correct answer is found,
# so that the machine continues to explore the range of possibilities, rather than
# exploiting its already acquired action policy.

# Let's move on to practice with OpenAI's Gymnasium, a module that allows the use 
# of reinforcement learning in an already created environment.

# Import
import gymnasium as gym
import numpy as np
import random

"""
# Environment creation and initialization
env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset(seed=42)

# For loop for repeating actions
for _ in range(1000):
    
    action = env.action_space.sample() # where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Once finished or stuck, reset the environment
    # for a new test
    if terminated or truncated:
        observation, info = env.reset()

env.close() # closing environment
"""

# Ship configuration:

# Four actions: 
    # - do nothing, 
    # - slow down during descent (lower engine),
    # - right tilt (left engine),
    # - left tilt (right engine)
    
# Status = position in space

# Rewards:
    # + 100 pts: descent and landing without damage
    # - 100 pts: crashes to the ground
    # loss of points each time the engine is activated
    
# New version of the spacecraft:

# Creation of the environment and reinitialization
env = gym.make("LunarLander-v3", render_mode="human")

# Hyper-paramètres
alpha = 0.1             # learning rate
gamma = 0.99            # discount factor (immediate/long-term rewards)
epsilon = 1.0           # exploration rate (choice: exploration 1.0, long exploration vs exploitation 0.0, greedy)
epsilon_decay = 0.995   # speed at which the agent stops exploring (allows balancing between explore (0.999) and exploit (0.99))
epsilon_min = 0.1       # guaranteed minimum exploration
episodes = 1000         # number of attempts (5,000/20,000 for serious training)
max_steps = 1000        # number of steps

# alpha         learning rate
# gamma            long-term vision
# epsilon        exploration
# epsilon_decay    transition speed to exploitation
# epsilon_min    residual exploration
# episodes        amount of training
# max_steps     limit per episode

# Ship state segmentation (like the agent in a coordinate maze)
state_bins = [np.linspace(-1.0, 1.0, 10) for _ in range (env.observation_space.shape[0])]

# env.observation_space.shape[0] = number of state variables
# (e.g., x-position, x-velocity, angle, etc.)
# np.linspace(-1.0, 1.0, 10) = creation of 9 intervals separated by 10 points

# Number of possible discrete values per dimension
n_bins = tuple(len(bins) + 1 for bins in state_bins)

# tuple()+1 = n+1 intervals for n terminals. E.g.: 10 + 1 possible zones
# If State = 2d, i.e. (11x11) -> 121 possible zones

# Creation of the q_table with a given discrete state + an action
q_table = np.zeros(n_bins + (env.action_space.n,))

# Discreet state function
def discretize_state(state):
    return tuple(np.digitize(state[i], state_bins[i]) for i in range(len(state)))


# Action loop + integration of choice between exploration and exploitation
for episode in range (episodes) :
    state, _ = env.reset(seed=42)
    state = discretize_state(state)
    total_reward = 0
    
    # Loop for each steps
    for step in range(max_steps):
        
        # Epsilon - greedy policy
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample( ) # Exploration
        else:
            action = np.argmax(q_table[state]) # Exploitation
            
        # Variables in order to choose the next action based on them
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize_state(next_state)
        
        # Retrieving old values and adding new ones to the table
        old_value = q_table[state][action] # current Q value
        next_max = np.max(q_table[next_state]) # maximum Q value of state S

        # Update table q with Bellman's formula
        new_value = (1 - alpha)*old_value + alpha*(reward + gamma*next_max)
        q_table[state][action] = new_value
        
        # Refresh status and total number of rewards
        state = next_state
        total_reward += reward
        
        # If finished or stuck, stop the loop
        if terminated or truncated :
            break
    
        # Reduction of Epsilon
        if epsilon > epsilon_min :
            epsilon *= epsilon_decay
    
        print(f"Episode {episode + 1}, Total Reward : {total_reward }")

# Closure of the environment
env.close()
