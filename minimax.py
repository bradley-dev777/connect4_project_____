import pytorch
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# DQN Parameters
num_episodes = 1000
max_steps_per_episode = 100
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
memory = deque(maxlen=2000)

# Define the DQN model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize DQN
state_dim = 2  # (x, y) coordinates
action_dim = 4  # up, down, left, right
policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Helper functions
def get_new_position(position, action):
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    new_position = (position[0] + actions[action][0],
                    position[1] + actions[action][1])
    if 0 <= new_position[0] < grid_size and 0 <= new_position[1] < grid_size:
        return new_position
    return position  # Stay in place if out of bounds

def get_reward(position):
    if position == goal_position:
        return 100
    elif position in obstacles:
        return -100
    return -1  # Small penalty per step to encourage faster solutions

def draw_grid():
    for x in range(grid_size):
        for y in range(grid_size):
            rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, colors["grid"], rect, 1)
            if (x, y) == goal_position:
                pygame.draw.rect(screen, colors["goal"], rect)
            elif (x, y) in obstacles:
                pygame.draw.rect(screen, colors["obstacle"], rect)

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(4))  # Explore action space
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = policy_net(state_tensor)
            return torch.argmax(q_values).item()  # Exploit learned values

def optimize_model():
    if len(memory) < batch_size:
        return
    
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states_tensor = torch.FloatTensor(states)
    actions_tensor = torch.LongTensor(actions)
    rewards_tensor = torch.FloatTensor(rewards)
    next_states_tensor = torch.FloatTensor(next_states)
    dones_tensor = torch.FloatTensor(dones)

    current_q_values = policy_net(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze()
    max_next_q_values = target_net(next_states_tensor).max(1)[0]
    target_q_values = rewards_tensor + gamma * max_next_q_values * (1 - dones_tensor)

    loss = loss_fn(current_q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Training loop with event handling
for episode in range(num_episodes):
    state = (0, 0)
    for step in range(max_steps_per_episode):
        # Check for pygame events during training
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                exit()  # Ensure the program fully exits if quit is requested