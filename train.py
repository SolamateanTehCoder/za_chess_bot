import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import h5py
import os
from collections import deque
from model import ChessNet
from chess_env import ChessEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Using device: {device}")

EPISODES = 5
BATCH_SIZE = 64 # Increased batch to saturate CUDA
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.95
LR = 1e-4
MEMORY_SIZE = 100000 # 100K Steps directly on SSD

class H5ReplayBuffer:
    def __init__(self, filename, capacity=100000):
        self.filename = filename
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        
        # Create or overwrite the HDF5 memory file on SSD
        if os.path.exists(self.filename):
            os.remove(self.filename)
            
        with h5py.File(self.filename, 'w') as f:
            f.create_dataset('states', (capacity, 14, 8, 8), dtype='f4')
            f.create_dataset('actions', (capacity,), dtype='i8')
            f.create_dataset('rewards', (capacity,), dtype='f4')
            f.create_dataset('next_states', (capacity, 14, 8, 8), dtype='f4')
            f.create_dataset('dones', (capacity,), dtype='f4')
            
    def push(self, state, action, reward, next_state, done):
        with h5py.File(self.filename, 'a') as f:
            f['states'][self.ptr] = state
            f['actions'][self.ptr] = action
            f['rewards'][self.ptr] = reward
            f['next_states'][self.ptr] = next_state
            f['dones'][self.ptr] = float(done)
            
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        idxs = np.random.choice(self.size, size=batch_size, replace=False)
        # Sort indices as required by h5py reading natively
        sorted_idxs = np.sort(idxs)
        
        with h5py.File(self.filename, 'r') as f:
            states = f['states'][sorted_idxs]
            actions = f['actions'][sorted_idxs]
            rewards = f['rewards'][sorted_idxs]
            next_states = f['next_states'][sorted_idxs]
            dones = f['dones'][sorted_idxs]
            
        # Shuffle back to remove sorting bias
        shuffle_idxs = np.random.permutation(batch_size)
        return (states[shuffle_idxs], actions[shuffle_idxs], 
                rewards[shuffle_idxs], next_states[shuffle_idxs], dones[shuffle_idxs])
    
    def __len__(self):
        return self.size

def move_to_idx(move):
    return move.from_square * 64 + move.to_square

def train_model(model, optimizer, memory):
    if len(memory) < BATCH_SIZE:
        return 0.0
        
    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
    
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
    
    q_values = model(states).gather(1, actions)
    
    with torch.no_grad():
        next_q_values = model(next_states).max(1)[0].unsqueeze(1)
        
    expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))
    
    loss = F.mse_loss(q_values, expected_q_values)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def main():
    env = ChessEnv()
    
    # Init Deeper Networks
    white_model = ChessNet(num_blocks=10, num_filters=128).to(device)
    black_model = ChessNet(num_blocks=10, num_filters=128).to(device)
    
    white_optimizer = optim.Adam(white_model.parameters(), lr=LR)
    black_optimizer = optim.Adam(black_model.parameters(), lr=LR)
    
    # SSD-Backed Replay Buffers
    print("Pre-allocating SSD SSD-Backed Buffers. This will act directly on the disk...")
    white_memory = H5ReplayBuffer(os.path.join(os.path.dirname(__file__), "white_buffer.h5"), MEMORY_SIZE)
    black_memory = H5ReplayBuffer(os.path.join(os.path.dirname(__file__), "black_buffer.h5"), MEMORY_SIZE)
    
    epsilon = EPSILON_START
    
    print("Starting Training Loop...")
    for episode in range(1, EPISODES + 1):
        state = env.reset()
        done = False
        steps = 0
        total_loss_w = 0
        total_loss_b = 0
        total_reward_w = 0
        total_reward_b = 0
        
        while not done and steps < 150: 
            is_white = env.board.turn
            model = white_model if is_white else black_model
            memory = white_memory if is_white else black_memory
            optimizer = white_optimizer if is_white else black_optimizer
            
            legal_moves = list(env.board.legal_moves)
            if not legal_moves:
                break
                
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            valid_moves_dict = {move_to_idx(m): m for m in legal_moves}
            
            if random.random() < epsilon:
                chosen_move = random.choice(legal_moves)
                action_idx = move_to_idx(chosen_move)
            else:
                with torch.no_grad():
                    q_vals = model(state_tensor).squeeze(0)
                    valid_actions = list(valid_moves_dict.keys())
                    mask = torch.ones(4096, dtype=torch.bool).to(device)
                    mask[valid_actions] = False
                    q_vals[mask] = -float('inf')
                    
                    action_idx = q_vals.argmax().item()
                    chosen_move = valid_moves_dict[action_idx]
            
            next_state, reward, done, info = env.step(chosen_move.uci())
            
            # This directly pushes to SSD
            memory.push(state, action_idx, reward, next_state, done)
            
            loss = train_model(model, optimizer, memory)
            if is_white:
                total_loss_w += loss
                total_reward_w += reward
            else:
                total_loss_b += loss
                total_reward_b += reward

            state = next_state
            steps += 1
            
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        print(f"Ep {episode}/{EPISODES} | Steps: {steps} | End: {env.board.result()} | R_W: {total_reward_w:.2f} R_B: {total_reward_b:.2f} | Eps: {epsilon:.2f}")
        
    print("Training Complete! Models are ready.")
    torch.save(white_model.state_dict(), "white_bot.pth")
    torch.save(black_model.state_dict(), "black_bot.pth")

if __name__ == "__main__":
    main()
