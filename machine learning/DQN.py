import gym
import torch.nn.functional as F
import numpy as np
import torch

# hyper parameters
BATCH_SIZE=32
LR=0.01
EPSILON=0.9
GAMMA=0.9
TARGET_REPLACE_ITER=100
MEMORY_CAPACITY=2000

