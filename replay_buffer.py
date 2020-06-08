import torch
import numpy as np
from collections import deque
from itertools import islice
import random

class ReplayBuffer(object):
    def __init__(self, env, device, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.env = env
        self.s_dim = env.s_dim
        self.a_dim = env.a_dim
        self.device = device

    def add(self, *args):
        experience = args
        self.memory.append(experience)

    def clear(self):
        self.memory.clear()

    def sample_sequence(self):
        """Ordered sequence replay with batch size: Do not shuffle indices. For on-policy methods"""

        min_start = len(self.memory) - self.batch_size  # If batch_size = episode length
        if min_start == 0: min_start = 1
        start_idx = np.random.randint(0, min_start)

        batch = deque(islice(self.memory, start_idx, start_idx + self.batch_size))

        # Pytorch replay buffer - squeeze 2nd dim (B, 1, x) -> (B, x)
        s_batch = np.array([_[0] for _ in batch]);         s_batch = np.reshape(s_batch, [-1, self.s_dim])
        a_batch = np.array([_[1] for _ in batch]);          a_batch = np.reshape(a_batch, [-1, self.a_dim])
        r_batch = np.array([_[2] for _ in batch]);         r_batch = np.reshape(r_batch, [-1, 1])
        s2_batch = np.array([_[3] for _ in batch]);        s2_batch = np.reshape(s2_batch, [-1, self.s_dim])
        term_batch = np.array([_[4] for _ in batch]);        term_batch = np.reshape(term_batch, [-1, 1])


        return s_batch, a_batch, r_batch, s2_batch, term_batch



    def __len__(self):
        return len(self.memory)