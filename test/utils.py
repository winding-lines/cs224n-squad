import random
import numpy as np
import torch

def init_random(value):
    """Initialize the random values"""
    random.seed(41)
    np.random.seed(41)
    torch.manual_seed(41)