import torch.nn as nn
import torch.nn.functional as F
from NdLinear import NdLinear

class NdDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(NdDQN, self).__init__()
        self.fc1 = NdLinear([state_size], [128])          # OK
        self.fc2 = NdLinear([128], [128])                 # FIXED
        self.fc3 = NdLinear([128], [action_size])         # FIXED

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
