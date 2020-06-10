import torch.nn as nn
import torch.nn.functional as F

class classifier(nn.Module):
    def __init__(self, num_layer):
        super(classifier, self).__init__()

        num_layer_to_input_size_dict = {
            1: 32*112*112,
            2: 16*112*112,
            3: 24*56*56,
            4: 24*56*56,
            5: 32*28*28,
            6: 32*28*28,
            7: 32*28*28,
            8: 64*14*14,
            9: 64*14*14,
            10: 64*14*14,
            11: 64*14*14,
            12: 96*14*14,
            13: 96*14*14,
            14: 96*14*14,
            15: 160*7*7,
            16: 160*7*7,
            17: 160*7*7,
            18: 320*7*7,
        }

        self.input_size = num_layer_to_input_size_dict[num_layer]
        self.fc1 = nn.Linear(self.input_size, self.input_size)
        self.fc2 = nn.Linear(self.input_size, self.input_size)
        self.fc3 = nn.Linear(self.input_size, 2)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features