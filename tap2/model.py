import torch
from torch import nn


class ElementExpert(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, dropout_rate: float):
        super(ElementExpert, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)


class ExpertRater(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size:int, dropout_rate: float):
        super(ExpertRater, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.layers(x)


class MoE2(nn.Module):
    def __init__(self, element_num: int, process_num:int, property_num: int, hidden_size: int, dropout_rate: float):
        super(MoE2, self).__init__()
        self.element_num = element_num
        self.process_num = process_num
        self.experts = nn.ModuleList(
            [ElementExpert(2 + process_num, property_num, hidden_size, dropout_rate) for _ in range(element_num - 1)]
        )
        self.rater = ExpertRater(element_num + process_num, element_num - 1, hidden_size, dropout_rate)

    def forward(self, x):
        score = self.rater(x)
        score = score.unsqueeze(-1)
        expert_outputs = [
            expert(x[:, [0, i + 1] + list(range(self.element_num, self.element_num + self.process_num))])
            for i, expert in enumerate(self.experts)
        ]
        expert_outputs = torch.stack(expert_outputs, dim=-1)
        output = expert_outputs @ score
        return output.squeeze(-1)


class AAModel(nn.Module):
    """
    Aluminum Alloy Performance Prediction Model V1: Simple Multilayer Perceptron
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        identity = x
        out = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(out + identity)
        identity = x
        out = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(out + identity)
        identity = x
        out = self.relu(self.bn4(self.fc4(x)))
        x = self.dropout(out + identity)
        x = self.fc5(x)
        return x
