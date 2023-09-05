import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation_fn=nn.ReLU):
        super().__init__()
        module_list = []
        last_in_dim = input_dim
        for hidden_dim in hidden_dims:
            module_list.append(nn.Linear(last_in_dim, hidden_dim))
            module_list.append(activation_fn())
            last_in_dim = hidden_dim
        module_list.append(nn.Linear(last_in_dim, output_dim))
        self.module_list = nn.Sequential(*module_list)


    def forward(self, *args):
        return self.module_list(torch.hstack(args))