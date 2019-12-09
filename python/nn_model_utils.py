import torch
import torch.nn as nn


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class MultilayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_layer, output_size, activation_name='ReLU'):
        super().__init__()
        
        modules = []
        layer_size = [input_size, *hidden_layer, output_size]
        
        if activation_name == 'ReLU':
            activation = nn.ReLU()
        elif activation_name == 'Sigmoid':
            activation = nn.Sigmoid()
        else:
            activation = nn.ReLU()
        
        num_layers = len(layer_size)
        for i in range(num_layers - 1):
            modules.append(nn.Linear(layer_size[i], layer_size[i+1]))
            modules.append(activation) 
        modules.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*modules)     
        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(x)