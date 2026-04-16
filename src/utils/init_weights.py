import torch.nn as nn

def init_weights_kaiming_uniform(m, mode='fan_in', nonlinearity='relu'):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode=mode, nonlinearity=nonlinearity)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.kaiming_uniform_(m.weight, mode=mode, nonlinearity=nonlinearity)

def init_weights_xavier_uniform(m, nonlinearity='relu'):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(nonlinearity))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(nonlinearity))

def init_weights_xavier_normal(m, nonlinearity='relu'):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain(nonlinearity))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain(nonlinearity))

def init_weights_zeros(m):
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.zeros_(m.weight) 