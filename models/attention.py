import torch
import torch.nn as nn


class global_attention(nn.Module):
    def __init__(self, hidden_size):
        super(global_attention, self).__init__()
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(2*hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, x, context):
        # x (1, batch, hidden_size) context (batch, seq_len_src, hidden_size)
        x = x.squeeze(0)
        gamma_x = self.linear_in(x).unsqueeze(2) # x (batch, hidden_size, 1)
        weights = torch.bmm(context, gamma_x).squeeze(2)    # (batch, seq_len_src)
        weights = self.softmax(weights)
        c_t = torch.bmm(weights.unsqueeze(1), context).squeeze(1)   # (batch, hidden_size)
        output = self.tanh(self.linear_out(torch.cat([c_t, x], 1))) # (batch, hidden_size)
        return output, weights


class dot(nn.Module):
    def __init__(self, hidden_size):
        super(dot, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.linear_out = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, x, context):
        # x (1, batch, hidden_size) context (batch, seq_len_src, hidden_size)
        x = x.squeeze(0)
        x_t = x.unsqueeze(2)
        # x (batch, hidden_size, 1)
        weights = torch.bmm(context, x_t).squeeze(2)  # (batch, seq_len_src)
        weights = self.softmax(weights)
        c_t = torch.bmm(weights.unsqueeze(1), context).squeeze(1)  # (batch, hidden_size)
        output = self.tanh(self.linear_out(torch.cat([c_t, x], 1)))  # (batch, hidden_size)
        return output, weights


class concat(nn.Module):
    def __init__(self, hidden_size):
        super(concat, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.linear_x = nn.Linear(hidden_size, hidden_size)
        self.linear_con = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, 1)
        self.linear_out = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, x, context):
        x = x.squeeze(0)
        gamma_x = self.linear_x(x).unsqueeze(1)
        gamma_con = self.linear_con(context)
        weights = self.linear_v(self.tanh(gamma_x + gamma_con)).squeeze(2) # (batch, seq_len_src)
        weights = self.softmax(weights)
        c_t = torch.bmm(weights.unsqueeze(1), context).squeeze(1)   # (batch, hidden_size)
        output = self.tanh(self.linear_out(torch.cat([c_t, x], 1)))  # (batch, hidden_size)
        return output, weights

