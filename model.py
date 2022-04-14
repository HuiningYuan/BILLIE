import torch
import torch.nn as nn
import torch.nn.functional as F


class BILLIE(nn.Module):
    def __init__(self, n_actions, num_of_components=1, embedding_size=128, hidden_size=128, device='cpu'):
        super(BILLIE, self).__init__()
        self.n_actions = n_actions
        self.num_of_components = num_of_components
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(n_actions + num_of_components, embedding_size)
        self.rnn = nn.LSTMCell(embedding_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, n_actions)
        self.to(device)
        self.embedding.to(device)
        self.rnn.to(device)
        self.decoder.to(device)

    def forward(self, input, h_t, c_t):
        input = self.embedding(input)
        h_t, c_t = self.rnn(input, (h_t, c_t))
        outputs = self.decoder(h_t)
        probs = F.softmax(outputs, dim=-1)
        log_probs = F.log_softmax(outputs, dim=-1)
        return h_t, c_t, probs, log_probs

    def autoregression(self, max_steps=20, component=1, cuda=True):
        # Build inputs
        buffer_log_probs = []
        buffer_entropy = []
        h_t = torch.zeros(1, self.hidden_size, dtype=torch.float)
        c_t = torch.zeros(1, self.hidden_size, dtype=torch.float)
        if cuda:
            h_t = h_t.cuda()
            c_t = c_t.cuda()
        next_symbol = self.n_actions + component
        flag = 0
        buffer_candidates = torch.zeros(0, dtype=torch.int64)
        if cuda:
            buffer_candidates = buffer_candidates.cuda()
        for j in range(max_steps):
            input = torch.tensor([next_symbol], dtype=torch.int64)
            if cuda:
                input = input.cuda()
            h_t, c_t, probs, log_probs = self.forward(input=input, h_t=h_t, c_t=c_t)
            next_symbol = torch.distributions.Categorical(probs[-1]).sample()
            buffer_candidates = torch.cat([buffer_candidates, input], -1)
            buffer_log_probs.append(log_probs[-1, next_symbol])
            buffer_entropy.append(torch.mul(probs, log_probs))
            for i in range(len(buffer_candidates)):
                if next_symbol == buffer_candidates[i]:
                    flag = 1
                    break
            if flag == 1:
                break
        return buffer_log_probs, buffer_entropy, buffer_candidates

