""" Graph-recurrent forecaster: at every hour each station mixes its own and its neighbours'
    features (graph convolution with self-loops), and a GRU carries that mix through time.
"""

import torch
import torch.nn as nn

def propagate(adj, x):
    """ Mix node features [n, ..., f] over the graph, for every hour at once.
    """
    return torch.sparse.mm(adj, x.reshape(x.shape[0], -1)).reshape(x.shape)

class GNNForecaster(nn.Module):
    """ Stacked graph-convolution + GRU layers. The encoder runs layer by layer over all 72
        hours at once (the graph step does not depend on the hidden state); the decoder is
        autoregressive (step by step) or direct (all hours in one pass).
    """

    def __init__(self, enc_dim, dec_dim, hidden, layers, dropout, autoregressive):
        super().__init__()
        self.autoregressive = autoregressive
        self.drop = nn.Dropout(dropout)
        dec_in = dec_dim + autoregressive  # + previous hour's PM2.5
        self.encoder = nn.ModuleDict({
            "lin": nn.ModuleList(nn.Linear(enc_dim if i == 0 else hidden, hidden)
                                 for i in range(layers)),
            "gru": nn.ModuleList(nn.GRU(hidden, hidden, batch_first=True) for _ in range(layers)),
        })
        self.dec_lin = nn.ModuleList(nn.Linear(dec_in if i == 0 else hidden, hidden)
                                     for i in range(layers))
        cell = nn.GRUCell if autoregressive else (lambda i, h: nn.GRU(i, h, batch_first=True))
        self.dec_rnn = nn.ModuleList(cell(hidden, hidden) for _ in range(layers))
        self.head = nn.Linear(hidden, 1)

    def encode(self, x_enc, adj, **_):
        """ Encode every node's lookback; return per-layer final states [layers, n, hidden], the
            (prebuilt) normalized adjacency, and the representation for MMD (top layer's final state).
        """
        x, states = x_enc, []
        layers = len(self.encoder["gru"])
        for i, (lin, gru) in enumerate(zip(self.encoder["lin"], self.encoder["gru"])):
            x, h = gru(propagate(adj, lin(x)))
            if i < layers - 1:
                x = self.drop(x)
            states.append(h[0])
        return torch.stack(states), adj, states[-1]

    def forward(self, x_enc, x_dec, adj, teacher=None, y=None, **_):
        """ Forecast [n,72] (normalized) for every node. `teacher` [n,72] marks the steps fed the
            true previous hour (autoregressive training only). Returns (prediction, representation).
        """
        states, adj, rep = self.encode(x_enc, adj)
        if not self.autoregressive:
            x = x_dec
            for i, (lin, gru) in enumerate(zip(self.dec_lin, self.dec_rnn)):
                x, _ = gru(propagate(adj, lin(x)), states[i:i + 1].contiguous())
                if i < len(self.dec_rnn) - 1:
                    x = self.drop(x)
            return self.head(x).squeeze(-1), rep

        h = list(states)
        prev = x_enc[:, -1, :1]  # last observed PM2.5 (encoder column 0)
        preds = []
        for t in range(x_dec.shape[1]):
            x = torch.cat([x_dec[:, t], prev], -1)
            for i, (lin, cell) in enumerate(zip(self.dec_lin, self.dec_rnn)):
                h[i] = cell(propagate(adj, lin(x)), h[i])
                x = self.drop(h[i]) if i < len(self.dec_rnn) - 1 else h[i]
            pred = self.head(x)
            preds.append(pred)
            prev = pred.detach()
            if teacher is not None:
                prev = torch.where(teacher[:, t:t + 1], y[:, t:t + 1], prev)
        return torch.cat(preds, 1), rep
