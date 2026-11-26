""" Encoder-decoder LSTM: one network shared by all stations, each window an independent sample.
"""

import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    """ Encode the 72h lookback and decode the 72h horizon, either autoregressively (each
        prediction feeds the next step) or directly (all hours from met/time in one pass).
    """

    def __init__(self, enc_dim, dec_dim, hidden, layers, dropout, autoregressive):
        super().__init__()
        self.autoregressive = autoregressive
        drop = dropout if layers > 1 else 0.0
        self.encoder = nn.LSTM(enc_dim, hidden, layers, batch_first=True, dropout=drop)
        # The autoregressive decoder gets one extra input: the previous hour's PM2.5.
        self.decoder = nn.LSTM(dec_dim + autoregressive, hidden, layers, batch_first=True,
                               dropout=drop)
        self.head = nn.Linear(hidden, 1)
        for name, p in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def encode(self, x_enc, **_):
        """ Encode a lookback; return the encoder state and its representation for MMD
            (the top layer's final hidden state, one row per window).
        """
        _, state = self.encoder(x_enc)
        return state, state[0][-1]

    def forward(self, x_enc, x_dec, teacher=None, y=None, **_):
        """ Forecast [n,72] (normalized). `teacher` [n,72] marks the steps fed the true
            previous hour instead of the prediction (autoregressive training only).
            Returns (prediction, encoder representation).
        """
        state, rep = self.encode(x_enc)
        if not self.autoregressive:
            out, _ = self.decoder(x_dec, state)
            return self.head(out).squeeze(-1), rep

        prev = x_enc[:, -1, :1]  # last observed PM2.5 (encoder column 0)
        preds = []
        for t in range(x_dec.shape[1]):
            out, state = self.decoder(torch.cat([x_dec[:, t], prev], -1).unsqueeze(1), state)
            pred = self.head(out[:, 0])
            preds.append(pred)
            prev = pred.detach()
            if teacher is not None:
                prev = torch.where(teacher[:, t:t + 1], y[:, t:t + 1], prev)
        return torch.cat(preds, 1), rep
