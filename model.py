# model.py
import torch
import torch.nn as nn

class MelodyLSTM(nn.Module):
    def __init__(self, vocab_size=129, emb_size=256, hidden_size=512, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)   # vocab includes rest token
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        # x: (B, T)
        emb = self.embedding(x)             # (B, T, E)
        out, hidden = self.lstm(emb, hidden)  # out: (B, T, H)
        logits = self.fc(out[:, -1, :])     # predict next token from last output
        return logits, hidden
