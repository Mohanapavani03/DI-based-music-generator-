# train.py
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from model import MelodyLSTM
from tqdm import tqdm
import os

class MelodyDataset(Dataset):
    def __init__(self, sequences, seq_len=32):
        self.inputs = []
        self.targets = []
        self.seq_len = seq_len
        for seq in sequences:
            # tokenization: map rest (-1) -> 0, pitch 0-127 -> pitch+1 (1..128)
            tokens = [(0 if int(x)==-1 else int(x)+1) for x in seq.tolist()]
            for i in range(len(tokens) - seq_len):
                self.inputs.append(np.array(tokens[i:i+seq_len], dtype=np.int64))
                self.targets.append(np.int64(tokens[i+seq_len]))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def train(data_file, save_path, batch=128, epochs=20, seq_len=32, lr=1e-3, device='cpu'):
    with open(data_file,'rb') as fh:
        sequences = pickle.load(fh)
    ds = MelodyDataset(sequences, seq_len=seq_len)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)
    vocab_size = 129  # 0..128 (0 is rest, 1..128 actual pitches)
    model = MelodyLSTM(vocab_size=vocab_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    for ep in range(epochs):
        running_loss = 0.0
        for xb, yb in tqdm(dl, desc=f"Epoch {ep+1}/{epochs}"):
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            running_loss += loss.item()
        print(f"Epoch {ep+1} loss: {running_loss / len(dl):.4f}")
        torch.save(model.state_dict(), save_path)
    print("Training finished. Saved:", save_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--save', required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--seq_len', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train(args.data, args.save, batch=args.batch, epochs=args.epochs, seq_len=args.seq_len, lr=args.lr, device=device)
