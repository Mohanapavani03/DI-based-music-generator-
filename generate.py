# generate.py
import argparse
import torch
import numpy as np
from model import MelodyLSTM
import pickle
from synth_utils import tokens_to_wav_bytes, tokens_to_midi

# helper: simple scales
SCALES = {
    'major': [0,2,4,5,7,9,11],
    'minor': [0,2,3,5,7,8,10],
    'pentatonic': [0,2,4,7,9]
}

def rule_based(seed_root=60, scale_name='major', length=64, bpm=120):
    # returns tokens list (0 = rest, 1..128 = midi pitch+1)
    scale_intervals = SCALES.get(scale_name, SCALES['major'])
    tokens = []
    import random
    for i in range(length):
        # prefer scale notes, sometimes rest
        if random.random() < 0.12:
            tokens.append(0)
            continue
        octave = random.choice([0,1,2])
        degree = random.choice(scale_intervals)
        pitch = seed_root + degree + 12*octave
        pitch = max(0, min(127, pitch))
        tokens.append(int(pitch + 1))
    return tokens

def load_model(path, device='cpu'):
    vocab_size = 129
    model = MelodyLSTM(vocab_size=vocab_size)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model

def generate_with_model(model, seed_seq=None, gen_len=128, temperature=1.0, device='cpu', bias_scale=None, key_root=60, scale_name='major'):
    # seed_seq: list of tokens (0..128). If None, start random.
    import torch.nn.functional as F
    if seed_seq is None:
        # random seed
        seq = [0]*32
    else:
        seq = seed_seq.copy()
    for _ in range(gen_len):
        inp = torch.tensor([seq[-32:]], dtype=torch.long).to(device)
        logits, _ = model(inp)
        logits = logits.detach().squeeze(0).cpu().numpy()  # (vocab,)
        # bias by scale if requested
        if bias_scale:
            allowed = [(p+1) for p in scale_pitches(key_root, scale_name)]
            # add small boost to scale notes
            for idx in range(len(logits)):
                if idx in allowed:
                    logits[idx] += bias_scale
        # temperature sampling
        probs = np.exp(logits / temperature)
        probs = probs / np.sum(probs)
        choice = np.random.choice(len(probs), p=probs)
        seq.append(int(choice))
    return seq

def scale_pitches(root_midi, scale_name):
    ints = []
    base = root_midi % 12
    for octave in range(-1, 8):
        for step in SCALES.get(scale_name, SCALES['major']):
            p = base + step + 12 * (octave + (root_midi // 12))
            if 0 <= p <= 127:
                ints.append(p)
    return sorted(list(set(ints)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['rule', 'model'], default='rule')
    ap.add_argument('--model', default=None)
    ap.add_argument('--length', type=int, default=128)
    ap.add_argument('--bpm', type=int, default=120)
    ap.add_argument('--out_wav', default='out.wav')
    ap.add_argument('--out_midi', default='out.mid')
    args = ap.parse_args()

    if args.mode == 'rule':
        tokens = rule_based(seed_root=60, scale_name='major', length=args.length, bpm=args.bpm)
    else:
        if args.model is None:
            raise SystemExit("Please provide --model")
        model = load_model(args.model, device='cpu')
        tokens = generate_with_model(model, seed_seq=None, gen_len=args.length, temperature=1.0, device='cpu', bias_scale=1.5, key_root=60, scale_name='major')

    wav_bytes = tokens_to_wav_bytes(tokens, bpm=args.bpm)
    with open(args.out_wav, 'wb') as fh:
        fh.write(wav_bytes)
    tokens_to_midi(tokens, bpm=args.bpm, out_file=args.out_midi)
    print("WAV and MIDI saved.")
