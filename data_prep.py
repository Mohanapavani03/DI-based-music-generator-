# data_prep.py
import os
import argparse
import pretty_midi
import numpy as np
from tqdm import tqdm
import pickle

def midi_to_sequence(path, step_seconds=0.125):
    pm = pretty_midi.PrettyMIDI(path)
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        notes.extend(inst.notes)
    if len(notes) == 0:
        return None
    # sort notes by start time
    notes.sort(key=lambda n: n.start)
    end = max(n.end for n in notes)
    steps = int(np.ceil(end / step_seconds))
    seq = []
    # We'll represent rest as -1, pitches as MIDI pitch 0-127
    for i in range(steps):
        t = i * step_seconds
        active = [n.pitch for n in notes if n.start <= t < n.end]
        if active:
            # choose the highest pitch (simple monophonic extraction)
            seq.append(int(max(active)))
        else:
            seq.append(-1)
    return seq

def build_dataset(input_dir, out_file, step=0.125, min_len=32):
    sequences = []
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.mid') or f.lower().endswith('.midi')]
    print(f"Found {len(files)} midi files.")
    for f in tqdm(files):
        try:
            seq = midi_to_sequence(f, step_seconds=step)
        except Exception as e:
            print("Skipping", f, "error:", e)
            continue
        if seq is None:
            continue
        if len(seq) >= min_len + 1:
            sequences.append(np.array(seq, dtype=np.int16))
    print(f"Prepared {len(sequences)} sequences.")
    # Save as a numpy object
    with open(out_file, 'wb') as fh:
        pickle.dump(sequences, fh)
    print("Saved to", out_file)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--step', type=float, default=0.125)
    args = ap.parse_args()
    build_dataset(args.input_dir, args.out, step=args.step)
