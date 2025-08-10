# synth_utils.py
import numpy as np
import pretty_midi
import io
import wave
import struct

def tokens_to_midi(tokens, bpm=120, step_subdivision=4, program=0, out_file=None):
    # tokens: list of ints where 0 is rest, 1..128 map to MIDI pitches 0..127
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)
    step_seconds = 60.0 / bpm / step_subdivision
    t = 0.0
    for tok in tokens:
        if tok == 0:
            t += step_seconds
            continue
        pitch = tok - 1
        note = pretty_midi.Note(velocity=100, pitch=int(pitch), start=t, end=t + step_seconds)
        instrument.notes.append(note)
        t += step_seconds
    pm.instruments.append(instrument)
    if out_file:
        pm.write(out_file)
        return out_file
    else:
        # return bytes
        bio = io.BytesIO()
        pm.write(bio)
        bio.seek(0)
        return bio.read()

def tokens_to_wav_bytes(tokens, bpm=120, sr=22050, step_subdivision=4, instrument='sine'):
    # Sine synth for each step. Returns WAV bytes buffer.
    step_seconds = 60.0 / bpm / step_subdivision
    total_seconds = step_seconds * len(tokens)
    total_samples = int(sr * total_seconds)
    audio = np.zeros(total_samples, dtype=np.float32)

    def note_wave(freq, dur, sr):
        t = np.linspace(0, dur, int(sr*dur), endpoint=False)
        wave = np.sin(2 * np.pi * freq * t)
        # simple ADSR-ish envelope
        a = 0.02
        sustain = 0.6
        env = np.ones_like(wave) * sustain
        attack_samples = max(1, int(len(wave)*a))
        env[:attack_samples] = np.linspace(0, sustain, attack_samples)
        env[-attack_samples:] *= np.linspace(1.0, 0.01, attack_samples)
        return wave * env

    for i, tok in enumerate(tokens):
        start_s = i * step_seconds
        start_idx = int(start_s * sr)
        end_idx = start_idx + int(step_seconds * sr)
        if tok == 0:
            continue
        pitch = tok - 1
        freq = 440.0 * (2 ** ((pitch - 69) / 12.0))
        w = note_wave(freq, step_seconds, sr)
        # in case of rounding issues:
        if end_idx > len(audio):
            end_idx = len(audio)
            w = w[:end_idx - start_idx]
        audio[start_idx:end_idx] += w

    # normalize
    maxv = np.max(np.abs(audio)) + 1e-9
    audio = audio / maxv * 0.95

    # write to bytes using wave module
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        int_samples = (audio * 32767).astype('<h')
        wf.writeframes(int_samples.tobytes())
    buf.seek(0)
    return buf.read()
