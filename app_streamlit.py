# app_streamlit_simple.py
import streamlit as st
from generate import rule_based
from synth_utils import tokens_to_wav_bytes, tokens_to_midi
import os

st.set_page_config(page_title="Simple Music Generator", layout="centered")
st.title("🎼 Simple AI Music Generator")

# Sidebar Inputs
mood = st.sidebar.selectbox("Mood", ["Happy", "Calm", "Sad", "Angry"])
bpm = st.sidebar.slider("Tempo (BPM)", 60, 150, 120)
length = st.sidebar.slider("Length (steps)", 16, 512, 128)
key_root = st.sidebar.selectbox("Key root (C=60)", list(range(48, 72)), index=12)

# Mood to scale map
scale_map = {"Happy": "major", "Calm": "pentatonic", "Sad": "minor", "Angry": "minor"}
scale_name = scale_map.get(mood, "major")

if st.button("Generate Music"):
    with st.spinner("Generating..."):
        tokens = rule_based(seed_root=key_root, scale_name=scale_name, length=length, bpm=bpm)

        wav_bytes = tokens_to_wav_bytes(tokens, bpm=bpm)
        midi_bytes = tokens_to_midi(tokens, bpm=bpm)

        # Save to local files
        with open("output.wav", "wb") as f:
            f.write(wav_bytes)
        with open("output.mid", "wb") as f:
            f.write(midi_bytes)

        st.success("Done! Files saved as output.wav and output.mid")

        # Audio preview and download buttons
        st.audio(wav_bytes, format='audio/wav')
        st.download_button("Download WAV", wav_bytes, file_name="output.wav", mime="audio/wav")
        st.download_button("Download MIDI", midi_bytes, file_name="output.mid", mime="audio/midi")
