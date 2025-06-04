import streamlit as st
import scipy.io.wavfile as wav
import numpy as np
import noisereduce as nr
import io

def apply_compression(audio, threshold=0.2, ratio=4, gain=1.5):
    compressed = np.copy(audio)
    over_threshold = np.abs(compressed) > threshold
    compressed[over_threshold] = np.sign(compressed[over_threshold]) * (
        threshold + (np.abs(compressed[over_threshold]) - threshold) / ratio
    )
    return compressed * gain

def apply_reverb(audio, decay=0.4, delay_samples=8000):
    reverb = np.copy(audio)
    for i in range(delay_samples, len(audio)):
        reverb[i] += decay * reverb[i - delay_samples]
    return reverb

def apply_delay(audio, delay_samples=10000, feedback=0.5, mix=0.3):
    delayed = np.copy(audio)
    for i in range(delay_samples, len(audio)):
        delayed[i] += feedback * delayed[i - delay_samples]
    return (1 - mix) * audio + mix * delayed

def process_audio(sr, audio):
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32) / np.max(np.abs(audio))

    audio_denoised = nr.reduce_noise(y=audio, sr=sr)
    audio_compressed = apply_compression(audio_denoised)
    audio_reverb = apply_reverb(audio_compressed)
    audio_final = apply_delay(audio_reverb)

    audio_final /= np.max(np.abs(audio_final))
    audio_final = (audio_final * 32767).astype(np.int16)

    return sr, audio_final

st.title("🎙️ Procesador Vocal Estilo Suno")
st.markdown("Sube tu voz cruda y obtén una versión con calidad profesional.")

uploaded_file = st.file_uploader("Sube tu archivo de voz (.wav)", type=["wav"])
if uploaded_file:
    sr, audio = wav.read(uploaded_file)
    sr_out, processed_audio = process_audio(sr, audio)

    st.audio(processed_audio, format="audio/wav", sample_rate=sr_out)

    output = io.BytesIO()
    wav.write(output, sr_out, processed_audio)
    st.download_button(
        label="Descargar voz procesada",
        data=output.getvalue(),
        file_name="voz_procesada.wav",
        mime="audio/wav"
    )
