import soundfile as sf
import numpy as np

def calculate_snr(clean_signal, noisy_signal):
    # Calculate signal power
    signal_power = np.mean(np.square(clean_signal))

    # Calculate noise power
    noise = noisy_signal - clean_signal
    noise_power = np.mean(np.square(noise))

    # Compute SNR in dB
    snr = 10 * np.log10(signal_power / noise_power)

    return snr

# Replace with the path to your clean and noisy wave files
clean_file = '../output0.wav'
noisy_file = '../outputrecon3.wav'

# Load wave files
clean_audio, clean_sr = sf.read(clean_file)
noisy_audio, noisy_sr = sf.read(noisy_file)

# Ensure both audio files have the same length
min_length = min(len(clean_audio), len(noisy_audio))
clean_audio = clean_audio[:min_length]
noisy_audio = noisy_audio[:min_length]

# Calculate SNR
snr_value = calculate_snr(clean_audio, noisy_audio)
print(f"SNR: {snr_value} dB")
