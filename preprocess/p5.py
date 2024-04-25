# Code to convolve our audio with impulse response
import numpy as np
import librosa
import soundfile as sf

def augment_with_rir(echo_file, rir_file, output_file):
    # Load echo and RIR audio
    echo, _ = librosa.load(echo_file, sr=None)
    rir, _ = librosa.load(rir_file, sr=None)

    # Normalize audio to have the same energy
    echo /= np.max(np.abs(echo))
    rir /= np.max(np.abs(rir))

    # Convolve the echo with the RIR
    augmented_audio = np.convolve(echo, rir, mode='full')

    # Save the augmented audio using soundfile
    sf.write(output_file, augmented_audio, samplerate=44100)  # Adjust samplerate as needed

# Example usage
echo_file = './echo/echo1.wav'
rir_file = 'Xx00y00.wav'
output_file = 'augmented_echo.wav'
augment_with_rir(echo_file, rir_file, output_file)
