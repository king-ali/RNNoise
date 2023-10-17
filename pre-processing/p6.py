from pydub import AudioSegment

# Load your two WAV files
audio1 = AudioSegment.from_file("output/combined_echo.wav")
audio2 = AudioSegment.from_file("output/combined_noise.wav")

audio2 = audio2 - 30

concatenated_audio = AudioSegment.empty()
concatenated_audio = audio1
concatenated_audio += audio2
concatenated_audio.export("concatenated.wav", format="wav")
# Optionally adjust the volume of audio1
# audio1 = audio1 + 5  # Increase by 5 dB

# Mix the two audio files
# mixed_audio = audio1.overlay(audio2)

# Export the mixed audio to a new WAV file
# mixed_audio.export("mixed_echo_noise.wav", format="wav")
