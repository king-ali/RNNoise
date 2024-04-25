import os
from pydub import AudioSegment

# Set the path to the folder containing the WAV files
folder_path = "echo"

# Create an empty audio segment to store the concatenated audio
concatenated_audio = AudioSegment.empty()

# Loop through the files in the folder, load each one, and concatenate it
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        audio = AudioSegment.from_file(os.path.join(folder_path, filename))
        concatenated_audio += audio

# Export the concatenated audio to a new file
concatenated_audio.export("concatenated_echo.wav", format="wav")
