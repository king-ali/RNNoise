# Code to combine all files
import wave
import os


# Define the folder where the WAV files are located
folder_path = "./reverb/reverb"

# Specify the output folder and filename for the combined WAV file
output_folder = "output"
output_filename = "combined_reverb.wav"

# List all WAV files in the folder
wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]

if len(wav_files) == 0:
    print("No WAV files found in the folder.")
else:
    # Open the first WAV file to get parameters
    first_wav = wave.open(os.path.join(folder_path, wav_files[0]), 'rb')
    params = first_wav.getparams()

    # Create a new WAV file for combining in the output folder
    os.makedirs(output_folder, exist_ok=True)
    combined_wav_path = os.path.join(output_folder, output_filename)
    combined_wav = wave.open(combined_wav_path, 'wb')
    combined_wav.setparams(params)

    # Loop through all other WAV files and append their data to the combined file
    for wav_file in wav_files:
        with wave.open(os.path.join(folder_path, wav_file), 'rb') as w:
            combined_wav.writeframes(w.readframes(w.getnframes()))

    # Close the combined WAV file
    combined_wav.close()

    print(f"WAV files combined successfully. Combined file saved at: {combined_wav_path}")





