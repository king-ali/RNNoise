# Combining all segments into one file
import wave
import os

# Define the folder where the segments are located
segment_folder = "echonoise_segment"

# List all WAV files in the segment folder
segment_files = [f for f in os.listdir(segment_folder) if f.endswith(".wav")]

if len(segment_files) == 0:
    print("No WAV files found in the segment folder.")
else:
    # Open the first segment WAV file to get parameters
    first_segment_path = os.path.join(segment_folder, segment_files[0])
    first_segment = wave.open(first_segment_path, 'rb')
    params = first_segment.getparams()

    # Create a new WAV file for combining
    combined_path = "combined_echonoise_segments.wav"
    combined_wav = wave.open(combined_path, 'wb')
    combined_wav.setparams(params)

    # Loop through all other segment WAV files and append their data to the combined file
    for segment_file in segment_files:
        segment_path = os.path.join(segment_folder, segment_file)
        with wave.open(segment_path, 'rb') as segment_wav:
            combined_wav.writeframes(segment_wav.readframes(segment_wav.getnframes()))

    # Close the combined WAV file
    combined_wav.close()

    print(f"All segment WAV files combined into {combined_path}")
