# Code to split into segments of 30 seconds
import wave
import os

# Specify the path to the combined WAV file
combined_wav_path = "output/combined_echonoise.wav"

# Define the folder where the segments will be saved
segment_folder = "echonoise_segment"

# Create the segment folder if it doesn't exist
os.makedirs(segment_folder, exist_ok=True)

# Open the combined WAV file
with wave.open(combined_wav_path, 'rb') as combined_wav:
    params = combined_wav.getparams()
    total_frames = combined_wav.getnframes()
    frame_rate = combined_wav.getframerate()
    frame_size = combined_wav.getsampwidth()

    # Calculate the number of frames for a 30-second segment
    frames_per_segment = int(frame_rate * 30)

    segment_index = 1
    start_frame = 0

    while start_frame + frames_per_segment <= total_frames:
        # Calculate the end frame for the current segment
        end_frame = start_frame + frames_per_segment

        # Read the data for the current segment
        combined_wav.setpos(start_frame)
        segment_data = combined_wav.readframes(frames_per_segment)

        # Create a new WAV file for the segment
        segment_filename = f"segment_{segment_index}.wav"
        segment_path = os.path.join(segment_folder, segment_filename)
        with wave.open(segment_path, 'wb') as segment_wav:
            segment_wav.setparams(params)
            segment_wav.writeframes(segment_data)

        print(f"Segment {segment_index} saved at: {segment_path}")

        # Update the start frame and segment index
        start_frame = end_frame
        segment_index += 1

print("Splitting completed.")
