from pydub import AudioSegment

# List your input WAV files in the order you want them concatenated.
filenames = ["tracheal-01.wav", "tracheal-02.wav", "tracheal-03.wav", "tracheal-04.wav", "tracheal-05.wav"]

# Initialize an empty audio segment.
combined = AudioSegment.empty()

# Loop over the file names and concatenate them.
for filename in filenames:
    audio = AudioSegment.from_wav(filename)
    combined += audio

# Export the combined audio to a new file.
combined.export("tracheal.wav", format="wav")
print("Combined file saved as 'tracheal.wav'")
