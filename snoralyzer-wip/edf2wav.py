import pyedflib
import numpy as np
import soundfile as sf

# Specify the EDF file path.
edf_path = 'datasets/00000995-100507-005.edf'

# Open the EDF file.
edf_reader = pyedflib.EdfReader(edf_path)

# Get the list of channel labels.
labels = edf_reader.getSignalLabels()
print("Signal labels found in EDF:", labels)

# Find the indices for 'Mic' and 'Tracheal'
try:
    mic_index = labels.index("Mic")
    tracheal_index = labels.index("Tracheal")
except ValueError as e:
    print("Could not find one or both required channels ('Mic' and 'Tracheal') in the EDF file.")
    edf_reader.close()
    raise e

# Read the signals.
mic_data = edf_reader.readSignal(mic_index)
tracheal_data = edf_reader.readSignal(tracheal_index)

# Get the sampling frequencies for each channel.
mic_fs = edf_reader.getSampleFrequency(mic_index)
tracheal_fs = edf_reader.getSampleFrequency(tracheal_index)

print(f"Mic channel: sampling frequency = {mic_fs} Hz")
print(f"Tracheal channel: sampling frequency = {tracheal_fs} Hz")

edf_reader.close()

# Optionally, ensure data is in floating point format.
mic_data = np.array(mic_data, dtype=np.float32)
tracheal_data = np.array(tracheal_data, dtype=np.float32)

# Save the channels as WAV files using SoundFile.
sf.write('datasets/mic-05.wav', mic_data, int(mic_fs))
sf.write('datasets/tracheal-05.wav', tracheal_data, int(tracheal_fs))

print("WAV files saved: 'mic.wav' and 'tracheal.wav'")

