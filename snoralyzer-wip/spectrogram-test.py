import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import matplotlib.animation as animation
import time

# -----------------------------
# 1. Load the WAV file
# -----------------------------
filename = 'tracheal.wav'  # Replace with your WAV file path
other_filename = 'extracted_tracheal.wave'
audio, sr = librosa.load(filename, sr=None)  # Load with original sample rate
duration = librosa.get_duration(y=audio, sr=sr)
print(f"Audio duration: {duration:.2f} seconds, Sample Rate: {sr}")

# -----------------------------
# 2. Compute the 80-bin Mel Spectrogram
# -----------------------------
n_fft = 400         # FFT window size
hop_length = 160    # Number of samples between successive frames
n_mels = 80         # Number of mel bins

# Compute mel spectrogram (power) and convert to dB scale
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr,
                                          n_fft=n_fft,
                                          hop_length=hop_length,
                                          n_mels=n_mels)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# -----------------------------
# 3. Set up the Visualization
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 4))
img = librosa.display.specshow(mel_spec_db,
                               x_axis='time',
                               y_axis='mel',
                               sr=sr,
                               hop_length=hop_length,
                               fmax=sr/2,
                               ax=ax)
fig.colorbar(img, ax=ax, format="%+2.0f dB")
ax.set_title("80-bin Mel Spectrogram (5 sec window)")

# Add a vertical red line to indicate playback position
play_line = ax.axvline(x=0, color='r', linewidth=2)

# Initially, set the x-axis to display only 5 seconds (or the full duration if shorter)
window_size = 5.0  # seconds
ax.set_xlim(0, min(window_size, duration))

# -----------------------------
# 4. Play the Audio and Animate the Spectrogram
# -----------------------------
sd.play(audio, sr)
start_time = time.time()

def update(frame):
    elapsed = time.time() - start_time
    play_line.set_xdata(elapsed)
    
    # Update x-axis limits to display a 5-second window centered around the current time.
    half_window = window_size / 2
    if elapsed < half_window:
        # If at the beginning, show from 0 to window_size.
        ax.set_xlim(0, min(window_size, duration))
    elif elapsed > duration - half_window:
        # Near the end, show the last window_size seconds.
        ax.set_xlim(max(0, duration - window_size), duration)
    else:
        # Otherwise, center the window around the elapsed time.
        ax.set_xlim(elapsed - half_window, elapsed + half_window)
    
    return play_line,

ani = animation.FuncAnimation(fig, update, interval=50, blit=True)
plt.show()