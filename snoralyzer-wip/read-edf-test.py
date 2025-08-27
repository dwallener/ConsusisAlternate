import pyedflib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Load the EDF file
edf_file = "S001R01.edf"  # Example filename
f = pyedflib.EdfReader(edf_file)

# Get channel labels and sampling frequencies
channel_labels = f.getSignalLabels()
num_channels = len(channel_labels)
sampling_rates = [f.getSampleFrequency(i) for i in range(num_channels)]

# Get total duration in seconds
total_duration = f.file_duration  # in seconds
window_size = 0.5 * 60  # 2 minutes in seconds
current_start = 0  # Start time in seconds

# Read all signals at once (to speed up scrolling)
signals = [f.readSignal(i) for i in range(num_channels)]
f.close()  # Close the file after reading

# Create a figure and axis
fig, ax = plt.subplots(num_channels, 1, figsize=(12, 8), sharex=True)
fig.subplots_adjust(bottom=0.1, top=0.95)

def plot_signals(start_time):
    """Plots a window of the EDF signals."""
    global current_start
    current_start = start_time
    ax[0].cla()  # Clear plots

    end_time = min(start_time + window_size, total_duration)
    
    for i in range(num_channels):
        fs = sampling_rates[i]
        start_idx = int(start_time * fs)
        end_idx = int(end_time * fs)
        time_axis = np.linspace(start_time, end_time, end_idx - start_idx)
        
        ax[i].cla()  # Clear each subplot
        ax[i].plot(time_axis, signals[i][start_idx:end_idx], label=channel_labels[i])
        ax[i].legend(loc="upper right")
        ax[i].set_ylabel(channel_labels[i])

    ax[-1].set_xlabel("Time (seconds)")
    fig.canvas.draw()

def on_key(event):
    """Handles key press events for scrolling."""
    global current_start
    if event.key == "right":
        if current_start + window_size < total_duration:
            plot_signals(current_start + window_size)
    elif event.key == "left":
        if current_start - window_size >= 0:
            plot_signals(current_start - window_size)

# Connect key events
fig.canvas.mpl_connect("key_press_event", on_key)

# Plot the first window
plot_signals(current_start)

plt.show()