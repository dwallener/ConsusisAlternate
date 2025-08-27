import time
import numpy as np
import math
import pyaudio
import torch
import torchaudio
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

inference_history = []

# -----------------------------
# Helper: Resample in Chunks
# -----------------------------
def resample_waveform_in_chunks(waveform, orig_freq, new_freq, chunk_size=1000000):
    """
    Resamples a waveform (Tensor of shape [channels, samples]) in chunks.
    Args:
      waveform: Tensor of shape (channels, samples)
      orig_freq: Original sample rate
      new_freq: Target sample rate
      chunk_size: Number of samples per chunk
    Returns:
      A Tensor with the resampled waveform.
    """

    import torchaudio.functional as F
    num_samples = waveform.size(1)
    out_chunks = []
    for start in range(0, num_samples, chunk_size):
        print(f"Resampling to 24khz...")
        end = min(num_samples, start + chunk_size)
        chunk = waveform[:, start:end]
        chunk_resampled = F.resample(chunk, orig_freq=orig_freq, new_freq=new_freq)
        out_chunks.append(chunk_resampled)
    return torch.cat(out_chunks, dim=1)

# -----------------------------
# Configuration
# -----------------------------
# Set SOURCE to either "external" (microphone) or "internal" (WAV file)
SOURCE = "internal"  # change to "external" for live mic input
INTERNAL_WAV_PATH = "datasets/internal_test.wav"

# -----------------------------
# Model Definition
# -----------------------------
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, model_dim=256, num_heads=4, num_layers=3, dropout=0.1):
        """
        A transformer-based classifier for audio segments.
        Args:
          input_dim (int): Number of mel bins (e.g., 40)
          model_dim (int): Hidden dimension of the transformer.
          num_heads (int): Number of attention heads.
          num_layers (int): Number of transformer encoder layers.
          dropout (float): Dropout rate.
        """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))
        self.pos_embedding = None  # Will be created dynamically
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(model_dim, 2)  # Binary classification

    def forward(self, x):
        """
        x: Tensor of shape (batch, T, input_dim)
        Returns logits of shape (batch, 2)
        """
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)  # (batch, T, model_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, model_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, T+1, model_dim)
        if self.pos_embedding is None or self.pos_embedding.shape[1] != x.shape[1]:
            self.pos_embedding = nn.Parameter(
                torch.randn(1, x.shape[1], x.shape[2], device=x.device),
                requires_grad=True
            )
        x = x + self.pos_embedding
        x = x.transpose(0, 1)  # (T+1, batch, model_dim)
        encoded = self.transformer_encoder(x)
        encoded = encoded.transpose(0, 1)  # (batch, T+1, model_dim)
        cls_encoded = encoded[:, 0, :]  # Use the CLS token output
        logits = self.classifier(cls_encoded)
        return logits

# -----------------------------
# Model Loading Utility
# -----------------------------
def load_model(checkpoint_path="trained_model.pth", device="cuda:0"):
    input_dim = 40  # Using 40 mel bins
    model = TransformerClassifier(input_dim=input_dim, model_dim=256, num_heads=4, num_layers=3, dropout=0.1)
    state_dict = torch.load(checkpoint_path, map_location=device)
    if "pos_embedding" in state_dict:
        del state_dict["pos_embedding"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

# -----------------------------
# Global Settings and Transforms
# -----------------------------
SAMPLE_RATE = 24000    # 24 kHz
N_MELS = 40            # 40 mel bins
F_MIN = 20
F_MAX = 4000
BUFFER_DURATION = 5.0  # 5-second buffer
STEP_DURATION = 1.0    # Process every 1 second

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_mels=N_MELS,
    f_min=F_MIN,
    f_max=F_MAX
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = load_model(checkpoint_path="trained_model.pth", device=device)

# -----------------------------
# Visualization Setup (Persistent Objects)
# -----------------------------
plt.ion()  # Interactive mode on
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
ax_waveform, ax_mel, ax_result = axes

# Persistent plot objects.
waveform_line = None    # For waveform plot
mel_im = None           # For mel spectrogram image
mel_colorbar = None     # For mel spectrogram colorbar

# Initialize result panel with a rectangle.
ax_result.axis("off")
result_patch = Rectangle((0, 0), 1, 1, color="grey")
ax_result.add_patch(result_patch)
ax_result.set_title("Inference Result", fontsize=14)

plt.tight_layout()
plt.draw()

# -----------------------------
# Process Function (updates persistent objects)
# -----------------------------
def process_last_5_seconds(audio_buffer):
    global waveform_line, mel_im, mel_colorbar, result_patch
    
    # --- Update waveform panel ---
    times = np.linspace(0, BUFFER_DURATION, len(audio_buffer))
    if waveform_line is None:
        waveform_line, = ax_waveform.plot(times, audio_buffer, color="blue")
        ax_waveform.set_title("Waveform (last 5 sec)")
        ax_waveform.set_xlabel("Time (sec)")
        ax_waveform.set_ylabel("Amplitude")
        ax_waveform.set_xlim(0, BUFFER_DURATION)
        #ax_waveform.set_ylim(-8000,8000)
    else:
        waveform_line.set_ydata(audio_buffer)
        #ax_waveform.set_ylim(-8000,8000)
    ax_waveform.relim()
    ax_waveform.autoscale_view()
    
    # --- Compute mel spectrogram ---
    waveform = torch.from_numpy(audio_buffer).float().unsqueeze(0)  # (1, samples)
    mel_spec = mel_transform(waveform)  # (1, n_mels, T)
    mel_spec = mel_spec.squeeze(0)       # (n_mels, T)
    mel_spec_db = 10 * torch.log10(mel_spec + 1e-10)  # dB scale
    
    # --- Update mel spectrogram panel ---
    if mel_im is None:
        mel_im = ax_mel.imshow(mel_spec_db.numpy(), aspect="auto", origin="lower", cmap="jet")
        ax_mel.set_title("Mel Spectrogram")
        ax_mel.set_xlabel("Time Frames")
        ax_mel.set_ylabel("Mel Frequency Bin")
        mel_colorbar = fig.colorbar(mel_im, ax=ax_mel)
    else:
        mel_im.set_data(mel_spec_db.numpy())
        mel_im.set_clim(vmin=np.min(mel_spec_db.numpy()), vmax=np.max(mel_spec_db.numpy()))
        if mel_colorbar is not None:
            mel_colorbar.update_normal(mel_im)
    
    # Define threshold for Apnea flag
    APNEA_THRESHOLD = 0.9

    # --- Prepare for inference ---
    mel_input = mel_spec.transpose(0, 1).unsqueeze(0).to(device)  # (1, T, n_mels)
    
    # --- Run inference ---
    with torch.no_grad():
        logits = model(mel_input)
        probabilities = torch.softmax(logits, dim=1)
        pred_prob_apnea  = probabilities[0,1].item()
        if pred_prob_apnea >= APNEA_THRESHOLD:
            pred_label = 1
        else:
            pred_label = 0
        # Old way, using straight probabilities from inference
        # pred_label = probabilities.argmax(dim=1).item()
    
    # buffer up to 600 most recent predictions
    global inference_history
    inference_history.append(pred_label)
    if len(inference_history) > 600:
        inference_history = inference_history[-600:]
    
    history_array = np.array(inference_history)
    num_rows = 20
    num_cols = math.ceil(len(history_array) / num_rows)
    # Pad the start with 0s
    pad_len = num_rows * num_cols - len(history_array)
    if pad_len > 0:
        history_array = np.pad(history_array, (pad_len, 0), mode='constant', constant_values=0)
    history_grid = history_array.reshape(num_rows, num_cols)

    '''
    # --- Update result panel ---
    ax_result.cla()
    ax_result.axis("off")
    color = "red" if pred_label == 1 else "green"
    result_patch = Rectangle((0, 0), 1, 1, color=color)
    ax_result.add_patch(result_patch)
    result_text = "Apnea" if pred_label == 1 else "Non-Apnea"
    ax_result.set_title(f"Inference Result: {result_text}", color=color, fontsize=16)
    '''
    
    # --- Update result panel as a checkerboard ---
    ax_result.cla()
    ax_result.axis("off")
    im_result = ax_result.imshow(history_grid, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    ax_result.set_title("Inference History (Checkerboard)", fontsize=16)
    result_text = "Apnea" if pred_label == 1 else "Non-Apnea"

    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)
    
    print(f"Inference: Predicted {result_text}, Probabilities: {probabilities.cpu().numpy()}")

# -----------------------------
# External (Microphone) Input
# -----------------------------
def run_external():
    global audio_buffer
    buffer_samples = int(SAMPLE_RATE * BUFFER_DURATION)
    audio_buffer = np.zeros(buffer_samples, dtype=np.int16)
    frames_per_buffer = 1024

    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16,
                     channels=1,
                     rate=SAMPLE_RATE,
                     input=True,
                     frames_per_buffer=frames_per_buffer)
    
    last_process_time = time.time()
    print("Recording from microphone... Press ESC in the plot window to exit.")
    
    exit_flag = False
    def on_key(event):
        nonlocal exit_flag
        if event.key == "escape":
            exit_flag = True
    fig.canvas.mpl_connect("key_press_event", on_key)
    
    try:
        while not exit_flag:
            data = stream.read(frames_per_buffer, exception_on_overflow=False)
            new_samples = np.frombuffer(data, dtype=np.int16)
            audio_buffer = np.roll(audio_buffer, -len(new_samples))
            audio_buffer[-len(new_samples):] = new_samples
            
            current_time = time.time()
            if current_time - last_process_time >= STEP_DURATION:
                process_last_5_seconds(audio_buffer)
                last_process_time = current_time
    except KeyboardInterrupt:
        print("Stopping recording...")
        
    stream.stop_stream()
    stream.close()
    pa.terminate()
    plt.ioff()
    plt.close()

# -----------------------------
# Internal (WAV File) Input
# -----------------------------
def run_internal():
    # Get information about the file without loading all the data.
    info = torchaudio.info(INTERNAL_WAV_PATH)
    total_frames = info.num_frames
    print(f"Internal WAV duration: {total_frames / info.sample_rate:.2f} seconds")
    
    # Process the file in 1-second steps with a rolling 5-second window.
    step_samples = int(STEP_DURATION * SAMPLE_RATE)
    buffer_samples = int(BUFFER_DURATION * SAMPLE_RATE)
    
    exit_flag = False
    def on_key(event):
        nonlocal exit_flag
        if event.key == "escape":
            exit_flag = True
    fig.canvas.mpl_connect("key_press_event", on_key)
    
    # Process segments from the file without loading the entire file.
    start = 0
    while start + buffer_samples <= total_frames and not exit_flag:
        # Print the timestamp of the current segment (in seconds).
        current_timestamp = start / SAMPLE_RATE
        print(f"Processing segment starting at {current_timestamp:.2f} seconds")
        
        # Load only the current 5-second segment.
        waveform, sr = torchaudio.load(
            INTERNAL_WAV_PATH,
            frame_offset=start,
            num_frames=buffer_samples,
            #normalize=False
        )

        # Check the maximum absolute value
        max_val = waveform.abs().max()
        print(f"Max value in waveform: {max_val}")

        #waveform = waveform.to(torch.float32)

        # Resample in chunks if needed.
        if sr != SAMPLE_RATE:
            waveform = resample_waveform_in_chunks(waveform, orig_freq=sr, new_freq=SAMPLE_RATE)
        # If stereo, take first channel.
        if waveform.shape[0] > 1:
            waveform = waveform[0:1]
        # Convert to numpy int16.
        audio_buffer = (waveform.squeeze(0).numpy()).astype(np.int16)
        
        process_last_5_seconds(audio_buffer)
        time.sleep(STEP_DURATION)
        start += step_samples
    plt.ioff()
    plt.close()


# -----------------------------
# Main Function
# -----------------------------
def main():
    if SOURCE == "external":
        run_external()
    elif SOURCE == "internal":
        run_internal()
    else:
        print("Unknown SOURCE. Set SOURCE to 'external' or 'internal'.")

if __name__ == "__main__":
    main()
