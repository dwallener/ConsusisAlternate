import os
import random
import torch
import torch.nn as nn
import torchaudio
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from torch.utils.data import Dataset

###########################
# Dataset Definition
###########################

class AudioApneaDataset(Dataset):
    def __init__(self, mic_wav_path, tracheal_wav_path, annotation_path,
                 segment_duration=5.0, sample_rate=48000, n_mels=80):
        """
        Args:
          mic_wav_path (str): Path to the mic audio file.
          tracheal_wav_path (str): Path to the tracheal audio file.
          annotation_path (str): Path to the .rml annotation file.
          segment_duration (float): Duration (in seconds) of each sample segment.
          sample_rate (int): Sample rate of the audio.
          n_mels (int): Number of mel bins.
        """
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.segment_samples = int(segment_duration * sample_rate)
        self.n_mels = n_mels

        # Load the two recordings.
        self.mic_waveform, sr1 = torchaudio.load(mic_wav_path)
        self.tracheal_waveform, sr2 = torchaudio.load(tracheal_wav_path)
        assert sr1 == sample_rate and sr2 == sample_rate, "Unexpected sample rate."

        # If stereo, take only one channel.
        if self.mic_waveform.shape[0] > 1:
            self.mic_waveform = self.mic_waveform[0:1]
        if self.tracheal_waveform.shape[0] > 1:
            self.tracheal_waveform = self.tracheal_waveform[0:1]

        # Ensure both recordings are the same length.
        if self.mic_waveform.shape[1] != self.tracheal_waveform.shape[1]:
            raise ValueError("Audio recordings are not the same length.")

        self.num_samples = self.mic_waveform.shape[1]
        self.apnea_events = self._parse_annotations(annotation_path)

        # Create the mel spectrogram transform.
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels)

        # Calculate the total number of segments.
        self.num_segments = self.num_samples // self.segment_samples

    def _parse_annotations(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        events = []
        for event in root.iter('Event'):
            if (event.attrib.get("Family") == "Respiratory" and 
                event.attrib.get("Type") == "ObstructiveApnea"):
                start = float(event.attrib.get("Start"))
                duration = float(event.attrib.get("Duration"))
                end = start + duration
                events.append((start, end))
        return events

    def _segment_label(self, seg_start_sec, seg_end_sec):
        for event_start, event_end in self.apnea_events:
            if event_start < seg_end_sec and event_end > seg_start_sec:
                return 1
        return 0

    def __len__(self):
        return self.num_segments

    def __getitem__(self, idx):
        start_sample = idx * self.segment_samples
        end_sample = start_sample + self.segment_samples

        mic_seg = self.mic_waveform[:, start_sample:end_sample]
        trac_seg = self.tracheal_waveform[:, start_sample:end_sample]

        mic_mel = self.mel_transform(mic_seg).squeeze(0)    # (n_mels, T)
        trac_mel = self.mel_transform(trac_seg).squeeze(0)    # (n_mels, T)

        # Concatenate along the frequency dimension and transpose:
        # shape becomes (T, 2*n_mels)
        mel_features = torch.cat([mic_mel, trac_mel], dim=0).transpose(0, 1)

        seg_start_sec = start_sample / self.sample_rate
        seg_end_sec = seg_start_sec + self.segment_duration
        label = self._segment_label(seg_start_sec, seg_end_sec)
        label = torch.tensor(label, dtype=torch.long)
        
        return mel_features, label

# Set the dataset variable using the same file names as in training.
dataset = AudioApneaDataset("mic.wav", "tracheal.wav", "annotations.rml",
                            segment_duration=5.0, sample_rate=48000, n_mels=80)
print(f"Dataset contains {len(dataset)} segments.")

###########################
# Model Definition
###########################

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, model_dim=256, num_heads=4, num_layers=3, dropout=0.1):
        """
        Args:
          input_dim (int): Dimensionality of input tokens (e.g., 2*n_mels).
          model_dim (int): Transformer model dimension.
          num_heads (int): Number of attention heads.
          num_layers (int): Number of transformer encoder layers.
          dropout (float): Dropout rate.
        """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        # Create a learnable [CLS] token.
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))
        # Dynamically created positional embedding (will be re-created as needed).
        self.pos_embedding = None
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head (binary classification).
        self.classifier = nn.Linear(model_dim, 2)

    def forward(self, x):
        """
        Args:
          x (Tensor): shape (batch, T, input_dim)
        Returns:
          logits (Tensor): shape (batch, 2)
        """
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        
        # Prepend the [CLS] token.
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Create (or reinitialize) positional embeddings on the same device as x.
        if self.pos_embedding is None or self.pos_embedding.shape[1] != x.shape[1]:
            self.pos_embedding = nn.Parameter(torch.randn(1, x.shape[1], x.shape[2], device=x.device), requires_grad=True)
        x = x + self.pos_embedding
        
        # Transformer expects input shape: (T+1, batch, model_dim)
        x = x.transpose(0, 1)
        encoded = self.transformer_encoder(x)
        encoded = encoded.transpose(0, 1)
        
        # Use the CLS token's representation for classification.
        cls_encoded = encoded[:, 0, :]
        logits = self.classifier(cls_encoded)
        return logits

# Initialize the model with the same parameters used during training.
input_dim = 2 * 80  # because we stacked two recordings (80 mel bins each)
model = TransformerClassifier(input_dim=input_dim, model_dim=256, num_heads=4, num_layers=3, dropout=0.1)

###########################
# Load the Saved Model
###########################

state_dict = torch.load("trained_model.pth", map_location="cuda:0" if torch.cuda.is_available() else "cpu")
# Remove DataParallel prefix if necessary.
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k[7:] if k.startswith("module.") else k
    new_state_dict[new_key] = v

# Load the state dict with strict=False to ignore keys like pos_embedding.
model.load_state_dict(new_state_dict, strict=False)

# Set up device and move the model.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

###########################
# Inference: Show Nonzero Predictions Only
###########################

# We'll try to find a few samples (e.g., 5) where the model predicts a nonzero label.
desired_nonzero_samples = 5
nonzero_samples_found = 0
max_attempts = 500
attempt = 0

print("Looking for samples with predicted label != 0. Press SPACEBAR to show next sample, or ESC to exit.")

while nonzero_samples_found < desired_nonzero_samples and attempt < max_attempts:
    idx = random.randrange(len(dataset))
    attempt += 1
    mel_features, true_label = dataset[idx]
    input_tensor = mel_features.unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_label = probabilities.argmax(dim=1).item()

    if predicted_label != 0:
        nonzero_samples_found += 1

        # Prepare spectrogram for display.
        spec = mel_features.transpose(0, 1).cpu().numpy()
        spec_db = 10 * np.log10(spec + 1e-10)  # convert to dB

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(spec_db, aspect='auto', origin='lower')
        fig.colorbar(im, ax=ax)
        title_str = (f"True Label: {true_label.item()} | Predicted: {predicted_label}\n"
                     f"Probabilities: {probabilities.cpu().numpy()}")
        ax.set_title(title_str)
        ax.axis('off')

        key_pressed = [None]

        def on_key(event):
            key_pressed[0] = event.key
            plt.close(fig)

        cid = fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()

        if key_pressed[0] == 'escape':
            print("ESC pressed. Exiting inference test.")
            break

if nonzero_samples_found == 0:
    print("No samples with predicted label != 0 were found.")
else:
    print(f"Displayed {nonzero_samples_found} samples with nonzero predicted labels.")
