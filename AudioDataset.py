import os
import torchaudio
from torch.utils.data import Dataset
import torch

class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_length=132500):
        self.root_dir = root_dir
        self.transform = transform
        self.max_length = max_length
        self.audio_data = []
        self.labels = []
        self._load_files_and_labels()

    def _load_files_and_labels(self):
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".wav"):
                    audio_path = os.path.join(root, file)
                    waveform, sample_rate = torchaudio.load(audio_path)

                    if self.transform:
                        waveform = self.transform(waveform)


                    if waveform.shape[1] < self.max_length:
                        pad_amount = self.max_length - waveform.shape[1]
                        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
                    elif waveform.shape[1] > self.max_length:
                        waveform = waveform[:, :self.max_length]

                    self.audio_data.append(waveform)

                    if "hc" in file.lower():
                        self.labels.append(0)  # 0 for healthy
                    elif "pd" in file.lower():
                        self.labels.append(1)  # 1 for non-healthy
    def __getitem__(self, idx):
        waveform = self.audio_data[idx]
        label = self.labels[idx]
        return waveform, label
    def __len__(self):
        return len(self.audio_data)
