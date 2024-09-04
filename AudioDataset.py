import os
import torchaudio
from torch.utils.data import Dataset
import torch
import librosa
from transformers import Wav2Vec2Processor, HubertModel
import torch
import torchaudio


class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_length=132500):
        self.root_dir = root_dir
        self.transform = transform
        self.max_length = max_length
        self.audio_data = []
        self.labels = []
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=16000, n_mfcc=13, melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23}
        )
        self._load_files_and_labels()

    def _load_files_and_labels(self):
        #This section is the mfcc feature extraction section
        # for root, _, files in os.walk(self.root_dir):
        #     for file in files:
        #         if file.endswith(".wav"):
        #             audio_path = os.path.join(root, file)
        #             # Load audio using librosa
        #             waveform, sample_rate = librosa.load(audio_path)
        #
        #             # Compute MFCCs using librosa
        #             mfcc = librosa.feature.mfcc(y=waveform, sr=16000, n_mfcc=13, n_fft=400, hop_length=160, n_mels=23)
        #             mfcc = torch.tensor(mfcc, dtype=torch.float32)
        #
        #             if self.transform:
        #                 mfcc = self.transform(mfcc)
        #
        #             # Ensure consistent length
        #             num_frames = mfcc.shape[1]
        #             if num_frames < self.max_length:
        #                 pad_amount = self.max_length - num_frames
        #                 mfcc = torch.nn.functional.pad(mfcc, (0, pad_amount))
        #             elif num_frames > self.max_length:
        #                 mfcc = mfcc[:, :self.max_length]
        #
        #             self.audio_data.append(mfcc)
        #
        #             if "hc" in file.lower():
        #                 self.labels.append(0)  # 0 for healthy
        #             elif "pd" in file.lower():
        #                 self.labels.append(1)  # 1 for non-healthy



        #This is the hubert feature extraction
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".wav"):
                    audio_path = os.path.join(root, file)

                    waveform, sample_rate = torchaudio.load(audio_path)

                    if sample_rate != 16000:
                        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                        waveform = resampler(waveform)

                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0).unsqueeze(0)

                    waveform = waveform / waveform.abs().max()
                    print("Waveform shape:", waveform.shape)
                    if self.transform:
                        waveform = self.transform(waveform)

                    # Ensure consistent length
                    # num_samples = waveform.shape[1]
                    # if num_samples < self.max_length:
                    #     pad_amount = self.max_length - num_samples
                    #     waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
                    # elif num_samples > self.max_length:
                    #     waveform = waveform[:, :self.max_length]

                    # Load HuBERT model and processor
                    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
                    model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
                    input_values = processor(waveform.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values

                    with torch.no_grad():
                        outputs = model(input_values)

                    # The output is a tuple, where the first element is the last hidden state
                    features = outputs.last_hidden_state

                    print("Extracted features shape:", features.shape)

                    self.audio_data.append(features)

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
