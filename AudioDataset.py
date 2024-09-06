import os
import torchaudio
from torch.utils.data import Dataset
import torch
import librosa
from transformers import Wav2Vec2Processor, HubertModel, WhisperProcessor, WhisperModel
import torch
import torchaudio


class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_length=280, feature_extractor="hubert", nmfcc=13):
        self.root_dir = root_dir
        self.transform = transform
        self.max_length = max_length
        self.audio_data = []
        self.labels = []
        self.feature_extractor = feature_extractor
        
        if self.feature_extractor == "mfcc":
            self.mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=16000, n_mfcc=nmfcc, melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23}
            )
        elif self.feature_extractor == "mel":
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_fft=400, hop_length=160, n_mels=nmfcc
            )

        self._load_files_and_labels()

    def _load_files_and_labels(self):
        i = 0
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".wav"):
                    audio_path = os.path.join(root, file)
                    waveform, sample_rate = self._load_audio(audio_path, self.feature_extractor)

                    if self.feature_extractor == "mfcc":
                        features = self._extract_mfcc(waveform, sample_rate)
                    elif self.feature_extractor == "mel":
                        features = self._extract_mel_spectrogram(waveform)
                    elif self.feature_extractor == "hubert":
                        features = self._extract_hubert(waveform)
                    elif self.feature_extractor == "whisper":
                        features = self._extract_whisper(waveform, sample_rate)
                    else:
                        raise ValueError("Invalid feature extractor")

                    if features.shape[0] == 49:
                        self.audio_data.append(features)
                        if "hc" in file.lower():
                            self.labels.append(0)  # 0 for healthy
                        elif "pd" in file.lower():
                            self.labels.append(1)  # 1 for non-healthy
                        i += 1
                        # if i == 50:
                        #     return
    def _load_audio(self, audio_path, feature_extractor):
        waveform, sample_rate = torchaudio.load(audio_path)
        if feature_extractor == "hubert":
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0).unsqueeze(0)
            waveform = waveform / waveform.abs().max()
        return waveform, sample_rate

    def _extract_mfcc(self, waveform, sample_rate):
        waveform = waveform.squeeze(0).numpy()
        mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13, n_fft=400, hop_length=160, n_mels=23)
        mfcc = torch.tensor(mfcc, dtype=torch.float32)
        mfcc = self._pad_or_truncate(mfcc)
        if self.transform:
            mfcc = self.transform(mfcc)
        return mfcc

    def _extract_mel_spectrogram(self, waveform):
        mel_spectrogram = self.mel_transform(waveform).squeeze(0)
        mel_spectrogram = self._pad_or_truncate(mel_spectrogram)
        if self.transform:
            mel_spectrogram = self.transform(mel_spectrogram)
        return mel_spectrogram

    def _extract_hubert(self, waveform):
        processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        input_values = processor(waveform.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values
        with torch.no_grad():
            outputs = model(input_values)
        features = outputs.last_hidden_state.squeeze(0)
        return features

    def _extract_whisper(self, waveform, sample_rate):
        # Load Whisper model and processor
        processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        model = WhisperModel.from_pretrained("openai/whisper-small")

        # Preprocess the audio using the Whisper processor
        input_features = processor(waveform.squeeze(0), return_tensors="pt", sampling_rate=sample_rate).input_features

        # Pass the input through the Whisper model
        with torch.no_grad():
            outputs = model(input_features)

        # Extract the last hidden states from the model
        features = outputs.last_hidden_state

        # Optionally truncate or pad the features to a fixed length
        features = self._pad_or_truncate(features.squeeze(0).transpose(0, 1))

        return features

    def _pad_or_truncate(self, feature_tensor):
        num_frames = feature_tensor.shape[1]
        if num_frames < self.max_length:
            pad_amount = self.max_length - num_frames
            feature_tensor = torch.nn.functional.pad(feature_tensor, (0, pad_amount))
        elif num_frames > self.max_length:
            feature_tensor = feature_tensor[:, :self.max_length]
        return feature_tensor


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
        # for root, _, files in os.walk(self.root_dir):
        #     for file in files:
        #         if file.endswith(".wav"):
        #             audio_path = os.path.join(root, file)
        #
        #             waveform, sample_rate = torchaudio.load(audio_path)
        #
        #             if sample_rate != 16000:
        #                 resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        #                 waveform = resampler(waveform)
        #
        #             if waveform.shape[0] > 1:
        #                 waveform = waveform.mean(dim=0).unsqueeze(0)
        #
        #             waveform = waveform / waveform.abs().max()
        #             print("Waveform shape:", waveform.shape)
        #             if self.transform:
        #                 waveform = self.transform(waveform)
        #
        #             # Ensure consistent length
        #             # num_samples = waveform.shape[1]
        #             # if num_samples < self.max_length:
        #             #     pad_amount = self.max_length - num_samples
        #             #     waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        #             # elif num_samples > self.max_length:
        #             #     waveform = waveform[:, :self.max_length]
        #
        #             # Load HuBERT model and processor
        #             processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        #             model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        #             input_values = processor(waveform.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values
        #
        #             with torch.no_grad():
        #                 outputs = model(input_values)
        #
        #             # The output is a tuple, where the first element is the last hidden state
        #             features = outputs.last_hidden_state
        #
        #             print("Extracted features shape:", features.shape)
        #
        #             self.audio_data.append(features)
        #
        #             if "hc" in file.lower():
        #                 self.labels.append(0)  # 0 for healthy
        #             elif "pd" in file.lower():
        #                 self.labels.append(1)  # 1 for non-healthy



    def __getitem__(self, idx):
        waveform = self.audio_data[idx]
        label = self.labels[idx]
        return waveform, label

    def __len__(self):
        return len(self.audio_data)
