import os
import torchaudio
from torch.utils.data import Dataset
import torch
import librosa
from transformers import Wav2Vec2Processor, HubertModel, WhisperProcessor, WhisperModel, WhisperFeatureExtractor
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

        self.cache_file = f"{feature_extractor}_dataset_cache.pt"

        if os.path.exists(self.cache_file):
            print(f"Loading dataset from {self.cache_file}...")
            self._load_from_cache()
        else:
            print("Processing dataset from audio files...")
            self._load_files_and_labels()
            print(f"Saving processed dataset to {self.cache_file}...")
            self._save_to_cache()

    def _save_to_cache(self):
        data = {
            "audio_data": self.audio_data,
            "labels": self.labels
        }
        torch.save(data, self.cache_file)

    def _load_from_cache(self):
        data = torch.load(self.cache_file)
        self.audio_data = data["audio_data"]
        self.labels = data["labels"]

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
                        features = self._extract_whisper(waveform)
                    else:
                        raise ValueError("Invalid feature extractor")

                    if (features.shape[0] == 49 and self.feature_extractor == "hubert") \
                            or (features.shape[1] == 384 and self.feature_extractor == "whisper"):
                        self.audio_data.append(features)
                        if "hc" in file.lower():
                            self.labels.append(0)  # 0 for healthy
                        elif "pd" in file.lower():
                            self.labels.append(1)  # 1 for non-healthy
                        if i % 100 == 0:
                            print(f"Processed {i} files")
                        i += 1
    def _load_audio(self, audio_path, feature_extractor):
        waveform, sample_rate = torchaudio.load(audio_path)
        if feature_extractor == "hubert" or feature_extractor == "whisper":
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

    def _extract_whisper(self, waveform):
        # Load Whisper model and processor
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = WhisperModel.from_pretrained("openai/whisper-tiny")

        # Preprocess the audio using the Whisper processor
        input_features = processor(waveform.squeeze(0), return_tensors="pt", sampling_rate=16000).input_features

        # Generate decoder input ids
        decoder_input_ids = torch.tensor([[processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")]])

        # Pass the input through the Whisper model
        with torch.no_grad():
            outputs = model(input_features, decoder_input_ids=decoder_input_ids)

        # Extract the last hidden states from the model
        features = outputs.last_hidden_state.squeeze(0)

        return features

    def _pad_or_truncate(self, feature_tensor):
        num_frames = feature_tensor.shape[1]
        if num_frames < self.max_length:
            pad_amount = self.max_length - num_frames
            feature_tensor = torch.nn.functional.pad(feature_tensor, (0, pad_amount))
        elif num_frames > self.max_length:
            feature_tensor = feature_tensor[:, :self.max_length]
        return feature_tensor

    def __getitem__(self, idx):
        waveform = self.audio_data[idx]
        label = self.labels[idx]
        return waveform, label

    def __len__(self):
        return len(self.audio_data)
