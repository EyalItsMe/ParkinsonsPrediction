from pydub import AudioSegment
from pydub.utils import make_chunks
import os
from torch.utils.data import DataLoader, random_split
from AudioDataset import AudioDataset

if __name__ == "__main__":
    #Creates new clips by cutting out the quiet speaker
    for root, dirs, files in os.walk("./dataset"):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                print(f"Found WAV file: {file} at {file_path}")
                x = root.split('_')[0]
                sound = AudioSegment.from_wav(file_path)
                chunks = make_chunks(sound, 600)

                louder_voice_chunks = []

                #gets rid of sections which is below silence threshold
                for i, chunk in enumerate(chunks):
                    chunk_loudness = chunk.dBFS
                    if chunk_loudness > -50:
                        print(f"Chunk {i} is loud enough: {chunk_loudness} dBFS")
                        louder_voice_chunks.append(chunk)
                    else:
                        print(f"Chunk {i} is too quiet: {chunk_loudness} dBFS")

                # Combine the louder chunks into one segment
                if louder_voice_chunks:
                    louder_voice_audio = louder_voice_chunks[0]
                    for chunk in louder_voice_chunks[1:]:
                        louder_voice_audio += chunk

                output_subdir = os.path.join("./new_dataset", os.path.relpath(root, "./dataset"))
                os.makedirs(output_subdir, exist_ok=True)

                output_subdir = os.path.join("./new_dataset", os.path.relpath(root, "./dataset"))
                os.makedirs(output_subdir, exist_ok=True)

                #create new segments and save them to new database
                if louder_voice_audio:
                    sec_chunks = make_chunks(louder_voice_audio, 3000)
                    for i, chunk in enumerate(sec_chunks):
                        chunk_filename = f"{os.path.splitext(file)[0]}_chunk{i}.wav"
                        chunk_path = os.path.join(output_subdir, chunk_filename)
                        chunk.export(chunk_path, format="wav")

    audio_dataset = AudioDataset(root_dir="./new_dataset")
    # dataloader = DataLoader(audio_dataset, batch_size=4, shuffle=True)

    # Define the split ratio
    train_ratio = 0.8
    test_ratio = 0.2

    # Calculate the lengths of train and test sets
    train_size = int(train_ratio * len(audio_dataset))
    test_size = len(audio_dataset) - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(audio_dataset, [train_size, test_size])

    # Create DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    for batch in train_loader:
        waveforms, labels = batch
        print(f"Train Waveforms: {waveforms.shape}")
        print(f"Train Labels: {labels}")

    # Example of iterating through the test DataLoader
    for batch in test_loader:
        waveforms, labels = batch
        print(f"Test Waveforms: {waveforms.shape}")
        print(f"Test Labels: {labels}")