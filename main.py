import os



from pydub import AudioSegment
from pydub.utils import make_chunks
from torch.utils.data import DataLoader, random_split
from AudioDataset import AudioDataset
from sklearn.metrics import accuracy_score
import torch
import torch.optim as optim
from BasicClassifier import BasicClassifier
import torch.nn as nn
from ConvultionCNN import MFCCCNN

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for mfccs, labels in dataloader:
        mfccs, labels = mfccs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(mfccs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * mfccs.size(0)
        print(running_loss)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for mfccs, labels in dataloader:
            mfccs, labels = mfccs.to(device), labels.to(device)
            outputs = model(mfccs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


if __name__ == "__main__":
    #Creates new clips by cutting out the quiet speaker
    # for root, dirs, files in os.walk("./dataset"):
    #     for file in files:
    #         if file.endswith(".wav"):
    #             file_path = os.path.join(root, file)
    #             print(f"Found WAV file: {file} at {file_path}")
    #             x = root.split('_')[0]
    #             sound = AudioSegment.from_wav(file_path)
    #             chunks = make_chunks(sound, 600)
    #
    #             louder_voice_chunks = []
    #
    #             #gets rid of sections which is below silence threshold
    #             for i, chunk in enumerate(chunks):
    #                 chunk_loudness = chunk.dBFS
    #                 if chunk_loudness > -50:
    #                     print(f"Chunk {i} is loud enough: {chunk_loudness} dBFS")
    #                     louder_voice_chunks.append(chunk)
    #                 else:
    #                     print(f"Chunk {i} is too quiet: {chunk_loudness} dBFS")
    #
    #             # Combine the louder chunks into one segment
    #             if louder_voice_chunks:
    #                 louder_voice_audio = louder_voice_chunks[0]
    #                 for chunk in louder_voice_chunks[1:]:
    #                     louder_voice_audio += chunk
    #
    #             output_subdir = os.path.join("./new_dataset", os.path.relpath(root, "./dataset"))
    #             os.makedirs(output_subdir, exist_ok=True)
    #
    #             output_subdir = os.path.join("./new_dataset", os.path.relpath(root, "./dataset"))
    #             os.makedirs(output_subdir, exist_ok=True)
    #
    #             #create new segments and save them to new database
    #             if louder_voice_audio:
    #                 sec_chunks = make_chunks(louder_voice_audio, 1000)
    #                 for i, chunk in enumerate(sec_chunks):
    #                     chunk_filename = f"{os.path.splitext(file)[0]}_chunk{i}.wav"
    #                     chunk_path = os.path.join(output_subdir, chunk_filename)
    #                     chunk.export(chunk_path, format="wav")

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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    input_dim = 13 * 132500  # Flattened MFCC dimension  # 13 MFCCs, 23 mel bands
    hidden_dim = 250
    output_dim = 2  # Healthy (0) or Parkinson's (1)
    learning_rate = 0.001
    num_epochs = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = BasicClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    model = MFCCCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training and evaluation loop
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_accuracy = evaluate(model, test_loader, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

