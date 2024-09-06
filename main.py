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
import matplotlib.pyplot as plt

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
        # print(running_loss)
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

def plot_training_results(train_losses, val_accuracies, model_name):
    epochs = range(1, len(train_losses) + 1)

    # Plotting the loss
    plt.figure(figsize=(14, 6))

    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, '-o', label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Training Loss')
    plt.grid(True)

    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, '-o', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} - Validation Accuracy')
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()


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

    # feature_extractor = "mfcc"
    # feature_extractor = "mel"
    feature_extractor = "hubert"
    # feature_extractor = "whisper"
    nmfcc = 49
    max_length = 280
    feature_dims = {
        "mfcc": nmfcc * max_length,        # MFCC: nmfcc x max_length (e.g., 13 x 280)
        "mel": nmfcc * max_length,           # Mel Spectrogram: 128 x max_length (standard Mel spec dimensions)
        "hubert": 49 * 1024,       # HuBERT: 1024 x max_length (based on large HuBERT model's output)
        "whisper": 1024 * max_length       # Whisper: 1024 x max_length (based on Whisper model's output)
    }
    audio_dataset = AudioDataset(root_dir="./new_dataset", feature_extractor=feature_extractor, nmfcc=nmfcc)
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
    input_dim = feature_dims[feature_extractor]
    hidden_dim = 256
    output_dim = 2  # Healthy (0) or Parkinson's (1)
    learning_rate = 0.0001
    num_epochs = 20


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "BasicClassifier"
    model = BasicClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    # input_dim_mfcccnn = 512
    # model = MFCCCNN(input_dim=input_dim_mfcccnn, nmfcc=nmfcc).to(device)
    # model_name = "MFCC-CNN"
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_accuracies = []
    # Training and evaluation loop
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_accuracy = evaluate(model, test_loader, device)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    # Plot the training results
    plot_training_results(train_losses, val_accuracies, model_name)