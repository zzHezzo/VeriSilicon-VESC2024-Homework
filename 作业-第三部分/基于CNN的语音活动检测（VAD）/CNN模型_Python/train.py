import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import numpy as np
from util import read_wav, read_txt, sample_rate_to_8K
from model import CNN
import random
from sklearn.metrics import confusion_matrix, precision_score, recall_score

class VADDataset(Dataset):
    def __init__(self, data_dir, label_dir, frame_len, sample_rate, augment=False):
        self.data_dir = Path(data_dir)
        self.label_dir = Path(label_dir)
        self.frame_len = frame_len
        self.sample_rate = sample_rate
        self.data_files = sorted(self.data_dir.glob("*.wav"))
        self.label_files = sorted(self.label_dir.glob("*.txt"))
        self.augment = augment

        self.data, self.labels = self.process_data()

    def process_data(self):
        data = []
        labels = []
        for data_file, label_file in zip(self.data_files, self.label_files):
            signal, signal_len, sample_rate = read_wav(str(data_file))
            signal, signal_len = sample_rate_to_8K(signal, sample_rate)
            label_data = read_txt(label_file)

            for start, end in label_data:
                for i in range(start, end, int(FS * FRAME_STEP)):
                    if i + self.frame_len > signal_len:
                        break
                    frame_data = signal[i:i + self.frame_len]
                    if self.augment:
                        frame_data = self.add_noise(frame_data)
                    label = 1 if (i >= start and i <= end) else 0
                    data.append(frame_data)
                    labels.append(label)

            non_voice_intervals = self.get_non_voice_intervals(signal_len, label_data)
            for start, end in non_voice_intervals:
                for i in range(start, end, int(FS * FRAME_STEP)):
                    if i + self.frame_len > signal_len:
                        break
                    frame_data = signal[i:i + self.frame_len]
                    if self.augment:
                        frame_data = self.add_noise(frame_data)
                    data.append(frame_data)
                    labels.append(0)

        return np.array(data), np.array(labels)

    def get_non_voice_intervals(self, signal_len, label_data):
        non_voice_intervals = []
        previous_end = 0
        for start, end in label_data:
            if start > previous_end:
                non_voice_intervals.append((previous_end, start))
            previous_end = end
        if previous_end < signal_len:
            non_voice_intervals.append((previous_end, signal_len))
        return non_voice_intervals

    def add_noise(self, data):
        noise = np.random.randn(len(data)) * 0.005
        augmented_data = data + noise
        return augmented_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.from_numpy(self.data[idx]).float().unsqueeze(0).unsqueeze(0)
        label = torch.tensor(self.labels[idx]).long()
        return sample, label

def train_vad(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        print(f'Validation Loss: {val_loss/len(val_loader)}')
        print(f'Accuracy: {100 * (np.array(all_labels) == np.array(all_preds)).sum() / len(all_labels)}%')
        print(f'Precision: {precision_score(all_labels, all_preds, average="binary", zero_division=0)}')
        print(f'Recall: {recall_score(all_labels, all_preds, average="binary", zero_division=0)}')
        print('Confusion Matrix:')
        print(confusion_matrix(all_labels, all_preds, labels=[0, 1]))

if __name__ == "__main__":
    FS = 8000
    FRAME_T = 0.03
    FRAME_STEP = 0.015
    frame_len = int(FRAME_T * FS)

    data_dir = "./data"
    label_dir = "./label"
    augment = True
    dataset = VADDataset(data_dir, label_dir, frame_len, FS, augment=augment)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_vad(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)

    torch.save(model.state_dict(), "./model/model_microphone.pth")
