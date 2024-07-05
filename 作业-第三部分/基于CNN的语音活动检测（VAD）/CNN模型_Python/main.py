import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from util import *
from model import CNN  # 确保导入模型类
import torch
from torch.utils.data import Dataset, DataLoader


class VADDataset(Dataset):
    def __init__(self, data_files, frame_len, sample_rate):
        self.data_files = data_files
        self.frame_len = frame_len
        self.sample_rate = sample_rate
        self.data, self.file_indices = self.process_data()

    def process_data(self):
        data = []
        file_indices = []
        for file_idx, data_file in enumerate(self.data_files):
            signal, signal_len, sample_rate = read_wav(str(data_file))
            signal, signal_len = sample_rate_to_8K(signal, sample_rate)

            for i in range(0, signal_len, int(FRAME_STEP * FS)):
                if i + self.frame_len > signal_len:
                    break
                frame_data = signal[i:i + self.frame_len]
                data.append(frame_data)
                file_indices.append((file_idx, i))

        return np.array(data), file_indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.from_numpy(self.data[idx]).float().unsqueeze(0).unsqueeze(0)
        file_idx, frame_idx = self.file_indices[idx]
        return sample, file_idx, frame_idx


def cal_voice_segment(pred_class, pred_idx_in_data, raw_data_len):
    if len(pred_class) != len(pred_idx_in_data):
        raise Exception("pred_class length must be pred_idx_in_data length!")

    all_voice_segment = np.array([])
    single_voice_segment = []
    diff_value = np.diff(pred_class)

    for i in range(len(diff_value)):
        if diff_value[i] == 1:
            single_voice_segment.append(pred_idx_in_data[i + 1])
        if diff_value[i] == -1:
            if len(single_voice_segment) == 0:
                single_voice_segment.append(0)
            single_voice_segment.append(pred_idx_in_data[i + 1])
        if len(single_voice_segment) == 2:
            if len(all_voice_segment) == 0:
                all_voice_segment = np.array(single_voice_segment).reshape(1, -1)
            else:
                all_voice_segment = np.concatenate((all_voice_segment, np.array(single_voice_segment).reshape(1, -1)),
                                                   axis=0)
            single_voice_segment = []

    if len(single_voice_segment) == 1:
        single_voice_segment.append(raw_data_len - 1)
        all_voice_segment = np.concatenate((all_voice_segment, np.array(single_voice_segment).reshape(1, -1)), axis=0)

    if all_voice_segment.size == 0 and np.all(pred_class == 1):
        all_voice_segment = np.array([[0, raw_data_len - 1]])

    return all_voice_segment


def vad_inference(data_dir: str, model_path: str, result_dir: str):
    data_files = sorted(Path(data_dir).glob("*.wav"))
    dataset = VADDataset(data_files, frame_len, FS)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = CNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    results = [[] for _ in range(len(data_files))]
    with torch.no_grad():
        for inputs, file_indices, frame_indices in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            for file_idx, frame_idx, pred in zip(file_indices, frame_indices, predicted):
                results[file_idx].append((frame_idx.item(), pred.item()))

    for file_idx, file_dir in enumerate(data_files):
        signal, signal_len, sample_rate = read_wav(str(file_dir))
        signal, signal_len = sample_rate_to_8K(signal, sample_rate)
        file_results = results[file_idx]
        frame_indices, preds = zip(*file_results)
        voice_segment = cal_voice_segment(np.array(preds), np.array(frame_indices), signal_len)

        plt.figure(1, figsize=(15, 7))
        plt.clf()
        draw_time_domain_image(signal, nframes=signal_len, framerate=sample_rate, line_style="b-")
        draw_result(signal, voice_segment)
        plt.grid()
        plt.show()

        result_dir_path = Path(result_dir)
        result_dir_path.mkdir(parents=True, exist_ok=True)
        result_file = result_dir_path / f"{file_dir.stem}_result.txt"
        np.savetxt(result_file, voice_segment, fmt="%d", delimiter=",")


if __name__ == "__main__":
    FS = 8000
    FRAME_T = 0.03
    FRAME_STEP = 0.015
    frame_len = int(FRAME_T * FS)

    model_path = "./model/model_microphone.pth"
    data_dir = "./maindata"
    result_dir = "./result"

    vad_inference(data_dir=data_dir, model_path=model_path, result_dir=result_dir)
