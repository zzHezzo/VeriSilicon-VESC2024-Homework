from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

def read_wav(fname):
    fs, signal = wavfile.read(fname)
    if len(signal.shape) != 1:
        print("convert stereo to mono")
        signal = signal[:,0]
    signal = signal.flatten()
    signal_len = len(signal)
    return signal,signal_len,fs

def read_txt(file_dir):
    data = []
    
    # read txt
    with open(file_dir, 'r') as file:
        lines = file.readlines()
        # process data
        for line in lines:
            # read data
            values = [int(value) for value in line.strip().split(",")]
            data.append(values)

    # set data format as numpy
    data = np.array(data)

    return data

def draw_time_domain_image(waveData, nframes, framerate, line_style):
    plt.plot(np.arange(0,nframes),waveData,line_style)
    plt.xlabel('point index')
    plt.ylabel('am')
    # plt.show()

def draw_result(raw_data,pred):
    """
    raw_data: (N,)
    pred: predict result,(M,2),for col:(start point,end point)
    """

    data_len = len(raw_data)

    x_value = np.zeros(data_len)
    for i in range(pred.shape[0]):
            x_value[pred[i][0]:pred[i][1]+1] = 1
    
    plt.plot(np.arange(0,data_len),x_value * np.max(raw_data),"r-")


def sample_rate_to_8K(signal,sample_rate):
    if sample_rate not in [8000,16000,24000,48000]:
        raise Exception("sample rate not in [8000,16000,48000]!")
    
    interval = int(sample_rate / 8000)
    sample_signal = signal[0::interval]

    signal_len = len(sample_signal)

    return sample_signal,signal_len