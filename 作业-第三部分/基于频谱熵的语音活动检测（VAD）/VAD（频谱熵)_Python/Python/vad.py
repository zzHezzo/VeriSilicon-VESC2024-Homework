import librosa
import numpy as np
from scipy.signal import medfilt

WAV = 'test_data/data/data_1.wav'
PREDICT_TXT_SAVE_PATH = 'test_data/predict/data_1.txt'
SAMPLE_RATE = 8000
FRAME_LENGTH = 240
HOP_LENGTH = 80
THR1 = 0.99
THR2 = 0.96
NIS = 20

def enframe(x, win, inc):
    nx = len(x)
    if isinstance(win, list) or isinstance(win, np.ndarray):
        nwin = len(win)
        nlen = nwin  # Frame length = window length
    elif isinstance(win, int):
        nwin = 1
        nlen = win  # Set to frame length
    if inc is None:
        inc = nlen
    nf = (nx - nlen + inc) // inc
    frameout = np.zeros((nf, nlen))
    indf = np.multiply(inc, np.array([i for i in range(nf)]))
    for i in range(nf):
        frameout[i, :] = x[indf[i]:indf[i] + nlen]
    if isinstance(win, list) or isinstance(win, np.ndarray):
        frameout = np.multiply(frameout, np.array(win))
    return frameout

def vad_specEN(data, wnd, inc, NIS, thr1, thr2, fs):
    x = enframe(data, wnd, inc)
    X = np.abs(np.fft.fft(x, axis=1))
    if len(wnd) == 1:
        wlen = wnd
    else:
        wlen = len(wnd)
    df = fs / wlen
    fx1 = int(250 // df + 1)
    fx2 = int(3500 // df + 1)
    K = 0.5
    E = np.zeros((X.shape[0], wlen // 2))
    E[:, fx1 + 1:fx2 - 1] = X[:, fx1 + 1:fx2 - 1]
    E = np.multiply(E, E)
    Esum = np.sum(E, axis=1, keepdims=True)
    P1 = np.divide(E, Esum)
    E = np.where(P1 >= 0.9, 0, E)
    Eb0 = E[:, 0::4]
    Eb1 = E[:, 1::4]
    Eb2 = E[:, 2::4]
    Eb3 = E[:, 3::4]
    Eb = Eb0 + Eb1 + Eb2 + Eb3
    prob = np.divide(Eb + K, np.sum(Eb + K, axis=1, keepdims=True))
    Hb = -np.sum(np.multiply(prob, np.log10(prob + 1e-10)), axis=1)
    Hb = medfilt(Hb, 5)
    Me = np.mean(Hb)
    eth = np.mean(Hb[:NIS])
    Det = eth - Me
    T1 = thr1 * Det + Me
    T2 = thr2 * Det + Me
    voiceseg = vad_revr(Hb, T1, T2)
    return voiceseg

def vad_revr(dst1, T1, T2):
    fn = len(dst1)
    maxsilence = 8
    minlen = 5
    status = 0
    count = np.zeros(fn)
    silence = np.zeros(fn)
    xn = 0
    x1 = np.zeros(fn)
    x2 = np.zeros(fn)
    for n in range(1, fn):
        if status == 0 or status == 1:
            if dst1[n] < T2:
                x1[xn] = max(1, n - count[xn] - 1)
                status = 2
                silence[xn] = 0
                count[xn] += 1
            elif dst1[n] < T1:
                status = 1
                count[xn] += 1
            else:
                status = 0
                count[xn] = 0
                x1[xn] = 0
                x2[xn] = 0
        if status == 2:
            if dst1[n] < T1:
                count[xn] += 1
            else:
                silence[xn] += 1
                if silence[xn] < maxsilence:
                    count[xn] += 1
                elif count[xn] < minlen:
                    status = 0
                    silence[xn] = 0
                    count[xn] = 0
                else:
                    status = 3
                    x2[xn] = x1[xn] + count[xn]
        if status == 3:
            status = 0
            xn += 1
            count[xn] = 0
            silence[xn] = 0
            x1[xn] = 0
            x2[xn] = 0
    el = len(x1[:xn])
    if x1[el - 1] == 0:
        el -= 1
    if x2[el - 1] == 0:
        print('Error: Not find ending point!\n')
        x2[el] = fn
    SF = np.zeros(fn)
    for i in range(el):
        SF[int(x1[i]):int(x2[i])] = 1
    voiceseg = findSegment(np.where(SF == 1)[0])
    return voiceseg

def findSegment(express):
    if express[0] == 0:
        voiceIndex = np.where(express)
    else:
        voiceIndex = express
    d_voice = np.where(np.diff(voiceIndex) > 1)[0]
    voiceseg = {}
    if len(d_voice) > 0:
        for i in range(len(d_voice) + 1):
            seg = {}
            if i == 0:
                st = voiceIndex[0]
                en = voiceIndex[d_voice[i]]
            elif i == len(d_voice):
                st = voiceIndex[d_voice[i - 1] + 1]
                en = voiceIndex[-1]
            else:
                st = voiceIndex[d_voice[i - 1] + 1]
                en = voiceIndex[d_voice[i]]
            seg['start'] = st
            seg['end'] = en
            seg['duration'] = en - st + 1
            voiceseg[i] = seg
    return voiceseg

def vad(wav):
    wav_input, sample_rate = librosa.load(wav, sr=SAMPLE_RATE)
    wav_input = wav_input - np.mean(wav_input)
    wav_input /= np.max(np.abs(wav_input))
    wnd = np.hamming(FRAME_LENGTH)
    voiceseg = vad_specEN(wav_input, wnd, HOP_LENGTH, NIS, THR1, THR2, sample_rate)
    vad_output = [[seg['start'] * HOP_LENGTH, seg['end'] * HOP_LENGTH] for seg in voiceseg.values()]
    return vad_output

if __name__ == '__main__':
    vad_output = vad(WAV)
    print(vad_output)
    with open(PREDICT_TXT_SAVE_PATH, "w") as f:
        for i in range(len(vad_output)):
            f.write(str(vad_output[i][0]))
            f.write(',')
            f.write(str(vad_output[i][1]))
            f.write('\n')
