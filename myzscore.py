import os
import pickle
import numpy as np
from python_speech_features import logfbank, delta
from scipy.io import wavfile

def read_IEMOCAP():
    train_num = 2928
    filter_num = 40
    base_dir = 'C:\\Users\\ken\\Desktop\\SESSION'
    train_data0 = np.empty((train_num * 300, filter_num))
    train_data1 = np.empty((train_num * 300, filter_num))
    train_data2 = np.empty((train_num * 300, filter_num))
    train_num = 0
    for session in os.listdir(base_dir):
        wav_dir = os.path.join(base_dir, session, 'sentences\\wav')
        emo_dir = os.path.join(base_dir, session, 'dialog\\EmoEvaluation')
        for ses in os.listdir(wav_dir):
            if (ses[7] == 'i'):
                emo_map = {}
                emo_txt = emo_dir + '\\' + ses + '.txt'
                with open(emo_txt, 'r') as file:
                    for line in file:
                        line = line.strip()
                        if line.startswith('['):
                            parts = line.split()  # 使用空格分割
                            key = parts[3]
                            value = parts[4]
                            emo_map[key] = value
                wav_dlg = os.path.join(wav_dir, ses)
                for sts in os.listdir(wav_dlg):
                    if (sts[-3:] != 'wav'):
                        break
                    sts_nm = sts[:-4]
                    emo = emo_map[sts_nm]
                    sts_cplt = os.path.join(wav_dlg, sts)
                    if (emo in ['hap', 'ang', 'neu', 'sad']):
                        rate, signal = wavfile.read(sts_cplt)
                        mel_spec = logfbank(signal, rate, nfilt=filter_num)
                        delta1 = delta(mel_spec, 2)
                        delta2 = delta(delta1, 2)
                        time = mel_spec.shape[0]
                        if (session in ['Session1', 'Session2', 'Session3', 'Session4']):
                            if (time <= 300):
                                pad_length = 300 - time
                                part0 = mel_spec
                                part1 = delta1
                                part2 = delta2
                                part0 = np.pad(part0, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
                                part1 = np.pad(part1, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
                                part2 = np.pad(part2, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
                                train_data0[train_num * 300:(train_num + 1) * 300] = part0
                                train_data1[train_num * 300:(train_num + 1) * 300] = part1
                                train_data2[train_num * 300:(train_num + 1) * 300] = part2
                                train_num += 1
                            else:
                                if (emo == 'hap'):
                                    steps = (time // 100) - 2
                                    for i in range(steps):
                                        part0 = mel_spec[i * 100:300 + i * 100, :]
                                        part1 = delta1[i * 100:300 + i * 100, :]
                                        part2 = delta2[i * 100:300 + i * 100, :]
                                        train_data0[train_num * 300:(train_num + 1) * 300] = part0
                                        train_data1[train_num * 300:(train_num + 1) * 300] = part1
                                        train_data2[train_num * 300:(train_num + 1) * 300] = part2
                                        train_num += 1
                                else:
                                    for i in range(2):
                                        if (i == 0):
                                            part0 = mel_spec[0:300, :]
                                            part1 = delta1[0:300, :]
                                            part2 = delta2[0:300, :]
                                            train_data0[train_num * 300:(train_num + 1) * 300] = part0
                                            train_data1[train_num * 300:(train_num + 1) * 300] = part1
                                            train_data2[train_num * 300:(train_num + 1) * 300] = part2
                                            train_num += 1
                                        else:
                                            part0 = mel_spec[time - 300:time, :]
                                            part1 = delta1[time - 300:time, :]
                                            part2 = delta2[time - 300:time, :]
                                            train_data0[train_num * 300:(train_num + 1) * 300] = part0
                                            train_data1[train_num * 300:(train_num + 1) * 300] = part1
                                            train_data2[train_num * 300:(train_num + 1) * 300] = part2
                                            train_num += 1
    mean0 = np.mean(train_data0, axis=0)
    std0 = np.std(train_data0, axis=0)
    mean1 = np.mean(train_data1, axis=0)
    std1 = np.std(train_data1, axis=0)
    mean2 = np.mean(train_data2, axis=0)
    std2 = np.std(train_data2, axis=0)

    with open('myzscore40.pkl', 'wb') as f:
        pickle.dump((mean0, std0, mean1, std1, mean2, std2), f)

if __name__ == '__main__':
    read_IEMOCAP()