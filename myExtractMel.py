import glob
import os
import pickle
import numpy as np
from python_speech_features import logfbank, delta
from scipy.io import wavfile

def trans_label(x):
    if (x == 'ang'):
        return 0
    elif (x == 'sad'):
        return 1
    elif (x == 'hap'):
        return 2
    elif (x == 'neu'):
        return 3

def load_data(file_path):
    with open(file_path, 'rb') as f:
        mean0, std0, mean1, std1, mean2, std2 = pickle.load(f)
    return mean0, std0, mean1, std1, mean2, std2

def read_IEMOCAP():
    eps = 1e-5
    tnum = 259
    vnum = 298
    test_num = 420
    valid_num = 436
    train_num = 2928
    filter_num = 40
    hapnum = 434
    angnum = 433
    neunum = 1262
    sadnum = 799
    pernum = 300
    mean0, std0, mean1, std1, mean2, std2 = load_data('myzscore40.pkl')
    base_dir = 'C:\\Users\\ken\\Desktop\\SESSION'
    train_data = np.empty((train_num, 300, filter_num, 3), dtype=np.float32)
    valid_data = np.empty((valid_num, 300, filter_num, 3), dtype=np.float32)
    test_data = np.empty((test_num, 300, filter_num, 3), dtype=np.float32)
    train_label = np.empty(train_num, dtype=np.int8)
    valid_label = np.empty(valid_num, dtype=np.int8)
    test_label = np.empty(test_num, dtype=np.int8)
    v_label = np.empty(vnum, dtype=np.int8)
    t_label = np.empty(tnum, dtype=np.int8)
    vsts_num = np.empty(vnum, dtype=np.int8)
    tsts_num = np.empty(tnum, dtype=np.int8)
    vnum = 0
    tnum = 0
    train_num = 0
    valid_num = 0
    test_num = 0

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
                file_dir = os.path.join(wav_dir, ses, '*.wav')
                files = glob.glob(file_dir)
                for filename in files:
                    wavname = filename.split("\\")[-1][:-4]
                    emo = emo_map[wavname]
                    if (emo in ['hap', 'ang', 'neu', 'sad']):
                        rate, signal = wavfile.read(filename)
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
                                train_data[train_num, :, :, 0] = (part0 - mean0) / (std0 + eps)
                                train_data[train_num, :, :, 1] = (part1 - mean1) / (std1 + eps)
                                train_data[train_num, :, :, 2] = (part2 - mean2) / (std2 + eps)
                                train_label[train_num] = trans_label(emo)
                                train_num += 1
                            else:
                                if (emo in ['ang', 'neu', 'sad']):
                                    for i in range(2):
                                        if (i == 0):
                                            part0 = mel_spec[0:300, :]
                                            part1 = delta1[0:300, :]
                                            part2 = delta2[0:300, :]
                                        else:
                                            part0 = mel_spec[time - 300:time, :]
                                            part1 = delta1[time - 300:time, :]
                                            part2 = delta2[time - 300:time, :]
                                        train_data[train_num, :, :, 0] = (part0 - mean0) / (std0 + eps)
                                        train_data[train_num, :, :, 1] = (part1 - mean1) / (std1 + eps)
                                        train_data[train_num, :, :, 2] = (part2 - mean2) / (std2 + eps)
                                        train_label[train_num] = trans_label(emo)
                                        train_num += 1
                                else:
                                    steps = divmod(time - 300, 100)[0] + 1
                                    for i in range(steps):
                                        part0 = mel_spec[i * 100:300 + i * 100, :]
                                        part1 = delta1[i * 100:300 + i * 100, :]
                                        part2 = delta2[i * 100:300 + i * 100, :]
                                        train_data[train_num, :, :, 0] = (part0 - mean0) / (std0 + eps)
                                        train_data[train_num, :, :, 1] = (part1 - mean1) / (std1 + eps)
                                        train_data[train_num, :, :, 2] = (part2 - mean2) / (std2 + eps)
                                        train_label[train_num] = trans_label(emo)
                                        train_num += 1
                        elif (session == 'Session5'):
                            if (wavname[-4] != 'M'):
                                v_label[vnum] = trans_label(emo)
                                if (time <= 300):
                                    vsts_num[vnum] = 1
                                    vnum += 1
                                    pad_length = 300 - time
                                    part0 = mel_spec
                                    part1 = delta1
                                    part2 = delta2
                                    part0 = np.pad(part0, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
                                    part1 = np.pad(part1, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
                                    part2 = np.pad(part2, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
                                    valid_data[valid_num, :, :, 0] = (part0 - mean0) / (std0 + eps)
                                    valid_data[valid_num, :, :, 1] = (part1 - mean1) / (std1 + eps)
                                    valid_data[valid_num, :, :, 2] = (part2 - mean2) / (std2 + eps)
                                    valid_label[valid_num] = trans_label(emo)
                                    valid_num += 1
                                else:
                                    vsts_num[vnum] = 2
                                    vnum += 1
                                    for i in range(2):
                                        if (i == 0):
                                            part0 = mel_spec[0:300, :]
                                            part1 = delta1[0:300, :]
                                            part2 = delta2[0:300, :]
                                        else:
                                            part0 = mel_spec[time - 300:time, :]
                                            part1 = delta1[time - 300:time, :]
                                            part2 = delta2[time - 300:time, :]
                                        valid_data[valid_num, :, :, 0] = (part0 - mean0) / (std0 + eps)
                                        valid_data[valid_num, :, :, 1] = (part1 - mean1) / (std1 + eps)
                                        valid_data[valid_num, :, :, 2] = (part2 - mean2) / (std2 + eps)
                                        valid_label[valid_num] = trans_label(emo)
                                        valid_num += 1
                            else:
                                t_label[tnum] = trans_label(emo)
                                if (time <= 300):
                                    tsts_num[tnum] = 1
                                    tnum += 1
                                    pad_length = 300 - time
                                    part0 = mel_spec
                                    part1 = delta1
                                    part2 = delta2
                                    part0 = np.pad(part0, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
                                    part1 = np.pad(part1, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
                                    part2 = np.pad(part2, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
                                    test_data[test_num, :, :, 0] = (part0 - mean0) / (std0 + eps)
                                    test_data[test_num, :, :, 1] = (part1 - mean1) / (std1 + eps)
                                    test_data[test_num, :, :, 2] = (part2 - mean2) / (std2 + eps)
                                    test_label[test_num] = trans_label(emo)
                                    test_num += 1
                                else:
                                    tsts_num[tnum] = 2
                                    tnum += 1
                                    for i in range(2):
                                        if (i == 0):
                                            part0 = mel_spec[0:300, :]
                                            part1 = delta1[0:300, :]
                                            part2 = delta2[0:300, :]
                                        else:
                                            part0 = mel_spec[time - 300:time, :]
                                            part1 = delta1[time - 300:time, :]
                                            part2 = delta2[time - 300:time, :]
                                        test_data[test_num, :, :, 0] = (part0 - mean0) / (std0 + eps)
                                        test_data[test_num, :, :, 1] = (part1 - mean1) / (std1 + eps)
                                        test_data[test_num, :, :, 2] = (part2 - mean2) / (std2 + eps)
                                        test_label[test_num] = trans_label(emo)
                                        test_num += 1
    a = 0
    s = 0
    h = 0
    n = 0
    ang_idx = np.empty(angnum, dtype=int)
    sad_idx = np.empty(sadnum, dtype=int)
    hap_idx = np.empty(hapnum, dtype=int)
    neu_idx = np.empty(neunum, dtype=int)
    for i in range(train_num):
        if (train_label[i] == 0):
            ang_idx[a] = i
            a += 1
        elif (train_label[i] == 1):
            sad_idx[s] = i
            s += 1
        elif (train_label[i] == 2):
            hap_idx[h] = i
            h += 1
        elif (train_label[i] == 3):
            neu_idx[n] = i
            n += 1
    np.random.shuffle(ang_idx)
    np.random.shuffle(sad_idx)
    np.random.shuffle(hap_idx)
    np.random.shuffle(neu_idx)
    Train_data = np.empty((pernum * 4, 300, filter_num, 3), dtype=np.float32)
    Train_label = np.empty(pernum * 4, dtype=np.int8)
    for i in range(pernum):

        Train_data[i * 4, :, :, :] = train_data[ang_idx[i], :, :, :]
        Train_data[i * 4 + 1, :, :, :] = train_data[sad_idx[i], :, :, :]
        Train_data[i * 4 + 2, :, :, :] = train_data[hap_idx[i], :, :, :]
        Train_data[i * 4 + 3, :, :, :] = train_data[neu_idx[i], :, :, :]
        Train_label[i * 4] = 0
        Train_label[i * 4 + 1] = 1
        Train_label[i * 4 + 2] = 2
        Train_label[i * 4 + 3] = 3

    with open('myIEMOCAP.pkl', 'wb') as f:
        pickle.dump((Train_data, Train_label, valid_data, valid_label, test_data, test_label, v_label, t_label, vsts_num, tsts_num), f)

if __name__ == '__main__':
    read_IEMOCAP()