import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import recall_score, confusion_matrix

from acrnn import acrnn

def load_data(file_path):
    with open(file_path, 'rb') as f:
        Train_data_s, Train_label_s, valid_data, valid_label, test_data, test_label, v_label, t_label, vsts_num, tsts_num = pickle.load(f)
    return Train_data_s, Train_label_s, valid_data, valid_label, test_data, test_label, v_label, t_label, vsts_num, tsts_num

def init_weights(m):
    if type(m) == torch.nn.Linear:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0.1)
    elif type(m) == torch.nn.Conv2d:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0.1)

def train():
    Train_data_s, Train_label_s, valid_data, valid_label, test_data, test_label, v_label, t_label, vsts_num, tsts_num = load_data('myIEMOCAP.pkl')
    Train_data_s = Train_data_s.transpose((0, 3, 1, 2))
    test_data = test_data.transpose((0, 3, 1, 2))
    valid_data = valid_data.transpose((0, 3, 1, 2))
    model = acrnn()
    model.apply(init_weights)
    criterion = nn.CrossEntropyLoss()

    device = 'cuda'
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)

    Train_data_s = torch.from_numpy(Train_data_s)
    Train_label_s = torch.from_numpy(Train_label_s)

    Train_data_s = Train_data_s.to(device)
    Train_label_s = Train_label_s.to(device)
    clip = 0
    num_epochs = 250
    batch_size = 60
    vnum = 298
    train_batches = Train_data_s.shape[0] // batch_size
    valid_batches = valid_data.shape[0] // batch_size
    valid_pred = np.empty((valid_data.shape[0], 4))
    v_pred = np.empty((vnum, 4))
    v_predfn = np.empty(vnum, dtype=np.int8)
    best_acc = 0

    for epoch in range(num_epochs):
        model.train()
        idx = np.arange(len(Train_label_s))
        np.random.shuffle(idx)
        for i in range(train_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            inputs = torch.tensor(Train_data_s[idx[start_idx:end_idx]]).to(device)
            labels = torch.tensor(Train_label_s[idx[start_idx:end_idx]], dtype=torch.long).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        model.eval()
        for i in range(valid_batches + 1):
            v_start = i * batch_size
            if (i == valid_batches): v_end = valid_data.shape[0]
            else: v_end = (i + 1) * batch_size
            with torch.no_grad():
                inputs = torch.tensor(valid_data[v_start:v_end]).to(device)
                labels = torch.tensor(valid_label[v_start:v_end], dtype=torch.long).to(device)
                outputs = model(inputs)
                valid_pred[v_start:v_end, :] = outputs.cpu().detach().numpy()
                valid_sumloss = criterion(outputs, labels).cpu().detach().numpy()
        valid_loss = valid_sumloss / valid_data.shape[0]
        v = 0
        for i in range(vnum):
            v_pred[i, :] = np.max(valid_pred[v:v + vsts_num[i], :], 0)
            v += vsts_num[i]
        v_predfn = np.argmax(v_pred, 1)
        v_acc = recall_score(v_label, v_predfn, average='macro')
        v_con = confusion_matrix(v_label, v_predfn)

        if (v_acc > best_acc):
            best_acc = v_acc
            best_con = v_con
            torch.save(model.state_dict(), 'model_weights.pth')

        print(f"Epoch {epoch + 1}")
        print(f"Validation Loss: {valid_loss:.4f}")
        print(f"Validation Accuracy: {v_acc:.2f}")
        print(f"Confusion matrix")
        print(v_con)
        print(f"Best Accuracy: {best_acc:.2f}")
        print(f"Best matrix")
        print(best_con)

if __name__ == '__main__':
    train()