import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit, TimeSeriesSplit
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tqdm.auto import tqdm, trange
from pathlib import Path

path = Path(r'F:/jaeho/SLCL/과제/M3DT/study/Biosignals/EMG-Torch')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Dataset
    #Class 0 : Rock / Class 1 : Scissors / Class 2 : Paper / Class 3 : Okay gestures
    #Each Gesture was recorded 6 times for 20s
    #In total 120s of each gestures being held in fixed position
    #Each dataset line has 8 consecutive readings of all 8 sensors. => 64 columns of EMG data, last column is a gesture class
    #Data was recorded 200Hz
    #Column's order : Sensor1_time1 / Sensor2_time1 / ... / Sensor8_time1 / Sensor1_time2 / Sensor2_time2 / ... / Sensor8_time2 / ... / Sensor8_time1 / Sensor8_time2 / ... / Sensor8_time8
df0 = pd.read_csv(path / '0.csv', header=None)
print("Class 0 Shape", df0.shape)
df1 = pd.read_csv(path / '1.csv', header=None)
print("Class 1 Shape", df1.shape)
df2 = pd.read_csv(path / '2.csv', header=None)
print("Class 2 Shape", df2.shape)
df3 = pd.read_csv(path / '3.csv', header=None)
print("Class 3 Shape", df3.shape)
df = pd.concat([df0, df1, df2, df3])
data = df.values
sc = MinMaxScaler(feature_range = (0,1)) # 데이터를 0-1 사이의 값으로 변환
print("Data Shape", data.shape)

# 데이터 시각화
def plot_data(data):
    
    X0, X1, X2, X3=[],[],[],[]
    data[:,:-1] = sc.fit_transform(data[:,:-1]) # 데이터 정규화
    
    for i in range(data.shape[0]): # 데이터 재구조화
        tmp = data[i,:-1].reshape((8,8))
        for j in range(8):
            
            if data[i,-1] == 0:
                X0.append(tmp[j,:])
            
            elif data[i,-1] == 1:
                X1.append(tmp[j,:])
            
            elif data[i,-1] == 2:
                X2.append(tmp[j,:])
            elif data[i,-1] == 3:
                X3.append(tmp[j,:])
    
    X0, X1, X2, X3 = np.array(X0), np.array(X1), np.array(X2), np.array(X3) # 데이터 분류
        
    fig, axes = plt.subplots(4,8, figsize=(30, 8), sharex=True, sharey=True) # 데이터 시각화
    for i in range(8):
        axes[0][i].plot(X0[:,i], label='Raw Ch '+str(i))
        axes[1][i].plot(X1[:,i], label='Raw Ch '+str(i))
        axes[2][i].plot(X2[:,i], label='Raw Ch '+str(i))
        axes[3][i].plot(X3[:,i], label='Raw Ch '+str(i))
    #plt.show()
plot_data(data)

#ClickNet : LSTM 기반 신경망 클래스
class ClickNet(nn.Module):
    
    def __init__(self, n_features, n_hidden, n_sequence, n_layers, n_classes): # 하이퍼파라미터 초기화
        super(ClickNet, self).__init__()
        
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_sequence = n_sequence
        self.n_layers = n_layers
        self.n_classes = n_classes
        
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_hidden, num_layers=n_layers, batch_first=True)
        
        self.linear_1 = nn.Linear(in_features=n_hidden, out_features=128) # 128개의 유닛으로 출력
        self.dropout_1 = nn.Dropout(p=0.2) # 드롭아웃 비율을 20%로 초기화
        self.linear_2 = nn.Linear(in_features=128, out_features=n_classes) # 128개의 유닛에서 n_Classes개의 클래스로 출력
        
    
    def forward(self, x): # forward propagation method
        
        #self.hidden 초기화
        self.hidden = (
            torch.zeros(self.n_layers, x.shape[0], self.n_hidden).to(device),
            torch.zeros(self.n_layers, x.shape[0], self.n_hidden).to(device)
        )
    
        out, (hs, cs) = self.lstm(x.view(len(x), self.n_sequence, -1),self.hidden)
        out = out[:,-1,:] # LSTM의 출력 중 마지막 시퀀스의 출력만 사용됨 => 각 시퀀스에 대한 최종 결과만 다음 레이어로 전달
        out = self.linear_1(out)
        out = self.dropout_1(out)
        out = self.linear_2(out) # 최종 출력은 2nd Fully Connected Layer를 통과하여 생성됨
        
        return out
    
# Train Model
def train_model(model, train_dataloader, n_epochs):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    
    for epoch in tqdm(range(n_epochs)):
        for i, (X_train, y_train) in enumerate(train_dataloader):
            y_hat = model(X_train)
            loss = loss_fn(y_hat.float(), y_train)
            
            # if i == 0 and (epoch+1)%10==0:
            #     print(f'Epoch {epoch+1} train loss: {loss.item()}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # 모델 파라미터 갱신
        
    return model

# one-hot encoding
def one_hot(a):
    a=a.astype(int)
    b = np.zeros(a.size, a.max()+1)
    b[np.arange(a.size),a] = 1
    return b

# Data Preparing
def prepare_data(data):
    data[:,:-1] = sc.fit_transform(data[:,:-1])
    #np.random.shuffle(data)
    X, y = data[:,:-1], data[:,-1]
    X = X.reshape(-1,8,8)
    
    # Random Shuffle 말고 Stratisfied Shuffle Split 방식을 적용해봄
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_index, test_index in sss.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)
    
    return X_train.float().to(device), y_train.long().to(device), X_test.float().to(device), y_test.long().to(device)

# def prepare_data_2(): # 각 class data를 따로 읽은 다음 total dataset을 shuffle하고 normalization함. 그리고 직접 data split
    rock_dataset = pd.read_csv(path / "0.csv", header=None) # class = 0
    scissors_dataset = pd.read_csv(path / "1.csv", header=None) # class = 1
    paper_dataset = pd.read_csv(path / "2.csv", header=None) # class = 2
    ok_dataset = pd.read_csv(path / "3.csv", header=None) # class = 3
    frames = [rock_dataset, scissors_dataset, paper_dataset, ok_dataset]
    dataset = pd.concat(frames)
    dataset_train = dataset.iloc[np.random.permutation(len(dataset))]
    dataset_train.reset_index(drop=True)
    X_train = []
    y_train = []
    for i in range(0, dataset_train.shape[0]):
        row = np.array(dataset_train.iloc[i:1+i, 0:64].values)
        X_train.append(np.reshape(row, (64, 1)))
        y_train.append(np.array(dataset_train.iloc[i:1+i, -1:])[0][0])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Reshape to one flatten vector
    X_train = X_train.reshape(X_train.shape[0]*X_train.shape[1], 1)
    X_train = sc.fit_transform(X_train)
    # Reshape again after normalization to (-1, 8, 8)
    X_train = X_train.reshape((-1, 8, 8))
    # Convert to one hot
    y_train = np.eye(np.max(y_train) + 1)[y_train]
    # Splitting Train/Test
    X_test = torch.from_numpy(X_train[7700:])
    y_test = torch.from_numpy(y_train[7700:])
    
    X_train = torch.from_numpy(X_train[0:7700])
    y_train = torch.from_numpy(y_train[0:7700])
    
    return X_train.float().to(device), y_train.float().to(device), X_test.float().to(device), y_test.float().to(device)

#
class EmgDataset(Dataset):
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.X.shape[0]
    
n_features=8
n_sequence=8
n_hidden=64
n_layers=1
n_classes=4
n_epochs = 500
n_batch_size = 256
model = ClickNet(n_features, n_hidden, n_sequence, n_layers, n_classes).to(device)
X_train, y_train, X_test, y_test = prepare_data(data)

# X_train, y_train, X_test, y_test = prepare_data_2()
print("Train Data Shape ",X_train.shape, y_train.shape)
print("Test Data Shape ",X_test.shape, y_test.shape)
train_dataset = EmgDataset(X_train, y_train)
train_dataloader = DataLoader(dataset = train_dataset, batch_size=n_batch_size, shuffle=False)
model = train_model(model, train_dataloader, n_epochs = n_epochs)

#Evaluation Model
def evaluateModel(prediction, y):
    prediction = torch.argmax(prediction, dim=1)
#     y = torch.argmax(y, dim=1)
    good = 0
    for i in range(len(y)):
        if (prediction[i] == y[i]):
            good = good +1
    return (good/len(y)) * 100.0
with torch.no_grad():
    y_hat_train = model(X_train)
    print("Train Accuracy ", evaluateModel(y_hat_train, y_train))
    y_hat_test = model(X_test)
    print("Test Accuracy ", evaluateModel(y_hat_test, y_test))
    
    precision_train = precision_score(y_train, torch.argmax(y_hat_train, dim=1).reshape(len(y_hat_train),1), average=None)
    precision_test = precision_score(y_test, torch.argmax(y_hat_test, dim=1).reshape(len(y_hat_test),1), average=None)
    
    for i, p in enumerate(precision_train):
        print(f"Class {i} Train Precision: {p:.2f}")
    
    for i, p in enumerate(precision_test):
        print(f"Class {i} Test Precision: {p:.2f}")
        
# def evalModel(label, pred):
#     pred = torch.argmax(, dim=1)
#     accuracy = accuracy_score(label, pred)
#     precision = precision_score(label, pred, average='macro')
#     recall = recall_score(label, pred, average='macro')
#     f1 = f1_score(label, pred, average='macro')