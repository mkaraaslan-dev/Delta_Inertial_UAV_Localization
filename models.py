
import torch
import torch.nn as nn
import torch.nn.functional as F



input_size = 16
output_size = 3
hidden_size = 256
num_layers = 2
dropout = 0.2   
learning_rate = 0.001
sequence_lenght = 1
batch_size = 64


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Yeni verileri standardize etme

# DNN modelini oluştur
class DNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,dropout):
        super(DNNModel, self).__init__()
        self.dropout = nn.Dropout(dropout)  
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x = self.dropout(x)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# GRU modelini oluştur
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,output_size,dropout):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # self.gru = nn.RNN(input_size, hidden_size, num_layers,batch_first=True,dropout=dropout)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,batch_first=True)
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.rnn(x, h0)
        
        out = self.fc(out[:, -1, :])
        return out


# LSTM modelini oluştur
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers, output_size,dropout):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers,batch_first=True)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,batch_first=True,dropout=dropout)
 
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # for lstm. we need an initial tensor for the cell state.

        out, _ = self.lstm(x, (h0,c0))
        out = self.fc(out[:, -1, :])
        return out
# BiLSTM modelini oluştur
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers, output_size,dropout):
        super(BiLSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers,batch_first=True, bidirectional=True,dropout=dropout)
        # self.bilstm = nn.LSTM(input_size, hidden_size, num_layers,batch_first=True, bidirectional=True)
        
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):

        out, _ = self.bilstm(x)
        out = self.fc(out[:, -1, :])
        return out
    
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,output_size,dropout):
        super(GRUModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers,batch_first=True,dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.gru(x, h0)
        
        out = self.fc(out[:, -1, :])
        return out

class AHLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout,attention_size=256):
        super(AHLSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        
        # İlk LSTM katmanı
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=dropout)
        # self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        
        # İkinci LSTM katmanı
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        
        # Attention mekanizması için katman
        self.attention = nn.Linear(hidden_size, attention_size)
        
        # Fully connected katman
        self.fc = nn.Linear(attention_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # İlk LSTM katmanı
        out1, _ = self.lstm1(x, (h0, c0))
        
        # İkinci LSTM katmanı
        out2, _ = self.lstm2(out1)
        
        # Attention mekanizması
        attention_weights = F.softmax(self.attention(out2), dim=1)
        attention_output = torch.sum(attention_weights * out2, dim=1)
        
        # Fully connected katman
        out = self.fc(attention_output)
        
        return out