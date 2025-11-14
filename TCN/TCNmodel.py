import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


class TemporalBlock(nn.Module):
    def __init__(self, nInputs, nOutputs, kernelSize, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(nInputs, nOutputs, kernelSize, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(nOutputs, nOutputs, kernelSize, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(nInputs, nOutputs, 1) if nInputs != nOutputs else None
        self.relu = nn.ReLU() # activation function relu maybe leaky relu or prelu can be used
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    def __init__(self, chompSize):
        super(Chomp1d, self).__init__()
        self.chompSize = chompSize
        
    def forward(self, x):
        return x[:, :, :-self.chompSize].contiguous()

class TemporalConvolutionalNetwork(nn.Module):
    def __init__(self, numInputs, numChannels, kernelSize=2, dropout=0.2):
        super(TemporalConvolutionalNetwork, self).__init__()
        
        layers = []
        numLevels = len(numChannels)
        
        for i in range(numLevels):
            dilationSize = 2 ** i
            inChannels = numInputs if i == 0 else numChannels[i-1]
            outChannels = numChannels[i]
            layers.append(TemporalBlock(inChannels, outChannels, kernelSize, stride=1, dilation=dilationSize, padding=(kernelSize-1) * dilationSize, dropout=dropout))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class TCNModel(nn.Module):
    def __init__(self, inputSize, outputSize, numChannels, kernelSize=2, dropout=0.2):
        super(TCNModel, self).__init__()
        
        self.tcn = TemporalConvolutionalNetwork(inputSize, numChannels, kernelSize, dropout)
        self.linear = nn.Linear(numChannels[-1], outputSize)
        
    def forward(self, x):
        y = self.tcn(x)
        return self.linear(y[:, :, -1])

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seqLength, predictionHorizon=1):
        self.data = data
        self.seqLength = seqLength
        self.predictionHorizon = predictionHorizon
        
    def __len__(self):
        return len(self.data) - self.seqLength - self.predictionHorizon + 1
    
    def __getitem__(self, idx):
        xSeq = self.data[idx:idx + self.seqLength]
        yTarget = self.data[idx + self.seqLength + self.predictionHorizon - 1]
        
        return torch.FloatTensor(xSeq).transpose(0, 1), torch.FloatTensor(yTarget)

class TCNPipeline:
    def __init__(self, inputSize=1, outputSize=1, numChannels=[25, 25, 25, 25], kernelSize=7, dropout=0.2, learningRate=0.001, seqLength=100, batchSize=32, device=None):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.numChannels = numChannels
        self.kernelSize = kernelSize
        self.dropout = dropout
        self.learningRate = learningRate
        self.seqLength = seqLength
        self.batchSize = batchSize
        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = TCNModel(inputSize, outputSize, numChannels, kernelSize, dropout).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learningRate)
        self.criterion = nn.MSELoss()
        self.scaler = StandardScaler()
        
        self.trainLosses = []
        self.valLosses = []
        
    def prepareData(self, timeseries, trainRatio=0.7, predictionHorizon=1):
        normalized = self.scaler.fit_transform(timeseries)
        
        splitIdx = int(len(normalized) * trainRatio)
        trainData = normalized[:splitIdx]
        testData = normalized[splitIdx:]
        
        trainDataset = TimeSeriesDataset(trainData, self.seqLength, predictionHorizon)
        testDataset = TimeSeriesDataset(testData, self.seqLength, predictionHorizon)
        
        trainLoader = DataLoader(trainDataset, batch_size=self.batchSize, shuffle=True)
        testLoader = DataLoader(testDataset, batch_size=self.batchSize, shuffle=False)
        
        return trainLoader, testLoader, trainData, testData
    
    def trainModel(self, trainLoader, valLoader, epochs=50, patience=10):
        bestValLoss = float('inf')
        patienceCounter = 0
        
        for epoch in range(epochs):
            self.model.train()
            trainLoss = 0.0
            
            for xBatch, yBatch in trainLoader:
                xBatch = xBatch.to(self.device)
                yBatch = yBatch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(xBatch)
                loss = self.criterion(outputs, yBatch)
                loss.backward()
                self.optimizer.step()
                
                trainLoss += loss.item()
            
            avgTrainLoss = trainLoss / len(trainLoader)
            self.trainLosses.append(avgTrainLoss)
            
            self.model.eval()
            valLoss = 0.0
            
            with torch.no_grad():
                for xBatch, yBatch in valLoader:
                    xBatch = xBatch.to(self.device)
                    yBatch = yBatch.to(self.device)
                    
                    outputs = self.model(xBatch)
                    loss = self.criterion(outputs, yBatch)
                    valLoss += loss.item()
            
            avgValLoss = valLoss / len(valLoader)
            self.valLosses.append(avgValLoss)
            
            if avgValLoss < bestValLoss:
                bestValLoss = avgValLoss
                patienceCounter = 0
                torch.save(self.model.state_dict(), 'best_tcn_model.pth')
            else:
                patienceCounter += 1
            
            if patienceCounter >= patience:
                break
        
        self.model.load_state_dict(torch.load('best_tcn_model.pth'))
        return self
    
    def predict(self, dataLoader):
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for xBatch, yBatch in dataLoader:
                xBatch = xBatch.to(self.device)
                outputs = self.model(xBatch)
                predictions.append(outputs.cpu().numpy())
                targets.append(yBatch.numpy())
        
        predictions = np.vstack(predictions)
        targets = np.vstack(targets)
        
        return predictions, targets
    
    def evaluateModel(self, testLoader):
        predictions, targets = self.predict(testLoader)
        
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        nrmse = rmse / (np.max(targets) - np.min(targets))
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'nrmse': nrmse
        }
        
        return predictions, targets, metrics
    
    def exportMetrics(self, metrics, filename='tcn_metrics.csv'):
        df = pd.DataFrame([metrics])
        df.to_csv(filename, index=False)
        return df
    
    def run(self, timeseries, trainRatio=0.7, epochs=50, patience=10):
        trainLoader, testLoader, trainData, testData = self.prepareData(timeseries, trainRatio)
        
        self.trainModel(trainLoader, testLoader, epochs, patience)
        
        predictions, targets, metrics = self.evaluateModel(testLoader)
        
        results = {
            'predictions': predictions,
            'targets': targets,
            'metrics': metrics,
            'trainLosses': self.trainLosses,
            'valLosses': self.valLosses
        }
        
        return results