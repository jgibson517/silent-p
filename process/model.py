#external package imports
import torch
import torchvision.transforms as T
from torchvision.io import read_image
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score

# internal imports
import os
from .data_module import CustomImageDataset, transforms

# create neural net object
class CustomNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # inspire by Turing award winning LeCun, Bengio and Hinton's paper from 1998
        # https://ieeexplore.ieee.org/document/726791 (cited more than 25,000 times!!!!!!!!!)
        # code from https://blog.paperspace.com/writing-lenet5-from-scratch-in-python/ 
        self.LeNet = nn.Sequential(     
            # convolutional layers            
            nn.Sequential(                                            # FIRST LAYER: (INPUT LAYER)
              nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),    # CONVOLUTION 
              nn.BatchNorm2d(6),
              nn.ReLU(),
              nn.MaxPool2d(kernel_size = 2, stride = 2)),             # POOLING
            nn.Sequential(                                            # SECOND LAYER: HIDDEN LAYER 1
              nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),   # CONVOLUTION 
              nn.BatchNorm2d(16),
              nn.ReLU(),
              nn.MaxPool2d(kernel_size = 2, stride = 2)),             # POOLING
            # fully connected layers
            nn.Flatten(),
            nn.Linear(250000, 120),                                   # THIRD LAYER: LINEAR YEAR, HIDDEN LAYER 2
            nn.ReLU(),                                                # HIDDEN LAYER's ACTIVATION FUNCION
            nn.Linear(120, 84),                                       # FOURTH LAYER: LINEAR YEAR, HIDDEN LAYER 3
            nn.ReLU(),                                                # HIDDEN LAYER's ACTIVATION FUNCION
            # output layer
            nn.Linear(84, 2)                                          # OUTPUT LAYER
        )


        # Jack: Math for the input size for first nn.Linear
        # Image features
        # Before forward: 1277760 (968 * 1320; height x width) 
        # First Convultion layer: adds 625176 (1902936)
        # Second layer: pools new features together: removes 652488 features (1250448)
        # Final Total before linear tranformation: 1250448 
            # layers: [0, 16, 239, 327] - same as our calculation 

        # 3: Define a Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x):
        out = self.LeNet(x)
        return out
    
    def train_model(self, train_dataloader, epochs=50):

        # Initalize NN model; could also go outside the function 
        #model = CustomNeuralNetwork()
       
        train_losses = []
        train_accuracies = []
        train_recall = []

        tot_pred = torch.empty(0)


        for epoch in range(epochs):  # loop over the dataset multiple times
            print('Epoch:', epoch)
            self.train()
            running_loss = 0.0
            total_correct = 0
            total_samples = 0

            tot_pred = torch.empty(0)
            all_labels = torch.empty(0)

            for i, data in enumerate(train_dataloader):
            # Reshapes inputs tensor to work in the NN
                inputs, labels = data     
                inputs = inputs.type(torch.float32)

    
            # zero the parameter gradients
                self.optimizer.zero_grad()

             # forward + backward + optimize
                outputs = self(inputs)
                self.loss = self.criterion(outputs, labels)
                self.loss.backward()
                self.optimizer.step()

                 # counts for acccuracy score
                _, predicted = torch.max(outputs.data, 1)
                # Concatate batch of prediction and labels togehter
                tot_pred = torch.cat((tot_pred, predicted))
                all_labels = torch.cat((all_labels, labels))

            # keep track of the loss
            running_loss += self.loss.item()
            
            # Calculate baseline accuracy: Correct/Total 
            total_correct = (tot_pred == all_labels).sum().item()
            total_samples = all_labels.size(0)
      
            avg_train_loss = running_loss / (i + 1)
            # CALCULATE AVERAGE ACCURACY METRIC
            avg_train_acc = total_correct / total_samples

            train_losses.append(avg_train_loss)
            train_accuracies.append(avg_train_acc)
            # May want to change poss label to zero to measure the Normal predictions
            train_recall.append(recall_score(tot_pred, all_labels, pos_label=1))

        return train_losses, train_accuracies, train_recall


    def evaluate_model(self, dataloader):

        # tills model not to track gradients
        self.eval()

        tot_pred = torch.empty(0)
        all_labels = torch.empty(0)

        #  for epoch in range(epochs): we need accuracy scores for every epoch the test data?

        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.type(torch.float32)
            
            with torch.no_grad():
                outputs = self(inputs)
               
            _, predicted = torch.max(outputs.data, 1)
            # Concatate batch of prediction and labels togehter
            tot_pred = torch.cat((tot_pred, predicted))
            all_labels = torch.cat((all_labels, labels))

        total_correct = (tot_pred == all_labels).sum().item()
        total_samples = all_labels.size(0)

        test_acc = total_correct / total_samples
        # Same note as above 
        test_recall = recall_score(tot_pred, all_labels, pos_label=1)

        return test_acc, test_recall


    def get_loss_graph(self, epochs, train_losses, test_losses=None):
                
        epochs_array = [i for i in range (0, epochs)]
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
        ax.plot(epochs_array, train_losses, color="orange", label="train losses", ls='dashed')
        #ax.plot(epochs_array, test_losses, color="blue", label="test losses")
        ax.grid(alpha=0.25)
        ax.set_axis_on()
        ax.legend(loc="lower right", fontsize=16)
        ax.set_xlabel("epochs", fontsize=16)
        ax.set_ylabel("loss", fontsize=16)
        # reset consistent axis for comparison purposes
        #plt.ylim([0, 1])
        plt.show()

    def get_accuracy_graph(self, epochs, train_accuracies, test_accuracies=None):
        
        epochs_array = [i for i in range (0, epochs)]
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
        ax.plot(epochs_array, train_accuracies, color="orange", label="train_accuracies", ls='dashed')
        #ax.plot(epochs_array, test_accuracies, color="blue", label="test_accuracies")
        ax.grid(alpha=0.25)
        ax.set_axis_on()
        ax.legend(loc="lower right", fontsize=16)
        ax.set_xlabel("epochs", fontsize=16)
        ax.set_ylabel("loss", fontsize=16)
        # reset consistent axis for comparison purposes
        #plt.ylim([0, 1])
        plt.show()

# 2: get dataloaders from the first checkpoint
training_data = CustomImageDataset("data/output/train.csv", "data/train/", transforms)
val_data = CustomImageDataset("data/output/val.csv", "data/val/", transforms)
test_data = CustomImageDataset("data/output/test.csv", "data/test/", transforms)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

