#external package imports
import torch
import torchvision.transforms as T
from torchvision.io import read_image
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
from torch.utils.data import DataLoader
import matplotlib as plt

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
            nn.Linear(1250448, 120),                                   # THIRD LAYER: LINEAR YEAR, HIDDEN LAYER 2
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

        for _ in range(epochs):  # loop over the dataset multiple times

            self.train()
            running_loss = 0.0
            
            for i in range(len(train_dataloader)):
            # Reshapes inputs tensor to work in the NN
                inputs, labels = train_dataloader[i]     
                inputs = inputs.type(torch.float32)
                inputs.unsqueeze_(0) 
        
            # zero the parameter gradients
                self.optimizer.zero_grad()

        # forward + backward + optimize
                outputs = self(inputs)
                self.loss = self.criterion(outputs, labels)
                self.loss.backward()
                self.optimizer.step()

            # keep track of the loss
            running_loss += self.loss.item()

      # ALSO CALCULATE YOUR ACCURACY METRIC
      
        avg_train_loss = running_loss / (i + 1)
        # CALCULATE AVERAGE ACCURACY METRIC
        avg_train_loss = None
        train_losses.append(avg_train_loss)

        return train_losses, train_accuracies


    def evaluate_model(self, dataloader, type):
            
        # CAN USE FOR EITHER VAL OR TEST DATALOADER
        if type == "val":
            self.eval()
        if type == "test":
            self.test()

        #initialize lists
        losses = []
        accuracies = []

        running_loss = 0.0
        correctly_predicted_normal = 0
        total_normal = 0
    
        for i, data in enumerate(dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            if labels == 'NORMAL':
                total_normal += 1

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)

            # keep track of the loss
            running_loss += loss.item()
            
        # ALSO CALCULATE YOUR ACCURACY METRIC
        avg_train_loss = running_loss / (i + 1)     # i + 1 gives us the total number of batches in train dataloader

        # CALCULATE AVERAGE ACCURACY METRIC
        avg_train_acc = correctly_predicted_normal / total_normal

        losses.append(avg_train_loss)
        accuracies.append(avg_train_acc)

        return losses, accuracies


    def get_loss_graph(epochs, train_losses, test_losses):
                
        epochs_array = [i for i in range (1, epochs, epochs/10)]
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
        ax.plot(epochs_array, train_losses, color="orange", label="train losses", ls='dashed')
        ax.plot(epochs_array, test_losses, color="blue", label="test losses")
        ax.grid(alpha=0.25)
        ax.set_axis_on()
        ax.legend(loc="lower right", fontsize=16)
        ax.set_xlabel("epochs", fontsize=16)
        ax.set_ylabel("loss", fontsize=16)
        # reset consistent axis for comparison purposes
        plt.ylim([0, 1])
        plt.show()

    def get_accuracy_graph(epochs, train_accuracies, test_accuracies):
        
        epochs_array = [i for i in range (1, epochs, epochs/10)]
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
        ax.plot(epochs_array, train_accuracies, color="orange", label="train accuracies", ls='dashed')
        ax.plot(epochs_array, test_accuracies, color="blue", label="train accuracies")
        ax.grid(alpha=0.25)
        ax.set_axis_on()
        ax.legend(loc="lower right", fontsize=16)
        ax.set_xlabel("epochs", fontsize=16)
        ax.set_ylabel("loss", fontsize=16)
        # reset consistent axis for comparison purposes
        plt.ylim([0, 1])
        plt.show()


# 2: get dataloaders from the first checkpoint
training_data = CustomImageDataset("data/output/train.csv", "data/train/", transforms)
val_data = CustomImageDataset("data/output/val.csv", "data/val/", transforms)
test_data = CustomImageDataset("data/output/test.csv", "data/test/", transforms)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

