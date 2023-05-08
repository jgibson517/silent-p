#external package imports
import torch
import torchvision.transforms as T
from torchvision.io import read_image
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
from torch.utils.data import DataLoader

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

        # 3: Define a Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x):
        # after writing the below out, I don't think it's actually needed

        # for layers in self.LeNet:
        #     # run x through each layer of the neural net as laid out above

        #     #if layer has multiple components, then go through each (not sure if this is needed)
        #     if len(layers) > 1:
        #         for layer in layers:
        #             x = layer(x)
            
        #     # if just one transformation, then reassign x to that new value
        #     x = layers(x)

        #     return x

        out = self.LeNet(x)
        return out
    
    def train(self, model, train_dataloader, epochs=50):

        # Initalize NN model; could also go outside the function 
        model = CustomNeuralNetwork()
       
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuarcies = []

        for _ in range(epochs):  # loop over the dataset multiple times

            model.train()
            running_loss = 0.0
            model.train()
            running_loss = 0.0
            
            for i in range(len(train_dataloader)):
            # Reshapes inputs tensor to work in the NN
                inputs, labels = train_dataloader[i]     
                inputs = inputs.type(torch.float32)
                inputs.unsqueeze_(0) 
        
            # zero the parameter gradients
                self.optimizer.zero_grad()

        # forward + backward + optimize
                outputs = model(inputs)
                self.loss = self.criterion(outputs, labels)
                self.loss.backward()
                self.optimizer.step()

            # keep track of the loss
            running_loss += self.loss.item()

      # ALSO CALCULATE YOUR ACCURACY METRIC
      
        avg_train_loss = running_loss / (i + 1)     # i + 1 gives us the total number of batches in train dataloader
        # CALCULATE AVERAGE ACCURACY METRIC
        avg_train_loss = None
        train_losses.append(avg_train_loss)


    def train(self, test_dataloader):
        # FOR TESTING YOU DON'T HAVE TO ITERATE OVER MULTIPLE EPOCHS
        # JUST ONE PASS OVER THE TEST DATALOADER!
        pass

    def analyze():
        pass
        
        # 6: ANAYLZE (i.e. 3RD OBJECTIVE)

        # YOU CAN MAKE GRAPHS of TRAIN AND VAL LOSSES OVER EPOCHS, etc!
        # YOU CAN ALSO DO MULTIPLE TRAININGS, CHOOSING A DIFFERENT LOSS FUNCTION
        # FOR EACH TRAINING RUN, AND THEN YOU COULD COMPARE HOW WHICH LOSS FUNCTION
        # LEADS TO THE BEST LOSSES OR BEST ACCURACIES

        # ALSO YOU COULD TRAIN USING DIFFERENT OPTIMIZERS!

        # SO MUCH YOU COULD DO!


# 2: get dataloaders from the first checkpoint
training_data = CustomImageDataset("data/output/train.csv", "data/train/", transforms)
val_data = CustomImageDataset("data/output/val.csv", "data/val/", transforms)
test_data = CustomImageDataset("data/output/test.csv", "data/test/", transforms)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

