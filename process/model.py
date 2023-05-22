#external package imports
import torch
import torchvision.transforms as T
from torchvision.io import read_image
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, confusion_matrix

# internal imports
import os
from .data_module import CustomImageDataset
from .transforms import base_transforms, color_transforms, edges_transforms, both_transforms

# create neural net object
class CustomNeuralNetwork(nn.Module):
    def __init__(self, eta):
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
            nn.Linear(59536, 120),                                   # THIRD LAYER: LINEAR YEAR, HIDDEN LAYER 2
            nn.ReLU(),                                                # HIDDEN LAYER's ACTIVATION FUNCION
            nn.Linear(120, 84),                                       # FOURTH LAYER: LINEAR YEAR, HIDDEN LAYER 3
            nn.ReLU(),                                               # HIDDEN LAYER's ACTIVATION FUNCION
            # output layer
            nn.Linear(84, 2)                                          # OUTPUT LAYER
        )


        # Jack: Math for the input size for first nn.Linear
        # Image features
        # Before forward: 1277760 (968 * 1320; height x width) 
        # First Convolution layer: adds 625176 (1902936)
        # Second layer: pools new features together: removes 652488 features (1250448)
        # Final Total before linear tranformation: 1250448 
            # layers: [0, 16, 239, 327] - same as our calculation

        # 3: Define a Loss function a nd optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=eta, momentum=0.9)

    def forward(self, x):
        out = self.LeNet(x)
        return out
    
    def train_model(self, train_dataloader, val_dataloader, epochs=15, epoch_step=1):
       
        train_losses = []
        train_accuracies = []
        train_recalls = []

        val_losses = []
        val_accuracies = []
        val_recalls = []

        for epoch in range(epochs):  # loop over the dataset multiple times
            print('Epoch:', epoch)
            self.train()
            running_loss = 0.0

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
            
            # only append to list if epoch is divisible by the steps to save
            if (epoch % epoch_step == 0):
                # train metrics

                # accuracy
                total_correct = (tot_pred == all_labels).sum().item()
                total_samples = all_labels.size(0)
                avg_train_acc = total_correct / total_samples
                train_accuracies.append(avg_train_acc)
        
                # loss
                avg_train_loss = running_loss / (i + 1)
                train_losses.append(avg_train_loss)

                # recall
                recall = recall_score(y_true = all_labels, 
                                      y_pred = tot_pred, 
                                      pos_label=0)
                train_recalls.append(recall)

                # validation metrics
                val_loss, val_acc, val_recall = self.evaluate_model(val_dataloader)

                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                val_recalls.append(val_recall)
        
        self.confusion_matrix = confusion_matrix(y_true= all_labels,
                                                 y_pred= tot_pred)

        return (train_losses, train_accuracies, train_recalls), \
                (val_losses, val_accuracies, val_recalls)


    def evaluate_model(self, dataloader):

        # tells model not to track gradients
        self.eval()

        tot_pred = torch.empty(0)
        all_labels = torch.empty(0)
        test_loss = 0.0
     
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.type(torch.float32)
            
            with torch.no_grad():
                outputs = self(inputs)
                batch_loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
        
                # Concatate batch of prediction and labels togehter
                tot_pred = torch.cat((tot_pred, predicted))
                all_labels = torch.cat((all_labels, labels))

                test_loss += batch_loss.item()

        # metrics

        # accuracy
        total_correct = (tot_pred == all_labels).sum().item()
        total_samples = all_labels.size(0)
        test_acc = total_correct / total_samples

        #loss
        avg_test_loss = test_loss / (i + 1)

        # recall
        test_recall = recall_score(y_true = all_labels, 
                                      y_pred = tot_pred, 
                                      pos_label=0)

        return avg_test_loss, test_acc, test_recall


    def create_graph(epochs, train_metric_list, val_metric_list, train_label, val_label, title):
                
        epochs_array = [i for i in range(epochs)]
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
        ax.plot(epochs_array, train_metric_list, color="orange", label=train_label, ls='dashed')
        ax.plot(epochs_array, val_metric_list, color="blue", label=val_label)
        ax.grid(alpha=0.25)
        ax.set_axis_on()
        ax.legend(loc="lower right", fontsize=16)
        ax.set_xlabel("Epochs", fontsize=16)
        ax.set_title(title, fontsize=16)

        plt.show()


