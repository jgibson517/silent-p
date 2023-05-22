################################################################################
# LIBRARIES
################################################################################

import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 
import torchvision.transforms as T
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
# from process.model import CustomNeuralNetwork   #internal
from process.data_module import CustomImageDataset  #internal
from process.transforms import base_transforms, edges_transforms, color_transforms, both_transforms 
from process.load import collect_image_files

################################################################################
# CONSTANTS
################################################################################

epochs = 15
num_classes = 2
#batch_size = 64
image_size = 256*256
learning_rate = 0.0001
input_size = image_size

################################################################################
# GET DATA
################################################################################

# Collect images
collect_image_files('test')
collect_image_files('train')
collect_image_files('val')

# Load original x-rays and apply transformations
training_data = CustomImageDataset("data2/paths/train.csv", "data2/train/", base_transforms)
val_data = CustomImageDataset("data2/paths/val.csv", "data2/val/", base_transforms)
test_data = CustomImageDataset("data2/paths/test.csv", "data2/test/", base_transforms)

# Load groups/batches of x-rays for analysis
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Describe dataset
print(len(training_data))
img_tensor, label = training_data[0]
print(img_tensor.shape, label)

################################################################################
# CREATE LOGISTIC REGRESSION CLASS AND HELPER FUNCTIONS
################################################################################

class LogisticRegression(torch.nn.Module):
    """
    """
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        xb = xb.view(xb.shape[0],-1)
        out = self.linear(xb)
        return out
    
    def train_step(self, batch):
        images, labels = batch
        out = self(images)                  
        loss = F.cross_entropy(out, labels)
        return loss

    def val_step(self, batch):
        images, labels = batch
        out = self(images)                  
        loss = F.cross_entropy(out, labels) 
        acc = accuracy(out, labels)
        return loss

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds==labels).item() / len(preds))

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

################################################################################
# INITIALIZE MODEL
################################################################################

model = LogisticRegression()

for images, labels in train_dataloader:
    print("images.shape: " , images.shape)
    print("img_tensor.shape: ", img_tensor.shape)
    output = model(images)                                          #ERROR MESSAGE!!!
    # break;
print("output.shape: ", output.shape)
print("output: ", output[:3].data)

#Probabilities
probs = F.softmax(output, dim=1)
print("Probability: \n" ,probs[122:126].data)

#Predictions
maxprob, preds = torch.max(probs, dim=1)
print(preds)
print(maxprob)

#Labels
labels

#Accuracy
accuracy(output, labels)

#Loss
loss_fn = F.cross_entropy
loss = loss_fn(output, labels) 
loss

#Evaluate model
evaluate(model, val_dataloader)
history = fit(5, learning_rate, model, training_data, val_dataloader)
accuracies = [r['val_acc'] for r in history]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs')


################################################################################
# END
################################################################################

# model = LogisticRegression(input_dim, output_dim)

# # 4. Loss function + optimizer
# criterion = torch.nn.BCELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# # Train model
# X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test)
# y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)

# losses_train = []
# losses_test = []
# Iterations = []
# iter = 0
# for epoch in tqdm(range(int(epochs)),desc='Training Epochs'):
#     x = X_train
#     labels = y_train
#     optimizer.zero_grad() # Setting our stored gradients equal to zero
#     outputs = model(X_train)
#     loss = criterion(torch.squeeze(outputs), labels) # [200,1] -squeeze-> [200]
    
#     loss.backward() # Computes the gradient of the given tensor w.r.t. graph leaves 
    
#     optimizer.step() # Updates weights and biases with the optimizer (SGD)
    
#     iter+=1
#     if iter%10000==0:
#         # calculate Accuracy
#         with torch.no_grad():
#             # Calculating the loss and accuracy for the test dataset
#             correct_test = 0
#             total_test = 0
#             outputs_test = torch.squeeze(model(X_test))
#             loss_test = criterion(outputs_test, y_test)
            
#             predicted_test = outputs_test.round().detach().numpy()
#             total_test += y_test.size(0)
#             correct_test += np.sum(predicted_test == y_test.detach().numpy())
#             accuracy_test = 100 * correct_test/total_test
#             losses_test.append(loss_test.item())
            
#             # Calculating the loss and accuracy for the train dataset
#             total = 0
#             correct = 0
#             total += y_train.size(0)
#             correct += np.sum(torch.squeeze(outputs).round().detach().numpy() == y_train.detach().numpy())
#             accuracy = 100 * correct/total
#             losses_train.append(loss.item())
#             Iterations.append(iter)
            
#             print(f"Iteration: {iter}. \nTest - Loss: {loss_test.item()}. Accuracy: {accuracy_test}")
#             print(f"Train -  Loss: {loss.item()}. Accuracy: {accuracy}\n")

#Hyperparameters
# batch_size = 64
# image_size = 256*256
# learning_rate = 0.0001

# #Other constants
# input_size = image_size
# num_classes = 2
# epochs = 15
# #input_dim = 2 # Two inputs x1 and x2 
# #output_dim = 1 # Single binary output 

# #Datasets
# train_df = DataLoader
# val_df = 
# test_df = 
