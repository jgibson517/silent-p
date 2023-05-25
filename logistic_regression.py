################################################################################
# LIBRARIES
################################################################################
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from process.data_module import CustomImageDataset
from process.transforms import base_transforms, edges_transforms, color_transforms, both_transforms 

################################################################################
# CONSTANTS
################################################################################

labels = 2
epochs = 15
image_size = 256*256
learning_rate = 0.0001

################################################################################
# GET DATA
################################################################################

# Load original x-rays and apply transformations
training_data = CustomImageDataset("data2/paths/train.csv", "data2/train/", base_transforms)
val_data = CustomImageDataset("data2/paths/val.csv", "data2/val/", base_transforms)
test_data = CustomImageDataset("data2/paths/test.csv", "data2/test/", base_transforms)

# training_data = CustomImageDataset("data2/paths/train.csv", "data2/train/", edges_transforms)
# val_data = CustomImageDataset("data2/paths/val.csv", "data2/val/", edges_transforms)
# test_data = CustomImageDataset("data2/paths/test.csv", "data2/test/", edges_transforms)

# training_data = CustomImageDataset("data2/paths/train.csv", "data2/train/", color_transforms)
# val_data = CustomImageDataset("data2/paths/val.csv", "data2/val/", color_transforms)
# test_data = CustomImageDataset("data2/paths/test.csv", "data2/test/", color_transforms)

# training_data = CustomImageDataset("data2/paths/train.csv", "data2/train/", both_transforms)
# val_data = CustomImageDataset("data2/paths/val.csv", "data2/val/", both_transforms)
# test_data = CustomImageDataset("data2/paths/test.csv", "data2/test/", both_transforms)


# # Load groups/batches of x-rays for analysis
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, num_workers=3, pin_memory=True)
val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True, num_workers=3, pin_memory=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=3, pin_memory=True)

################################################################################
# CREATE LOGISTIC REGRESSION CLASS AND HELPER FUNCTIONS
################################################################################

class LogisticRegression(nn.Module):
    """
    This class creates a logistic regression model, that will serve as a 
    baseline to compare the performance of our CNN model.
    """
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(image_size, labels)

    def forward(self, xb):
        xb = xb.view(xb.shape[0],-1)
        xb = xb.type(torch.float32)
        out = self.linear(xb)
        return out
    
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                  
        loss = F.cross_entropy(out, labels) 
        acc = accuracy(out, labels)
        rec = recall(out, labels)
        return {'loss': loss.detach(), 'accuracy': acc.detach(), 'recall': rec.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['accuracy'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        batch_recs = [x['recall'] for x in outputs]
        epoch_rec = torch.stack(batch_recs).mean()      # Combine recalls
        return {'loss': epoch_loss.item(), 'accuracy': epoch_acc.item(), 'recall': epoch_rec.item()}
    
    def epoch_end(self, epoch):
        print("Epoch {}".format(epoch))    

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds==labels).item() / len(preds))

def recall(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    TP = torch.sum((preds == labels) & (labels == 1)).item()
    total_positive = torch.sum(labels == 1).item()
    return torch.tensor(TP / total_positive)

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=optim.SGD):
    metrics = []
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
        model.epoch_end(epoch)
        if epoch == epochs-1:
            metrics.append(result)
    return metrics

################################################################################
# INITIALIZE MODEL
################################################################################

model = LogisticRegression()

for images, _ in train_dataloader:
    output = model(images.type(torch.float32))

fit(epochs, learning_rate, model, train_dataloader, test_dataloader)
