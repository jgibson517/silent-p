################################################################################
# LOGISTIC REGRESSION
################################################################################

# import torch
# import torch.nn as nn 
# import torch.optim as optim
# import torch.nn.functional as F 
# from torch.utils.data import DataLoader
# from process.data_module import CustomImageDataset
# from process.transforms import base_transforms, edges_transforms, color_transforms, both_transforms 

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

    def testing_step(self, batch):
        images, labels = batch
        out = self(images)                  
        loss = F.cross_entropy(out, labels) 
        acc = accuracy(out, labels)
        rec = recall(out, labels)
        return {'loss': loss.detach(), 'accuracy': acc.detach(), 'recall': rec.detach()}

    def testing_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['accuracy'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean() 
        batch_recs = [x['recall'] for x in outputs]
        epoch_rec = torch.stack(batch_recs).mean()
        return {'accuracy': epoch_acc.item(), 'recall': epoch_rec.item()}
    
    def epoch_print(self, epoch):
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
    outputs = [model.testing_step(batch) for batch in val_loader]
    return model.testing_epoch_end(outputs)

def fit(epochs, lr, model, train_dataloader, test_dataloader, opt=optim.SGD):
    metrics = []
    optimizer = opt(model.parameters(), lr)
    for epoch in range(epochs):
        for batch in train_dataloader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result = evaluate(model, test_dataloader)
        model.epoch_print(epoch)
        if epoch == epochs-1:
            metrics.append(result)
    return metrics