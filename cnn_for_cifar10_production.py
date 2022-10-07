

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import PIL
import os
from os import walk
import glob

tr = transforms.Compose([transforms.Resize((64,64)),
                                transforms.ToTensor()])

print(os.listdir("D:/neural networks/cifar10/dataset/cifar10/cifar10"))

for (dirpath, dirnames, filenames) in walk("D:/neural networks/cifar10/dataset/cifar10/cifar10"):
    print("Directory path: ", dirpath)
    print("Folder name: ", dirnames)
    break

test_datasets=torchvision.datasets.ImageFolder("D:/neural networks/cifar10/dataset/cifar10/cifar10/test",transform=tr)

train_datasets=torchvision.datasets.ImageFolder("D:/neural networks/cifar10/dataset/cifar10/cifar10/train",transform=tr)

batch_size=64
train_loader=torch.utils.data.DataLoader(dataset=train_datasets,
                                         batch_size=batch_size,
                                         shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=train_datasets,
                                         batch_size=batch_size,
                                         shuffle=True)

# number of classes
# K = len(set(train_dataset.targets.numpy()))
K = 10
print("number of classes:", K)

# Define the model
class CNN(nn.Module):
  def __init__(self, K):
    super(CNN, self).__init__()
#     The same model! Using the newly introduced "Flatten"
    self.model = nn.Sequential(
    nn.Conv2d(in_channels = 3, out_channels=32, kernel_size=3, stride=2),
    nn.ReLU(),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
    nn.ReLU(),
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Dropout(0.2),
    nn.Linear(128 * 7 * 7, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, K)   
)
  
  def forward(self, X):
    out = self.model(X)
    return out

model=CNN(K)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs):
  train_losses = np.zeros(epochs)
  test_losses = np.zeros(epochs)

  for it in range(epochs):
    model.train()
    t0 = datetime.now()
    train_loss = []
    for inputs, targets in train_loader:
      # move data to GPU
      inputs, targets = inputs.to(device), targets.to(device)

      # zero the parameter gradients
      optimizer.zero_grad()

      # Forward pass
      outputs = model(inputs)
      loss = criterion(outputs, targets)
        
      # Backward and optimize
      loss.backward()
      optimizer.step()

      train_loss.append(loss.item())

    # Get train loss and test loss
    train_loss = np.mean(train_loss) # a little misleading
    
    model.eval()
    test_loss = []
    for inputs, targets in test_loader:
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = model(inputs)
      loss = criterion(outputs, targets)
      test_loss.append(loss.item())
    test_loss = np.mean(test_loss)

    # Save losses
    train_losses[it] = train_loss
    test_losses[it] = test_loss
    
    dt = datetime.now() - t0
    print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, \
      Test Loss: {test_loss:.4f}, Duration: {dt}')
  
  return train_losses, test_losses

def cb():
  train_losses, test_losses = batch_gd(
      model, criterion, optimizer, train_loader, test_loader, epochs=30)

  # Plot the train loss and test loss per iteration
  plt.plot(train_losses, label='train loss')
  plt.plot(test_losses, label='test loss')
  plt.legend()
  plt.show()

  # Accuracy

  model.eval()
  n_correct = 0.
  n_total = 0.
  for inputs, targets in train_loader:
    # move data to GPU
    inputs, targets = inputs.to(device), targets.to(device)

    # Forward pass
    outputs = model(inputs)

    # Get prediction
    # torch.max returns both max and argmax
    _, predictions = torch.max(outputs, 1)
    
    # update counts
    n_correct += (predictions == targets).sum().item()
    n_total += targets.shape[0]

  train_acc = n_correct / n_total


  n_correct = 0.
  n_total = 0.
  for inputs, targets in test_loader:
    # move data to GPU
    inputs, targets = inputs.to(device), targets.to(device)

    # Forward pass
    outputs = model(inputs)

    # Get prediction
    # torch.max returns both max and argmax
    _, predictions = torch.max(outputs, 1)
    
    # update counts
    n_correct += (predictions == targets).sum().item()
    n_total += targets.shape[0]

  test_acc = n_correct / n_total
  print(f"Train acc: {train_acc*100:.4f}, Test acc: {test_acc*100:.4f}")

  img = glob.glob("D:/neural networks/cifar10/dataset/cifar10/cifar10/test/bird/0001.png")
  for image in img:
      images=PIL.Image.open(image)
      trans=transforms.ToPILImage()
      trans1=transforms.ToTensor()
      img_req = (trans1(images))
      plt.imshow(trans(trans1(images)))

  type(img_req)
  model.eval()
  img_req.shape

  tr
  rgb_im = images.convert('RGB')
  img = tr(rgb_im)
  img = img.unsqueeze(dim=0)
  img.shape
  img = img.to(device)

  item_labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
  print(model(img))
  max = torch.argmax(model(img))
  print(f'Predicted image is {item_labels[max]}')

torch.save(model.state_dict(),'cifarmodel.pt')
if __name__ =="__main__":
  cb()






