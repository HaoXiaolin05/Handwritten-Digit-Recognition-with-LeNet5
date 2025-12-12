import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# === LeNet-5 GỐC: KHÔNG augmentation, KHÔNG normalize ===
class LeNet5(nn.Module):
  def __init__(self):
    super(LeNet5, self).__init__()

    self.conv1 = nn.Conv2d(1, 6, 5)
    self.pool = nn.AvgPool2d(2, 2)   # AvgPool đúng bản gốc
    self.conv2 = nn.Conv2d(6, 16, 5)

    self.fc1 = nn.Linear(16*4*4, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(F.tanh(self.conv1(x)))   # Tanh đúng bản gốc
    x = self.pool(F.tanh(self.conv2(x)))
    x = x.view(-1, 16*4*4)
    x = F.tanh(self.fc1(x))
    x = F.tanh(self.fc2(x))
    x = self.fc3(x)
    return x
  

if __name__ == "__main__":
  transform = transforms.ToTensor()

  trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
  testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)

  trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
  testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

  correct = 0
  total = 0

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = LeNet5().to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.01)

  for epoch in range(10):
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss:.3f}")

  print("Training completed!")

  with torch.no_grad():
    for images, labels in testloader:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)

      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  accuracy = 100 * correct / total
  print(f"Baseline LeNet-5 Accuracy: {accuracy:.2f}%")

  torch.save(model.state_dict(), 'lenet_original.pth')
  print("Model saved to 'lenet_original.pth'")