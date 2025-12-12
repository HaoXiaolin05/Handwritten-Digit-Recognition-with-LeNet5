import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset

class LeNet5(nn.Module):
  def __init__(self):
    super(LeNet5, self).__init__()

    #Convolution Layer 1
    self.conv1 = nn.Conv2d(1, 6, 5)

    #Max Pooling --> Giảm kích thước ảnh → giữ thông tin mạnh nhất
    self.pool = nn.MaxPool2d(2, 2)

    #Convolution Layer 2 --> Trích xuất đặc trưng sâu hơn
    self.conv2 = nn.Conv2d(6, 16, 5)

    #Fully Connected Layers --> Chuyển đặc trưng thành xác suất 10 chữ số (0→9)
    self.fc1 = nn.Linear(16*4*4, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  # hàm forward
  # Ảnh → Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → FC → FC → Output
  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16*4*4)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
  
if __name__ == "__main__":
  transform_base = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
  ])

  # Left: Rotate exactly -45 degrees
  transform_left = transforms.Compose([
      transforms.RandomRotation((-45, -45)), 
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
  ])

  # Right: Rotate exactly +45 degrees
  transform_right = transforms.Compose([
      transforms.RandomRotation((45, 45)),
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
  ])

  train_original = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_base)
  train_left = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_left)
  train_right = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_right)

  final_trainset = ConcatDataset([train_original, train_left, train_right])

  # Note: For testing, usually we stick to the standard original dataset 
  testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_base)

  trainloader = torch.utils.data.DataLoader(final_trainset, batch_size=64, shuffle=True)
  testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = LeNet5().to(device)

  criterion = nn.CrossEntropyLoss() # use cross-entropy for classification
  optimizer = optim.Adam(model.parameters(), lr=0.001)

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

  correct = 0
  total = 0

  with torch.no_grad():
    for images, labels in testloader:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)

      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  accuracy = 100 * correct / total
  print(f"Test Accuracy: {accuracy:.2f}%")

  torch.save(model.state_dict(), 'lenet_improve.pth')
  print("Model saved to 'lenet_improve.pth'")