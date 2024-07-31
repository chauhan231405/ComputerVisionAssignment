import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.edsr import EDSR
from utils.dataset import CustomDataset

# Initialize data transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Initialize dataset and dataloader
dataset = CustomDataset(image_folder=r'/config/workspace/data/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize the model, loss function, optimizer, and learning rate scheduler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EDSR().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        targets = nn.functional.interpolate(targets, size=outputs.shape[2:])
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(dataset)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    scheduler.step()

# Save the model weights
os.makedirs(r'/config/workspace/models', exist_ok=True)
torch.save(model.state_dict(), r'/config/workspace/models/trained_model.pth')
print('Model weights saved to /config/workspace/models')
