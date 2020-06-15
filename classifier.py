import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# print(torch.cuda.get_device_name(0))


# Hyper-parameters
num_epochs = 4
batch_size = 4
learning_rate = 0.001

def load_data(data_path):
    # data loading
    train_dataset = torchvision.datasets.ImageFolder(
        root = data_path,
        transform = transforms.Compose(
    [transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     ])
    )

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                    batch_size = 1,
                                                    shuffle = True)

    return train_loader


train_loader = load_data('train')



# dataiter = iter(train_loader)
# images, labels =  dataiter.next()
# print(images.shape)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)


    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

if os.path.exists('model.pth'):
    model.load_state_dict(torch.load('model.pth'))

else:

    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # origin shape: [4, 3, 32, 32] = 4, 3, 1024
            # input_layer: 3 input channels, 6 output channels, 5 kernel size
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 2000 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    print('Training Finished')

    torch.save(model.state_dict(), 'model.pth')

image = Image.open('test1/5.jpg')
transform = transforms.Compose(
    [transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     ])

input_image = transform(image).unsqueeze(0)

_, output = torch.max(model(input_image),1)

print(input_image.shape)



