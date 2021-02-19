import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import random

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

for i in range(len(trainset.targets)):
    trainset.targets[i] = random.randint(0,9)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True,num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False,num_workers=2)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

num_epochs = 10

training_loss = []

training_accuracy = []

test_accuracy = []

correct = 0

total = 0

for epoch in range(num_epochs):

    running_loss=0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch +1, i+1, running_loss/2000))
            running_loss = 0.0

        grad_all = 0.0
        for p in net.parameters():
            grad = 0.0
            if p.grad is not None:
                grad = (p.grad.cpu().data.numpy() ** 2).sum()
            grad_all += grad

    training_loss.append(loss)

    _,predicted = torch.max(outputs.data,1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    training_accuracy.append(correct/total)

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _,predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy.append(correct/total)


print('done')

plt.plot(range(num_epochs),training_accuracy,label='train accuracy')
plt.plot(range(num_epochs),test_accuracy,label='test accuracy')
plt.xlabel ("epoch")
plt.ylabel("accuracy")
plt.title("model accuracy through training")
plt.legend()

plt.show()

plt.plot(range(num_epochs),training_loss)
plt.xlabel ("epoch")
plt.ylabel("loss")
plt.title("Loss change through training")

plt.show()