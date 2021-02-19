import torch
import torch.nn as nn
import math
import numpy as np
from matplotlib import pyplot as plt

x = torch.linspace(-math.pi,math.pi,2000)

y = torch.cos(12*math.pi*x)/12*math.pi*x 

p = torch.tensor([1, 2, 3])

xx = x.unsqueeze(-1).pow(p)

model = nn.Sequential(
    nn.Linear(3,5),
    nn.Linear(5,10),
    nn.Linear(10,10),
    nn.Linear(10,10),
    nn.Linear(10,10),
    nn.Linear(10,5),
    nn.Linear(5,1),
    nn.Flatten(0,1),
    )

model2 = nn.Sequential(
    nn.Linear(3,10),
    nn.Linear(10,18),
    nn.Linear(18,15),
    nn.Linear(15,4),
    nn.Linear(4,1),
    nn.Flatten(0,1)
)

model3 = nn.Sequential(
    nn.Linear(3,190),
    nn.Linear(190,1),
    nn.Flatten(0,1)
)

loss_fn = nn.MSELoss(reduction='sum')
loss_fn2 = nn.MSELoss(reduction='sum')
loss_fn3 = nn.MSELoss(reduction='sum')

learning_rate = 1e-6

loss_all = []
loss_all2 = []
loss_all3 = []

grad_graph = []

num_epochs = 2000

for t in range(num_epochs):

    y_hat = model(xx)

    loss = loss_fn(y_hat,y)

    loss_all.append(loss)

    model.zero_grad()

    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

    grad_all = 0.0
    for p in model.parameters():
        grad = 0.0
        if p.grad is not None:
            grad = (p.grad.cpu().data.numpy() ** 2).sum()
        grad_all += grad

    grad_norm = grad_all ** 0.5
    grad_graph.append(grad_norm)


for t in range(num_epochs):

    y_hat2 = model2(xx)

    loss2 = loss_fn2(y_hat2,y)

    loss_all2.append(loss2)

    model2.zero_grad()

    loss2.backward()

    with torch.no_grad():
        for param in model2.parameters():
            param -= learning_rate * param.grad


for t in range(num_epochs):
    y_hat3 = model3(xx)

    loss3 = loss_fn3(y_hat3,y)

    loss_all3.append(loss3)

    model3.zero_grad()

    loss3.backward()

    with torch.no_grad():
        for param in model3.parameters():
            param -= learning_rate * param.grad


plt.plot(range(num_epochs),loss_all,label='model 1')
plt.plot(range(num_epochs),loss_all2,label='model 2')
plt.plot(range(num_epochs),loss_all3, label='model 3')
plt.xlabel ("epoch")
plt.ylabel("loss")
plt.title("Loss change through training")
plt.legend()

plt.show()

plt.plot(x,y,label='ground-truth')
plt.title('accuracy vs truth')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(x,y_hat.detach().numpy(),label='model 1')
plt.plot(x,y_hat2.detach().numpy(),label='model 2')
plt.plot(x,y_hat3.detach().numpy(),label='model 3')
plt.xlim([-math.pi,math.pi])
plt.legend()

plt.show()

plt.plot(range(num_epochs), grad_graph)
plt.xlabel('epoch')
plt.ylabel('grad')
plt.title('grad change through training')

plt.show()

plt.plot(range(num_epochs),loss_all,label='model 1')
plt.xlabel ("epoch")
plt.ylabel("loss")
plt.title("Loss change through training")

plt.show()
