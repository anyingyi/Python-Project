import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # transform 1 dim to 2 dim
y = x.pow(2) + 0.2 * torch.rand(x.size())

# torch.unsqueeze: [1,2,3,4] -> [[1,2,3,4]]

plt.scatter(x, y)
plt.show()


class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)  # nn.Linear is full-connected network
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)  # want to keep the big range of prediction,not use activation function
        return x


net = Net(1, 10, 1)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)   # SGD:stochastic gradient descent
loss_func = torch.nn.MSELoss()  # MSEloss: mean square of (predict-label)

for t in range(100):
    prediction = net(x)

    loss = loss_func(prediction, y)

    optimizer.zero_grad()  # set gradient to zero, different calculate method
    loss.backward()
    optimizer.step()  # w=w+lr*gradient

    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
