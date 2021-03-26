import torch
import torch.nn.functional  as F
import torch.nn as nn
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)

plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()


# method 1
class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.softmax(self.predict(x))
        return x


net1 = Net(2, 10, 2)
print(net1)

#method 2
net2=torch.nn.Sequential(
    nn.Linear(2,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,2)
)
print(net2)

plt.ion()
plt.show()

optimizer = torch.optim.SGD(net2.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()  # for classify problem

for t in range(100):
    out = net2(x)

    loss = loss_func(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 2 == 0:
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x[:, 0], x[:, 1], c=pred_y, s=100, lw=0, cmap="RdYlGn")
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, "accuracy=%.2f" % accuracy, fontdict={"size": 20, "color": "red"})
        plt.pause(0.1)

plt.ioff()
plt.show()
