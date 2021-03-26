import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as Data
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#hyper parameters
LR=0.01
BATCH_SIZE=32
EPOCH=12

x=torch.unsqueeze(torch.linspace(-1,1,1000),dim=1)
y=x**2+0.1*torch.normal(torch.zeros(*x.size()))

plt.plot(x,y,".")
plt.show()

# minibatch
torch_dataset=Data.TensorDataset(x,y)
loader=Data.DataLoader(dataset=torch_dataset,batch_size=BATCH_SIZE,shuffle=True)


class Net(nn.Module):
    def __init__(self, n_feature=1, n_hidden=20, n_output=1):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)  # nn.Linear is full-connected network
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)  # want to keep the big range of prediction,not use activation function
        return x

# different nets
net_SGD=Net()
net_Momentum=Net()
net_RMSprop=Net()
net_Adam=Net()
nets=[net_SGD,net_Momentum,net_RMSprop,net_Adam]

opt_SGD=torch.optim.SGD(net_SGD.parameters(),lr=LR)
opt_Momentum=torch.optim.SGD(net_Momentum.parameters(),lr=LR,momentum=0.8)
opt_RMSprop=torch.optim.RMSprop(net_RMSprop.parameters(),lr=LR,alpha=0.9)
opt_Adam=torch.optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9,0.99))
optimizers=[opt_SGD,opt_Momentum,opt_RMSprop,opt_Adam]

loss_func=torch.nn.MSELoss()
losses_hit=[[],[],[],[]]

for epoch in range(EPOCH):
    for step,(batch_x,batch_y) in enumerate(loader):
        for net,opt,l_hit in zip(nets,optimizers,losses_hit):
            output=net(batch_x)
            loss=loss_func(output,batch_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            l_hit.append(loss.data.numpy())

labels=["SGD","Momentum","RMSprop","Adam"]
for i,l_hit in enumerate(losses_hit):
    plt.plot(l_hit,label=labels[i])
plt.legend(loc="best")
plt.xlabel("steps")
plt.ylabel("loss")
plt.ylim((0,0.2))
plt.show()




#optimizer=torch.optim.SGD()
