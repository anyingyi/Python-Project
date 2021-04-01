import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#hyper parameters
EPOCH=1
BATCH_SIZE=64
TIME_STEP=28        # run time step/image height
INPUT_SIZE=28       # run input size/image width
LR=0.01
DOWNLOAD_MNIST=False    # set to True if not download the data

train_data=dsets.MNIST(
    root="./mnist",
    train=True,         # if True, get train data;else get test data
    transform=transforms.ToTensor(),    # normalize value:(0,255)->(0,1)
    download=DOWNLOAD_MNIST
)
train_loader=torch.utils.data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)

test_data=dsets.MNIST(root="./mnist",train=False,transform=transforms.ToTensor())
test_x=test_data.test_data.type(torch.FloatTensor)[:2000]/255
test_y=test_data.test_labels.numpy()[:2000]


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn=nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True,       # if True:(batch,time step,input); else:(time step,batch,input)
        )
        self.out=nn.Linear(64,10)   # Linear:full connected network

    def forward(self,x):
        r_out,(h_n,h_c)=self.rnn(x,None)    # h_n:data of shor time memory; h_c:data of long time memory
        out=self.out(r_out[:,-1,:])     # (batch,time step,input)
        return out


rnn=RNN()
print(rnn)

optimizer=torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss()     # transform result to label of one-hot automaticly

for epoch in range(EPOCH):
    for step,(b_x,b_y) in enumerate(train_loader):
        b_x=b_x.view(-1,28,28)      # reshape b_x to (batch,time_step,input)

        output=rnn(b_x)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%50==0:
            test_output=rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print("Epoch:", epoch, "| train loss:%.4f" % loss.data.numpy(),
                  "| test accuracy:%.4f" % accuracy)

# print 10 predictions from test data
test_output=rnn(test_x[:10].view(-1,28,28))
pred_y=torch.max(test_output,1)[1].data.numpy().squeeze()
print(pred_y,"prediction number")
print(test_y[:10],"real number")


