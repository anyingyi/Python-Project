"""
from multiprocessing import Process,freeze_support

def process_data(fileList):
    for filepath in fileList:
        print("processing {} ...".format(filepath))

if __name__=='__main__':
    freeze_support()

    full_list=["xiao","wen","love","miao","ran"]
    n_processes=3
    length=len(full_list)/n_processes

    indices=[int(round(i*length)) for i in range(n_processes+1)]

    sublists=[full_list[indices[i]:indices[i+1]] for i in range(n_processes)]

    processes=[Process(target=process_data,args=(x,)) for x in sublists]

    for p in processes:
        p.start()
    for p in processes:
        p.join()
"""
'''
def getPos(x):
    return x[1]

heros=[("superman",99),
       ("batman",100),
       ("joker",85)]

heros.sort(key=getPos)
'''
'''
from math import cos,pi
print(cos(pi))

with open("new.txt",'r') as fread,open("age_name.txt",'w') as fwrite:
    line=fread.readline()
    while line:
        name,age=line.rstrip().split(',')
        fwrite.write("{},{}\n".format(age,name))
        print("{} is {} years old".format(name,age))
        line=fread.readline()
'''
"""
#multiprocessing
from multiprocessing import Process,freeze_support
def process_data(filelist):
    for filepath in filelist:
        print("processing {} ".format(filepath))
    print("\n")

if __name__=="__main__":
    full_list=["new.txt","xiao","age_name.txt","wen","love","miao","ran"]
    freeze_support() #only need in windows
    n_total=len(full_list)
    n_processes=3

    length=n_total/n_processes

    indices=[int(round(i*length)) for i in range(n_processes+1)]

    sublists=[full_list[indices[i]:indices[i+1]] for i in range(n_processes)]

    processes=[Process(target=process_data,args=(x,)) for x in sublists]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
        


import os

label_map={
    "cat":0,"dog":1,"bat":2
}
with open("data.txt",'w') as f:
    for root,dirs,files in os.walk('data'):
        for filename in files:
            filepath=os.sep.join([root,filename])
            dirname=root.split(os.sep)[-1]
            label=label_map[dirname]
            line='{},{}\n'.format(filepath,label)
            f.write(line)
            

import os, shutil

filepath0="data/bat/download.jpg"
filepath1="data/bat/download_rename.jpg"

os.system("mv {} {}".format(filepath0,filepath1))

dirname="data_samples"
os.system("mkdir -p {}".format(dirname))

os.system("cp {} {}".format(filepath1,dirname))


import numpy.random as random


random.seed(42)
n_tests=10000

winning_doors=random.randint(0,3,n_tests)

change_mind_wins=0

insist_wins=0

for winning_door in winning_doors:
    first_try=random.randint(0,3)

    remaining_choices=[i for i in range(3) if i != first_try]

    wrong_choices=[i for i in range(3) if i != winning_door]

    if first_try in wrong_choices:
        wrong_choices.remove(first_try)

    screened_out=random.choice(wrong_choices)
    remaining_choices.remove(screened_out)

    change_mind_try=remaining_choices[0]

    if change_mind_try== winning_door:
        change_mind_wins+=1
    if first_try==winning_door:
        insist_wins+=1

print("you win {1} out of {0} if you changed your mind\n""you win {2} out of {0} if you insist".format(n_tests,change_mind_wins,insist_wins))

import torch,torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d(1,6,3)
        self.conv2=nn.Conv2d(6,16,3)

        self.fc1=nn.Linear(16*6*6,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=x.view(-1,self.num_flat_features(x))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

    def num_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        return num_features


net=Net()
print(net)

params=list(net.parameters())
print(len(params))
print(params[0].size())

input=torch.randn(1,1,32,32)
out=net(input)
print(out)
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# torch.manual_seed(1)    # reproducible
# np.random.seed(1)

# Hyper Parameters
BATCH_SIZE = 64
LR_G = 0.0001  # learning rate for generator
LR_D = 0.0001  # learning rate for discriminator
N_IDEAS = 5  # think of this as number of ideas for generating an art work (Generator)
ART_COMPONENTS = 15  # it could be total point G can draw in the canvas
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])


# show our beautiful painting range
# plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
# plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
# plt.legend(loc='upper right')
# plt.show()


def artist_works():  # painting from the famous artist (real target)
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + (a - 1)
    paintings = torch.from_numpy(paintings).float()
    return paintings


G = nn.Sequential(  # Generator
    nn.Linear(N_IDEAS, 128),  # random ideas (could from normal distribution)
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENTS),  # making a painting from these random ideas
)

D = nn.Sequential(  # Discriminator
    nn.Linear(ART_COMPONENTS, 128),  # receive art work either from the famous artist or a newbie like G
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),  # tell the probability that the art work is made by artist
)

opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

plt.ion()  # something about continuous plotting

for step in range(10000):
    artist_paintings = artist_works()  # real painting from artist
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS, requires_grad=True)  # random ideas\n
    G_paintings = G(G_ideas)  # fake painting from G (random ideas)
    prob_artist1 = D(G_paintings)  # D try to reduce this prob
    G_loss = torch.mean(torch.log(1. - prob_artist1))
    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    prob_artist0 = D(artist_paintings)  # D try to increase this prob
    prob_artist1 = D(G_paintings.detach())  # D try to reduce this prob
    D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)  # reusing computational graph
    opt_D.step()

    if step % 50 == 0:  # plotting
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting', )
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(),
                 fontdict={'size': 13})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        plt.ylim((0, 3));
        plt.legend(loc='upper right', fontsize=10);
        plt.draw();
        plt.pause(0.01)

plt.ioff()
plt.show()

























