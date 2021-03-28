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
import matplotlib.pyplot as plt
import numpy.random as random
import numpy as np
from mpl_toolkits.mplot3d import Axes3D     # nessary for 3D plot

np.random.seed(42)
n_graids=51
c=n_graids/2
nf=2

x=np.linspace(0,1,n_graids)
y=np.linspace(0,1,n_graids)

X,Y=np.meshgrid(x,y)

spectrum=np.zeros((n_graids,n_graids),dtype=np.complex)

noise=[np.complex(x,y) for x,y in np.random.uniform((-1,1,((2*nf+1)**2/2,2)))]

noisy_block=np.concatenate((noise,[0j],np.conjugate((noise[::-1]))))
spectrum[c-nf:c+nf+1,c-nf:c+nf+1]=noisy_block.reshape((2*nf+1),2*nf+1)

Z=np.real(np.fft.ifft2(np.fft.ifftshift(spectrum)))

fig=plt.figure("3D surface and wire")
ax=fig.add_subplot(1,2,1,projection="3d")

ax.plot_surface(X,Y,Z,alpha=0.7,cmap="jet",rstride=1,cstride=1,lw=0)

ax=fig.add_subplot(1,2,2,projection="3d")
ax.plot_wireframe(X,Y,Z,rstride=3,cstride=3,lw=0.5)
plt.show()























