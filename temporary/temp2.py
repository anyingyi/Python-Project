# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 15:12:16 2020

@author: Anyingyi
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#mpl.rcParams["xtick.labelsize"]=24
#mpl.rcParams["ytick.labelsize"]=24

speed_map={
    "dog":(48,"#7199cf"),
    "cat":(45,"#4fc4aa"),
    "cheetah":(120,"#e1a7a2")
}

fig=plt.figure("bar chart & pie chart")

ax=fig.add_subplot(121)
ax.set_title("running speed-bar chart")

xticks=np.arange(3)

bar_width=0.5

animals=speed_map.keys()

speeds=[x[0] for x in speed_map.values()]

colors=[x[1] for x in speed_map.values()]

bars=ax.bar(xticks,speeds,width=bar_width,edgecolor="none")

ax.set_ylabel("speed")
#ax.set_xlabel()

ax=fig.add_subplot(122)
ax.set_title("running speed-pie chart")

labels=["{}\n{} km/h".format(a,s) for a,s in zip(animals,speeds)]
ax.pie(speeds,labels=labels,colors=colors)

plt.axis("equal")
plt.show()



