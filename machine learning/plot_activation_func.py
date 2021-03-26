import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#fake data
x=torch.linspace(-5,5,200)
x_np=x.numpy()

y_relu=F.relu(x).numpy()
y_sigmoid=torch.sigmoid(x)
y_tanh=torch.tanh(x)
y_softplus=F.softmax(x).numpy()

plt.figure()
plt.subplot(221)
plt.plot(x_np,y_relu,c="red",label="relu")
plt.ylim((-1,5))
plt.legend(loc="best")

plt.subplot(222)
plt.plot(x_np,y_sigmoid,c="red",label="sigmoid")
plt.xlim((-2,2))
plt.ylim((-2,2))
plt.legend(loc="best")

plt.subplot(223)
plt.plot(x_np,y_tanh,c="red",label="tanh")
plt.ylim((-2,2))
plt.legend(loc="best")

plt.subplot(224)
plt.plot(x_np,y_softplus,c="red",label="softplus")
plt.ylim((-1,5))
plt.legend(loc="best")
plt.show()