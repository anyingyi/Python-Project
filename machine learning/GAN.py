import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#torch.manual_seed(1)
#np.random.seed(1)

# hyper parameters
BATCH_SIZE = 64
LR_G = 0.0001  # learning rate for generator
LR_D = 0.0001  # learning rate for discriminator
N_IDEAS = 5  # think of this as number of ideas of generator
ART_COMPONENTS = 15
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])


# show beautiful painting range
# plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c="#74BCFF", lw=3, label="up limit")
# plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c="#FF9359", lw=3, label="down limit")
# plt.legend(loc="upper right")
# plt.show()


def artist_works():
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + (a - 1)
    paintings = torch.from_numpy(paintings).float()
    return paintings


G = nn.Sequential(      # generator
    nn.Linear(N_IDEAS, 128),
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENTS)
)

D = nn.Sequential(      # discriminator
    nn.Linear(ART_COMPONENTS, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()        # transform data to (0,1)
)

opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)
opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)

plt.ion()  # something about continuous plotting
# plt.show()

for step in range(10000):
    artist_paintings = artist_works()
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS)
    G_paintings = G(G_ideas)
    prob_generator=D(G_paintings)
    G_loss=torch.mean(torch.log(1.-prob_generator))
    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    prob_artist = D(artist_paintings)
    # prob_generator = D(G_paintings)
    prob_generator2 = D(G_paintings.detach())

    # the calculation in gradient is minimize
    D_loss = -torch.mean(torch.log(prob_artist) + torch.log(1. - prob_generator2))
    # G_loss = torch.mean(torch.log(1. - prob_generator))

    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)  # reusing computational graph
    opt_D.step()







    if step % 50 == 0:
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], lw=3, label="generated painting")
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c="#74BCFF", lw=3, label="up limit")
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c="#FF9359", lw=3, label="down limit")
        plt.ylim((0, 3))
        plt.legend(loc="upper right", fontsize=10)
        plt.draw()
        plt.pause(0.01)

plt.ioff()
plt.show()
