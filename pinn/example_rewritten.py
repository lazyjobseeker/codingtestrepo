import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
plt.ion()

from torch.autograd import Variable

import pandas as pd
import torch
torch.manual_seed(42)
import torch.nn as nn
from torch.func import functional_call, grad, vmap, jacrev, jacfwd, hessian
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
plt.ion()
import tqdm
from tkinter.filedialog import asksaveasfilename

g = 9.8
mu = 0.3

def fun(t, V):
    vx, vy = V
    drag = mu*np.sqrt(vx**2+vy**2)
    return [-drag*vx, -drag*vy-g]

V_0 = [30, 10]
ts = np.linspace(0, 2, 10000)
dt = ts[1]-ts[0]

result = solve_ivp(fun, (0, 100), V_0, t_eval=ts)
sx = np.cumsum(result.y[0])*dt
sy = np.cumsum(result.y[1])*dt

ts_train = np.linspace(0, 0.2, 2000)
dt_train = ts_train[1]-ts_train[0]
result_train = solve_ivp(fun, (0, 100), V_0, t_eval=ts_train)
t_train = result_train.t
sx_train = np.cumsum(result_train.y[0])*dt_train
sy_train = np.cumsum(result_train.y[1])*dt_train

noise_x = np.random.normal(0, 0.03, len(t_train))
noise_y = np.random.normal(0, 0.02, len(t_train))

#sx_train = [x+n for x, n in zip(sx_train, noise_x)]
sy_train = [y+n for y, n in zip(sy_train, noise_y)]

plt.plot(sx, sy, c='b', alpha=0.3, linewidth=2)
plt.scatter(sx_train, sy_train, c='r', s=3)
plt.xlim(-0.2, np.max(sx)*1.1)
plt.ylim(-0.2, np.max(sy)*1.1)