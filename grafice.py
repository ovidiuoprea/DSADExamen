import math

import numpy as np
from matplotlib import pyplot as plt
from seaborn import heatmap, kdeplot
from sklearn.metrics import silhouette_score


def corelograma(x):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Corelograma")
    heatmap(data=x, vmin=-1, vmax=1, cmap="bwr", annot=True, ax=ax)

def cerc_corelatii(corelatii):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Cerc corelatii")
    theta = np.arange(0, 2*math.pi, 0.01)
    ax.plot(np.sin(theta), np.cos(theta))
    ax.plot(0.7*np.sin(theta), 0.7*np.cos(theta))
    ax.scatter(corelatii[:, 0], corelatii[:, 1], c="r")

def biplot(x, y):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Biplot")
    ax.scatter(x[:, 0], x[:, 1], c="orange")
    ax.scatter(y[:, 0], y[:, 1], c="b")

def plot_distributii(scoruri, axa):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Plot distributii in axe determinante")
    ax.set_xlabel("Z" + str(axa+1))
    kdeplot(x=scoruri[:, axa], fill=True, ax=ax)

def show():
    plt.show()
